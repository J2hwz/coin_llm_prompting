import json
import logging
import re
import traceback
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
from minigrid.minigrid_env import MiniGridEnv
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential

from coinenv.agents.agent_abc import Agent
from coinenv.agents.llm_templates import ActionResponse, ActionWithNoteResponse, Message
from coinenv.datatypes import Action
from coinenv.environment_generator.wrappers.text_obs_wrapper import (
    FogOfWarTextWrapper,
    FullObservabilityTextWrapper,
)
from coinenv.llm_interface import BaseLLMInterface

logger = logging.getLogger(__name__)


def _read_coins_collected(env) -> int | None:
    """Read the cumulative coin collection count from the environment.

    Returns int for coin envs (0/1 for CoinNavigationEnv, 0/1/2 for TwoCoinNavigationEnv),
    or None for environments with no coin mechanics.
    """
    if hasattr(env, "coins_collected"):
        return env.coins_collected
    if hasattr(env, "coin_collected"):
        return int(env.coin_collected)
    return None


class LLMAgent(Agent, BaseLLMInterface):
    """An agent that uses a Language Model to select actions in a MiniGrid environment."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        name: Optional[str] = None,
        template_path: Optional[Path] = None,
        response_format: BaseModel = ActionResponse,
        track_history: bool = False,
    ) -> None:
        """
        Args:
            model_name: Identifier understood by ``litellm`` (e.g. ``"gpt-4"``).
            temperature: Temperature for the model (keep at 0.0 for consistent action selection).
            name: Optional name for the agent. Defaults to class name.
            track_history: If True, record (position, action, coins_collected) per step and
                inject the history into each prompt. Requires a template with a
                ``{% if history %}`` block to render it.
        """
        if template_path is None:
            template_path = (
                Path(__file__).parent.parent
                / "templates"
                / "grid_full_observability.j2"
            )

        Agent.__init__(self, name)
        BaseLLMInterface.__init__(
            self,
            model_name=model_name,
            temperature=temperature,
            template_path=template_path,
        )
        # Cost tracking
        self.total_cost = 0.0
        self.call_count = 0
        self.response_format = response_format
        self.track_history = track_history
        self.history: list[tuple] = []

    def add_to_history(
        self,
        action: "Action | int",
        position: Tuple[int, int],
        coins_collected: int | None = None,
    ) -> None:
        """Append a (position, action, coins_collected) entry to the step history.

        Accepts either an Action enum or a raw int action code (invalid actions
        with value < 0 are silently skipped). coins_collected is None for non-coin
        environments, 0/1 for CoinNavigationEnv, and 0/1/2 for TwoCoinNavigationEnv.
        """
        if isinstance(action, int):
            if action < 0:
                return
            action_str = Action(action).to_str()
        else:
            action_str = action.to_str()
        if isinstance(position[0], np.int64):
            new_position = (int(position[1].item()), int(position[0].item()))
        else:
            new_position = (position[1], position[0])
        self.history.append((new_position, action_str, coins_collected))

    def _get_agent_pos(self, env: MiniGridEnv | FogOfWarTextWrapper) -> Any:
        base_env = getattr(env, "unwrapped", env)
        return (
            tuple(base_env.agent_pos) if hasattr(base_env, "agent_pos") else "unknown"
        )

    def _build_request_params(
        self, return_logprobs: bool, top_logprobs: int
    ) -> tuple[Optional[BaseModel], dict]:
        extra_kwargs: dict[str, Any] = {}
        response_format: Optional[BaseModel] = self.response_format
        extra_kwargs["reasoning_effort"] = "low"
        if "qwen" in self.model_name and "fireworks" in self.model_name:
            extra_kwargs["reasoning_effort"] = 2048
        extra_kwargs["allowed_openai_params"] = [
            "logprobs",
            "top_logprobs",
            "reasoning_effort",
        ]
        response_format = None
        if return_logprobs and top_logprobs > 0:
            extra_kwargs["logprobs"] = True
            extra_kwargs["top_logprobs"] = top_logprobs
            response_format = None
        return response_format, extra_kwargs

    def _finalize_cost_and_logprobs(
        self, cost: float, raw_response: Any, return_logprobs: bool
    ) -> Optional[list[dict]]:
        # Track costs
        self.total_cost += cost
        self.call_count += 1

        # Capture logprobs if requested
        if return_logprobs:
            logprobs_raw = raw_response.choices[0].logprobs.content
            # for some models, the logprobs are in the top_logprobs field instead of the content field
            if logprobs_raw is None:
                assert raw_response.choices[0].logprobs.top_logprobs is not None
                logprobs_raw = raw_response.choices[0].logprobs.top_logprobs
                return self._serialize_logprobs_old(logprobs_raw)
            return self._serialize_logprobs(logprobs_raw)
        return None

    def _build_base_metadata(
        self, action_value: int, cost: float, logprobs_serialized: Optional[list[dict]]
    ) -> dict:
        return {
            "agents_name": self.name,
            "llm_response": action_value,
            "call_cost": cost,
            "total_cost": self.total_cost,
            "call_count": self.call_count,
            "logprobs": logprobs_serialized,
        }

    def _try_to_clean_response(self, response: Any) -> Any:
        """Try to clean the response by removing any trailing commas and other invalid JSON.

        Also attempts to extract a JSON object from the end of the response,
        since some models return extra text before the JSON output.
        """

        if isinstance(response, str):
            # Remove any trailing commas before closing objects/arrays
            response = re.sub(r",(\s*[}\]])", r"\1", response)

            # Try to extract JSON object from the end of the response
            # Some models output text before the JSON, so we look for the last { and }
            last_brace = response.rfind("}")
            if last_brace != -1:
                # Find the matching opening brace by counting braces from the end
                brace_count = 0
                start_pos = None
                for i in range(last_brace, -1, -1):
                    if response[i] == "}":
                        brace_count += 1
                    elif response[i] == "{":
                        brace_count -= 1
                        if brace_count == 0:
                            start_pos = i
                            break

                if start_pos is not None and start_pos > 0:
                    # There's text before the JSON object, try to extract just the JSON
                    potential_json = response[start_pos : last_brace + 1]
                    try:
                        # Verify it's valid JSON before using it
                        json.loads(potential_json)
                        response = potential_json
                    except json.JSONDecodeError:
                        # Keep the original cleaned response if extraction fails
                        pass
        return response

    def _request_action_via_prompt(
        self, prompt: str, return_logprobs: bool, top_logprobs: int
    ) -> tuple[Any, float, Any, Optional[list[dict]]]:
        response_format, extra_kwargs = self._build_request_params(
            return_logprobs, top_logprobs
        )
        response, cost, raw_response = self._make_completion_request(
            prompt, response_format=response_format, **extra_kwargs
        )
        if response_format is None:
            response = self._try_to_clean_response(response)
            action_response = self.response_format.model_validate_json(response)
        else:
            action_response = response
        logprobs_serialized = self._finalize_cost_and_logprobs(
            cost, raw_response, return_logprobs
        )
        return action_response, cost, raw_response, logprobs_serialized

    def _serialize_logprobs(self, logprobs_obj: Any) -> Optional[list[dict]]:
        """Convert model logprobs object(s) to JSON-serializable dictionaries.

        The LiteLLM/OpenAI responses expose rich objects like ChatCompletionTokenLogprob
        and TopLogprob. Those are not JSON-serializable out of the box. This helper
        extracts the relevant fields into plain Python types.
        """
        if logprobs_obj is None:
            return None

        def serialize_top(top: Any) -> dict:
            return {
                "token": getattr(top, "token", None),
                "bytes": getattr(top, "bytes", None),
                "logprob": getattr(top, "logprob", None),
                "token_id": getattr(top, "token_id", None),
            }

        def serialize_token_logprob(item: Any) -> dict:
            top_list = getattr(item, "top_logprobs", None) or []
            return {
                "token": getattr(item, "token", None),
                "bytes": getattr(item, "bytes", None),
                "logprob": getattr(item, "logprob", None),
                "token_id": getattr(item, "token_id", None),
                "text_offset": getattr(item, "text_offset", None),
                "top_logprobs": [serialize_top(t) for t in top_list],
            }

        try:
            return [serialize_token_logprob(x) for x in list(logprobs_obj)]
        except Exception:
            # Fall back: if it's already a list[dict] or otherwise JSON-ready, return as-is
            return logprobs_obj  # type: ignore[return-value]

    def _serialize_logprobs_old(self, logprobs_obj: Any) -> Optional[list[dict]]:
        """Convert old-format logprobs (list of token->logprob dicts) to JSON-serializable format.

        The old format is a list of dictionaries where each dict maps token strings
        to their log probabilities, e.g.:
            [{'{\n': -0.019, '{': -4.03, ...}, {' ': 0, '   ': -35, ...}, ...]

        This converts to the same output structure as _serialize_logprobs.
        """
        if logprobs_obj is None:
            return None

        def serialize_position(token_logprob_dict: dict) -> dict:
            # Sort by logprob descending to get the top token first
            sorted_tokens = sorted(
                token_logprob_dict.items(), key=lambda x: x[1], reverse=True
            )
            top_token, top_logprob = sorted_tokens[0] if sorted_tokens else (None, None)

            top_logprobs_list = [
                {
                    "token": token,
                    "bytes": None,
                    "logprob": logprob,
                    "token_id": None,
                }
                for token, logprob in sorted_tokens
            ]

            return {
                "token": top_token,
                "bytes": None,
                "logprob": top_logprob,
                "token_id": None,
                "text_offset": None,
                "top_logprobs": top_logprobs_list,
            }

        try:
            return [serialize_position(pos) for pos in logprobs_obj]
        except Exception:
            # Fall back: if it's already JSON-ready, return as-is
            return logprobs_obj  # type: ignore[return-value]

    def select_action(
        self,
        env: MiniGridEnv,
        return_logprobs: bool = False,
        top_logprobs: int = 20,
        **kwargs: Any,
    ) -> Tuple[int, dict]:
        """
        Select an action using the LLM based on the current environment state.
        """
        try:
            return self._select_action(env, return_logprobs, top_logprobs, **kwargs)
        except Exception as e:
            logger.error(
                f"Error getting action for position {env.agent_pos} from LLM after retrying: {e}"
            )
            return -1, {"agents_name": self.name}

    # We need to retry if the model response is not valid json
    @retry(
        stop=stop_after_attempt(3),
        # 1-10 seconds between attempts, to help avoid rate limiting
        wait=wait_random_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _select_action(
        self,
        env: MiniGridEnv,
        return_logprobs: bool = False,
        top_logprobs: int = 20,
        **kwargs: Any,
    ) -> Tuple[int, dict]:
        """
        Select an action using the LLM based on the current environment state.

        Args:
            env: The environment to interact with
            return_logprobs: Whether to return logprobs
            top_logprobs: Number of top logprobs to return (default: 5)
            **kwargs: Additional arguments (ignored by this agent)

        Returns:
            The selected action and a dictionary with related metadata.
        """
        base_env = getattr(env, "unwrapped", env)
        agent_pos = (
            tuple(base_env.agent_pos) if hasattr(base_env, "agent_pos") else "unknown"
        )

        logger.info(
            f"Getting action from LLM model {self.model_name} for position {agent_pos}"
        )
        prompt = self._generate_action_query_prompt(env)

        try:
            action_response, cost, raw_response, logprobs_serialized = (
                self._request_action_via_prompt(prompt, return_logprobs, top_logprobs)
            )

            logger.info(
                f"LLM selected action {action_response.action}, cost: ${cost:.6f}, total: ${self.total_cost:.6f}"
            )
            meta = self._build_base_metadata(
                action_response.action.value, cost, logprobs_serialized
            )
            if self.track_history:
                coins = _read_coins_collected(base_env)
                self.add_to_history(action_response.action, agent_pos, coins)
                meta["history"] = self.history.copy()
            return action_response.action.value, meta

        except Exception as e:
            logger.error(
                f"Error getting action from LLM: {e}\n{traceback.format_exc()}"
            )
            raise

    def _get_text_observation(self, env: MiniGridEnv) -> str:
        """Get text observation of the environment using existing wrapper."""
        # Wrap env temporarily to get text observation
        text_env = FullObservabilityTextWrapper(env)
        return text_env.observation(None)

    def _generate_action_query_prompt(self, env: MiniGridEnv) -> str:
        """Generate a prompt for the LLM to select an action."""
        # Get text observation using existing wrapper
        obs_text = self._get_text_observation(env)

        # Check if agent is carrying a key (only relevant for templates that use it)
        base_env = getattr(env, "unwrapped", env)
        carrying_key = False
        if hasattr(base_env, "carrying") and base_env.carrying is not None:
            carrying_key = base_env.carrying.type == "key"

        prompt = self.render_template(
            grid_state=obs_text,
            carrying_key=carrying_key,
            history=self.history,
        )
        return prompt

    def reset(self) -> None:
        """Reset the agent's internal state and cost tracking."""
        super().reset()
        self.total_cost = 0.0
        self.call_count = 0
        self.history = []

    def get_cost_summary(self) -> dict:
        """Get a summary of costs incurred by this agent.

        Returns:
            Dictionary with total_cost, call_count, and avg_cost_per_call
        """
        avg_cost = self.total_cost / self.call_count if self.call_count > 0 else 0.0
        return {
            "total_cost": self.total_cost,
            "call_count": self.call_count,
            "avg_cost_per_call": avg_cost,
        }


class PartiallyObservableLLMAgent(LLMAgent):
    """An agent that uses a Language Model to select actions in a MiniGrid environment with partial observability."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        name: Optional[str] = None,
        template_path: Optional[Path] = None,
        response_format: BaseModel = ActionResponse,
    ) -> None:
        if template_path is None:
            template_path = (
                Path(__file__).parent.parent
                / "templates"
                / "grid_partial_observability.j2"
            )

        super().__init__(model_name, temperature, name, template_path, response_format)

        self.history = []

    def _generate_action_query_prompt(self, env: FogOfWarTextWrapper) -> str:
        """Generate a prompt for the LLM to select an action."""
        # Get text observation using existing wrapper
        obs_text = self._get_text_observation(env)

        prompt = self.render_template(
            grid_state=obs_text,
            history=self.history,
        )
        return prompt

    def _get_text_observation(self, env: FogOfWarTextWrapper) -> str:
        """Get text observation of the environment using existing wrapper."""
        return env.observation(None)

    def add_to_history(self, action: Action, position: Tuple[int, int]) -> None:
        """Add an action and position to the history."""
        if isinstance(position[0], np.int64):
            new_position = (int(position[1].item()), int(position[0].item()))
        else:
            new_position = (position[1], position[0])
        self.history.append((new_position, action.to_str()))

    def select_action(
        self,
        env: FogOfWarTextWrapper,
        return_logprobs: bool = False,
        top_logprobs: int = 20,
        **kwargs: Any,
    ) -> Tuple[int, dict]:
        """
        Select an action using the LLM based on the current environment state.
        """
        try:
            return self._select_action(env, return_logprobs, top_logprobs, **kwargs)
        except Exception as e:
            logger.error(
                f"Error getting action for position {env.agent_pos} from LLM after retrying: {e}"
            )
            return -1, {"agents_name": self.name}

    # We need to retry if the model response is not valid json
    @retry(
        stop=stop_after_attempt(3),
        # 1-10 seconds between attempts, to help avoid rate limiting
        wait=wait_random_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _select_action(
        self,
        env: FogOfWarTextWrapper,
        return_logprobs: bool = False,
        top_logprobs: int = 20,
        **kwargs: Any,
    ) -> Tuple[int, dict]:
        """
        Select an action using the LLM based on the current environment state.

        Args:
            env: The environment to interact with
            return_logprobs: Whether to return logprobs
            top_logprobs: Number of top logprobs to return (default: 5)
            **kwargs: Additional arguments (ignored by this agent)

        Returns:
            The selected action and a dictionary with related metadata.
        """
        base_env = getattr(env, "unwrapped", env)
        agent_pos = (
            tuple(base_env.agent_pos) if hasattr(base_env, "agent_pos") else "unknown"
        )

        logger.info(
            f"Getting action from LLM model {self.model_name} for position {agent_pos}"
        )
        prompt = self._generate_action_query_prompt(env)

        try:
            action_response, cost, raw_response, logprobs_serialized = (
                self._request_action_via_prompt(prompt, return_logprobs, top_logprobs)
            )

            logger.info(
                f"LLM selected action {action_response.action}, cost: ${cost:.6f}, total: ${self.total_cost:.6f}"
            )
            self.add_to_history(action_response.action, agent_pos)
            meta = self._build_base_metadata(
                action_response.action.value, cost, logprobs_serialized
            )
            meta["history"] = self.history.copy()
            return action_response.action.value, meta

        except Exception as e:
            logger.error(
                f"Error getting action from LLM: {e}\n{traceback.format_exc()}"
            )
            raise

    def reset(self) -> None:
        """Reset the agent's internal state and cost tracking."""
        super().reset()
        self.total_cost = 0.0
        self.call_count = 0
        self.history = []


class PartiallyObservableWithNoteLLMAgent(LLMAgent):
    """An agent that uses a Language Model to select actions in a MiniGrid environment with partial observability."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        name: Optional[str] = None,
        template_path: Optional[Path] = None,
        response_format: BaseModel = ActionWithNoteResponse,
    ) -> None:
        if template_path is None:
            template_path = (
                Path(__file__).parent.parent
                / "templates"
                / "grid_partial_observability_with_note.j2"
            )

        super().__init__(model_name, temperature, name, template_path, response_format)

        self.current_note = ""

    def update_note(self, note: str) -> None:
        """Update the current note."""
        if note == "KEEP":
            return
        if note == "N/A":
            self.current_note = ""
            return
        self.current_note = note

    def _generate_action_query_prompt(self, env: FogOfWarTextWrapper) -> str:
        """Generate a prompt for the LLM to select an action."""
        # Get text observation using existing wrapper
        obs_text = self._get_text_observation(env)

        prompt = self.render_template(
            grid_state=obs_text,
            note=self.current_note,
        )
        return prompt

    def _get_text_observation(self, env: FogOfWarTextWrapper) -> str:
        """Get text observation of the environment using existing wrapper."""
        return env.observation(None)

    def select_action(
        self,
        env: FogOfWarTextWrapper,
        return_logprobs: bool = False,
        top_logprobs: int = 20,
        **kwargs: Any,
    ) -> Tuple[int, str, dict]:
        """
        Select an action using the LLM based on the current environment state.
        """
        try:
            return self._select_action(env, return_logprobs, top_logprobs, **kwargs)
        except Exception as e:
            logger.error(
                f"Error getting action for position {env.agent_pos} from LLM after retrying: {e}"
            )
            return -1, "", {"agents_name": self.name}

    # We need to retry if the model response is not valid json
    @retry(
        stop=stop_after_attempt(3),
        # 1-10 seconds between attempts, to help avoid rate limiting
        wait=wait_random_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _select_action(
        self,
        env: FogOfWarTextWrapper,
        return_logprobs: bool = False,
        top_logprobs: int = 20,
        **kwargs: Any,
    ) -> Tuple[int, str, dict]:
        """
        Select an action using the LLM based on the current environment state.

        Args:
            env: The environment to interact with
            return_logprobs: Whether to return logprobs
            top_logprobs: Number of top logprobs to return (default: 5)
            **kwargs: Additional arguments (ignored by this agent)

        Returns:
            The selected action and a dictionary with related metadata.
        """
        base_env = getattr(env, "unwrapped", env)
        agent_pos = (
            tuple(base_env.agent_pos) if hasattr(base_env, "agent_pos") else "unknown"
        )

        logger.info(
            f"Getting action from LLM model {self.model_name} for position {agent_pos}"
        )
        prompt = self._generate_action_query_prompt(env)

        try:
            action_response, cost, raw_response, logprobs_serialized = (
                self._request_action_via_prompt(prompt, return_logprobs, top_logprobs)
            )

            logger.info(
                f"LLM selected action {action_response.action}, cost: ${cost:.6f}, total: ${self.total_cost:.6f}"
            )
            self.update_note(action_response.note)
            meta = self._build_base_metadata(
                action_response.action.value, cost, logprobs_serialized
            )
            meta["note"] = action_response.note
            return action_response.action.value, action_response.note, meta

        except Exception as e:
            logger.error(
                f"Error getting action from LLM: {e}\n{traceback.format_exc()}"
            )
            raise

    def reset(self) -> None:
        """Reset the agent's internal state and cost tracking."""
        super().reset()
        self.total_cost = 0.0
        self.call_count = 0
        self.current_note = ""


class PartiallyObservableWithChatHistoryLLMAgent(PartiallyObservableLLMAgent):
    """LLM agent with partial observability that also maintains full chat history."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        name: Optional[str] = None,
        template_path: Optional[Path] = None,
        response_format: BaseModel = ActionResponse,
    ) -> None:
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            name=name,
            template_path=template_path,
            response_format=response_format,
        )
        # Full conversation history as Message BaseModels
        self.chat_messages: list[Message] = []

    def _request_action_via_messages(
        self, messages_payload: list[dict], return_logprobs: bool, top_logprobs: int
    ) -> tuple[Any, float, Any, Optional[list[dict]]]:
        response_format, extra_kwargs = self._build_request_params(
            return_logprobs, top_logprobs
        )
        response, cost, raw_response = self._make_chat_completion_request(
            messages=messages_payload,
            response_format=response_format,
            **extra_kwargs,
        )
        if response_format is None:
            action_response = self.response_format.model_validate_json(response)
        else:
            action_response = response
        logprobs_serialized = self._finalize_cost_and_logprobs(
            cost, raw_response, return_logprobs
        )
        return action_response, cost, raw_response, logprobs_serialized

    def select_action(
        self,
        env: FogOfWarTextWrapper,
        return_logprobs: bool = False,
        top_logprobs: int = 20,
        **kwargs: Any,
    ) -> Tuple[int, dict]:
        """Select an action using accumulated chat messages + state/action history."""
        try:
            return self._select_action(env, return_logprobs, top_logprobs, **kwargs)
        except Exception as e:
            logger.error(
                f"Error getting action for position {env.agent_pos} from LLM after retrying: {e}"
            )
            return -1, {"agents_name": self.name}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _select_action(
        self,
        env: FogOfWarTextWrapper,
        return_logprobs: bool = False,
        top_logprobs: int = 20,
        **kwargs: Any,
    ) -> Tuple[int, dict]:
        """Use the LLM with full chat history; also includes state/action history in prompt."""
        base_env = getattr(env, "unwrapped", env)
        agent_pos = (
            tuple(base_env.agent_pos) if hasattr(base_env, "agent_pos") else "unknown"
        )

        logger.info(
            f"Getting action from LLM model {self.model_name} for position {agent_pos}"
        )

        # Build the new user prompt including current observation and state/action history
        prompt = self._generate_action_query_prompt(env)

        try:
            # Convert Message BaseModels to dicts and append current user prompt
            messages_payload = [m.model_dump() for m in self.chat_messages]
            messages_payload.append({"role": "user", "content": prompt})

            action_response, cost, raw_response, logprobs_serialized = (
                self._request_action_via_messages(
                    messages_payload, return_logprobs, top_logprobs
                )
            )

            logger.info(
                f"LLM selected action {action_response.action}, cost: ${cost:.6f}, total: ${self.total_cost:.6f}"
            )

            # Record assistant message content and user message into chat history
            # Append user message for this turn
            self.chat_messages.append(Message(role="user", content=prompt))
            # Append assistant message (raw content)
            assistant_content = raw_response.choices[0].message.content
            self.chat_messages.append(
                Message(
                    role="assistant",
                    content=assistant_content,
                )
            )

            # Record state/action pair history
            self.add_to_history(action_response.action, agent_pos)

            meta = self._build_base_metadata(
                action_response.action.value, cost, logprobs_serialized
            )
            meta["history"] = self.history.copy()
            return action_response.action.value, meta

        except Exception as e:
            logger.error(
                f"Error getting action from LLM: {e}\n{traceback.format_exc()}"
            )
            raise

    def reset(self) -> None:
        """Reset agent state, including chat and state/action histories and cost tracking."""
        super().reset()
        self.chat_messages = []
