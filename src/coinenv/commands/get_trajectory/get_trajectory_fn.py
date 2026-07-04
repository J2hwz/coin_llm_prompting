"""Generate and save agent trajectories in navigation environments with detailed token-level analysis."""

import copy
import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
from minigrid.core.world_object import Ball, Goal, Wall

from coinenv.agents.coin_astar_agent import CoinAStarAgent
from coinenv.agents.llm_agent import LLMAgent
from coinenv.commands.get_trajectory.compact_json_encoder import CompactJSONEncoder
from coinenv.commands.get_trajectory.get_trajectory_utils import (
    DEFAULT_TRANSFORM_NAMES,
    annotate_output_tokens,
    generate_trajectory,
    get_transformed_environments,
    to_dic_list,
    upload_directory_to_huggingface,
    upload_files_to_huggingface,
    upload_to_huggingface,
)
from coinenv.commands.get_trajectory.rate_limiter import (
    RateLimiter,
)
from coinenv.environment_generator.custom_minigrid import (
    CoinNavigationEnv,
    Simple2DNavigationEnv,
    TwoCoinNavigationEnv,
)
from coinenv.environment_generator.env_transformations import (
    ReflectEnv,
    RotateEnv,
    StartGoalSwap,
    TransposeEnv,
)
from coinenv.environment_generator.utils import (
    clone_env,
    find_coin_pos,
    get_all_dead_ends,
    manhattan_distance,
)
from coinenv.environment_generator.wrappers.text_obs_wrapper import (
    FullObservabilityTextWrapper,
)

logger = logging.getLogger(__file__)

logging.getLogger("LiteLLM").setLevel(logging.WARNING)

DATA_DIR = Path(__file__).parents[4] / "data"


def _resolve_output(path: str) -> Path:
    """Resolve path relative to DATA_DIR unless it is already absolute."""
    p = Path(path)
    return p if p.is_absolute() else DATA_DIR / p


def _env_to_grid_list(env_unwrapped) -> list[list[str]]:
    """Render the unwrapped env as a 2D list of cell symbols."""
    grid_list = []
    for j in range(env_unwrapped.height):
        row = []
        for i in range(env_unwrapped.width):
            cell = env_unwrapped.grid.get(i, j)
            if (i, j) == tuple(env_unwrapped.agent_pos):
                row.append("A")
            elif cell is None:
                row.append("_")
            elif cell.type == "wall":
                row.append("#")
            elif cell.type == "goal":
                row.append("G")
            elif cell.type == "ball":
                row.append("C")
            else:
                row.append("?")
        grid_list.append(row)
    return grid_list


def _reshuffle_walls(
    layout: dict,
    max_attempts: int = 200,
) -> "CoinNavigationEnv | None":
    """Generate a fresh maze with the same complexity but different wall layout.

    Keeps agent start, goal, and coin positions fixed. On each attempt the maze
    is regenerated randomly at the same complexity. When all anchors land on
    naturally empty cells the maze is accepted immediately (complexity is
    preserved exactly). If any anchor is walled off and retries are exhausted,
    a fallback is applied: the conflicting anchor cells are forcibly cleared and
    an equal number of non-anchor empty interior cells are re-walled, preserving
    the total wall count.

    Returns the configured (unwrapped) CoinNavigationEnv, or None if reachability
    between all anchor positions could not be satisfied.
    """
    from collections import deque

    size = layout["grid_size"]
    complexity = layout["grid_complexity"]
    start_pos = tuple(layout["agent_start_pos"])
    goal_pos = tuple(layout["goal_pos"])
    coin_pos = tuple(layout["coin_pos"]) if layout.get("coin_pos") else None
    agent_dir = layout.get("agent_start_dir", 0)

    anchor_positions: set[tuple[int, int]] = {start_pos, goal_pos}
    if coin_pos is not None:
        anchor_positions.add(coin_pos)

    for attempt in range(max_attempts + 1):
        is_fallback = attempt == max_attempts

        env = CoinNavigationEnv(size=size, complexity=complexity)
        env.reset()

        # Remove the randomly-placed goal so only wall/empty structure remains.
        if env.goal_pos is not None:
            env.grid.set(*env.goal_pos, None)

        # Identify anchor positions that the new maze walled off.
        conflicting = [
            pos
            for pos in anchor_positions
            if env.grid.get(*pos) is not None and env.grid.get(*pos).type == "wall"
        ]

        if conflicting and not is_fallback:
            continue  # Try a fresh maze

        if conflicting:
            # Fallback: clear conflicting anchors, compensate by re-walling an
            # equal number of non-anchor empty interior cells (wall count preserved).
            non_anchor_empty = [
                (x, y)
                for x in range(1, env.width - 1)
                for y in range(1, env.height - 1)
                if (x, y) not in anchor_positions and env.grid.get(x, y) is None
            ]
            random.shuffle(non_anchor_empty)
            for pos in conflicting:
                env.grid.set(*pos, None)
            for pos in non_anchor_empty[: len(conflicting)]:
                env.grid.set(*pos, Wall())

        # BFS from start_pos — all other anchors must be reachable.
        reachable: set[tuple[int, int]] = {start_pos}
        queue: deque = deque([start_pos])
        while queue:
            x, y = queue.popleft()
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in reachable or nx < 0 or ny < 0:
                    continue
                if nx >= env.width or ny >= env.height:
                    continue
                cell = env.grid.get(nx, ny)
                if cell is None or cell.type != "wall":
                    reachable.add((nx, ny))
                    queue.append((nx, ny))

        if not (anchor_positions - {start_pos}).issubset(reachable):
            if is_fallback:
                return None  # Reachability unresolvable
            continue

        # Place fixed objects.
        env.put_obj(Goal(), *goal_pos)
        env.goal_pos = goal_pos
        env.agent_pos = np.array(start_pos)
        env.agent_dir = agent_dir
        env._initial_agent_pos = np.array(start_pos)
        env._initial_agent_dir = agent_dir
        if coin_pos is not None:
            env.put_obj(Ball("yellow"), *coin_pos)

        return env

    return None  # Unreachable; loop above always returns on last iteration


def get_trajectory(
    grid_size: int = 5,
    grid_complexity: float = 0.0,
    max_steps_per_trajectory: int = 50,
    max_tokens: int = 10000,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_logprobs: int = 5,
    seed: int = 42,
    reasoning_effort: Literal["low", "medium", "high"] = "low",
    model_name: str = "together_ai/openai/gpt-oss-20b",
    observation_placeholders: list[str] = ["grid_state"],
    output_path: str = "get_trajectory_example_output.json",
    verbose: bool = False,
    enable_dynamic_max_steps: bool = False,
    hf_repo_id: str | None = None,
    hf_path_in_repo: str | None = None,
    hf_token: str | None = None,
    env: FullObservabilityTextWrapper | None = None,
    use_safe_reset: bool = False,
    transform_type: str = "base",
    template_name: str | None = None,
    track_history: bool = False,
):
    """Generate an agent trajectory in a 2D navigation environment and save detailed results to JSON.

    Creates a Simple2D navigation environment, runs an LLM agent to generate a trajectory,
    and saves comprehensive information including grid parameters, model parameters, prompt
    template with token-level analysis, and trajectory steps with token probabilities.

    Args:
        grid_size: Size of the square grid environment.
        grid_complexity: Complexity level of obstacles in the grid (higher = more obstacles).
        max_steps_per_trajectory: Maximum number of steps to generate in the trajectory.
        max_tokens: Maximum tokens for model generation per step.
        temperature: Sampling temperature for the model (higher = more random).
        top_p: Nucleus sampling parameter (cumulative probability threshold).
        top_logprobs: Number of top log probabilities to return for each token.
        seed: Random seed for reproducibility.
        reasoning_effort: Reasoning effort level for the model ("low", "medium", or "high").
        model_name: Name of the model in format "provider/model_id".
        observation_placeholders: List of placeholder names in the prompt template.
        output_path: Path to save the output JSON file.
        verbose: If True, print detailed logging during trajectory generation.
        enable_dynamic_max_steps: If True, override max_steps_per_trajectory with
            a dynamic value based on 1.5x the A* optimal path length.
        hf_repo_id: Hugging Face repository ID to upload to (e.g., "username/repo-name").
            If None, no upload is performed.
        hf_path_in_repo: Path within the HF repo where the file will be stored.
            If None, uses the output filename.
        hf_token: Hugging Face API token. If None, uses HF_TOKEN env var or cached credentials.
        env: Optional pre-created environment. If provided, grid_size and grid_complexity
            are ignored. Useful for generating multiple trajectories on the same grid.
        use_safe_reset: If True, use safe_reset() which resets agent position without
            regenerating the grid. Only applicable when env is provided.
        transform_type: Type of environment transform applied ("base", "RotateEnv",
            "ReflectEnv", "TransposeEnv", "StartGoalSwap"). Stored in grid_params.
        template_name: Jinja2 template filename to use (e.g.
            "grid_full_observability_hidden_goals.j2"). If None, uses the LLMAgent
            default (grid_full_observability.j2).

    Returns:
        str | None: URL of uploaded file if hf_repo_id is provided, otherwise None.
        Results are always saved to the specified output_path.

    The output JSON structure follows the format expected by the trace viewer: https://github.com/SPAR-Telos/interp/tree/trace-viewer
        - grid_params: Grid configuration (size, complexity, start/goal positions, A* distance, legend)
        - model_params: Model configuration (name, provider, sampling parameters, seed)
        - prompt: Prompt template with token-level annotations (prefix, suffix, placeholder tokens)
        - steps: List of trajectory steps, each containing:
            - step_id: Step number
            - grid_state: Grid visualization as list of strings
            - grid_state_tokens: Tokenized grid state with annotations
            - prompt_suffix_tokens: Tokenized prompt suffix
            - agent_action: Action taken by the agent
            - output_text: Model's generated output text
            - output_tokens: Tokenized output with probabilities and annotations
    """
    output_path = str(_resolve_output(output_path))
    # Use provided env or create a new one
    if env is None:
        base_env = FullObservabilityTextWrapper(
            Simple2DNavigationEnv(size=grid_size, complexity=grid_complexity)
        )
        use_safe_reset = False  # Cannot use safe_reset on a fresh env
    else:
        base_env = env
        # Get grid_size from the provided env
        grid_size = base_env.unwrapped.width
        grid_complexity = base_env.unwrapped.complexity

    if template_name is not None:
        template_path = (
            Path(__file__).parent.parent.parent / "templates" / template_name
        )
        agent = LLMAgent(model_name=model_name, template_path=template_path, track_history=track_history)
    else:
        agent = LLMAgent(model_name, track_history=track_history)
    model_id = "/".join(model_name.split("/")[1:])
    provider = model_name.split("/")[0]
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_id)

    traj = generate_trajectory(
        env=base_env,
        agent=agent,
        max_steps_per_trajectory=max_steps_per_trajectory,
        generation_kwargs={
            "top_logprobs": top_logprobs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "reasoning_effort": reasoning_effort,
            "seed": seed,
        },
        verbose=verbose,
        enable_dynamic_max_steps=enable_dynamic_max_steps,
        use_safe_reset=use_safe_reset,
    )

    grid_params = {}

    grid_params["grid_width"] = grid_size
    grid_params["grid_height"] = grid_size
    grid_params["grid_complexity"] = grid_complexity
    grid_params["fully_observable"] = True
    grid_params["transform_type"] = transform_type
    grid_params["astar_distance"] = traj.traj_metadata["astar_distance"]
    grid_params["agent_start_coordinates"] = traj.traj_metadata[
        "agent_start_coordinates"
    ]
    grid_params["goal_coordinates"] = traj.traj_metadata["goal_coordinates"]
    grid_params["legend"] = base_env.grid_cells

    grid_symbols = [cell["symbol"] for cell in base_env.grid_cells.values()]

    model_params = {}

    model_params["model_id"] = model_id
    model_params["provider"] = provider
    model_params["interface"] = "litellm"
    model_params["n_interactions_in_context"] = 0
    model_params["max_tokens"] = max_tokens
    model_params["max_steps_per_trajectory"] = max_steps_per_trajectory
    model_params["temperature"] = temperature
    model_params["reasoning_effort"] = reasoning_effort
    model_params["top_p"] = top_p
    model_params["top_logprobs"] = top_logprobs
    model_params["seed"] = seed

    prompt = {}

    render_kwargs = {"grid_state": "{{grid_state}}"}
    template = agent._template.render(**render_kwargs)
    formatted_template: str = tokenizer.apply_chat_template(
        [{"role": "user", "content": template}],
        tokenize=False,
        add_generation_prompt=True,
    )
    template_tokens = to_dic_list(formatted_template, tokenizer)

    # TODO: Handle multiple observation placeholders if needed (for now, we default to [0] only)
    observation_placeholder = "{{" + observation_placeholders[0] + "}}"
    prefix, suffix = formatted_template.split(observation_placeholder)
    raw_prefix, raw_suffix = template.split(observation_placeholder)
    prompt["prompt_template"] = formatted_template
    prompt["prompt_template_n_tokens"] = len(template_tokens)
    prompt["prompt_prefix_tokens"] = to_dic_list(prefix, tokenizer)
    raw_prefix_tokens = to_dic_list(raw_prefix, tokenizer)
    start_raw_prefix_idx = len(prompt["prompt_prefix_tokens"]) - len(raw_prefix_tokens)
    for i in range(start_raw_prefix_idx):
        prompt["prompt_prefix_tokens"][i]["token_groups"] += ["template"]
    prompt["prompt_prefix_n_tokens"] = len(prompt["prompt_prefix_tokens"])
    prompt["prompt_placeholder_tokens"] = to_dic_list(
        observation_placeholder, tokenizer, groups=["prompt", "placeholder"]
    )
    prompt["prompt_placeholder_n_tokens"] = len(prompt["prompt_placeholder_tokens"])
    prompt["prompt_suffix_tokens"] = to_dic_list(suffix, tokenizer)
    raw_suffix_tokens = to_dic_list(raw_suffix, tokenizer)
    start_raw_suffix_idx = (
        len(prompt["prompt_suffix_tokens"]) - len(raw_suffix_tokens) + 1
    )
    for i in range(
        len(prompt["prompt_suffix_tokens"]) - start_raw_suffix_idx,
        len(prompt["prompt_suffix_tokens"]) - 1,
    ):
        prompt["prompt_suffix_tokens"][i]["token_groups"] += ["template"]
    prompt["prompt_suffix_n_tokens"] = len(prompt["prompt_suffix_tokens"])

    steps = []

    for step_id, traj_step in enumerate(traj.steps):
        step_dic = {}
        step_dic["step_id"] = step_id
        step_dic["grid_state"] = traj_step.observation.split("\n")
        step_dic["grid_state_tokens"] = to_dic_list(
            traj_step.observation, tokenizer, groups=["prompt", "grid_state"]
        )
        step_dic["grid_state_n_tokens"] = len(step_dic["grid_state_tokens"])

        for i, t in enumerate(step_dic["grid_state_tokens"]):
            if any(sym in t["token"] for sym in grid_symbols):
                step_dic["grid_state_tokens"][i]["token_groups"] += ["grid_tile"]

        step_dic["prompt_suffix_tokens"] = prompt["prompt_suffix_tokens"]
        step_dic["prompt_suffix_n_tokens"] = len(step_dic["prompt_suffix_tokens"])
        step_dic["agent_action"] = traj_step.metadata["action"]
        if "coin_collected" in traj_step.metadata:
            step_dic["coin_collected"] = traj_step.metadata["coin_collected"]

        out_tokens = [t["token"] for t in traj_step.metadata["logprobs"]]
        step_dic["output_text"] = tokenizer.convert_tokens_to_string(out_tokens)
        step_dic["output_tokens"] = to_dic_list(
            step_dic["output_text"], tokenizer, groups=["output"]
        )
        step_dic["output_n_tokens"] = len(step_dic["output_tokens"])
        step_dic["output_tokens"] = annotate_output_tokens(
            model_name, step_dic["output_tokens"]
        )

        # Note: output_tokens (from local tokenizer) and logprobs (from API) may differ
        # in length due to tokenization differences, so we need bounds checking
        api_logprobs = traj_step.metadata["logprobs"]
        for i, t in enumerate(step_dic["output_tokens"]):
            if i >= len(api_logprobs):
                # Local tokenizer produced more tokens than API returned
                break
            if "top_logprobs" not in api_logprobs[i] or "template" in t["token_groups"]:
                continue
            curr_probs = {}
            for logprob_dic in api_logprobs[i]["top_logprobs"]:
                curr_probs[logprob_dic["token"]] = np.round(
                    np.exp(logprob_dic["logprob"]), 4
                )
            step_dic["output_tokens"][i]["probabilities"] = curr_probs

        steps.append(step_dic)

    out = {
        "grid_params": grid_params,
        "model_params": model_params,
        "prompt": prompt,
        "steps": steps,
    }
    with open(output_path, "w") as f:
        json.dump(out, f, cls=CompactJSONEncoder, ensure_ascii=False, indent=4)

    # Upload to Hugging Face if repo_id is provided
    if hf_repo_id is not None:
        return upload_to_huggingface(
            file_path=output_path,
            repo_id=hf_repo_id,
            path_in_repo=hf_path_in_repo,
            hf_token=hf_token,
        )
    return None


def get_trajectories(
    grid_sizes: list[int] = [5],
    grid_complexities: list[float] = [0.0],
    max_steps_per_trajectory: int = 50,
    max_tokens: int = 10000,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_logprobs: int = 5,
    seed: int = 42,
    reasoning_effort: Literal["low", "medium", "high"] = "low",
    model_names: list[str] = ["together_ai/openai/gpt-oss-20b"],
    observation_placeholders: list[str] = ["grid_state"],
    output_dir: str = ".",
    verbose: bool = False,
    enable_dynamic_max_steps: bool = False,
    num_examples: int = 1,
    max_workers: int | None = None,
    enable_rate_limit: bool = False,
    rate_limit: int = 1000,
    rate_limit_period: float = 300.0,
    hf_repo_id: str | None = None,
    hf_path_prefix: str = "",
    hf_token: str | None = None,
    track_history: bool = False,
):
    """Generate multiple agent trajectories across parameter combinations in parallel.

    Creates trajectories for all combinations of grid_sizes, grid_complexities, and model_names,
    running the specified number of examples per combination in parallel. Each trajectory is saved
    to a separate JSON file with a name based on the parameters.

    Args:
        grid_sizes: List of grid sizes to use for trajectory generation.
        grid_complexities: List of grid complexity levels to use.
        max_steps_per_trajectory: Maximum number of steps to generate in each trajectory.
        max_tokens: Maximum tokens for model generation per step.
        temperature: Sampling temperature for the model (higher = more random).
        top_p: Nucleus sampling parameter (cumulative probability threshold).
        top_logprobs: Number of top log probabilities to return for each token.
        seed: Base random seed for reproducibility. Each example uses seed + example_id.
        reasoning_effort: Reasoning effort level for the model ("low", "medium", or "high").
        model_names: List of model names in format "provider/model_id".
        observation_placeholders: List of placeholder names in the prompt template.
        output_dir: Directory to save the output JSON files.
        verbose: If True, print detailed logging during trajectory generation.
        enable_dynamic_max_steps: If True, override max_steps_per_trajectory with
            a dynamic value based on 1.5x the A* optimal path length.
        num_examples: Number of different examples to generate per parameter combination.
        max_workers: Maximum number of parallel workers. If None, uses min(32, total_tasks).
        enable_rate_limit: If True, enforce rate limiting on API requests.
        rate_limit: Maximum number of requests allowed per rate_limit_period.
        rate_limit_period: Time period in seconds for rate limiting (default: 300 = 5 minutes).
        hf_repo_id: Hugging Face repository ID to upload to (e.g., "username/repo-name").
            If None, no upload is performed.
        hf_path_prefix: Path prefix within the HF repo (e.g., "trajectories/" to put files
            in a subfolder).
        hf_token: Hugging Face API token. If None, uses HF_TOKEN env var or cached credentials.

    Returns:
        list[str] | None: List of URLs if uploaded to HF, otherwise None.
        Results are always saved to individual JSON files in output_dir with format:
        {model_sanitized}_size{grid_size}_comp{grid_complexity}_{example_id}.json
    """
    output_dir = str(_resolve_output(output_dir))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate all parameter combinations
    all_combinations = list(
        product(grid_sizes, grid_complexities, model_names, range(num_examples))
    )

    total_tasks = len(all_combinations)
    logger.info(
        f"Generating {total_tasks} trajectories across {len(grid_sizes)} grid sizes, "
        f"{len(grid_complexities)} complexities, {len(model_names)} models, "
        f"with {num_examples} examples each."
    )

    # Create rate limiter if enabled
    rate_limiter: RateLimiter | None = None
    if enable_rate_limit:
        rate_limiter = RateLimiter(rate_limit=rate_limit, period=rate_limit_period)
        logger.info(
            f"Rate limiting enabled: {rate_limit} requests per {rate_limit_period} seconds "
            f"({rate_limit / rate_limit_period:.2f} requests/second)"
        )

    def _generate_single_task(params: tuple) -> dict:
        """Generate a single trajectory for given parameters."""
        # Acquire rate limit token if enabled
        if rate_limiter is not None:
            rate_limiter.acquire()

        grid_size, grid_complexity, model_name, example_id = params
        task_seed = seed + example_id

        # Sanitize model name for filename
        model_sanitized = model_name.replace("/", "_").replace(".", "_")

        output_filename = (
            f"{model_sanitized}_size{grid_size}_comp{grid_complexity}_{example_id}.json"
        )
        output_path = str(Path(output_dir) / output_filename)

        if verbose:
            logger.info(
                f"Starting task: model={model_name}, size={grid_size}, "
                f"complexity={grid_complexity}, example={example_id}, seed={task_seed}"
            )

        try:
            get_trajectory(
                grid_size=grid_size,
                grid_complexity=grid_complexity,
                max_steps_per_trajectory=max_steps_per_trajectory,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_logprobs=top_logprobs,
                seed=task_seed,
                reasoning_effort=reasoning_effort,
                model_name=model_name,
                observation_placeholders=observation_placeholders,
                output_path=output_path,
                verbose=verbose,
                enable_dynamic_max_steps=enable_dynamic_max_steps,
                track_history=track_history,
            )
            return {"status": "success", "output_path": output_path, "params": params}
        except Exception as e:
            logger.error(f"Failed to generate trajectory for {params}: {e}")
            return {"status": "error", "error": str(e), "params": params}

    # Set default max_workers if not specified
    if max_workers is None:
        max_workers = min(32, total_tasks)

    # Adjust max_workers based on rate limit to avoid excessive idle workers
    if enable_rate_limit and rate_limiter is not None:
        # Calculate the sustainable number of workers based on request rate
        # If each task takes at least 1 second, don't exceed the rate limit per second
        sustainable_workers = min(max_workers, int(rate_limiter.tokens_per_second * 2))
        if sustainable_workers < max_workers:
            logger.info(
                f"Adjusting max_workers from {max_workers} to {sustainable_workers} "
                f"based on rate limit ({rate_limiter.tokens_per_second:.2f} req/sec)"
            )
            max_workers = sustainable_workers

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_generate_single_task, combo): combo
            for combo in all_combinations
        }

        for future in tqdm(
            as_completed(futures),
            total=total_tasks,
            desc="Generating trajectories",
            unit="trajectory",
        ):
            combo = futures[future]
            try:
                result = future.result()
                results.append(result)
                if result["status"] != "success":
                    tqdm.write(
                        f"Failed for {combo}: {result.get('error', 'Unknown error')}"
                    )
            except Exception as e:
                tqdm.write(f"Exception for {combo}: {e}")
                results.append({"status": "error", "error": str(e), "params": combo})

    success_count = sum(1 for r in results if r["status"] == "success")
    logger.info(f"Completed {success_count}/{total_tasks} trajectories successfully.")

    # Upload to Hugging Face if repo_id is provided
    if hf_repo_id is not None and success_count > 0:
        successful_paths = [
            r["output_path"] for r in results if r["status"] == "success"
        ]
        return upload_files_to_huggingface(
            file_paths=successful_paths,
            repo_id=hf_repo_id,
            path_prefix=hf_path_prefix,
            hf_token=hf_token,
        )
    return None


def get_trajectories_multiple_per_grid(
    grid_sizes: list[int] = [5],
    grid_complexities: list[float] = [0.0],
    max_steps_per_trajectory: int = 50,
    max_tokens: int = 10000,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_logprobs: int = 5,
    seed: int = 42,
    reasoning_effort: Literal["low", "medium", "high"] = "low",
    model_names: list[str] = ["together_ai/openai/gpt-oss-20b"],
    observation_placeholders: list[str] = ["grid_state"],
    output_dir: str = ".",
    verbose: bool = False,
    enable_dynamic_max_steps: bool = False,
    num_trajectories_per_grid: int = 5,
    num_grids_per_config: int = 1,
    max_workers: int | None = None,
    max_workers_per_grid: int | None = None,
    enable_rate_limit: bool = False,
    rate_limit: int = 1000,
    rate_limit_period: float = 300.0,
    hf_repo_id: str | None = None,
    hf_path_prefix: str = "",
    hf_token: str | None = None,
    include_transforms: bool = False,
    transform_names: list[str] | None = None,
    track_history: bool = False,
):
    """Generate multiple trajectories on the same grid layout for each configuration.

    Creates trajectories where multiple runs are performed on the same grid without
    regenerating it. This is useful for studying variability in agent behavior on
    identical environments.

    For each (grid_size, grid_complexity, model_name) combination:
    - Creates `num_grids_per_config` unique grid layouts
    - For each grid, generates `num_trajectories_per_grid` trajectories in parallel

    Note: Trajectories within the same grid use deepcopy of the environment to enable
    parallel execution while ensuring identical grid layouts. Different grids are also
    processed in parallel.

    Args:
        grid_sizes: List of grid sizes to use for trajectory generation.
        grid_complexities: List of grid complexity levels to use.
        max_steps_per_trajectory: Maximum number of steps to generate in each trajectory.
        max_tokens: Maximum tokens for model generation per step.
        temperature: Sampling temperature for the model (higher = more random).
        top_p: Nucleus sampling parameter (cumulative probability threshold).
        top_logprobs: Number of top log probabilities to return for each token.
        seed: Base random seed for reproducibility. Each grid uses seed + grid_id.
        reasoning_effort: Reasoning effort level for the model ("low", "medium", or "high").
        model_names: List of model names in format "provider/model_id".
        observation_placeholders: List of placeholder names in the prompt template.
        output_dir: Directory to save the output JSON files.
        verbose: If True, print detailed logging during trajectory generation.
        enable_dynamic_max_steps: If True, override max_steps_per_trajectory with
            a dynamic value based on 1.5x the A* optimal path length.
        num_trajectories_per_grid: Number of trajectories to generate per grid layout.
        num_grids_per_config: Number of different grid layouts per (size, complexity, model) combo.
        max_workers: Maximum number of parallel workers for processing different grids.
            If None, uses min(32, total_grid_tasks).
        max_workers_per_grid: Maximum number of parallel workers for trajectories within each grid.
            If None, uses num_trajectories_per_grid (full parallelism within each grid).
        enable_rate_limit: If True, enforce rate limiting on API requests.
        rate_limit: Maximum number of requests allowed per rate_limit_period.
        rate_limit_period: Time period in seconds for rate limiting (default: 300 = 5 minutes).
        hf_repo_id: Hugging Face repository ID to upload to (e.g., "username/repo-name").
            If None, no upload is performed.
        hf_path_prefix: Path prefix within the HF repo (e.g., "trajectories/" to put files
            in a subfolder).
        hf_token: Hugging Face API token. If None, uses HF_TOKEN env var or cached credentials.
        include_transforms: If True, also generate trajectories for transformed versions
            of each grid (RotateEnv, ReflectEnv, TransposeEnv, StartGoalSwap). Each
            transform gets num_trajectories_per_grid trajectories.
        transform_names: List of transform names to include when include_transforms=True.
            If None, uses all available transforms. Options: "RotateEnv", "ReflectEnv",
            "TransposeEnv", "StartGoalSwap".

    Returns:
        list[str] | None: List of URLs if uploaded to HF, otherwise None.
        Results are saved to individual JSON files in output_dir with format:
        {model_sanitized}_size{grid_size}_comp{grid_complexity}_grid{grid_id}_{transform}_traj{traj_id}.json
    """
    output_dir = str(_resolve_output(output_dir))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Generate all grid configurations (each will have multiple trajectories)
    grid_configs = list(
        product(
            grid_sizes,
            grid_complexities,
            model_names,
            range(num_grids_per_config),
        )
    )

    total_grid_tasks = len(grid_configs)

    # Determine number of transforms (including base)
    if include_transforms:
        used_transform_names = (
            transform_names if transform_names else DEFAULT_TRANSFORM_NAMES
        )
        num_transforms = 1 + len(used_transform_names)  # base + transforms
    else:
        used_transform_names = []
        num_transforms = 1  # just base

    total_trajectories = total_grid_tasks * num_trajectories_per_grid * num_transforms
    transform_info = (
        f" with transforms ({', '.join(['base'] + used_transform_names)})"
        if include_transforms
        else ""
    )
    logger.info(
        f"Generating {total_trajectories} trajectories across {len(grid_sizes)} grid sizes, "
        f"{len(grid_complexities)} complexities, {len(model_names)} models, "
        f"with {num_grids_per_config} grids each and {num_trajectories_per_grid} trajectories per grid"
        f"{transform_info}."
    )

    # Create rate limiter if enabled
    rate_limiter: RateLimiter | None = None
    if enable_rate_limit:
        rate_limiter = RateLimiter(rate_limit=rate_limit, period=rate_limit_period)
        logger.info(
            f"Rate limiting enabled: {rate_limit} requests per {rate_limit_period} seconds "
            f"({rate_limit / rate_limit_period:.2f} requests/second)"
        )

    def _generate_trajectories_for_grid(config: tuple) -> dict:
        """Generate multiple trajectories on a single grid in parallel using deepcopy.

        When include_transforms is True, also generates trajectories for transformed
        versions of the grid (RotateEnv, ReflectEnv, TransposeEnv, StartGoalSwap).
        """
        grid_size, grid_complexity, model_name, grid_id = config
        grid_seed = seed + grid_id

        # Sanitize model name for filename
        model_sanitized = model_name.replace("/", "_").replace(".", "_")

        # Create the master environment once for this grid
        np.random.seed(grid_seed)
        master_env = FullObservabilityTextWrapper(
            Simple2DNavigationEnv(size=grid_size, complexity=grid_complexity)
        )
        master_env.reset()

        # Get all environments to generate trajectories for (base + transforms if enabled)
        if include_transforms:
            env_variants = get_transformed_environments(
                master_env,
                include_base=True,
                transform_names=used_transform_names if used_transform_names else None,
            )
        else:
            env_variants = [("base", master_env)]

        # Save grid layout for base environment
        grid_paths = []
        base_grid_path = _save_grid_layout(
            env=master_env,
            grid_size=grid_size,
            grid_complexity=grid_complexity,
            grid_id=grid_id,
            grid_seed=grid_seed,
            model_sanitized=model_sanitized,
            transform_type="base",
        )
        grid_paths.append(base_grid_path)

        # Save grid layouts for transformed environments
        if include_transforms:
            for transform_name, transformed_env in env_variants:
                if transform_name == "base":
                    continue  # Already saved
                transform_grid_path = _save_grid_layout(
                    env=transformed_env,
                    grid_size=grid_size,
                    grid_complexity=grid_complexity,
                    grid_id=grid_id,
                    grid_seed=grid_seed,
                    model_sanitized=model_sanitized,
                    transform_type=transform_name,
                )
                grid_paths.append(transform_grid_path)

        def _generate_single_trajectory(
            traj_id: int, transform_name: str, env_to_use: FullObservabilityTextWrapper
        ) -> dict:
            """Generate a single trajectory using a deepcopy of the given environment."""
            # Acquire rate limit token if enabled
            if rate_limiter is not None:
                rate_limiter.acquire()

            # Offset seeds for different trajectories and transforms
            transform_offset = hash(transform_name) % 10000
            traj_seed = grid_seed + traj_id * 1000 + transform_offset

            # Deep copy the environment to avoid race conditions
            env_copy = copy.deepcopy(env_to_use)

            output_filename = (
                f"{model_sanitized}_size{grid_size}_comp{grid_complexity}"
                f"_grid{grid_id}_{transform_name}_traj{traj_id}.json"
            )
            output_path = str(Path(output_dir) / output_filename)

            if verbose:
                logger.info(
                    f"Starting trajectory: model={model_name}, size={grid_size}, "
                    f"complexity={grid_complexity}, grid={grid_id}, "
                    f"transform={transform_name}, traj={traj_id}"
                )

            try:
                get_trajectory(
                    grid_size=grid_size,
                    grid_complexity=grid_complexity,
                    max_steps_per_trajectory=max_steps_per_trajectory,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_logprobs=top_logprobs,
                    seed=traj_seed,
                    reasoning_effort=reasoning_effort,
                    model_name=model_name,
                    observation_placeholders=observation_placeholders,
                    output_path=output_path,
                    verbose=verbose,
                    enable_dynamic_max_steps=enable_dynamic_max_steps,
                    env=env_copy,
                    use_safe_reset=True,
                    transform_type=transform_name,
                    track_history=track_history,
                )
                return {
                    "status": "success",
                    "output_path": output_path,
                    "config": config,
                    "traj_id": traj_id,
                    "transform_type": transform_name,
                }
            except Exception as e:
                logger.error(
                    f"Failed to generate trajectory for grid {grid_id}, "
                    f"transform {transform_name}, traj {traj_id}: {e}"
                )
                return {
                    "status": "error",
                    "error": str(e),
                    "config": config,
                    "traj_id": traj_id,
                    "transform_type": transform_name,
                }

        # Parallelize trajectories within this grid (across all transforms)
        total_tasks_per_grid = num_trajectories_per_grid * len(env_variants)
        inner_workers = (
            max_workers_per_grid
            if max_workers_per_grid is not None
            else min(total_tasks_per_grid, num_trajectories_per_grid * 2)
        )
        trajectory_results = []

        with ThreadPoolExecutor(max_workers=inner_workers) as inner_executor:
            futures = {}
            for transform_name, env_variant in env_variants:
                for traj_id in range(num_trajectories_per_grid):
                    future = inner_executor.submit(
                        _generate_single_trajectory,
                        traj_id,
                        transform_name,
                        env_variant,
                    )
                    futures[future] = (traj_id, transform_name)

            for future in as_completed(futures):
                traj_id, transform_name = futures[future]
                try:
                    result = future.result()
                    trajectory_results.append(result)
                except Exception as e:
                    logger.error(
                        f"Exception for grid {grid_id}, transform {transform_name}, "
                        f"traj {traj_id}: {e}"
                    )
                    trajectory_results.append(
                        {
                            "status": "error",
                            "error": str(e),
                            "config": config,
                            "traj_id": traj_id,
                            "transform_type": transform_name,
                        }
                    )

        return {
            "trajectory_results": trajectory_results,
            "grid_paths": grid_paths,
        }

    def _save_grid_layout(
        env: FullObservabilityTextWrapper,
        grid_size: int,
        grid_complexity: float,
        grid_id: int,
        grid_seed: int,
        model_sanitized: str,
        transform_type: str = "base",
    ) -> str:
        """Save the grid layout to a JSON file."""
        unwrapped = env.unwrapped

        # Build grid representation as list of lists
        grid_list = []
        for j in range(unwrapped.height):
            row = []
            for i in range(unwrapped.width):
                cell = unwrapped.grid.get(i, j)
                if (i, j) == tuple(unwrapped.agent_pos):
                    row.append("A")
                elif cell is None:
                    row.append("_")
                elif cell.type == "wall":
                    row.append("#")
                elif cell.type == "goal":
                    row.append("G")
                else:
                    row.append("?")
            grid_list.append(row)

        # Get agent position - handle both array and tuple forms
        agent_pos = unwrapped.agent_pos
        if hasattr(agent_pos, "tolist"):
            agent_start_pos = agent_pos.tolist()
        else:
            agent_start_pos = list(agent_pos)

        # Get initial agent position if available
        if (
            hasattr(unwrapped, "_initial_agent_pos")
            and unwrapped._initial_agent_pos is not None
        ):
            initial_pos = unwrapped._initial_agent_pos
            if hasattr(initial_pos, "tolist"):
                agent_start_pos = initial_pos.tolist()
            else:
                agent_start_pos = list(initial_pos)

        # Get initial agent direction if available
        agent_start_dir = getattr(unwrapped, "_initial_agent_dir", unwrapped.agent_dir)

        grid_data = {
            "grid_id": grid_id,
            "grid_seed": grid_seed,
            "grid_size": grid_size,
            "grid_complexity": grid_complexity,
            "grid_width": unwrapped.width,
            "grid_height": unwrapped.height,
            "transform_type": transform_type,
            "agent_start_pos": agent_start_pos,
            "agent_start_dir": agent_start_dir,
            "goal_pos": list(unwrapped.goal_pos),
            "grid_layout": grid_list,
            "grid_text": env._render(),
            "legend": env.grid_cells,
        }

        grid_filename = (
            f"{model_sanitized}_size{grid_size}_comp{grid_complexity}"
            f"_grid{grid_id}_{transform_type}.json"
        )
        grid_path = str(Path(output_dir) / grid_filename)

        with open(grid_path, "w") as f:
            json.dump(grid_data, f, indent=2)

        if verbose:
            logger.info(f"Saved grid layout to {grid_path}")

        return grid_path

    # Set default max_workers if not specified
    if max_workers is None:
        max_workers = min(32, total_grid_tasks)

    # Adjust max_workers based on rate limit to avoid excessive idle workers
    if enable_rate_limit and rate_limiter is not None:
        sustainable_workers = min(max_workers, int(rate_limiter.tokens_per_second * 2))
        if sustainable_workers < max_workers:
            logger.info(
                f"Adjusting max_workers from {max_workers} to {sustainable_workers} "
                f"based on rate limit ({rate_limiter.tokens_per_second:.2f} req/sec)"
            )
            max_workers = sustainable_workers

    all_trajectory_results = []
    all_grid_paths = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_generate_trajectories_for_grid, config): config
            for config in grid_configs
        }

        for future in tqdm(
            as_completed(futures),
            total=total_grid_tasks,
            desc="Processing grids",
            unit="grid",
        ):
            config = futures[future]
            try:
                result = future.result()
                grid_results = result["trajectory_results"]
                grid_paths = result["grid_paths"]

                all_trajectory_results.extend(grid_results)
                all_grid_paths.extend(grid_paths)

                # Report any failures for this grid
                failures = [r for r in grid_results if r["status"] != "success"]
                for failure in failures:
                    transform_info = failure.get("transform_type", "base")
                    tqdm.write(
                        f"Failed for grid {config}, transform {transform_info}, "
                        f"traj {failure['traj_id']}: {failure.get('error', 'Unknown error')}"
                    )
            except Exception as e:
                tqdm.write(f"Exception for grid {config}: {e}")
                # Mark all trajectories for this grid as failed
                transforms_to_fail = ["base"]
                if include_transforms:
                    transforms_to_fail.extend(
                        used_transform_names
                        if used_transform_names
                        else DEFAULT_TRANSFORM_NAMES
                    )
                for transform_name in transforms_to_fail:
                    for traj_id in range(num_trajectories_per_grid):
                        all_trajectory_results.append(
                            {
                                "status": "error",
                                "error": str(e),
                                "config": config,
                                "traj_id": traj_id,
                                "transform_type": transform_name,
                            }
                        )

    success_count = sum(1 for r in all_trajectory_results if r["status"] == "success")
    logger.info(
        f"Completed {success_count}/{total_trajectories} trajectories successfully."
    )
    logger.info(f"Saved {len(all_grid_paths)} grid layout files.")

    # Upload to Hugging Face if repo_id is provided
    if hf_repo_id is not None and (success_count > 0 or all_grid_paths):
        all_paths_to_upload = []

        # Add successful trajectory files
        successful_traj_paths = [
            r["output_path"] for r in all_trajectory_results if r["status"] == "success"
        ]
        all_paths_to_upload.extend(successful_traj_paths)

        # Add grid layout files
        all_paths_to_upload.extend(all_grid_paths)

        if all_paths_to_upload:
            return upload_files_to_huggingface(
                file_paths=all_paths_to_upload,
                repo_id=hf_repo_id,
                path_prefix=hf_path_prefix,
                hf_token=hf_token,
            )
    return None


def get_single_trajectory_coin_env(
    grid_size: int = 11,
    grid_complexity: float = 0.8,
    target_dead_ends: int | None = None,
    place_at_dead_ends: bool = False,
    coin_placement: Literal["random", "dead_end"] | None = None,
    max_attempts: int = 1000,
    min_manhattan_distance: int = 3,
    max_steps_per_trajectory: int = 50,
    max_tokens: int = 10000,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_logprobs: int = 5,
    seed: int = 42,
    reasoning_effort: Literal["low", "medium", "high"] = "low",
    model_name: str = "together_ai/openai/gpt-oss-20b",
    template_name: str = "grid_full_observability_hidden_goals.j2",
    observation_placeholders: list[str] = ["grid_state"],
    output_path: str = "get_trajectory_deadend_example_output.json",
    verbose: bool = False,
    enable_dynamic_max_steps: bool = False,
    track_history: bool = False,
):
    """Generate an agent trajectory in a dead-end constrained navigation environment and save detailed results to JSON.

    Creates a Simple2DNavigationEnv with an optional dead-end count constraint and optional coin
    placement. Environments are repeatedly generated until one matching the target_dead_ends count
    is found (up to max_attempts). Follows the same output format as get_trajectory.

    Args:
        grid_size: Size of the square grid environment.
        grid_complexity: Complexity level of the grid (0.0 = open room, 1.0 = perfect maze).
        target_dead_ends: Exact number of dead ends required in the generated maze. If None,
            the first generated environment is used without constraint.
        place_at_dead_ends: If True, place the agent start and goal at dead-end cells.
        coin_placement: Where to place the coin. "random" places it in any open cell;
            "dead_end" places it at a dead-end cell (falls back to random if none available);
            None places no coin.
        max_attempts: Maximum number of generation attempts when target_dead_ends is set.
        max_steps_per_trajectory: Maximum number of steps to generate in the trajectory.
        max_tokens: Maximum tokens for model generation per step.
        temperature: Sampling temperature for the model (higher = more random).
        top_p: Nucleus sampling parameter (cumulative probability threshold).
        top_logprobs: Number of top log probabilities to return for each token.
        seed: Random seed for reproducibility.
        reasoning_effort: Reasoning effort level for the model ("low", "medium", or "high").
        model_name: Name of the model in format "provider/model_id".
        template_name: Name of the Jinja2 template file to use for prompts.
        observation_placeholders: List of placeholder names in the prompt template.
        output_path: Path to save the output JSON file.
        verbose: If True, print detailed logging during trajectory generation.
        enable_dynamic_max_steps: If True, override max_steps_per_trajectory with a dynamic
            value based on 1.5x the A* optimal path length.

    Returns:
        None. Results are saved to the specified output_path.

    The output JSON structure follows the format expected by the trace viewer:
        - grid_params: Grid configuration including dead-end count, coin placement, start/goal
            positions, A* distance, and legend.
        - model_params: Model configuration (name, provider, sampling parameters, seed).
        - prompt: Prompt template with token-level annotations.
        - steps: List of trajectory steps with grid state, action, output tokens, and probabilities.
    """
    output_path = str(_resolve_output(output_path))

    # Use CoinNavigationEnv only when a coin is requested; Simple2DNavigationEnv otherwise
    if coin_placement is not None:
        env_class = CoinNavigationEnv
    else:
        env_class = Simple2DNavigationEnv

    base_env_unwrapped = None
    coin_pos = None
    for _ in range(max_attempts):
        candidate = env_class(
            size=grid_size,
            complexity=grid_complexity,
            place_at_dead_ends=place_at_dead_ends,
        )
        candidate.reset()

        # Reject grids that don't meet the dead-end count requirement
        if (
            target_dead_ends is not None
            and len(get_all_dead_ends(candidate)) != target_dead_ends
        ):
            candidate.close()
            continue

        start = tuple(candidate.agent_pos)
        goal = tuple(candidate.goal_pos)

        # Reject grids where start and goal are too close together
        if manhattan_distance(start, goal) < min_manhattan_distance:
            candidate.close()
            continue

        # Find valid coin positions satisfying distance constraints
        resolved_coin_pos = None
        if coin_placement is not None:
            excluded = {start, goal}
            effective_placement = coin_placement

            # Prefer dead-end cells; fall back to random if none available
            if effective_placement == "dead_end":
                raw = [
                    (x, y)
                    for (x, y) in get_all_dead_ends(candidate)
                    if (x, y) not in excluded
                ]
                if not raw:
                    logger.warning(
                        "No valid dead end for coin placement; falling back to random."
                    )
                    effective_placement = "random"
            if effective_placement == "random":
                raw = [
                    (x, y)
                    for x in range(1, candidate.width - 1)
                    for y in range(1, candidate.height - 1)
                    if candidate.grid.get(x, y) is None and (x, y) not in excluded
                ]

            # Filter to positions that are far enough from both start and goal
            valid = [
                p
                for p in raw
                if manhattan_distance(start, p) >= min_manhattan_distance
                and manhattan_distance(goal, p) >= min_manhattan_distance
            ]

            # Reject grids with no valid coin position
            if not valid:
                candidate.close()
                continue

            resolved_coin_pos = random.choice(valid)
            candidate.put_obj(Ball("yellow"), *resolved_coin_pos)

        base_env_unwrapped = candidate
        coin_pos = resolved_coin_pos
        break

    if base_env_unwrapped is None:
        raise RuntimeError(
            f"Could not generate an environment satisfying all constraints after {max_attempts} "
            f"attempts (size={grid_size}, complexity={grid_complexity}, "
            f"target_dead_ends={target_dead_ends}, min_manhattan_distance={min_manhattan_distance})."
        )

    actual_dead_ends = len(get_all_dead_ends(base_env_unwrapped))

    base_env = FullObservabilityTextWrapper(base_env_unwrapped)

    # Load template and create agent
    template_path = Path(__file__).parent.parent.parent / "templates" / template_name
    agent = LLMAgent(model_name=model_name, template_path=template_path, track_history=track_history)
    model_id = "/".join(model_name.split("/")[1:])
    provider = model_name.split("/")[0]
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_id)

    traj = generate_trajectory(
        env=base_env,
        agent=agent,
        max_steps_per_trajectory=max_steps_per_trajectory,
        generation_kwargs={
            "top_logprobs": top_logprobs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "reasoning_effort": reasoning_effort,
            "seed": seed,
        },
        verbose=verbose,
        enable_dynamic_max_steps=enable_dynamic_max_steps,
        skip_reset=True,
    )

    grid_params = {}
    grid_params["grid_width"] = base_env_unwrapped.width
    grid_params["grid_height"] = base_env_unwrapped.height
    grid_params["grid_complexity"] = grid_complexity
    grid_params["target_dead_ends"] = target_dead_ends
    grid_params["actual_dead_ends"] = actual_dead_ends
    grid_params["place_at_dead_ends"] = place_at_dead_ends
    grid_params["coin_placement"] = coin_placement
    grid_params["coin_pos"] = list(coin_pos) if coin_pos is not None else None
    grid_params["fully_observable"] = True
    grid_params["astar_distance"] = traj.traj_metadata["astar_distance"]
    grid_params["agent_start_coordinates"] = traj.traj_metadata[
        "agent_start_coordinates"
    ]
    grid_params["goal_coordinates"] = traj.traj_metadata["goal_coordinates"]
    grid_params["legend"] = base_env.grid_cells

    grid_symbols = [cell["symbol"] for cell in base_env.grid_cells.values()]

    model_params = {}
    model_params["model_id"] = model_id
    model_params["provider"] = provider
    model_params["interface"] = "litellm"
    model_params["template_name"] = template_name
    model_params["n_interactions_in_context"] = 0
    model_params["max_tokens"] = max_tokens
    model_params["max_steps_per_trajectory"] = max_steps_per_trajectory
    model_params["temperature"] = temperature
    model_params["reasoning_effort"] = reasoning_effort
    model_params["top_p"] = top_p
    model_params["top_logprobs"] = top_logprobs
    model_params["seed"] = seed

    prompt = {}
    render_kwargs = {"grid_state": "{{grid_state}}"}
    template = agent._template.render(**render_kwargs)
    formatted_template: str = tokenizer.apply_chat_template(
        [{"role": "user", "content": template}],
        tokenize=False,
        add_generation_prompt=True,
    )
    template_tokens = to_dic_list(formatted_template, tokenizer)

    observation_placeholder = "{{" + observation_placeholders[0] + "}}"
    prefix, suffix = formatted_template.split(observation_placeholder)
    raw_prefix, raw_suffix = template.split(observation_placeholder)
    prompt["prompt_template"] = formatted_template
    prompt["prompt_template_n_tokens"] = len(template_tokens)
    prompt["prompt_prefix_tokens"] = to_dic_list(prefix, tokenizer)
    raw_prefix_tokens = to_dic_list(raw_prefix, tokenizer)
    start_raw_prefix_idx = len(prompt["prompt_prefix_tokens"]) - len(raw_prefix_tokens)
    for i in range(start_raw_prefix_idx):
        prompt["prompt_prefix_tokens"][i]["token_groups"] += ["template"]
    prompt["prompt_prefix_n_tokens"] = len(prompt["prompt_prefix_tokens"])
    prompt["prompt_placeholder_tokens"] = to_dic_list(
        observation_placeholder, tokenizer, groups=["prompt", "placeholder"]
    )
    prompt["prompt_placeholder_n_tokens"] = len(prompt["prompt_placeholder_tokens"])
    prompt["prompt_suffix_tokens"] = to_dic_list(suffix, tokenizer)
    raw_suffix_tokens = to_dic_list(raw_suffix, tokenizer)
    start_raw_suffix_idx = (
        len(prompt["prompt_suffix_tokens"]) - len(raw_suffix_tokens) + 1
    )
    for i in range(
        len(prompt["prompt_suffix_tokens"]) - start_raw_suffix_idx,
        len(prompt["prompt_suffix_tokens"]) - 1,
    ):
        prompt["prompt_suffix_tokens"][i]["token_groups"] += ["template"]
    prompt["prompt_suffix_n_tokens"] = len(prompt["prompt_suffix_tokens"])

    steps = []
    for step_id, traj_step in enumerate(traj.steps):
        step_dic = {}
        step_dic["step_id"] = step_id
        step_dic["grid_state"] = traj_step.observation.split("\n")
        step_dic["grid_state_tokens"] = to_dic_list(
            traj_step.observation, tokenizer, groups=["prompt", "grid_state"]
        )
        step_dic["grid_state_n_tokens"] = len(step_dic["grid_state_tokens"])

        for i, t in enumerate(step_dic["grid_state_tokens"]):
            if any(sym in t["token"] for sym in grid_symbols):
                step_dic["grid_state_tokens"][i]["token_groups"] += ["grid_tile"]

        step_dic["prompt_suffix_tokens"] = prompt["prompt_suffix_tokens"]
        step_dic["prompt_suffix_n_tokens"] = len(step_dic["prompt_suffix_tokens"])
        step_dic["agent_action"] = traj_step.metadata["action"]
        if "coin_collected" in traj_step.metadata:
            step_dic["coin_collected"] = traj_step.metadata["coin_collected"]

        out_tokens = [t["token"] for t in traj_step.metadata["logprobs"]]
        step_dic["output_text"] = tokenizer.convert_tokens_to_string(out_tokens)
        step_dic["output_tokens"] = to_dic_list(
            step_dic["output_text"], tokenizer, groups=["output"]
        )
        step_dic["output_n_tokens"] = len(step_dic["output_tokens"])
        step_dic["output_tokens"] = annotate_output_tokens(
            model_name, step_dic["output_tokens"]
        )

        api_logprobs = traj_step.metadata["logprobs"]
        for i, t in enumerate(step_dic["output_tokens"]):
            if i >= len(api_logprobs):
                break
            if "top_logprobs" not in api_logprobs[i] or "template" in t["token_groups"]:
                continue
            curr_probs = {}
            for logprob_dic in api_logprobs[i]["top_logprobs"]:
                curr_probs[logprob_dic["token"]] = np.round(
                    np.exp(logprob_dic["logprob"]), 4
                )
            step_dic["output_tokens"][i]["probabilities"] = curr_probs

        steps.append(step_dic)

    out = {
        "grid_params": grid_params,
        "model_params": model_params,
        "prompt": prompt,
        "steps": steps,
    }
    with open(output_path, "w") as f:
        json.dump(out, f, cls=CompactJSONEncoder, ensure_ascii=False, indent=4)


def get_multiple_trajectories_coin_env(
    grid_sizes: list[int] = [11],
    grid_complexities: list[float] = [0.8],
    target_dead_ends: int | None = None,
    place_at_dead_ends: bool = False,
    coin_placement: Literal["random", "dead_end"] | None = "random",
    min_manhattan_distance: int = 3,
    max_attempts: int = 1000,
    num_trajectories_per_grid: int = 5,
    num_grids_per_config: int = 1,
    max_steps_per_trajectory: int = 50,
    max_tokens: int = 10000,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_logprobs: int = 5,
    seed: int = 42,
    reasoning_efforts: list[Literal["low", "medium", "high"]] = ["low"],
    model_names: list[str] = ["together_ai/openai/gpt-oss-20b"],
    observation_placeholders: list[str] = ["grid_state"],
    output_dir: str = ".",
    verbose: bool = False,
    enable_dynamic_max_steps: bool = False,
    max_workers: int | None = None,
    max_workers_per_grid: int | None = None,
    enable_rate_limit: bool = False,
    rate_limit: int = 1000,
    rate_limit_period: float = 300.0,
    hf_repo_id: str | None = None,
    hf_path_prefix: str = "",
    hf_token: str | None = None,
    template_name: str = "grid_full_observability_hidden_goals.j2",
    track_history: bool = False,
):
    """Generate multiple trajectories on coin environments for each size × complexity × model combination.

    Like get_trajectories_multiple_per_grid but uses CoinNavigationEnv and supports
    dead-end constraints and coin placement. For each (grid_size, grid_complexity,
    model_name) combination, generates num_grids_per_config unique grid layouts and
    runs num_trajectories_per_grid trajectories on each layout in parallel.

    Each trajectory uses a deepcopy of the master environment, so the coin and grid
    layout are identical across trajectories on the same grid.

    Args:
        grid_sizes: List of grid sizes to use.
        grid_complexities: List of grid complexity levels.
        target_dead_ends: Exact dead-end count required. If None, any environment is accepted.
        place_at_dead_ends: If True, place agent start and goal at dead-end cells.
        coin_placement: Where to place the coin. "random" = any open cell; "dead_end" = a
            dead-end cell (falls back to random if none available); None = no coin.
        min_manhattan_distance: Minimum pairwise Manhattan distance required between agent
            start, coin, and goal. Grids that don't satisfy this are rejected and retried.
        max_attempts: Maximum generation attempts when target_dead_ends is set.
        num_trajectories_per_grid: Number of trajectories to generate per grid layout.
        num_grids_per_config: Number of distinct grid layouts per (size, complexity, model).
        max_steps_per_trajectory: Maximum steps per trajectory.
        max_tokens: Maximum tokens per model generation step.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        top_logprobs: Number of top log probabilities to return.
        seed: Base random seed; each grid uses seed + grid_id.
        reasoning_efforts: List of reasoning effort levels to run on each grid. Each
            level produces num_trajectories_per_grid trajectories.
        model_names: List of model names in "provider/model_id" format.
        observation_placeholders: Placeholder names in the prompt template.
        output_dir: Directory to save output JSON files.
        verbose: If True, print detailed logging.
        enable_dynamic_max_steps: If True, override max_steps_per_trajectory with
            1.5× the A* optimal path length.
        max_workers: Max parallel workers for grid-level parallelism. Defaults to
            min(32, total_grid_tasks).
        max_workers_per_grid: Max parallel workers within each grid. Defaults to
            num_trajectories_per_grid.
        enable_rate_limit: If True, enforce API rate limiting.
        rate_limit: Max requests per rate_limit_period.
        rate_limit_period: Rate limit time window in seconds.
        hf_repo_id: Hugging Face repo to upload results to. None = no upload.
        hf_path_prefix: Path prefix within the HF repo.
        hf_token: Hugging Face API token.

    Returns:
        list[str] | None: List of HF URLs if uploaded, otherwise None.
        Results are saved to individual JSON files in output_dir with format:
        {model_sanitized}_size{grid_size}_comp{grid_complexity}_grid{grid_id}_coin_{effort}_traj{traj_id}.json
        Grid layouts are saved as:
        {model_sanitized}_size{grid_size}_comp{grid_complexity}_grid{grid_id}_coin_layout.json
    """
    output_dir = str(_resolve_output(output_dir))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    grid_configs = list(
        product(
            grid_sizes,
            grid_complexities,
            model_names,
            range(num_grids_per_config),
        )
    )

    total_grid_tasks = len(grid_configs)
    total_trajectories = (
        total_grid_tasks * num_trajectories_per_grid * len(reasoning_efforts)
    )

    logger.info(
        f"Generating {total_trajectories} coin-env trajectories across {len(grid_sizes)} "
        f"grid sizes, {len(grid_complexities)} complexities, {len(model_names)} models, "
        f"with {num_grids_per_config} grids each, {num_trajectories_per_grid} trajectories per grid, "
        f"and reasoning efforts {reasoning_efforts}."
    )

    rate_limiter: RateLimiter | None = None
    if enable_rate_limit:
        rate_limiter = RateLimiter(rate_limit=rate_limit, period=rate_limit_period)

    def _build_coin_env(
        grid_size: int, grid_complexity: float, rng_seed: int
    ) -> tuple[FullObservabilityTextWrapper, list | None, int]:
        """Build and reset a coin environment, retrying until all constraints are met.

        Returns (wrapped_env, coin_pos, actual_dead_ends). The env is already reset
        and has the coin placed — do not call reset() again or the grid will regenerate.
        """
        np.random.seed(rng_seed)

        # Use CoinNavigationEnv only when a coin is requested; Simple2DNavigationEnv otherwise
        if coin_placement is not None:
            env_class = CoinNavigationEnv
        else:
            env_class = Simple2DNavigationEnv

        base_env_unwrapped = None
        coin_pos = None

        for _ in range(max_attempts):
            candidate = env_class(
                size=grid_size,
                complexity=grid_complexity,
                place_at_dead_ends=place_at_dead_ends,
            )
            candidate.reset()

            # Reject grids that don't meet the dead-end count requirement
            if (
                target_dead_ends is not None
                and len(get_all_dead_ends(candidate)) != target_dead_ends
            ):
                candidate.close()
                continue

            start = tuple(candidate.agent_pos)
            goal = tuple(candidate.goal_pos)

            # Reject grids where start and goal are too close together
            if manhattan_distance(start, goal) < min_manhattan_distance:
                candidate.close()
                continue

            # Find valid coin positions satisfying distance constraints
            resolved_coin_pos = None
            if coin_placement is not None:
                excluded = {start, goal}
                effective_placement = coin_placement

                # Prefer dead-end cells; fall back to random if none available
                if effective_placement == "dead_end":
                    raw = [
                        (x, y)
                        for (x, y) in get_all_dead_ends(candidate)
                        if (x, y) not in excluded
                    ]
                    if not raw:
                        logger.warning(
                            "No valid dead end for coin placement; falling back to random."
                        )
                        effective_placement = "random"
                if effective_placement == "random":
                    raw = [
                        (x, y)
                        for x in range(1, candidate.width - 1)
                        for y in range(1, candidate.height - 1)
                        if candidate.grid.get(x, y) is None and (x, y) not in excluded
                    ]

                # Filter to positions that are far enough from both start and goal
                valid = [
                    p
                    for p in raw
                    if manhattan_distance(start, p) >= min_manhattan_distance
                    and manhattan_distance(goal, p) >= min_manhattan_distance
                ]

                # Reject grids with no valid coin position
                if not valid:
                    candidate.close()
                    continue

                resolved_coin_pos = random.choice(valid)
                candidate.put_obj(Ball("yellow"), *resolved_coin_pos)

            base_env_unwrapped = candidate
            coin_pos = resolved_coin_pos
            break

        if base_env_unwrapped is None:
            raise RuntimeError(
                f"Could not generate an environment satisfying all constraints after {max_attempts} "
                f"attempts (size={grid_size}, complexity={grid_complexity}, "
                f"target_dead_ends={target_dead_ends}, min_manhattan_distance={min_manhattan_distance})."
            )

        actual_dead_ends = len(get_all_dead_ends(base_env_unwrapped))
        return (
            FullObservabilityTextWrapper(base_env_unwrapped),
            coin_pos,
            actual_dead_ends,
        )

    def _save_coin_grid_layout(
        env: FullObservabilityTextWrapper,
        grid_size: int,
        grid_complexity: float,
        grid_id: int,
        grid_seed: int,
        model_sanitized: str,
        coin_pos: list | None,
        actual_dead_ends: int,
    ) -> str:
        unwrapped = env.unwrapped

        grid_list = []
        for j in range(unwrapped.height):
            row = []
            for i in range(unwrapped.width):
                cell = unwrapped.grid.get(i, j)
                if (i, j) == tuple(unwrapped.agent_pos):
                    row.append("A")
                elif cell is None:
                    row.append("_")
                elif cell.type == "wall":
                    row.append("#")
                elif cell.type == "goal":
                    row.append("G")
                elif cell.type == "ball":
                    row.append("C")
                else:
                    row.append("?")
            grid_list.append(row)

        agent_pos = unwrapped.agent_pos
        agent_start_pos = (
            agent_pos.tolist() if hasattr(agent_pos, "tolist") else list(agent_pos)
        )
        if (
            hasattr(unwrapped, "_initial_agent_pos")
            and unwrapped._initial_agent_pos is not None
        ):
            p = unwrapped._initial_agent_pos
            agent_start_pos = p.tolist() if hasattr(p, "tolist") else list(p)

        agent_start_dir = getattr(unwrapped, "_initial_agent_dir", unwrapped.agent_dir)

        grid_data = {
            "grid_id": grid_id,
            "grid_seed": grid_seed,
            "grid_size": grid_size,
            "grid_complexity": grid_complexity,
            "grid_width": unwrapped.width,
            "grid_height": unwrapped.height,
            "target_dead_ends": target_dead_ends,
            "actual_dead_ends": actual_dead_ends,
            "place_at_dead_ends": place_at_dead_ends,
            "coin_placement": coin_placement,
            "coin_pos": list(coin_pos) if coin_pos is not None else None,
            "agent_start_pos": agent_start_pos,
            "agent_start_dir": agent_start_dir,
            "goal_pos": list(unwrapped.goal_pos),
            "grid_layout": grid_list,
            "grid_text": env._render(),
            "legend": env.grid_cells,
        }

        grid_filename = (
            f"{model_sanitized}_size{grid_size}_comp{grid_complexity}"
            f"_grid{grid_id}_coin_layout.json"
        )
        grid_path = str(Path(output_dir) / grid_filename)
        with open(grid_path, "w") as f:
            json.dump(grid_data, f, indent=2)

        if verbose:
            logger.info(f"Saved coin grid layout to {grid_path}")

        return grid_path

    def _generate_trajectories_for_grid(config: tuple) -> dict:
        grid_size, grid_complexity, model_name, grid_id = config
        grid_seed = seed + grid_id
        model_sanitized = model_name.replace("/", "_").replace(".", "_")

        master_env, coin_pos, actual_dead_ends = _build_coin_env(
            grid_size, grid_complexity, grid_seed
        )

        grid_path = _save_coin_grid_layout(
            env=master_env,
            grid_size=grid_size,
            grid_complexity=grid_complexity,
            grid_id=grid_id,
            grid_seed=grid_seed,
            model_sanitized=model_sanitized,
            coin_pos=coin_pos,
            actual_dead_ends=actual_dead_ends,
        )

        def _generate_single_trajectory(traj_id: int, effort: str) -> dict:
            if rate_limiter is not None:
                rate_limiter.acquire()

            traj_seed = grid_seed + traj_id * 1000 + hash(effort) % 10000
            env_copy = copy.deepcopy(master_env)

            output_filename = (
                f"{model_sanitized}_size{grid_size}_comp{grid_complexity}"
                f"_grid{grid_id}_coin_{effort}_traj{traj_id}.json"
            )
            output_path = str(Path(output_dir) / output_filename)

            if verbose:
                logger.info(
                    f"Starting coin trajectory: model={model_name}, size={grid_size}, "
                    f"complexity={grid_complexity}, grid={grid_id}, effort={effort}, traj={traj_id}"
                )

            try:
                get_trajectory(
                    grid_size=grid_size,
                    grid_complexity=grid_complexity,
                    max_steps_per_trajectory=max_steps_per_trajectory,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_logprobs=top_logprobs,
                    seed=traj_seed,
                    reasoning_effort=effort,
                    model_name=model_name,
                    observation_placeholders=observation_placeholders,
                    output_path=output_path,
                    verbose=verbose,
                    enable_dynamic_max_steps=enable_dynamic_max_steps,
                    env=env_copy,
                    use_safe_reset=True,
                    template_name=template_name,
                    track_history=track_history,
                )
                return {
                    "status": "success",
                    "output_path": output_path,
                    "config": config,
                    "traj_id": traj_id,
                    "effort": effort,
                }
            except Exception as e:
                logger.error(
                    f"Failed coin trajectory grid={grid_id}, effort={effort}, traj={traj_id}: {e}"
                )
                return {
                    "status": "error",
                    "error": str(e),
                    "config": config,
                    "traj_id": traj_id,
                    "effort": effort,
                }

        inner_workers = (
            max_workers_per_grid
            if max_workers_per_grid is not None
            else num_trajectories_per_grid * len(reasoning_efforts)
        )
        trajectory_results = []

        with ThreadPoolExecutor(max_workers=inner_workers) as inner_executor:
            futures = {
                inner_executor.submit(_generate_single_trajectory, traj_id, effort): (
                    traj_id,
                    effort,
                )
                for effort in reasoning_efforts
                for traj_id in range(num_trajectories_per_grid)
            }
            for future in as_completed(futures):
                traj_id, effort = futures[future]
                try:
                    trajectory_results.append(future.result())
                except Exception as e:
                    logger.error(
                        f"Exception for grid {grid_id}, effort {effort}, traj {traj_id}: {e}"
                    )
                    trajectory_results.append(
                        {
                            "status": "error",
                            "error": str(e),
                            "config": config,
                            "traj_id": traj_id,
                            "effort": effort,
                        }
                    )

        return {"trajectory_results": trajectory_results, "grid_paths": [grid_path]}

    if max_workers is None:
        max_workers = min(32, total_grid_tasks)

    if enable_rate_limit and rate_limiter is not None:
        sustainable_workers = min(max_workers, int(rate_limiter.tokens_per_second * 2))
        if sustainable_workers < max_workers:
            logger.info(
                f"Adjusting max_workers from {max_workers} to {sustainable_workers} "
                f"based on rate limit ({rate_limiter.tokens_per_second:.2f} req/sec)"
            )
            max_workers = sustainable_workers

    all_trajectory_results = []
    all_grid_paths = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_generate_trajectories_for_grid, config): config
            for config in grid_configs
        }
        for future in tqdm(
            as_completed(futures),
            total=total_grid_tasks,
            desc="Processing coin grids",
            unit="grid",
        ):
            config = futures[future]
            try:
                result = future.result()
                all_trajectory_results.extend(result["trajectory_results"])
                all_grid_paths.extend(result["grid_paths"])
                failures = [
                    r for r in result["trajectory_results"] if r["status"] != "success"
                ]
                for failure in failures:
                    tqdm.write(
                        f"Failed for grid {config}, traj {failure['traj_id']}: "
                        f"{failure.get('error', 'Unknown error')}"
                    )
            except Exception as e:
                tqdm.write(f"Exception for grid {config}: {e}")
                for traj_id in range(num_trajectories_per_grid):
                    all_trajectory_results.append(
                        {
                            "status": "error",
                            "error": str(e),
                            "config": config,
                            "traj_id": traj_id,
                        }
                    )

    success_count = sum(1 for r in all_trajectory_results if r["status"] == "success")
    logger.info(
        f"Completed {success_count}/{total_trajectories} coin-env trajectories successfully."
    )
    logger.info(f"Saved {len(all_grid_paths)} grid layout files.")

    if hf_repo_id is not None and (success_count > 0 or all_grid_paths):
        all_paths_to_upload = [
            r["output_path"] for r in all_trajectory_results if r["status"] == "success"
        ]
        all_paths_to_upload.extend(all_grid_paths)
        if all_paths_to_upload:
            return upload_files_to_huggingface(
                file_paths=all_paths_to_upload,
                repo_id=hf_repo_id,
                path_prefix=hf_path_prefix,
                hf_token=hf_token,
            )
    return None


def get_single_trajectory_two_coin_env(
    grid_size: int = 11,
    grid_complexity: float = 0.8,
    target_dead_ends: int | None = None,
    place_at_dead_ends: bool = False,
    coin_placement: Literal["random", "dead_end"] = "random",
    max_attempts: int = 1000,
    min_manhattan_distance: int = 3,
    max_steps_per_trajectory: int = 50,
    max_tokens: int = 10000,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_logprobs: int = 5,
    seed: int = 42,
    reasoning_effort: Literal["low", "medium", "high"] = "low",
    model_name: str = "together_ai/openai/gpt-oss-20b",
    template_name: str = "grid_full_observability_two_coins_collect_one.j2",
    observation_placeholders: list[str] = ["grid_state"],
    output_path: str = "get_trajectory_two_coin_example_output.json",
    verbose: bool = False,
    enable_dynamic_max_steps: bool = False,
    track_history: bool = False,
):
    """Generate an agent trajectory in a two-coin navigation environment and save results to JSON.

    Creates a TwoCoinNavigationEnv with two coins placed according to coin_placement, with
    minimum Manhattan distance constraints between start, goal, and both coins. Environments
    are retried up to max_attempts until all constraints are satisfied.

    Two coins are always placed. The template_name parameter controls what objective the agent
    is given — use grid_full_observability_two_coins_collect_one.j2 to instruct the agent to
    collect exactly one coin, or grid_full_observability_two_coins_collect_all.j2 to collect
    both. The avoid-coin template can also be used (grid_full_observability_avoid_coin.j2).

    Args:
        grid_size: Size of the square grid environment.
        grid_complexity: Complexity level of the grid (0.0 = open room, 1.0 = perfect maze).
        target_dead_ends: Exact number of dead ends required. If None, no constraint.
        place_at_dead_ends: If True, place agent start and goal at dead-end cells.
        coin_placement: Strategy for placing each coin: "random" = any open cell;
            "dead_end" = dead-end cell (falls back to random if insufficient dead ends).
        max_attempts: Maximum generation attempts.
        min_manhattan_distance: Minimum pairwise Manhattan distance required between agent
            start, goal, and both coin positions.
        max_steps_per_trajectory: Maximum steps per trajectory.
        max_tokens: Maximum tokens per model generation step.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        top_logprobs: Number of top log probabilities to return per token.
        seed: Random seed for reproducibility.
        reasoning_effort: Reasoning effort level ("low", "medium", or "high").
        model_name: Model name in "provider/model_id" format.
        template_name: Jinja2 template to use. Determines the agent's coin objective.
        observation_placeholders: Placeholder names in the prompt template.
        output_path: Path to save the output JSON file.
        verbose: If True, print detailed logging.
        enable_dynamic_max_steps: If True, override max_steps_per_trajectory with 1.5×
            the A* optimal path length.

    Returns:
        None. Results are saved to output_path.
    """
    output_path = str(_resolve_output(output_path))

    base_env_unwrapped = None
    coin1_pos = None
    coin2_pos = None

    np.random.seed(seed)

    for _ in range(max_attempts):
        candidate = TwoCoinNavigationEnv(
            size=grid_size,
            complexity=grid_complexity,
            place_at_dead_ends=place_at_dead_ends,
        )
        candidate.reset()

        if (
            target_dead_ends is not None
            and len(get_all_dead_ends(candidate)) != target_dead_ends
        ):
            candidate.close()
            continue

        start = tuple(candidate.agent_pos)
        goal = tuple(candidate.goal_pos)

        if manhattan_distance(start, goal) < min_manhattan_distance:
            candidate.close()
            continue

        excluded = {start, goal}
        effective_placement = coin_placement

        if effective_placement == "dead_end":
            raw = [
                (x, y)
                for (x, y) in get_all_dead_ends(candidate)
                if (x, y) not in excluded
            ]
            if len(raw) < 2:
                logger.warning(
                    "Fewer than 2 dead ends for two-coin placement; falling back to random."
                )
                effective_placement = "random"
        if effective_placement == "random":
            raw = [
                (x, y)
                for x in range(1, candidate.width - 1)
                for y in range(1, candidate.height - 1)
                if candidate.grid.get(x, y) is None and (x, y) not in excluded
            ]

        valid_1 = [
            p
            for p in raw
            if manhattan_distance(start, p) >= min_manhattan_distance
            and manhattan_distance(goal, p) >= min_manhattan_distance
        ]
        if not valid_1:
            candidate.close()
            continue

        resolved_coin1_pos = random.choice(valid_1)
        excluded.add(resolved_coin1_pos)

        valid_2 = [
            p
            for p in raw
            if p not in excluded
            and manhattan_distance(start, p) >= min_manhattan_distance
            and manhattan_distance(goal, p) >= min_manhattan_distance
            and manhattan_distance(resolved_coin1_pos, p) >= min_manhattan_distance
        ]
        if not valid_2:
            candidate.close()
            continue

        resolved_coin2_pos = random.choice(valid_2)
        candidate.put_obj(Ball("yellow"), *resolved_coin1_pos)
        candidate.put_obj(Ball("yellow"), *resolved_coin2_pos)

        base_env_unwrapped = candidate
        coin1_pos = resolved_coin1_pos
        coin2_pos = resolved_coin2_pos
        break

    if base_env_unwrapped is None:
        raise RuntimeError(
            f"Could not generate a two-coin environment satisfying all constraints after {max_attempts} "
            f"attempts (size={grid_size}, complexity={grid_complexity}, "
            f"target_dead_ends={target_dead_ends}, min_manhattan_distance={min_manhattan_distance})."
        )

    actual_dead_ends = len(get_all_dead_ends(base_env_unwrapped))
    base_env = FullObservabilityTextWrapper(base_env_unwrapped)

    template_path = Path(__file__).parent.parent.parent / "templates" / template_name
    agent = LLMAgent(model_name=model_name, template_path=template_path, track_history=track_history)
    model_id = "/".join(model_name.split("/")[1:])
    provider = model_name.split("/")[0]
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_id)

    traj = generate_trajectory(
        env=base_env,
        agent=agent,
        max_steps_per_trajectory=max_steps_per_trajectory,
        generation_kwargs={
            "top_logprobs": top_logprobs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "reasoning_effort": reasoning_effort,
            "seed": seed,
        },
        verbose=verbose,
        enable_dynamic_max_steps=enable_dynamic_max_steps,
        skip_reset=True,
    )

    grid_params = {}
    grid_params["grid_width"] = base_env_unwrapped.width
    grid_params["grid_height"] = base_env_unwrapped.height
    grid_params["grid_complexity"] = grid_complexity
    grid_params["target_dead_ends"] = target_dead_ends
    grid_params["actual_dead_ends"] = actual_dead_ends
    grid_params["place_at_dead_ends"] = place_at_dead_ends
    grid_params["coin_placement"] = coin_placement
    grid_params["coin_pos_1"] = list(coin1_pos)
    grid_params["coin_pos_2"] = list(coin2_pos)
    grid_params["fully_observable"] = True
    grid_params["astar_distance"] = traj.traj_metadata["astar_distance"]
    grid_params["agent_start_coordinates"] = traj.traj_metadata[
        "agent_start_coordinates"
    ]
    grid_params["goal_coordinates"] = traj.traj_metadata["goal_coordinates"]
    grid_params["legend"] = base_env.grid_cells

    grid_symbols = [cell["symbol"] for cell in base_env.grid_cells.values()]

    model_params = {}
    model_params["model_id"] = model_id
    model_params["provider"] = provider
    model_params["interface"] = "litellm"
    model_params["template_name"] = template_name
    model_params["n_interactions_in_context"] = 0
    model_params["max_tokens"] = max_tokens
    model_params["max_steps_per_trajectory"] = max_steps_per_trajectory
    model_params["temperature"] = temperature
    model_params["reasoning_effort"] = reasoning_effort
    model_params["top_p"] = top_p
    model_params["top_logprobs"] = top_logprobs
    model_params["seed"] = seed

    prompt = {}
    render_kwargs = {"grid_state": "{{grid_state}}"}
    template = agent._template.render(**render_kwargs)
    formatted_template: str = tokenizer.apply_chat_template(
        [{"role": "user", "content": template}],
        tokenize=False,
        add_generation_prompt=True,
    )
    template_tokens = to_dic_list(formatted_template, tokenizer)

    observation_placeholder = "{{" + observation_placeholders[0] + "}}"
    prefix, suffix = formatted_template.split(observation_placeholder)
    raw_prefix, raw_suffix = template.split(observation_placeholder)
    prompt["prompt_template"] = formatted_template
    prompt["prompt_template_n_tokens"] = len(template_tokens)
    prompt["prompt_prefix_tokens"] = to_dic_list(prefix, tokenizer)
    raw_prefix_tokens = to_dic_list(raw_prefix, tokenizer)
    start_raw_prefix_idx = len(prompt["prompt_prefix_tokens"]) - len(raw_prefix_tokens)
    for i in range(start_raw_prefix_idx):
        prompt["prompt_prefix_tokens"][i]["token_groups"] += ["template"]
    prompt["prompt_prefix_n_tokens"] = len(prompt["prompt_prefix_tokens"])
    prompt["prompt_placeholder_tokens"] = to_dic_list(
        observation_placeholder, tokenizer, groups=["prompt", "placeholder"]
    )
    prompt["prompt_placeholder_n_tokens"] = len(prompt["prompt_placeholder_tokens"])
    prompt["prompt_suffix_tokens"] = to_dic_list(suffix, tokenizer)
    raw_suffix_tokens = to_dic_list(raw_suffix, tokenizer)
    start_raw_suffix_idx = (
        len(prompt["prompt_suffix_tokens"]) - len(raw_suffix_tokens) + 1
    )
    for i in range(
        len(prompt["prompt_suffix_tokens"]) - start_raw_suffix_idx,
        len(prompt["prompt_suffix_tokens"]) - 1,
    ):
        prompt["prompt_suffix_tokens"][i]["token_groups"] += ["template"]
    prompt["prompt_suffix_n_tokens"] = len(prompt["prompt_suffix_tokens"])

    steps = []
    for step_id, traj_step in enumerate(traj.steps):
        step_dic = {}
        step_dic["step_id"] = step_id
        step_dic["grid_state"] = traj_step.observation.split("\n")
        step_dic["grid_state_tokens"] = to_dic_list(
            traj_step.observation, tokenizer, groups=["prompt", "grid_state"]
        )
        step_dic["grid_state_n_tokens"] = len(step_dic["grid_state_tokens"])

        for i, t in enumerate(step_dic["grid_state_tokens"]):
            if any(sym in t["token"] for sym in grid_symbols):
                step_dic["grid_state_tokens"][i]["token_groups"] += ["grid_tile"]

        step_dic["prompt_suffix_tokens"] = prompt["prompt_suffix_tokens"]
        step_dic["prompt_suffix_n_tokens"] = len(step_dic["prompt_suffix_tokens"])
        step_dic["agent_action"] = traj_step.metadata["action"]
        if "coins_collected" in traj_step.metadata:
            step_dic["coins_collected"] = traj_step.metadata["coins_collected"]

        out_tokens = [t["token"] for t in traj_step.metadata["logprobs"]]
        step_dic["output_text"] = tokenizer.convert_tokens_to_string(out_tokens)
        step_dic["output_tokens"] = to_dic_list(
            step_dic["output_text"], tokenizer, groups=["output"]
        )
        step_dic["output_n_tokens"] = len(step_dic["output_tokens"])
        step_dic["output_tokens"] = annotate_output_tokens(
            model_name, step_dic["output_tokens"]
        )

        api_logprobs = traj_step.metadata["logprobs"]
        for i, t in enumerate(step_dic["output_tokens"]):
            if i >= len(api_logprobs):
                break
            if "top_logprobs" not in api_logprobs[i] or "template" in t["token_groups"]:
                continue
            curr_probs = {}
            for logprob_dic in api_logprobs[i]["top_logprobs"]:
                curr_probs[logprob_dic["token"]] = np.round(
                    np.exp(logprob_dic["logprob"]), 4
                )
            step_dic["output_tokens"][i]["probabilities"] = curr_probs

        steps.append(step_dic)

    out = {
        "grid_params": grid_params,
        "model_params": model_params,
        "prompt": prompt,
        "steps": steps,
    }
    with open(output_path, "w") as f:
        json.dump(out, f, cls=CompactJSONEncoder, ensure_ascii=False, indent=4)


def get_multiple_trajectories_two_coin_env(
    grid_sizes: list[int] = [11],
    grid_complexities: list[float] = [0.8],
    target_dead_ends: int | None = None,
    place_at_dead_ends: bool = False,
    coin_placement: Literal["random", "dead_end"] = "random",
    min_manhattan_distance: int = 3,
    max_attempts: int = 1000,
    num_trajectories_per_grid: int = 5,
    num_grids_per_config: int = 1,
    max_steps_per_trajectory: int = 50,
    max_tokens: int = 10000,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_logprobs: int = 5,
    seed: int = 42,
    reasoning_efforts: list[Literal["low", "medium", "high"]] = ["low"],
    model_names: list[str] = ["together_ai/openai/gpt-oss-20b"],
    observation_placeholders: list[str] = ["grid_state"],
    output_dir: str = ".",
    verbose: bool = False,
    enable_dynamic_max_steps: bool = False,
    max_workers: int | None = None,
    max_workers_per_grid: int | None = None,
    enable_rate_limit: bool = False,
    rate_limit: int = 1000,
    rate_limit_period: float = 300.0,
    hf_repo_id: str | None = None,
    hf_path_prefix: str = "",
    hf_token: str | None = None,
    template_name: str = "grid_full_observability_two_coins_collect_one.j2",
    track_history: bool = False,
):
    """Generate multiple trajectories on two-coin environments for each size × complexity × model combination.

    Like get_multiple_trajectories_coin_env but uses TwoCoinNavigationEnv and places two coins
    per grid. For each (grid_size, grid_complexity, model_name) combination, generates
    num_grids_per_config unique grid layouts and runs num_trajectories_per_grid trajectories on
    each layout in parallel.

    Both coins must satisfy the min_manhattan_distance constraint relative to start, goal, and
    each other. The template_name parameter controls the coin objective given to the agent.

    Args:
        grid_sizes: List of grid sizes to use.
        grid_complexities: List of grid complexity levels.
        target_dead_ends: Exact dead-end count required. If None, any environment is accepted.
        place_at_dead_ends: If True, place agent start and goal at dead-end cells.
        coin_placement: Strategy for placing coins: "random" = any open cell; "dead_end" =
            dead-end cells (falls back to random if fewer than 2 dead ends available).
        min_manhattan_distance: Minimum pairwise Manhattan distance required between agent
            start, goal, and both coin positions.
        max_attempts: Maximum generation attempts per grid.
        num_trajectories_per_grid: Number of trajectories to generate per grid layout.
        num_grids_per_config: Number of distinct grid layouts per (size, complexity, model).
        max_steps_per_trajectory: Maximum steps per trajectory.
        max_tokens: Maximum tokens per model generation step.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        top_logprobs: Number of top log probabilities to return.
        seed: Base random seed; each grid uses seed + grid_id.
        reasoning_efforts: List of reasoning effort levels to run on each grid.
        model_names: List of model names in "provider/model_id" format.
        observation_placeholders: Placeholder names in the prompt template.
        output_dir: Directory to save output JSON files.
        verbose: If True, print detailed logging.
        enable_dynamic_max_steps: If True, override max_steps_per_trajectory with 1.5×
            the A* optimal path length.
        max_workers: Max parallel workers for grid-level parallelism.
        max_workers_per_grid: Max parallel workers within each grid.
        enable_rate_limit: If True, enforce API rate limiting.
        rate_limit: Max requests per rate_limit_period.
        rate_limit_period: Rate limit time window in seconds.
        hf_repo_id: Hugging Face repo to upload results to. None = no upload.
        hf_path_prefix: Path prefix within the HF repo.
        hf_token: Hugging Face API token.
        template_name: Jinja2 template to use. Determines the agent's coin objective.

    Returns:
        list[str] | None: List of HF URLs if uploaded, otherwise None.
        Results are saved to individual JSON files in output_dir with format:
        {model_sanitized}_size{grid_size}_comp{grid_complexity}_grid{grid_id}_twocoin_{effort}_traj{traj_id}.json
        Grid layouts are saved as:
        {model_sanitized}_size{grid_size}_comp{grid_complexity}_grid{grid_id}_twocoin_layout.json
    """
    output_dir = str(_resolve_output(output_dir))
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    grid_configs = list(
        product(
            grid_sizes,
            grid_complexities,
            model_names,
            range(num_grids_per_config),
        )
    )

    total_grid_tasks = len(grid_configs)
    total_trajectories = (
        total_grid_tasks * num_trajectories_per_grid * len(reasoning_efforts)
    )

    logger.info(
        f"Generating {total_trajectories} two-coin-env trajectories across {len(grid_sizes)} "
        f"grid sizes, {len(grid_complexities)} complexities, {len(model_names)} models, "
        f"with {num_grids_per_config} grids each, {num_trajectories_per_grid} trajectories per grid, "
        f"and reasoning efforts {reasoning_efforts}."
    )

    rate_limiter: RateLimiter | None = None
    if enable_rate_limit:
        rate_limiter = RateLimiter(rate_limit=rate_limit, period=rate_limit_period)

    def _build_two_coin_env(
        grid_size: int, grid_complexity: float, rng_seed: int
    ) -> tuple[FullObservabilityTextWrapper, list, list, int]:
        """Build and reset a two-coin environment, retrying until all constraints are met.

        Returns (wrapped_env, coin1_pos, coin2_pos, actual_dead_ends). The env is already
        reset and has both coins placed — do not call reset() again.
        """
        np.random.seed(rng_seed)

        resolved_coin1_pos = None
        resolved_coin2_pos = None
        base_env_unwrapped = None

        for _ in range(max_attempts):
            candidate = TwoCoinNavigationEnv(
                size=grid_size,
                complexity=grid_complexity,
                place_at_dead_ends=place_at_dead_ends,
            )
            candidate.reset()

            if (
                target_dead_ends is not None
                and len(get_all_dead_ends(candidate)) != target_dead_ends
            ):
                candidate.close()
                continue

            start = tuple(candidate.agent_pos)
            goal = tuple(candidate.goal_pos)

            if manhattan_distance(start, goal) < min_manhattan_distance:
                candidate.close()
                continue

            excluded = {start, goal}
            effective_placement = coin_placement

            if effective_placement == "dead_end":
                raw = [
                    (x, y)
                    for (x, y) in get_all_dead_ends(candidate)
                    if (x, y) not in excluded
                ]
                if len(raw) < 2:
                    logger.warning(
                        "Fewer than 2 dead ends for two-coin placement; falling back to random."
                    )
                    effective_placement = "random"
            if effective_placement == "random":
                raw = [
                    (x, y)
                    for x in range(1, candidate.width - 1)
                    for y in range(1, candidate.height - 1)
                    if candidate.grid.get(x, y) is None and (x, y) not in excluded
                ]

            valid_1 = [
                p
                for p in raw
                if manhattan_distance(start, p) >= min_manhattan_distance
                and manhattan_distance(goal, p) >= min_manhattan_distance
            ]
            if not valid_1:
                candidate.close()
                continue

            c1 = random.choice(valid_1)
            excluded.add(c1)

            valid_2 = [
                p
                for p in raw
                if p not in excluded
                and manhattan_distance(start, p) >= min_manhattan_distance
                and manhattan_distance(goal, p) >= min_manhattan_distance
                and manhattan_distance(c1, p) >= min_manhattan_distance
            ]
            if not valid_2:
                candidate.close()
                continue

            c2 = random.choice(valid_2)
            candidate.put_obj(Ball("yellow"), *c1)
            candidate.put_obj(Ball("yellow"), *c2)

            base_env_unwrapped = candidate
            resolved_coin1_pos = c1
            resolved_coin2_pos = c2
            break

        if base_env_unwrapped is None:
            raise RuntimeError(
                f"Could not generate a two-coin environment satisfying all constraints after {max_attempts} "
                f"attempts (size={grid_size}, complexity={grid_complexity}, "
                f"target_dead_ends={target_dead_ends}, min_manhattan_distance={min_manhattan_distance})."
            )

        actual_dead_ends = len(get_all_dead_ends(base_env_unwrapped))
        return (
            FullObservabilityTextWrapper(base_env_unwrapped),
            resolved_coin1_pos,
            resolved_coin2_pos,
            actual_dead_ends,
        )

    def _save_two_coin_grid_layout(
        env: FullObservabilityTextWrapper,
        grid_size: int,
        grid_complexity: float,
        grid_id: int,
        grid_seed: int,
        model_sanitized: str,
        coin1_pos: list,
        coin2_pos: list,
        actual_dead_ends: int,
    ) -> str:
        unwrapped = env.unwrapped

        grid_list = []
        for j in range(unwrapped.height):
            row = []
            for i in range(unwrapped.width):
                cell = unwrapped.grid.get(i, j)
                if (i, j) == tuple(unwrapped.agent_pos):
                    row.append("A")
                elif cell is None:
                    row.append("_")
                elif cell.type == "wall":
                    row.append("#")
                elif cell.type == "goal":
                    row.append("G")
                elif cell.type == "ball":
                    row.append("C")
                else:
                    row.append("?")
            grid_list.append(row)

        agent_pos = unwrapped.agent_pos
        agent_start_pos = (
            agent_pos.tolist() if hasattr(agent_pos, "tolist") else list(agent_pos)
        )
        if (
            hasattr(unwrapped, "_initial_agent_pos")
            and unwrapped._initial_agent_pos is not None
        ):
            p = unwrapped._initial_agent_pos
            agent_start_pos = p.tolist() if hasattr(p, "tolist") else list(p)

        agent_start_dir = getattr(unwrapped, "_initial_agent_dir", unwrapped.agent_dir)

        grid_data = {
            "grid_id": grid_id,
            "grid_seed": grid_seed,
            "grid_size": grid_size,
            "grid_complexity": grid_complexity,
            "grid_width": unwrapped.width,
            "grid_height": unwrapped.height,
            "target_dead_ends": target_dead_ends,
            "actual_dead_ends": actual_dead_ends,
            "place_at_dead_ends": place_at_dead_ends,
            "coin_placement": coin_placement,
            "coin_pos_1": list(coin1_pos),
            "coin_pos_2": list(coin2_pos),
            "agent_start_pos": agent_start_pos,
            "agent_start_dir": agent_start_dir,
            "goal_pos": list(unwrapped.goal_pos),
            "grid_layout": grid_list,
            "grid_text": env._render(),
            "legend": env.grid_cells,
        }

        grid_filename = (
            f"{model_sanitized}_size{grid_size}_comp{grid_complexity}"
            f"_grid{grid_id}_twocoin_layout.json"
        )
        grid_path = str(Path(output_dir) / grid_filename)
        with open(grid_path, "w") as f:
            json.dump(grid_data, f, indent=2)

        if verbose:
            logger.info(f"Saved two-coin grid layout to {grid_path}")

        return grid_path

    def _generate_trajectories_for_grid(config: tuple) -> dict:
        grid_size, grid_complexity, model_name, grid_id = config
        grid_seed = seed + grid_id
        model_sanitized = model_name.replace("/", "_").replace(".", "_")

        master_env, coin1_pos, coin2_pos, actual_dead_ends = _build_two_coin_env(
            grid_size, grid_complexity, grid_seed
        )

        grid_path = _save_two_coin_grid_layout(
            env=master_env,
            grid_size=grid_size,
            grid_complexity=grid_complexity,
            grid_id=grid_id,
            grid_seed=grid_seed,
            model_sanitized=model_sanitized,
            coin1_pos=coin1_pos,
            coin2_pos=coin2_pos,
            actual_dead_ends=actual_dead_ends,
        )

        def _generate_single_trajectory(traj_id: int, effort: str) -> dict:
            if rate_limiter is not None:
                rate_limiter.acquire()

            traj_seed = grid_seed + traj_id * 1000 + hash(effort) % 10000
            env_copy = copy.deepcopy(master_env)

            output_filename = (
                f"{model_sanitized}_size{grid_size}_comp{grid_complexity}"
                f"_grid{grid_id}_twocoin_{effort}_traj{traj_id}.json"
            )
            output_path = str(Path(output_dir) / output_filename)

            if verbose:
                logger.info(
                    f"Starting two-coin trajectory: model={model_name}, size={grid_size}, "
                    f"complexity={grid_complexity}, grid={grid_id}, effort={effort}, traj={traj_id}"
                )

            try:
                get_trajectory(
                    grid_size=grid_size,
                    grid_complexity=grid_complexity,
                    max_steps_per_trajectory=max_steps_per_trajectory,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_logprobs=top_logprobs,
                    seed=traj_seed,
                    reasoning_effort=effort,
                    model_name=model_name,
                    observation_placeholders=observation_placeholders,
                    output_path=output_path,
                    verbose=verbose,
                    enable_dynamic_max_steps=enable_dynamic_max_steps,
                    env=env_copy,
                    use_safe_reset=True,
                    template_name=template_name,
                    track_history=track_history,
                )
                return {
                    "status": "success",
                    "output_path": output_path,
                    "config": config,
                    "traj_id": traj_id,
                    "effort": effort,
                }
            except Exception as e:
                logger.error(
                    f"Failed two-coin trajectory grid={grid_id}, effort={effort}, traj={traj_id}: {e}"
                )
                return {
                    "status": "error",
                    "error": str(e),
                    "config": config,
                    "traj_id": traj_id,
                    "effort": effort,
                }

        inner_workers = (
            max_workers_per_grid
            if max_workers_per_grid is not None
            else num_trajectories_per_grid * len(reasoning_efforts)
        )
        trajectory_results = []

        with ThreadPoolExecutor(max_workers=inner_workers) as inner_executor:
            futures = {
                inner_executor.submit(_generate_single_trajectory, traj_id, effort): (
                    traj_id,
                    effort,
                )
                for effort in reasoning_efforts
                for traj_id in range(num_trajectories_per_grid)
            }
            for future in as_completed(futures):
                traj_id, effort = futures[future]
                try:
                    trajectory_results.append(future.result())
                except Exception as e:
                    logger.error(
                        f"Exception for grid {grid_id}, effort {effort}, traj {traj_id}: {e}"
                    )
                    trajectory_results.append(
                        {
                            "status": "error",
                            "error": str(e),
                            "config": config,
                            "traj_id": traj_id,
                            "effort": effort,
                        }
                    )

        return {"trajectory_results": trajectory_results, "grid_paths": [grid_path]}

    if max_workers is None:
        max_workers = min(32, total_grid_tasks)

    if enable_rate_limit and rate_limiter is not None:
        sustainable_workers = min(max_workers, int(rate_limiter.tokens_per_second * 2))
        if sustainable_workers < max_workers:
            logger.info(
                f"Adjusting max_workers from {max_workers} to {sustainable_workers} "
                f"based on rate limit ({rate_limiter.tokens_per_second:.2f} req/sec)"
            )
            max_workers = sustainable_workers

    all_trajectory_results = []
    all_grid_paths = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_generate_trajectories_for_grid, config): config
            for config in grid_configs
        }
        for future in tqdm(
            as_completed(futures),
            total=total_grid_tasks,
            desc="Processing two-coin grids",
            unit="grid",
        ):
            config = futures[future]
            try:
                result = future.result()
                all_trajectory_results.extend(result["trajectory_results"])
                all_grid_paths.extend(result["grid_paths"])
                failures = [
                    r for r in result["trajectory_results"] if r["status"] != "success"
                ]
                for failure in failures:
                    tqdm.write(
                        f"Failed for grid {config}, traj {failure['traj_id']}: "
                        f"{failure.get('error', 'Unknown error')}"
                    )
            except Exception as e:
                tqdm.write(f"Exception for grid {config}: {e}")
                for traj_id in range(num_trajectories_per_grid):
                    all_trajectory_results.append(
                        {
                            "status": "error",
                            "error": str(e),
                            "config": config,
                            "traj_id": traj_id,
                        }
                    )

    success_count = sum(1 for r in all_trajectory_results if r["status"] == "success")
    logger.info(
        f"Completed {success_count}/{total_trajectories} two-coin-env trajectories successfully."
    )
    logger.info(f"Saved {len(all_grid_paths)} grid layout files.")

    if hf_repo_id is not None and (success_count > 0 or all_grid_paths):
        all_paths_to_upload = [
            r["output_path"] for r in all_trajectory_results if r["status"] == "success"
        ]
        all_paths_to_upload.extend(all_grid_paths)
        if all_paths_to_upload:
            return upload_files_to_huggingface(
                file_paths=all_paths_to_upload,
                repo_id=hf_repo_id,
                path_prefix=hf_path_prefix,
                hf_token=hf_token,
            )
    return None


def augment_from_layouts(
    layout_dir: str,
    augmentations: list[
        Literal["start_goal_swap", "rotate", "reflect", "transpose", "random_starts"]
    ] = ["start_goal_swap", "rotate", "reflect", "transpose"],
    num_random_starts: int = 3,
    num_trajectories_per_variant: int = 5,
    max_steps_per_trajectory: int = 50,
    max_tokens: int = 10000,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_logprobs: int = 5,
    reasoning_efforts: list[Literal["low", "medium", "high"]] = ["low"],
    model_names: list[str] = ["together_ai/openai/gpt-oss-20b"],
    observation_placeholders: list[str] = ["grid_state"],
    output_dir: str = ".",
    verbose: bool = False,
    enable_dynamic_max_steps: bool = False,
    max_workers: int | None = None,
    enable_rate_limit: bool = False,
    rate_limit: int = 1000,
    rate_limit_period: float = 300.0,
    hf_repo_id: str | None = None,
    hf_path_prefix: str = "",
    hf_token: str | None = None,
    template_name: str = "grid_full_observability_hidden_goals.j2",
):
    """Load existing grid layout JSONs and generate trajectories on augmented variants.

    For each ``*_layout.json`` file in ``layout_dir``, reconstructs the environment and
    applies the requested augmentations:

    - ``"start_goal_swap"``: swap agent start and goal positions (coin stays fixed)
    - ``"rotate"``: 90° counter-clockwise rotation of the grid
    - ``"reflect"``: vertical reflection of the grid
    - ``"transpose"``: transpose of the grid
    - ``"random_starts"``: sample up to ``num_random_starts`` valid agent start positions
      on the same grid (same walls, goal, and coin)

    A new layout JSON is saved for each augmented variant, and
    ``num_trajectories_per_variant × len(reasoning_efforts)`` trajectories are generated
    per variant per model.

    Args:
        layout_dir: Directory containing ``*_layout.json`` files to augment.
        augmentations: Which augmentations to apply. Each produces a separate set of
            trajectories.
        num_random_starts: Number of random start positions to sample when
            ``"random_starts"`` is in augmentations.
        num_trajectories_per_variant: Number of trajectories per (variant, model, effort).
        max_steps_per_trajectory: Maximum steps per trajectory.
        max_tokens: Maximum tokens per model generation step.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        top_logprobs: Number of top log probabilities to return.
        reasoning_efforts: List of reasoning effort levels.
        model_names: List of model names in ``"provider/model_id"`` format.
        observation_placeholders: Placeholder names in the prompt template.
        output_dir: Directory to save output JSON files.
        verbose: If True, print detailed logging.
        enable_dynamic_max_steps: If True, override max_steps with 1.5× A* optimal length.
        max_workers: Max parallel workers. Defaults to min(32, total_variant_tasks).
        enable_rate_limit: If True, enforce API rate limiting.
        rate_limit: Max requests per rate_limit_period.
        rate_limit_period: Rate limit window in seconds.
        hf_repo_id: Hugging Face repo to upload results to. None = no upload.
        hf_path_prefix: Path prefix within the HF repo.
        hf_token: Hugging Face API token.
        template_name: Jinja2 template filename.

    Returns:
        list[str] | None: HF upload URLs if hf_repo_id is set, otherwise None.
        Layout JSONs are saved as: ``{original_stem}_{variant_name}_layout.json``
        Trajectory JSONs are saved as:
        ``{original_stem}_{variant_name}_{effort}_traj{traj_id}.json``
    """
    from collections import deque

    output_dir_path = Path(str(_resolve_output(output_dir)))
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_dir_str = str(output_dir_path)

    layout_dir_path = Path(str(_resolve_output(layout_dir)))
    layout_files = sorted(layout_dir_path.glob("*_layout.json"))
    if not layout_files:
        raise ValueError(f"No *_layout.json files found in {layout_dir_path}")

    _transform_map: dict[str, tuple[str, object]] = {
        "start_goal_swap": ("StartGoalSwap", StartGoalSwap()),
        "rotate": ("RotateEnv", RotateEnv()),
        "reflect": ("ReflectEnv", ReflectEnv()),
        "transpose": ("TransposeEnv", TransposeEnv()),
    }

    rate_limiter: RateLimiter | None = None
    if enable_rate_limit:
        rate_limiter = RateLimiter(rate_limit=rate_limit, period=rate_limit_period)

    def _reconstruct_env(layout: dict):
        """Reconstruct a live (unwrapped) env from a saved layout JSON dict."""
        if layout.get("coin_placement") is not None:
            env_class = CoinNavigationEnv
        else:
            env_class = Simple2DNavigationEnv
        env = env_class(size=layout["grid_size"], complexity=layout["grid_complexity"])
        env.reset()
        # set_env_from_list handles '#', '_', 'A', 'G' but not 'C' (coin)
        env.set_env_from_list(layout["grid_layout"])
        # Place coin separately using the saved coin_pos
        if layout.get("coin_pos") is not None:
            env.put_obj(Ball("yellow"), *layout["coin_pos"])
        return env

    def _build_variants(base_env) -> list[tuple[FullObservabilityTextWrapper, str]]:
        """Apply each requested augmentation and return (wrapped_env, variant_name) pairs."""
        variants = []
        for aug in augmentations:
            if aug in _transform_map:
                variant_name, transform = _transform_map[aug]
                aug_env = transform(base_env)
                variants.append((FullObservabilityTextWrapper(aug_env), variant_name))

            elif aug == "random_starts":
                # Compute which cells are reachable from the goal via BFS — any cell
                # reachable from the goal can also reach the goal (undirected grid)
                goal = tuple(base_env.goal_pos)
                current_start = tuple(base_env.agent_pos)
                coin_pos = find_coin_pos(base_env)
                excluded = {goal, current_start}
                if coin_pos is not None:
                    excluded.add(coin_pos)

                reachable: set = {goal}
                queue: deque = deque([goal])
                while queue:
                    x, y = queue.popleft()
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if (nx, ny) in reachable:
                            continue
                        if (
                            nx < 0
                            or ny < 0
                            or nx >= base_env.width
                            or ny >= base_env.height
                        ):
                            continue
                        cell = base_env.grid.get(nx, ny)
                        if cell is None or (
                            hasattr(cell, "can_overlap") and cell.can_overlap()
                        ):
                            reachable.add((nx, ny))
                            queue.append((nx, ny))

                # Only consider truly empty interior cells not already in use
                valid_starts = [
                    (x, y)
                    for x in range(1, base_env.width - 1)
                    for y in range(1, base_env.height - 1)
                    if (x, y) not in excluded
                    and base_env.grid.get(x, y) is None
                    and (x, y) in reachable
                ]

                sampled = random.sample(
                    valid_starts, min(num_random_starts, len(valid_starts))
                )
                for pos in sampled:
                    env_copy = clone_env(base_env)
                    env_copy.agent_pos = np.array(pos)
                    env_copy._initial_agent_pos = np.array(pos)
                    variants.append(
                        (
                            FullObservabilityTextWrapper(env_copy),
                            f"RandomStart_{pos[0]}_{pos[1]}",
                        )
                    )

            else:
                logger.warning(f"Unknown augmentation '{aug}' — skipping.")

        return variants

    def _save_variant_layout(
        wrapped_env: FullObservabilityTextWrapper,
        original_layout: dict,
        variant_name: str,
        original_stem: str,
    ) -> str:
        env_u = wrapped_env.unwrapped
        coin_pos = find_coin_pos(env_u)

        agent_pos = env_u.agent_pos
        agent_start_pos = (
            agent_pos.tolist() if hasattr(agent_pos, "tolist") else list(agent_pos)
        )
        if (
            hasattr(env_u, "_initial_agent_pos")
            and env_u._initial_agent_pos is not None
        ):
            p = env_u._initial_agent_pos
            agent_start_pos = p.tolist() if hasattr(p, "tolist") else list(p)

        layout_data = {
            "original_layout_file": original_layout.get("_source_filename", ""),
            "variant": variant_name,
            "grid_id": original_layout["grid_id"],
            "grid_seed": original_layout["grid_seed"],
            "grid_size": original_layout["grid_size"],
            "grid_complexity": original_layout["grid_complexity"],
            "grid_width": env_u.width,
            "grid_height": env_u.height,
            "target_dead_ends": original_layout.get("target_dead_ends"),
            "actual_dead_ends": len(get_all_dead_ends(env_u)),
            "place_at_dead_ends": original_layout.get("place_at_dead_ends", False),
            "coin_placement": original_layout.get("coin_placement"),
            "coin_pos": list(coin_pos) if coin_pos is not None else None,
            "agent_start_pos": agent_start_pos,
            "agent_start_dir": getattr(env_u, "_initial_agent_dir", env_u.agent_dir),
            "goal_pos": list(env_u.goal_pos) if env_u.goal_pos is not None else None,
            "grid_layout": _env_to_grid_list(env_u),
            "grid_text": wrapped_env._render(),
            "legend": wrapped_env.grid_cells,
        }

        layout_filename = f"{original_stem}_{variant_name}_layout.json"
        layout_path = str(Path(output_dir_str) / layout_filename)
        with open(layout_path, "w") as f:
            json.dump(layout_data, f, indent=2)
        if verbose:
            logger.info(f"Saved augmented layout to {layout_path}")
        return layout_path

    # Build all (wrapped_env, variant_name, original_stem, model_name) tasks up front
    variant_tasks: list[tuple] = []
    all_layout_paths: list[str] = []

    for layout_path in layout_files:
        with open(layout_path) as f:
            layout = json.load(f)
        layout["_source_filename"] = layout_path.name
        original_stem = layout_path.name.removesuffix("_layout.json")

        base_env_unwrapped = _reconstruct_env(layout)
        variants = _build_variants(base_env_unwrapped)

        for model_name in model_names:
            for wrapped_env, variant_name in variants:
                saved_path = _save_variant_layout(
                    wrapped_env, layout, variant_name, original_stem
                )
                all_layout_paths.append(saved_path)
                variant_tasks.append(
                    (wrapped_env, variant_name, original_stem, model_name)
                )

    total_variants = len(variant_tasks)
    total_trajectories = (
        total_variants * num_trajectories_per_variant * len(reasoning_efforts)
    )
    logger.info(
        f"Loaded {len(layout_files)} layout files; built {total_variants} variants. "
        f"Generating {total_trajectories} trajectories."
    )

    def _run_variant(task: tuple) -> dict:
        wrapped_env, variant_name, original_stem, model_name = task
        trajectory_results = []

        for effort in reasoning_efforts:
            for traj_id in range(num_trajectories_per_variant):
                if rate_limiter is not None:
                    rate_limiter.acquire()

                env_copy = copy.deepcopy(wrapped_env)
                output_filename = (
                    f"{original_stem}_{variant_name}_{effort}_traj{traj_id}.json"
                )
                output_path = str(Path(output_dir_str) / output_filename)

                if verbose:
                    logger.info(
                        f"Starting trajectory: variant={variant_name}, model={model_name}, "
                        f"effort={effort}, traj={traj_id}"
                    )

                try:
                    get_trajectory(
                        max_steps_per_trajectory=max_steps_per_trajectory,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_logprobs=top_logprobs,
                        seed=traj_id,
                        reasoning_effort=effort,
                        model_name=model_name,
                        observation_placeholders=observation_placeholders,
                        output_path=output_path,
                        verbose=verbose,
                        enable_dynamic_max_steps=enable_dynamic_max_steps,
                        env=env_copy,
                        use_safe_reset=True,
                        transform_type=variant_name,
                        template_name=template_name,
                    )
                    trajectory_results.append(
                        {"status": "success", "output_path": output_path}
                    )
                except Exception as e:
                    logger.error(
                        f"Failed: variant={variant_name}, effort={effort}, traj={traj_id}: {e}"
                    )
                    trajectory_results.append(
                        {"status": "error", "error": str(e), "output_path": output_path}
                    )

        return {"trajectory_results": trajectory_results}

    effective_workers = (
        max_workers if max_workers is not None else min(32, total_variants)
    )
    all_trajectory_results: list[dict] = []

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {executor.submit(_run_variant, task): task for task in variant_tasks}
        for future in tqdm(
            as_completed(futures),
            total=total_variants,
            desc="Augmenting variants",
            unit="variant",
        ):
            task = futures[future]
            try:
                result = future.result()
                all_trajectory_results.extend(result["trajectory_results"])
                failures = [
                    r for r in result["trajectory_results"] if r["status"] != "success"
                ]
                for failure in failures:
                    tqdm.write(
                        f"Failed for variant {task[1]}: {failure.get('error', 'Unknown error')}"
                    )
            except Exception as e:
                tqdm.write(f"Exception for variant {task[1]}: {e}")
                all_trajectory_results.append({"status": "error", "error": str(e)})

    success_count = sum(1 for r in all_trajectory_results if r["status"] == "success")
    logger.info(
        f"Completed {success_count}/{total_trajectories} augmented trajectories successfully. "
        f"Saved {len(all_layout_paths)} augmented layout files."
    )

    if hf_repo_id is not None and (success_count > 0 or all_layout_paths):
        paths_to_upload = [
            r["output_path"] for r in all_trajectory_results if r["status"] == "success"
        ]
        paths_to_upload.extend(all_layout_paths)
        if paths_to_upload:
            return upload_files_to_huggingface(
                file_paths=paths_to_upload,
                repo_id=hf_repo_id,
                path_prefix=hf_path_prefix,
                hf_token=hf_token,
            )
    return None


def reshuffle_walls_from_layouts(
    layout_dir: str,
    num_permutations: int = 5,
    num_trajectories_per_permutation: int = 5,
    max_attempts_per_permutation: int = 200,
    max_steps_per_trajectory: int = 50,
    max_tokens: int = 10000,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_logprobs: int = 5,
    reasoning_efforts: list[Literal["low", "medium", "high"]] = ["low"],
    model_names: list[str] = ["together_ai/openai/gpt-oss-20b"],
    observation_placeholders: list[str] = ["grid_state"],
    output_dir: str = ".",
    verbose: bool = False,
    enable_dynamic_max_steps: bool = False,
    max_workers: int | None = None,
    enable_rate_limit: bool = False,
    rate_limit: int = 1000,
    rate_limit_period: float = 300.0,
    template_name: str = "grid_full_observability_hidden_goals.j2",
) -> None:
    """Load existing coin layout JSONs and generate trajectories on wall-reshuffled variants.

    For each ``*_layout.json`` file in ``layout_dir``, generates ``num_permutations``
    structurally distinct mazes that share the same grid size, complexity, agent start,
    goal, and coin positions but have different internal wall layouts. Trajectories are
    then collected on each permuted layout.

    Wall generation strategy:
        Each permutation re-runs the randomised DFS maze generator at the original
        complexity. If all anchor positions (start, goal, coin) land on naturally empty
        cells the maze is used as-is (complexity preserved exactly). If any anchor is
        walled off, the attempt is retried up to ``max_attempts_per_permutation`` times.
        As a last resort a fallback is applied: conflicting anchor cells are cleared and
        an equal number of non-anchor empty interior cells are re-walled so the total
        wall count is preserved.

    Args:
        layout_dir: Directory containing ``*_layout.json`` files to permute.
        num_permutations: Number of wall permutations to generate per layout.
        num_trajectories_per_permutation: Trajectories per (permutation, model, effort).
        max_attempts_per_permutation: Max maze regeneration attempts before the wall-count
            fallback is applied.
        max_steps_per_trajectory: Maximum steps per trajectory.
        max_tokens: Maximum tokens per model generation step.
        temperature: Sampling temperature.
        top_p: Nucleus sampling parameter.
        top_logprobs: Number of top log probabilities to return.
        reasoning_efforts: List of reasoning effort levels.
        model_names: List of model names in ``"provider/model_id"`` format.
        observation_placeholders: Placeholder names in the prompt template.
        output_dir: Directory to save output JSON files.
        verbose: If True, print detailed logging.
        enable_dynamic_max_steps: If True, override max_steps with 1.5× A* optimal length.
        max_workers: Max parallel workers. Defaults to min(32, total_tasks).
        enable_rate_limit: If True, enforce API rate limiting.
        rate_limit: Max requests per rate_limit_period.
        rate_limit_period: Rate limit window in seconds.
        template_name: Jinja2 template filename.

    Output filenames:
        Layout JSONs:     ``{original_stem}_walls{N}_layout.json``
        Trajectory JSONs: ``{original_stem}_walls{N}_{effort}_traj{traj_id}.json``
    """
    output_dir_path = Path(str(_resolve_output(output_dir)))
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_dir_str = str(output_dir_path)

    layout_dir_path = Path(str(_resolve_output(layout_dir)))
    layout_files = sorted(layout_dir_path.glob("*_layout.json"))
    if not layout_files:
        raise ValueError(f"No *_layout.json files found in {layout_dir_path}")

    rate_limiter: RateLimiter | None = None
    if enable_rate_limit:
        rate_limiter = RateLimiter(rate_limit=rate_limit, period=rate_limit_period)

    # Build all (wrapped_env, variant_name, original_stem, model_name) tasks.
    variant_tasks: list[tuple] = []
    all_layout_paths: list[str] = []

    for layout_path in layout_files:
        with open(layout_path) as f:
            layout = json.load(f)
        layout["_source_filename"] = layout_path.name
        original_stem = layout_path.name.removesuffix("_layout.json")

        for perm_idx in range(num_permutations):
            variant_name = f"walls{perm_idx}"
            env_unwrapped = _reshuffle_walls(
                layout, max_attempts=max_attempts_per_permutation
            )

            if env_unwrapped is None:
                logger.warning(
                    f"Could not generate reachable wall permutation {perm_idx} "
                    f"for {layout_path.name} — skipping."
                )
                continue

            wrapped_env = FullObservabilityTextWrapper(env_unwrapped)

            # Save layout JSON
            coin_pos = find_coin_pos(env_unwrapped)
            agent_pos = env_unwrapped.agent_pos
            agent_start_pos = (
                agent_pos.tolist() if hasattr(agent_pos, "tolist") else list(agent_pos)
            )
            if (
                hasattr(env_unwrapped, "_initial_agent_pos")
                and env_unwrapped._initial_agent_pos is not None
            ):
                p = env_unwrapped._initial_agent_pos
                agent_start_pos = p.tolist() if hasattr(p, "tolist") else list(p)

            layout_data = {
                "original_layout_file": layout.get("_source_filename", ""),
                "variant": variant_name,
                "permutation_index": perm_idx,
                "grid_id": layout["grid_id"],
                "grid_seed": layout["grid_seed"],
                "grid_size": layout["grid_size"],
                "grid_complexity": layout["grid_complexity"],
                "grid_width": env_unwrapped.width,
                "grid_height": env_unwrapped.height,
                "target_dead_ends": layout.get("target_dead_ends"),
                "actual_dead_ends": len(get_all_dead_ends(env_unwrapped)),
                "place_at_dead_ends": layout.get("place_at_dead_ends", False),
                "coin_placement": layout.get("coin_placement"),
                "coin_pos": list(coin_pos) if coin_pos is not None else None,
                "agent_start_pos": agent_start_pos,
                "agent_start_dir": getattr(
                    env_unwrapped, "_initial_agent_dir", env_unwrapped.agent_dir
                ),
                "goal_pos": list(env_unwrapped.goal_pos)
                if env_unwrapped.goal_pos is not None
                else None,
                "grid_layout": _env_to_grid_list(env_unwrapped),
                "grid_text": wrapped_env._render(),
                "legend": wrapped_env.grid_cells,
            }

            layout_filename = f"{original_stem}_{variant_name}_layout.json"
            layout_save_path = str(output_dir_path / layout_filename)
            with open(layout_save_path, "w") as f:
                json.dump(layout_data, f, indent=2)
            all_layout_paths.append(layout_save_path)
            if verbose:
                logger.info(f"Saved reshuffled layout: {layout_filename}")

            for model_name in model_names:
                variant_tasks.append(
                    (wrapped_env, variant_name, original_stem, model_name)
                )

    total_variants = len(variant_tasks)
    total_trajectories = (
        total_variants * num_trajectories_per_permutation * len(reasoning_efforts)
    )
    logger.info(
        f"Loaded {len(layout_files)} layout files; built {total_variants} wall permutations. "
        f"Generating {total_trajectories} trajectories."
    )

    def _run_variant(task: tuple) -> dict:
        wrapped_env, variant_name, original_stem, model_name = task
        trajectory_results = []

        for effort in reasoning_efforts:
            for traj_id in range(num_trajectories_per_permutation):
                if rate_limiter is not None:
                    rate_limiter.acquire()

                env_copy = copy.deepcopy(wrapped_env)
                output_filename = (
                    f"{original_stem}_{variant_name}_{effort}_traj{traj_id}.json"
                )
                output_path = str(Path(output_dir_str) / output_filename)

                if verbose:
                    logger.info(
                        f"Starting trajectory: variant={variant_name}, model={model_name}, "
                        f"effort={effort}, traj={traj_id}"
                    )

                try:
                    get_trajectory(
                        max_steps_per_trajectory=max_steps_per_trajectory,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_logprobs=top_logprobs,
                        seed=traj_id,
                        reasoning_effort=effort,
                        model_name=model_name,
                        observation_placeholders=observation_placeholders,
                        output_path=output_path,
                        verbose=verbose,
                        enable_dynamic_max_steps=enable_dynamic_max_steps,
                        env=env_copy,
                        use_safe_reset=True,
                        transform_type=variant_name,
                        template_name=template_name,
                    )
                    trajectory_results.append(
                        {"status": "success", "output_path": output_path}
                    )
                except Exception as e:
                    logger.error(
                        f"Failed: variant={variant_name}, effort={effort}, traj={traj_id}: {e}"
                    )
                    trajectory_results.append(
                        {"status": "error", "error": str(e), "output_path": output_path}
                    )

        return {"trajectory_results": trajectory_results}

    effective_workers = (
        max_workers if max_workers is not None else min(32, total_variants)
    )
    all_trajectory_results: list[dict] = []

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {executor.submit(_run_variant, task): task for task in variant_tasks}
        for future in tqdm(
            as_completed(futures),
            total=total_variants,
            desc="Reshuffling walls",
            unit="variant",
        ):
            task = futures[future]
            try:
                result = future.result()
                all_trajectory_results.extend(result["trajectory_results"])
                failures = [
                    r for r in result["trajectory_results"] if r["status"] != "success"
                ]
                for failure in failures:
                    tqdm.write(
                        f"Failed for variant {task[1]}: {failure.get('error', 'Unknown error')}"
                    )
            except Exception as e:
                tqdm.write(f"Exception for variant {task[1]}: {e}")
                all_trajectory_results.append({"status": "error", "error": str(e)})

    success_count = sum(1 for r in all_trajectory_results if r["status"] == "success")
    logger.info(
        f"Completed {success_count}/{total_trajectories} reshuffled-wall trajectories successfully. "
        f"Saved {len(all_layout_paths)} permuted layout files."
    )


def upload_trajectories_dir(
    directory: str,
    hf_repo_id: str,
    glob_pattern: str = "*.json",
    hf_token: str | None = None,
    num_workers: int = 8,
) -> str:
    """Upload a directory of trajectory and grid JSON files to Hugging Face.

    Uses upload_large_folder for reliable uploads of large directories with many
    files. This method is resumable (can be re-run if interrupted), uses multiple
    workers for parallel uploads, and automatically handles batching to avoid
    timeouts.

    This is useful for uploading trajectory and grid files generated by
    get_trajectories_multiple_per_grid after they have been generated.

    Note: Files are uploaded to the root of the repository. If you need files
    in a subdirectory, organize them locally in a parent folder first.

    Args:
        directory: Path to the directory containing JSON files to upload.
        hf_repo_id: Hugging Face repository ID (e.g., "username/repo-name").
        glob_pattern: Glob pattern to match files (default: "*.json").
            Use "*.json" to upload all JSON files, or a more specific pattern
            like "*_traj*.json" to upload only trajectory files.
        hf_token: Hugging Face API token. If None, uses HF_TOKEN env var
            or cached credentials.
        num_workers: Number of parallel workers for uploading (default: 8).
            Increase for faster uploads on fast connections.

    Returns:
        str: URL of the repository on Hugging Face Hub.

    Raises:
        ValueError: If directory doesn't exist or no matching files found.

    Example:
        # Upload all JSON files from output directory
        upload_trajectories_dir(
            directory="./trajectory_output",
            hf_repo_id="myuser/trajectories",
        )

        # Upload only trajectory files (excluding grid layout files)
        upload_trajectories_dir(
            directory="./trajectory_output",
            hf_repo_id="myuser/trajectories",
            glob_pattern="*_traj*.json",
        )

        # Use more workers for faster upload
        upload_trajectories_dir(
            directory="./trajectory_output",
            hf_repo_id="myuser/trajectories",
            num_workers=16,
        )
    """
    directory = str(_resolve_output(directory))
    return upload_directory_to_huggingface(
        directory=directory,
        repo_id=hf_repo_id,
        glob_pattern=glob_pattern,
        hf_token=hf_token,
        repo_type="dataset",
        num_workers=num_workers,
    )


# =============================================================================
# Supervised Fine-Tuning (SFT) dataset generation
# =============================================================================

TEMPLATES_DIR = Path(__file__).parents[2] / "templates"
_ACTION_INT_TO_NAME = {0: "LEFT", 1: "RIGHT", 2: "UP", 3: "DOWN"}


def generate_sft_dataset(
    num_episodes_per_combination: int = 100,
    grid_sizes: list[int] = [7, 9, 11],
    grid_complexities: list[float] = [0.0, 0.2, 0.4, 0.6],
    coin_placement: Literal["random", "dead_end"] = "random",
    place_at_dead_ends: bool = False,
    target_dead_ends: int | None = None,
    min_manhattan_distance: int = 3,
    max_attempts: int = 200,
    template_name: str = "grid_full_observability_hidden_goals.j2",
    output_path: str = "sft_dataset.jsonl",
    seed: int = 42,
) -> None:
    """Generate a JSONL SFT dataset from optimal coin-then-goal trajectories.

    Iterates over every (grid_size, grid_complexity) combination and generates
    num_episodes_per_combination episodes for each. Total episodes =
    num_episodes_per_combination * len(grid_sizes) * len(grid_complexities).

    Each JSONL line is one step, formatted to match LLMAgent's single-user-message
    prompt format so train and inference distributions are identical:
        {"messages": [
            {"role": "user",      "content": "<full rendered template + grid>"},
            {"role": "assistant", "content": "{\"action\": \"RIGHT\"}"}
        ]}

    The oracle agent (CoinAStarAgent) plans start→coin then coin→goal via A*.

    Args:
        num_episodes_per_combination: Episodes per (grid_size, grid_complexity) pair.
        grid_sizes: Grid sizes to generate data for.
        grid_complexities: Complexities to generate data for.
        coin_placement: How to place the coin — "random" or "dead_end".
        place_at_dead_ends: If True, agent/goal are also placed at dead ends.
        target_dead_ends: If set, only accept grids with exactly this many dead ends.
        min_manhattan_distance: Minimum Manhattan distance between agent, goal, and coin.
        max_attempts: Max retries to find a valid environment per episode.
        template_name: Jinja2 template file name.
        output_path: Output JSONL path (relative to DATA_DIR or absolute).
        seed: Base random seed.
    """
    output_path_resolved = _resolve_output(output_path)
    jinja_env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    template = jinja_env.get_template(template_name)

    random.seed(seed)
    np.random.seed(seed)

    combinations = list(product(grid_sizes, grid_complexities))
    total_episodes = num_episodes_per_combination * len(combinations)
    total_steps = 0
    skipped = 0

    output_path_resolved.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path_resolved, "w", encoding="utf-8") as f:
        episode_idx = 0
        for grid_size, grid_complexity in combinations:
            for combo_episode in tqdm(
                range(num_episodes_per_combination),
                desc=f"SFT {grid_size}x{grid_size} complexity={grid_complexity}",
                leave=False,
            ):
                rng_seed = seed + episode_idx
                episode_idx += 1

                np.random.seed(rng_seed)
                random.seed(rng_seed)

                # Build a valid coin environment (retry until constraints are met)
                base_env = None
                coin_pos = None

                for _ in range(max_attempts):
                    candidate = CoinNavigationEnv(
                        size=grid_size,
                        complexity=grid_complexity,
                        place_at_dead_ends=place_at_dead_ends,
                    )
                    candidate.reset()

                    if (
                        target_dead_ends is not None
                        and len(get_all_dead_ends(candidate)) != target_dead_ends
                    ):
                        candidate.close()
                        continue

                    start = tuple(candidate.agent_pos)
                    goal = tuple(candidate.goal_pos)
                    excluded = {start, goal}

                    if manhattan_distance(start, goal) < min_manhattan_distance:
                        candidate.close()
                        continue

                    # Resolve coin position
                    if coin_placement == "dead_end":
                        raw_positions = [
                            (x, y)
                            for (x, y) in get_all_dead_ends(candidate)
                            if (x, y) not in excluded
                        ]
                        if not raw_positions:
                            raw_positions = [
                                (x, y)
                                for x in range(1, candidate.width - 1)
                                for y in range(1, candidate.height - 1)
                                if candidate.grid.get(x, y) is None and (x, y) not in excluded
                            ]
                    else:
                        raw_positions = [
                            (x, y)
                            for x in range(1, candidate.width - 1)
                            for y in range(1, candidate.height - 1)
                            if candidate.grid.get(x, y) is None and (x, y) not in excluded
                        ]

                    valid_positions = [
                        p
                        for p in raw_positions
                        if manhattan_distance(start, p) >= min_manhattan_distance
                        and manhattan_distance(goal, p) >= min_manhattan_distance
                    ]

                    if not valid_positions:
                        candidate.close()
                        continue

                    coin_pos = random.choice(valid_positions)
                    candidate.put_obj(Ball("yellow"), *coin_pos)
                    base_env = candidate
                    break

                if base_env is None:
                    logger.warning(
                        f"Episode {episode_idx} (size={grid_size}, complexity={grid_complexity}): "
                        f"could not build a valid env after {max_attempts} attempts — skipping."
                    )
                    skipped += 1
                    continue

                text_env = FullObservabilityTextWrapper(base_env)
                agent = CoinAStarAgent()

                # Get initial observation without resetting (reset would lose the coin)
                grid_text = text_env.observation(base_env.gen_obs())

                terminated = False
                truncated = False
                step_count = 0
                max_steps = base_env.max_steps

                while not (terminated or truncated) and step_count < max_steps:
                    current_grid_text = grid_text

                    action_int, _ = agent.select_action(text_env)
                    action_name = _ACTION_INT_TO_NAME.get(action_int)
                    if action_name is None:
                        logger.warning(
                            f"Episode {episode_idx} step {step_count}: "
                            f"CoinAStarAgent returned invalid action {action_int} — stopping episode."
                        )
                        break

                    rendered_prompt = template.render(grid_state=current_grid_text)
                    record = {
                        "messages": [
                            {"role": "user", "content": rendered_prompt},
                            {"role": "assistant", "content": json.dumps({"action": action_name})},
                        ]
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_steps += 1

                    grid_text, _, terminated, truncated, _ = text_env.step(action_int)

                    # text_env.step returns the observation (str) as first element
                    # when wrapped with FullObservabilityTextWrapper; re-render if not
                    if not isinstance(grid_text, str):
                        grid_text = text_env.observation(base_env.gen_obs())

                    step_count += 1

    logger.info(
        f"SFT dataset written to {output_path_resolved} "
        f"({total_steps} steps from {total_episodes - skipped}/{total_episodes} episodes across "
        f"{len(combinations)} combinations, {skipped} skipped)."
    )
