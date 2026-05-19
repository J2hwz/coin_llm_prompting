from typing import Any, Optional, Tuple

from minigrid.minigrid_env import MiniGridEnv

from coinenv.agents.agent_abc import Agent


class RandomAgent(Agent):
    """
    A simple agent that selects actions randomly from the environment's action space.

    This agent doesn't maintain any state and simply samples uniformly from the available actions at each step.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the random agent.

        Args:
            name: Optional name for the agent. Defaults to class name.
        """
        super().__init__(name)

    def select_action(self, env: MiniGridEnv, **kwargs: Any) -> Tuple[int, dict]:
        """
        Select a random action from the environment's action space.

        Args:
            env: The environment to interact with
            **kwargs: Additional arguments (ignored by this agent)

        Returns:
            A randomly sampled action from the environment's action space and a dictionary with related metadata.

        Raises:
            ValueError: If the environment doesn't have an action_space attribute
        """
        if not hasattr(env, "action_space"):
            raise ValueError("Environment must have an action_space attribute")

        return int(env.action_space.sample()), {"agents_name": self.name}
