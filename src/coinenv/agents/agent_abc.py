from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

from minigrid.minigrid_env import MiniGridEnv


class Agent(ABC):
    """
    An abstract base class for an agent that navigates in a 2D environment.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the agent
        Args:
            name: agent's name
        """
        self.name = name if name else self.__class__.__name__

    @abstractmethod
    def select_action(self, env: MiniGridEnv, **kwargs: Any) -> Tuple[int, dict]:
        """
        Selects an action based on the current state of the environment.
        This is the core decision-making method of the agent.

        Args:
            env: the environment object

        Returns:
            The action chosen by the agent and a dictionary with related metadata.
        """
        pass

    def update(self, **kwargs: Any) -> None:
        """
        Updates the agent based on the last transition.

        Args:
            **kwargs (Any): A dictionary of transition information.
        """
        pass

    def reset(self) -> None:
        """
        Resets the agent's internal state at the beginning of a new episode.

        This is optional and can be overridden by agents that require
        an internal state reset (e.g., clearing memory in an LLM agent).
        """
        pass
