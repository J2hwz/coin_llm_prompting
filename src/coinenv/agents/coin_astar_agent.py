from typing import Any, Optional, Tuple

from minigrid.minigrid_env import MiniGridEnv

from coinenv.agents.alpha_start_agent import AlphaStarAgent
from coinenv.environment_generator.utils import find_coin_pos


class CoinAStarAgent(AlphaStarAgent):
    """Two-phase A* oracle for CoinNavigationEnv: navigates start→coin→goal.

    Phase 1 (coin not yet collected): plans toward the coin position.
    Phase 2 (coin collected): plans toward the real goal.

    Produces optimal trajectories suitable for SFT data generation. The Ball
    cell is temporarily removed from the grid before A* planning so that the
    parent's is_passable() check (which uses can_overlap()) treats the coin
    cell as passable. It is restored immediately after the action is selected.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)

    def select_action(self, env: MiniGridEnv, **kwargs: Any) -> Tuple[int, dict]:
        base_env = getattr(env, "unwrapped", env)
        coin_collected: bool = getattr(base_env, "coin_collected", True)

        if coin_collected:
            return super().select_action(env, **kwargs)

        coin_pos = find_coin_pos(base_env)
        if coin_pos is None:
            # Coin already removed from grid; fall through to phase 2
            return super().select_action(env, **kwargs)

        real_goal_pos = base_env.goal_pos
        coin_cell = base_env.grid.get(*coin_pos)
        try:
            # Temporarily clear the coin cell so A*'s is_passable() allows it
            # (Ball.can_overlap() returns False in MiniGrid by default)
            base_env.grid.set(*coin_pos, None)
            base_env.goal_pos = coin_pos
            # Invalidate cached plan so parent replans toward coin
            setattr(base_env, "_astar_plan_for", None)
            action, metadata = super().select_action(env, **kwargs)
        finally:
            base_env.grid.set(*coin_pos, coin_cell)
            base_env.goal_pos = real_goal_pos

        return action, metadata
