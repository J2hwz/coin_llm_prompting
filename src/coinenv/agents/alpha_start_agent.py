from typing import Any, Dict, List, Optional, Tuple
import heapq

from coinenv.agents.agent_abc import Agent
from coinenv.agents.random_agent import RandomAgent
from minigrid.minigrid_env import MiniGridEnv


class AlphaStarAgent(Agent):
    """
    An agent that selects an optimal move based on the AlphaStar algorithm.
    """

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        """
        Initialize the AlphaStar agent.

        Args:
            name: Optional name for the agent. Defaults to class name.
        """
        super().__init__(name)

    def select_action(self, env: MiniGridEnv, **kwargs: Any) -> Tuple[int, dict]:
        """
        Select an optimal action based on AlphaStar.

        Computes a shortest path (Manhattan optimal) from the agent's current
        position to the goal on the grid, avoiding walls. The resulting plan is
        cached on the base environment and one action is returned per call.

        Falls back to random_policy if planning fails for any reason.

        Args:
            env: The environment to interact with
            **kwargs: Additional arguments

        Returns:
            The selected action and a dictionary with related metadata.
        """

        # Resolve the base (unwrapped) env to access grid/positions through wrappers
        base_env = getattr(env, "unwrapped", env)

        start: Tuple[int, int] = tuple(base_env.agent_pos)
        goal: Tuple[int, int] = tuple(base_env.goal_pos)
        grid = base_env.grid

        # If a cached plan exists for this (start, goal), reuse it
        plan_key = "_astar_action_plan"
        plan_ctx_key = "_astar_plan_for"
        cached_for = getattr(base_env, plan_ctx_key, None)
        plan: List[int] = getattr(base_env, plan_key, [])
        if cached_for != (start, goal) or not plan:
            # Recompute plan using A*
            def is_passable(x: int, y: int) -> bool:
                cell = grid.get(x, y)
                return (cell is None) or (getattr(cell, "can_overlap", lambda: False)())

            width, height = grid.width, grid.height

            # Neighbor deltas and their corresponding actions
            neighbors: List[Tuple[int, int, int]] = [
                (-1, 0, 0),  # LEFT
                (1, 0, 1),  # RIGHT
                (0, -1, 2),  # UP
                (0, 1, 3),  # DOWN
            ]

            def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
                return abs(a[0] - b[0]) + abs(a[1] - b[1])

            # A* open set: (f_score, g_score, (x,y))
            open_heap: List[Tuple[int, int, Tuple[int, int]]] = []
            heapq.heappush(open_heap, (heuristic(start, goal), 0, start))
            came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
            g_score: Dict[Tuple[int, int], int] = {start: 0}

            closed: set[Tuple[int, int]] = set()
            found = False

            while open_heap:
                f, g, current = heapq.heappop(open_heap)
                if current in closed:
                    continue
                if current == goal:
                    found = True
                    break
                closed.add(current)

                cx, cy = current
                for dx, dy, _ in neighbors:
                    nx, ny = cx + dx, cy + dy
                    # Bounds check
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    if not is_passable(nx, ny):
                        continue
                    tentative_g = g + 1
                    if tentative_g < g_score.get((nx, ny), 1_000_000_000):
                        came_from[(nx, ny)] = current
                        g_score[(nx, ny)] = tentative_g
                        f_score = tentative_g + heuristic((nx, ny), goal)
                        heapq.heappush(open_heap, (f_score, tentative_g, (nx, ny)))

            action_plan: List[int] = []
            if found:
                # Reconstruct path of positions from goal back to start
                path: List[Tuple[int, int]] = []
                cur = goal
                path.append(cur)
                while cur != start and cur in came_from:
                    cur = came_from[cur]
                    path.append(cur)
                if path[-1] != start:
                    # No valid path reconstructed
                    path = []

                # Convert position path (reversed) to forward order and then to actions
                if path:
                    path.reverse()  # start -> ... -> goal
                    for (x1, y1), (x2, y2) in zip(path[:-1], path[1:]):
                        dx, dy = x2 - x1, y2 - y1
                        if dx == -1 and dy == 0:
                            action_plan.append(0)  # LEFT
                        elif dx == 1 and dy == 0:
                            action_plan.append(1)  # RIGHT
                        elif dx == 0 and dy == -1:
                            action_plan.append(2)  # UP
                        elif dx == 0 and dy == 1:
                            action_plan.append(3)  # DOWN
                        else:
                            # Unexpected move (shouldn't happen on 4-connected grid)
                            pass

            # Cache plan (might be empty if not found)
            setattr(base_env, plan_key, action_plan)
            setattr(base_env, plan_ctx_key, (start, goal))
            plan = action_plan

        # If plan is empty (e.g., already at goal or planning failed), fallback
        if not plan:
            return RandomAgent().select_action(env)

        # Pop and return the next action
        next_action = plan.pop(0)

        # Predict next start position for cache coherence to avoid recompute next call
        dx_dy_for_action = {
            0: (-1, 0),  # LEFT
            1: (1, 0),  # RIGHT
            2: (0, -1),  # UP
            3: (0, 1),  # DOWN
        }
        dx, dy = dx_dy_for_action.get(next_action, (0, 0))
        predicted_next_start = (start[0] + dx, start[1] + dy)

        setattr(base_env, plan_key, plan)  # update remaining plan
        setattr(base_env, plan_ctx_key, (predicted_next_start, goal))

        return next_action, {"agents_name": self.name}
