import numpy as np
from gymnasium.core import ObservationWrapper
from minigrid.minigrid_env import MiniGridEnv


# TODO: convert logic of seen_mask to fog_mask to disambiguate wrappers
class FogOfWarWrapper(ObservationWrapper):
    """
    An abstract base class for MiniGrid wrappers that adds a cumulative,
    line-of-sight-based fog of war.

    This class manages the 'seen_mask' but does not modify the observation
    itself. Subclasses are responsible for using the mask to render the
    final observation (e.g., as an RGB image or text).
    """

    def __init__(self, env, view_radius=None):
        super().__init__(env)
        self.gridenv: MiniGridEnv = self.unwrapped  # type: ignore

        if view_radius is None:
            # Default radius is based on the env's triangular view size.
            # For agent_view_size=7, this results in a radius of 3 (a 7x7 view).
            self.view_radius = (self.gridenv.agent_view_size - 1) // 2
        else:
            self.view_radius = view_radius

        self.seen_mask = None

    def reset(self, **kwargs):
        """Resets the environment and the seen mask."""
        obs, info = self.env.reset(**kwargs)
        self.seen_mask = np.zeros((self.gridenv.height, self.gridenv.width), dtype=bool)
        # The observation method is called by the subclass reset
        return obs, info

    def observation(self, observation):
        """Updates the seen mask based on the agent's current position."""
        self._update_seen_mask()
        return observation

    @staticmethod
    def _trace_line(x0, y0, x1, y1):
        """
        Yields all cells on the line from (x0, y0) to (x1, y1)
        using Bresenham's line algorithm.
        """
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy

        while True:
            yield (x0, y0)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def _update_seen_mask(self):
        """
        Updates the seen_mask by casting rays from the agent's position
        to the edge of its view radius, accounting for walls.
        """
        agent_x, agent_y = self.gridenv.agent_pos

        min_x = max(0, agent_x - self.view_radius)
        max_x = min(self.gridenv.width - 1, agent_x + self.view_radius)
        min_y = max(0, agent_y - self.view_radius)
        max_y = min(self.gridenv.height - 1, agent_y + self.view_radius)

        perimeter_points = set()
        for x in range(min_x, max_x + 1):
            perimeter_points.add((x, min_y))
            perimeter_points.add((x, max_y))
        for y in range(min_y + 1, max_y):
            perimeter_points.add((min_x, y))
            perimeter_points.add((max_x, y))

        for px, py in perimeter_points:
            for x, y in self._trace_line(agent_x, agent_y, px, py):
                if self.seen_mask is not None:
                    self.seen_mask[y, x] = True
                cell = self.gridenv.grid.get(x, y)
                if cell and cell.type == "wall":
                    break
