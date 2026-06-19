import operator
from collections import deque
import random
from typing import Tuple, List, Optional

import numpy as np

orientations = WEST, SOUTH, EAST, NORTH = [(-1, 0), (0, -1), (1, 0), (0, 1)]
STAY = (0, 0)
turns = LEFT, RIGHT = (+1, -1)


def turn_heading(heading, inc, headings=orientations):
    return headings[(headings.index(heading) + inc) % len(headings)]


def turn_right(heading):
    if heading == STAY:
        return STAY
    return turn_heading(heading, RIGHT)


def turn_left(heading):
    if heading == STAY:
        return STAY
    return turn_heading(heading, LEFT)


def shortest_dist(point1: Tuple[int, int], point2: Tuple[int, int]):
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


def sample_action(pi: dict, s: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """Sample an action from a stochastic policy."""
    if pi[s] is None:
        return None
    return random.choices(list(pi[s].keys()), list(pi[s].values()), k=1)[0]


def shortest_path(start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Given two points, find a shortest path between them with breadth first search."""
    queue = deque([[start]])
    seen = set([start])
    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if (x, y) == end:
            return path
        for x2, y2 in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if (x2, y2) not in seen:
                queue.append(path + [(x2, y2)])
                seen.add((x2, y2))


def vector_add(a, b):
    """Component-wise addition of two vectors."""
    return tuple(map(operator.add, a, b))


def scale(data: List[float], new_min: float = 0, new_max: float = 1) -> List[float]:
    old_min = np.min(data)
    old_max = np.max(data)

    old_range = old_max - old_min
    new_range = new_max - new_min

    return [(new_min + (num - old_min) * new_range / old_range) for num in data]
