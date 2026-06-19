import random
from copy import deepcopy
from typing import List, Tuple, Dict

import numpy as np

from grid_utils import vector_add, orientations, turn_right, turn_left, STAY


class GridMdpOld:
    """This class now lacks unittest coverage, because I moved the unittests to use GridMDP."""

    """From AIMA. A two-dimensional grid MDP, as in [Figure 17.1]. All you have to do is
    specify the grid as a list of lists of rewards; use None for an obstacle
    (unreachable state). Also, you should specify the terminal states.
    An action is an (x, y) unit vector; e.g. (1, 0) means move east."""

    def __init__(self, grid, terminal_locs=[], gamma=0.99, windy=False):
        if not (0 < gamma <= 1):
            raise ValueError("An MDP must have 0 < gamma <= 1")

        grid = list(reversed(grid))  # because we want row 0 on bottom, not on top
        rewards = {}
        states = set()
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.grid = grid
        self.gamma = gamma
        self.terminals = terminal_locs  # For consistency with GridMDP
        self.terminal_locs = terminal_locs  # For consistency with GridMDP

        # TODO: wrap this in method with unittest
        # Build up state space and rewards
        for x in range(self.cols):
            for y in range(self.rows):
                if grid[y][x] is not None and (x, y) not in states:
                    states.add((x, y))
                    rewards[(x, y)] = grid[y][x]
        self.rewards = rewards
        self.oto_rewards = {}

        self.states = states
        self.locs = states
        self.windy = windy
        self.actlist = orientations + [STAY]

        # TODO: wrap this in method with unittest
        # Build up transition matrix
        transitions = {}
        for s in self.states:
            transitions[s] = {}
            for a in self.actlist:
                transitions[s][a] = self.calculate_T(s, a)
        self.transitions = transitions

        self.check_consistency()

    def check_consistency(self):
        # check reward for each state
        assert set(self.rewards.keys()) == set(self.states)

        # check that all terminals are valid states
        assert all(t in self.states for t in self.terminals)

        # check that probability distributions for all actions sum to 1
        for s1, actions in self.transitions.items():
            for a in actions.keys():
                s = 0
                for o in actions[a]:
                    s += o[0]
                assert abs(s - 1) < 0.001

    def go(self, state, direction):
        """Return the state that results from going in this direction."""
        state1 = vector_add(state, direction)
        return state1 if state1 in self.states else state

    def calculate_T(self, state, action):
        if self.windy:
            return [
                (0.8, self.go(state, action)),
                (0.1, self.go(state, turn_right(action))),
                (0.1, self.go(state, turn_left(action))),
            ]
        else:
            return [(1, self.go(state, action))]

    def R(self, state):
        """Return a numeric reward for this state."""
        return self.rewards[state]

    def T(self, state, action) -> List[Tuple[float, Tuple]]:
        """Returns list of (prob, state) tuples."""
        return self.transitions[state][action] if action else [(0.0, state)]

    def transitions_matrix(self) -> np.ndarray:
        """Different IRL methods expect transitions in different formats.

        Shape is (rows*cols, n_actions, rows*cols) so that state_to_int indices
        (which are row*cols + col) are always in bounds, even with wall cells.
        """
        n = self.rows * self.cols
        matrix = np.zeros(shape=(n, len(self.actlist), n))
        for s in self.transitions:
            distrib = self.transitions[s]
            for a in distrib:
                a_ind = self.actlist.index(a)
                for elem in distrib[a]:  # List of prob, s_prime pairs
                    prob, s_prime = elem
                    matrix[self.state_to_int(s), a_ind, self.state_to_int(s_prime)] = prob
        return matrix

    def actions(self, state):
        """Return a list of actions that can be performed in this state. By default, a
        fixed list of actions, except for terminal states. Override this
        method if you need to specialize by state."""

        if state in self.terminals:
            return [STAY]
        else:
            return self.actlist

    def new_state(self, state, action):
        assert action is not None
        distrib = self.T(state, action)
        new_state = random.choices([x[1] for x in distrib], weights=[x[0] for x in distrib])
        assert len(new_state) == 1
        return new_state[0]

    def get_oto_rewards(self) -> Dict[Tuple, float]:
        return {}

    def reward_grid(self):
        """Used when plotting rewards as a heatmap."""
        grid = []
        for y in range(self.rows):
            row = []
            for x in range(self.cols):
                row.append(self.rewards[(x, y)] if (x, y) in self.rewards else None)
            grid.append(row)
        return list(reversed(grid))

    def state_to_int(self, s: Tuple[int, int]) -> int:
        """
        Return 0-indexed int representing state. e.g. in a 2x2:
        (0,1) (1,1)              2  3
        (0,0) (1,0)      ->      0  1
        """
        if not (0 <= s[0] < self.cols) or not (0 <= s[1] < self.rows):
            raise KeyError  # Not a state in this MDP
        return (s[1] * self.cols) + s[0]

    def int_to_state(self, s: int):
        """Inverse of point_to_int"""
        # TODO: do this properly
        for x in range(self.cols):
            for y in range(self.rows):
                if self.state_to_int((x, y)) == s:
                    return x, y
        raise KeyError

    def new_current_state(self, successor_state):
        return successor_state


def construct_mdp_old(mdp: GridMdpOld, rewards: List[float]) -> GridMdpOld:
    """Some IRL methods return rewards as a vector. Build an MDP out of a given MDP and a given rewards vector."""
    rewards_dict = {}
    for i in range(len(rewards)):
        state = mdp.int_to_state(i)
        rewards_dict[state] = rewards[i]
    mdp_new = deepcopy(mdp)
    mdp_new.rewards = rewards_dict
    return mdp_new
