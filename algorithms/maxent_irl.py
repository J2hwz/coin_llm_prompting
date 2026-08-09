"""
maxent_irl.py
-------------
Maximum Causal Entropy IRL for subgoal inference — self-contained wrapper.

Algorithm
---------
Ziebart et al. (2008) Maximum Entropy IRL; causal-entropy variant from
Ziebart's thesis (2010). Core implementation adapted from
https://github.com/qzed/irl-maxent (numpy-only).

Note: the basic irl() function has documented issues with negative reward
states. run_maxent() uses irl_causal() which is more numerically stable for
gridworlds with negative step costs.

Public API
----------
    run_maxent(paths, layout, discount=0.99, eps=1e-4, eps_svf=1e-5)
        -> (score_grid_display, predicted_json_coords)

Coordinate conventions
----------------------
- JSON coords : (col, row_from_top)
- MDP coords  : (x, y_mdp) — y=0 = bottom row
"""

from copy import deepcopy
from itertools import chain
from typing import List, Tuple
import sys
from pathlib import Path

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_ALGO_DIR = Path(__file__).resolve().parent
if str(_ALGO_DIR) not in sys.path:
    sys.path.insert(0, str(_ALGO_DIR))

from gridworld import GridMdpOld, construct_mdp_old
from mdp_funcs import action_to_int
from algorithms_util import to_mdp, to_json, build_gridmdp_old, json_path_to_mdp_traj


# =============================================================================
# Core algorithm (from qzed/irl-maxent)
# =============================================================================

def expected_svf_from_policy(p_transition, p_initial, terminal, p_action, eps=1e-5):
    """
    Compute the expected state visitation frequency using the given local
    action probabilities.

    This is the forward pass of Algorithm 1 of the Maximum Entropy IRL paper
    by Ziebart et al. (2008). Alternatively, it can also be found as
    Algorithm 9.3 in in Ziebart's thesis (2010).

    Args:
        p_transition: [from, to, action] transition probabilities.
        p_initial: [state] -> probability of being an initial state.
        terminal: list of terminal state integers.
        p_action: [state, action] -> local action probability.
        eps: convergence threshold.

    Returns:
        Expected state visitation frequencies [state] -> svf.
    """
    n_states, _, n_actions = p_transition.shape

    p_transition = np.copy(p_transition)
    p_transition[terminal, :, :] = 0.0

    p_transition = [np.array(p_transition[:, :, a]) for a in range(n_actions)]

    d = np.zeros(n_states)

    delta = np.inf
    while delta > eps:
        d_ = [p_transition[a].T.dot(p_action[:, a] * d) for a in range(n_actions)]
        d_ = p_initial + np.array(d_).sum(axis=0)

        delta, d = np.max(np.abs(d_ - d)), d_

    return d


def local_action_probabilities(p_transition, terminal, reward):
    """
    Compute local action probabilities (backward pass of Algorithm 1,
    Ziebart et al. 2008).

    Args:
        p_transition: [from, to, action] transition probabilities.
        terminal: list of terminal state integers.
        reward: [state] -> reward.

    Returns:
        [state, action] -> probability.
    """
    n_states, _, n_actions = p_transition.shape

    er = np.exp(reward)
    p = [np.array(p_transition[:, :, a]) for a in range(n_actions)]

    zs = np.zeros(n_states)
    zs[terminal] = 1.0

    for _ in range(2 * n_states):
        za = np.array([er * p[a].dot(zs) for a in range(n_actions)]).T
        zs = za.sum(axis=1)

    return za / zs[:, None]


def compute_expected_svf(p_transition, p_initial, terminal, reward, eps=1e-5):
    """Combine backward + forward pass for MaxEnt IRL (Algorithm 1, Ziebart 2008)."""
    p_action = local_action_probabilities(p_transition, terminal, reward)
    return expected_svf_from_policy(p_transition, p_initial, terminal, p_action, eps)


def softmax(x1, x2):
    """Element-wise soft maximum."""
    x_max = np.maximum(x1, x2)
    x_min = np.minimum(x1, x2)
    return x_max + np.log(1.0 + np.exp(x_min - x_max))


def local_causal_action_probabilities(p_transition, terminal, reward, discount, eps=1e-5):
    """
    Compute local action probabilities for MaxCausalEntropy IRL.
    Algorithm 9.1 from Ziebart's thesis (2010) with discounting.

    Args:
        p_transition: [from, to, action] transition probabilities.
        terminal: list of terminal state integers OR terminal reward function.
        reward: [state] -> reward.
        discount: discounting factor.
        eps: convergence threshold for state partition function.

    Returns:
        [state, action] -> probability.
    """
    n_states, _, n_actions = p_transition.shape

    if len(terminal) == n_states:
        reward_terminal = np.array(terminal, dtype=float)
    else:
        reward_terminal = -np.inf * np.ones(n_states)
        reward_terminal[terminal] = 0.0

    p = [np.array(p_transition[:, :, a]) for a in range(n_actions)]

    v = -1e200 * np.ones(n_states)

    delta = np.inf
    while delta > eps:
        v_old = v

        q = np.array([reward + discount * p[a].dot(v_old) for a in range(n_actions)]).T

        v = reward_terminal
        for a in range(n_actions):
            v = softmax(v, q[:, a])

        v = np.array(v, dtype=float)

        delta = np.max(np.abs(v - v_old))

    return np.exp(q - v[:, None])


def compute_expected_causal_svf(
    p_transition, p_initial, terminal, reward, discount, eps_lap=1e-5, eps_svf=1e-5
):
    """Combine Algorithm 9.1 + 9.3 for MaxCausalEntropy IRL (Ziebart 2010)."""
    p_action = local_causal_action_probabilities(p_transition, terminal, reward, discount, eps_lap)
    return expected_svf_from_policy(p_transition, p_initial, terminal, p_action, eps_svf)


def feature_expectation_from_trajectories(features, trajectories):
    """Average feature counts over trajectories."""
    n_states, n_features = features.shape
    fe = np.zeros(n_features)
    for t in trajectories:
        for s in t.states():
            fe += features[s, :]
    return fe / len(trajectories)


def initial_probabilities_from_trajectories(n_states, trajectories):
    """Empirical distribution over initial states."""
    p = np.zeros(n_states)
    for t in trajectories:
        p[t.transitions()[0][0]] += 1.0
    return p / len(trajectories)


def linear_decay(lr0=0.2, decay_rate=1.0, decay_steps=1):
    """Linear learning-rate decay: lr(k) = lr0 / (1 + decay_rate * floor(k/decay_steps))."""
    def _lr(k):
        return lr0 / (1.0 + decay_rate * np.floor(k / decay_steps))
    return _lr


class Initializer:
    def __init__(self):
        pass

    def initialize(self, shape):
        raise NotImplementedError

    def __call__(self, shape):
        return self.initialize(shape)


class Constant(Initializer):
    def __init__(self, value=1.0):
        super().__init__()
        self.value = value

    def initialize(self, shape):
        if callable(self.value):
            return np.ones(shape) * self.value(shape)
        else:
            return np.ones(shape) * self.value


class Optimizer:
    def __init__(self):
        self.parameters = None

    def reset(self, parameters):
        self.parameters = parameters

    def step(self, grad, *args, **kwargs):
        raise NotImplementedError

    def normalize_grad(self, ord=None):
        return NormalizeGrad(self, ord)


class NormalizeGrad(Optimizer):
    def __init__(self, opt, ord=None):
        super().__init__()
        self.opt = opt
        self.ord = ord

    def reset(self, parameters):
        super().reset(parameters)
        self.opt.reset(parameters)

    def step(self, grad, *args, **kwargs):
        return self.opt.step(grad / np.linalg.norm(grad, self.ord), *args, **kwargs)


class ExpSga(Optimizer):
    """Exponentiated stochastic gradient ascent (Ziebart 2010, Algorithm 10.5)."""

    def __init__(self, lr, normalize=False):
        super().__init__()
        self.lr = lr
        self.normalize = normalize
        self.k = 0

    def reset(self, parameters):
        super().reset(parameters)
        self.k = 0

    def step(self, grad, *args, **kwargs):
        lr = self.lr if not callable(self.lr) else self.lr(self.k)
        self.k += 1
        self.parameters *= np.exp(lr * grad)
        if self.normalize:
            self.parameters /= self.parameters.sum()


class Trajectory:
    """From qzed/irl-maxent"""

    def __init__(self, transitions: List[Tuple]):
        self._t = transitions

    def transitions(self) -> List[Tuple]:
        return self._t

    def states(self):
        return map(lambda x: x[0], chain(self._t, [(self._t[-1][2], 0, 0)]))

    def __repr__(self):
        return "Trajectory({})".format(repr(self._t))

    def __str__(self):
        return "{}".format(self._t)

    def __eq__(self, other):
        return self._t == other._t


def transform_trajectory(mdp: GridMdpOld, traj: List[Tuple]):
    """Convert [(state, action), ...] to a Trajectory of integer indices."""
    transform = []
    for i in range(len(traj) - 1):
        s, a = traj[i]
        if a is None:
            continue
        s_int = mdp.state_to_int(s)
        a_int = action_to_int(a)
        if a_int is None:
            continue
        s_prime = mdp.state_to_int(traj[i + 1][0])
        transform.append((s_int, a_int, s_prime))
    return Trajectory(transform)


def irl(p_transition, features, terminal, trajectories, optim, init, eps=1e-4, eps_esvf=1e-5):
    """
    Maximum Entropy IRL (Ziebart et al. 2008).

    Note: Has known issues with all-negative reward states (negative step costs).
    Prefer irl_causal for gridworlds with step cost -1.

    Args:
        p_transition: [from, to, action] transition probabilities.
        features: [n_states, n_features] feature matrix.
        terminal: list of terminal state integers.
        trajectories: list of Trajectory instances.
        optim: Optimizer instance.
        init: Initializer instance.

    Returns:
        Per-state reward array [n_states].
    """
    n_states, _, n_actions = p_transition.shape
    _, n_features = features.shape

    e_features = feature_expectation_from_trajectories(features, trajectories)
    p_initial = initial_probabilities_from_trajectories(n_states, trajectories)

    theta = init(n_features)
    delta = np.inf

    optim.reset(theta)
    while delta > eps:
        theta_old = theta.copy()

        reward = features.dot(theta)

        e_svf = compute_expected_svf(p_transition, p_initial, terminal, reward, eps_esvf)
        grad = e_features - features.T.dot(e_svf)

        optim.step(grad)

        delta = np.max(np.abs(theta_old - theta))

    return features.dot(theta)


def irl_causal(
    p_transition,
    features,
    terminal,
    trajectories,
    optim,
    init,
    discount,
    eps=1e-4,
    eps_svf=1e-5,
    eps_lap=1e-5,
):
    """
    Maximum Causal Entropy IRL (Ziebart 2010 thesis, Algorithm 9.1).

    More numerically stable than irl() for environments with negative step costs.

    Args:
        p_transition: [from, to, action] transition probabilities.
        features: [n_states, n_features] feature matrix.
        terminal: list of terminal state integers OR terminal reward function.
        trajectories: list of Trajectory instances.
        optim: Optimizer instance.
        init: Initializer instance.
        discount: discount factor.

    Returns:
        Per-state reward array [n_states].
    """
    n_states, _, n_actions = p_transition.shape
    _, n_features = features.shape

    e_features = feature_expectation_from_trajectories(features, trajectories)
    p_initial = initial_probabilities_from_trajectories(n_states, trajectories)

    theta = init(n_features)
    delta = np.inf

    optim.reset(theta)
    while delta > eps:
        theta_old = theta.copy()

        reward = features.dot(theta)

        e_svf = compute_expected_causal_svf(
            p_transition, p_initial, terminal, reward, discount, eps_lap, eps_svf
        )

        grad = e_features - features.T.dot(e_svf)

        optim.step(grad)
        delta = np.max(np.abs(theta_old - theta))

    return features.dot(theta)


# =============================================================================
# Public wrapper
# =============================================================================

def run_maxent(paths, layout, discount=0.99, eps=1e-4, eps_svf=1e-5):
    """
    MaxCausalEntropy IRL subgoal inference.

    Uses irl_causal (Ziebart 2010) for stability with negative step-cost rewards.
    Identity feature matrix gives one reward parameter per grid cell (tabular).

    Parameters
    ----------
    paths : list of list of (col, row_top)
    layout : dict  — grid_layout, goal_pos, coin_pos, agent_start_pos
    discount : float   discount factor for the causal entropy backward pass
    eps : float        convergence threshold for reward parameters
    eps_svf : float    convergence threshold for SVF

    Returns
    -------
    score_grid : np.ndarray [n_rows, n_cols]   inferred rewards (display coords)
    predicted  : (col, row_top)
    """
    n_rows = len(layout["grid_layout"])
    n_cols = len(layout["grid_layout"][0])
    goal_j = tuple(layout["goal_pos"])

    mdp = build_gridmdp_old(layout)

    trajs_mdp = [json_path_to_mdp_traj(p, n_rows, terminal_marker=True)
                 for p in paths if len(p) >= 2]
    trajs_mdp = [t for t in trajs_mdp if t]
    if not trajs_mdp:
        return np.full((n_rows, n_cols), np.nan), goal_j

    traj_objects = [transform_trajectory(mdp, t) for t in trajs_mdp]
    traj_objects = [t for t in traj_objects if t.transitions()]
    if not traj_objects:
        return np.full((n_rows, n_cols), np.nan), goal_j

    # Build transition matrix directly from the transitions dict.
    # We use rows*cols (not len(states)) so state_to_int indices are always in bounds
    # even when wall cells are present.
    n_states = mdp.rows * mdp.cols
    n_actions = len(mdp.actlist)
    tm = np.zeros((n_states, n_actions, n_states))
    for s in mdp.transitions:
        si = mdp.state_to_int(s)
        for ai, a in enumerate(mdp.actlist):
            for prob, s_prime in mdp.transitions[s][a]:
                tm[si, ai, mdp.state_to_int(s_prime)] = prob
    # irl_causal expects [from, to, action]
    p_transition = np.transpose(tm, (0, 2, 1))

    terminals = [mdp.state_to_int(s) for s in mdp.terminals]
    features = np.eye(n_states)
    optim = ExpSga(lr=linear_decay(lr0=0.2))
    init = Constant(1.0)

    reward = irl_causal(
        p_transition, features, terminals, traj_objects,
        optim, init, discount, eps=eps, eps_svf=eps_svf, eps_lap=eps,
    )

    mdp_inferred = construct_mdp_old(mdp, reward)

    start_set = {to_mdp(*p[0], n_rows) for p in paths if p}
    goal_m = to_mdp(*goal_j, n_rows)

    score = np.full((n_rows, n_cols), np.nan)
    best, best_r = goal_j, -np.inf
    for s in mdp.states:
        col, y = s
        if s in start_set or s == goal_m:
            continue
        rt = n_rows - 1 - y
        r = mdp_inferred.R(s)
        if 0 <= rt < n_rows and 0 <= col < n_cols:
            score[rt, col] = r
            if r > best_r:
                best_r = r
                best = to_json(col, y, n_rows)

    return score, best
