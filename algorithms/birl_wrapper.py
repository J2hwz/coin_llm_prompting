"""
birl_wrapper.py
---------------
Bayesian IRL (PolicyWalk MCMC) for subgoal inference — self-contained wrapper.

Algorithm
---------
Ramachandran & Amir (2007). Bayesian inverse reinforcement learning.
PolicyWalk samples from P(R | O) ∝ e^(α·E(O,R)) · P(R) via Metropolis-Hastings,
walking over a discretised reward space R^|S|_δ.

Uses GridMdpOld (simple (x,y) states) and the stochastic policy_iteration /
q_value from mdp_funcs.py, which work with any MDP sharing the same interface.

Public API
----------
    run_birl(paths, layout, **kwargs)
        -> (score_grid_display, predicted_json_coords)

Coordinate conventions
----------------------
- JSON coords : (col, row_from_top)
- MDP coords  : (x, y_mdp) — y=0 = bottom row
"""

import copy
from copy import deepcopy
import random
from enum import Enum
from itertools import chain
from typing import List, Tuple
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import multivariate_normal
from tqdm import trange

# ── Path setup ────────────────────────────────────────────────────────────────
_ALGO_DIR = Path(__file__).resolve().parent
if str(_ALGO_DIR) not in sys.path:
    sys.path.insert(0, str(_ALGO_DIR))

from mdp_funcs import policy_iteration, q_value, value_function, policy_evaluation, expected_utility
from gridworld import GridMdpOld, construct_mdp_old
from algorithms_util import to_mdp, to_json
from inv_planning import build_gridmdp_old, json_path_to_mdp_traj


# =============================================================================
# Step size schedules
# =============================================================================

class Step(Enum):
    CONSTANT = 0
    LINEAR = 1
    EXPONENTIAL = 2


def linear_decrease_step(value_start, value_end, max_i, i):
    step = (value_start - value_end) / (max_i - 1)
    return value_start - (i * step)


def constant_step(value_start, value_end=None, max_i=None, i=None):
    return value_start


def exponential_decrease_step(a, b, n, i):
    k = math.log(b / a) / (n - 1)
    return a * math.exp(i * k)


# =============================================================================
# Prior distributions  (Ramachandran & Amir 2007, Section 3.2)
# =============================================================================

def improper_prior(mdp: GridMdpOld = None, r_min: float = None, r_max: float = None) -> float:
    return np.log(1)


def beta_prior(mdp: GridMdpOld, r_min: float, r_max: float):
    epsilon = 1e-10
    log_prob = 0
    for s in mdp.states:
        r = mdp.rewards[s]
        log_prob += -math.log((r / (r_max + epsilon)) ** 0.5) - math.log(
            (1 - (r / (r_max + epsilon))) ** 0.5
        )
    return log_prob


def uniform_prior(mdp: GridMdpOld, r_min: float, r_max: float, allow_flat=False) -> float:
    log_prob = 0
    for s in mdp.states:
        if r_min <= mdp.rewards[s] <= r_max:
            log_prob += -np.log(len(mdp.rewards))
        else:
            return np.NINF
    return log_prob


# =============================================================================
# PolicyWalk MCMC sampler  (Ramachandran & Amir 2007, Figure 3)
# =============================================================================

class PolicyWalk:
    def __init__(
        self,
        mdp_true: GridMdpOld,
        trajectories,
        r_max: float,
        step_size: float,
        alpha,
        prior,
        r_min: float = None,
        step_size_end: float = None,
        step_type: Step = Step.CONSTANT,
        iterations: int = 1000,
        write_filename=None,
    ):
        self.trajectories = list(chain.from_iterable(trajectories))
        self.r_max = r_max
        self.r_min = r_min if r_min is not None else -1 * self.r_max
        self.step_size = step_size
        self.step_size_end = step_size_end
        self.mdp_true = mdp_true
        self.alpha = alpha
        self.iterations = iterations
        self.prior = prior
        self.current_i = 0
        self.step_type = step_type
        self.write_filename = write_filename

        if self.step_type == Step.CONSTANT:
            self.step_func = constant_step
        elif self.step_type == Step.LINEAR:
            self.step_func = linear_decrease_step
        elif self.step_type == Step.EXPONENTIAL:
            self.step_func = exponential_decrease_step

    def random_rewards(self) -> dict:
        space = np.arange(self.r_min, self.r_max, self.step_size)
        return {s: np.random.choice(space) for s in self.mdp_true.states}

    def get_rewards_neighbour(self, rewards: dict):
        step_size = self.step_func(self.step_size, self.step_size_end, self.iterations, self.current_i)
        rewards_new = deepcopy(rewards)
        step = random.choice([-step_size, step_size])
        state = random.choice(list(rewards.keys()))
        rewards_new[state] = np.clip(rewards[state] + step, a_max=self.r_max, a_min=self.r_min)
        return rewards_new

    def compute_likelihood(self, mdp, pi):
        V = value_function(mdp, pi)
        q = [q_value(mdp=mdp, s=s, a=a, V=V) for s, a in self.trajectories]
        return self.alpha * np.sum(q)

    def compute_posterior(self, mdp: GridMdpOld, pi: dict):
        return self.compute_likelihood(mdp, pi) + self.prior(mdp, self.r_min, self.r_max)

    def compute_ratio(self, mdp: GridMdpOld, mdp_tilde: GridMdpOld, pi, pi_tilde=None) -> float:
        if pi_tilde:
            upper = self.compute_posterior(mdp_tilde, pi_tilde)
        else:
            upper = self.compute_posterior(mdp_tilde, pi)
        lower = self.compute_posterior(mdp, pi)
        return np.exp(upper - lower)

    def run_policy_walk(self) -> GridMdpOld:
        mdp = copy.copy(self.mdp_true)
        mdp.rewards = self.random_rewards()
        pi = policy_iteration(mdp)

        for i in trange(self.iterations, desc="BIRL PolicyWalk", leave=False):
            self.current_i = i

            mdp_tilde = copy.copy(mdp)
            mdp_tilde.rewards = self.get_rewards_neighbour(mdp.rewards)

            if not pi_is_optimal(mdp_tilde, pi):
                pi_tilde = policy_iteration(mdp_tilde, deepcopy(pi))
                prob = min(1.0, self.compute_ratio(mdp=mdp, mdp_tilde=mdp_tilde, pi=pi, pi_tilde=pi_tilde))
                if np.random.random() < prob:
                    mdp = copy.copy(mdp_tilde)
                    pi = pi_tilde
            else:
                prob = min(1.0, self.compute_ratio(mdp=mdp, mdp_tilde=mdp_tilde, pi=pi))
                if np.random.random() < prob:
                    mdp = copy.copy(mdp_tilde)

        return mdp


def pi_is_optimal(mdp: GridMdpOld, pi) -> bool:
    V = {s: 0 for s in mdp.states}
    V = policy_evaluation(pi, V, mdp)
    for s in mdp.states:
        if s in mdp.terminals:
            continue
        action_utils = {a: expected_utility(a, s, V, mdp) for a in mdp.actions(s)}
        max_u = max(action_utils.values())
        best = [a for a, u in action_utils.items() if u == max_u]
        optimal_probs = {a: (1 / len(best) if a in best else 0) for a in mdp.actions(s)}
        if pi[s] != optimal_probs:
            return False
    return True


def irl_with_birl(mdp: GridMdpOld, trajectories: List[List[Tuple]], **kwargs) -> GridMdpOld:
    r_max        = kwargs.get("r_max")
    r_min        = kwargs.get("r_min")
    step_size    = kwargs.get("step_size")
    step_size_end = kwargs.get("step_size_end")
    step_type    = kwargs.get("step_type")
    alpha        = kwargs.get("alpha")
    prior        = kwargs.get("prior")
    iterations   = kwargs.get("iterations")
    write_filename = kwargs.get("write_filename", None)

    policy_walk = PolicyWalk(
        mdp_true=mdp,
        trajectories=trajectories,
        r_max=r_max,
        r_min=r_min,
        step_size=step_size,
        step_size_end=step_size_end,
        step_type=step_type,
        alpha=alpha,
        prior=prior,
        iterations=iterations,
        write_filename=write_filename,
    )
    mdp_hat = policy_walk.run_policy_walk()
    del mdp_hat.oto_rewards
    del mdp_hat.grid
    return mdp_hat


birl_default_config = {
    "r_max": -0.1,
    "r_min": -2.1,
    "step_size": 0.5,
    "alpha": 20,
    "iterations": 1500,
    "step_type": Step.CONSTANT,
    "prior": uniform_prior,
}


# =============================================================================
# Public wrapper
# =============================================================================

def run_birl(paths, layout, **kwargs):
    """
    BIRL subgoal inference.

    Builds a GridMdpOld from `layout`, runs PolicyWalk MCMC to infer a reward
    function, and returns the highest-reward non-terminal, non-start cell.

    Parameters
    ----------
    paths : list of list of (col, row_top)
    layout : dict  — grid_layout, goal_pos, coin_pos, agent_start_pos
    **kwargs : override any key in birl_default_config

    Returns
    -------
    score_grid : np.ndarray [n_rows, n_cols]   inferred rewards (display coords)
    predicted  : (col, row_top)
    """
    n_rows = len(layout["grid_layout"])
    n_cols = len(layout["grid_layout"][0])
    goal_j = tuple(layout["goal_pos"])

    mdp = build_gridmdp_old(layout)

    # Convert paths to [(state_mdp, action_mdp), ...] without terminal marker
    trajs = [json_path_to_mdp_traj(p, n_rows, terminal_marker=False)
             for p in paths if len(p) >= 2]
    trajs = [t for t in trajs if t]
    if not trajs:
        return np.full((n_rows, n_cols), np.nan), goal_j

    config = {**birl_default_config, **kwargs}
    mdp_inferred = irl_with_birl(mdp, trajs, **config)

    score = np.full((n_rows, n_cols), np.nan)
    start_set = {to_mdp(*p[0], n_rows) for p in paths if p}
    goal_m = to_mdp(*goal_j, n_rows)

    best, best_r = goal_j, -np.inf
    for s, r in mdp_inferred.rewards.items():
        col, y = s
        if s in start_set or s == goal_m:
            continue
        rt = n_rows - 1 - y
        if 0 <= rt < n_rows and 0 <= col < n_cols:
            score[rt, col] = r
            if r > best_r:
                best_r = r
                best = to_json(col, y, n_rows)

    return score, best
