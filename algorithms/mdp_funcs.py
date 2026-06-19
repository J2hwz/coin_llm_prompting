"""
mdp_funcs.py
------------
Generic MDP utility functions. Work with any MDP that implements:
    .states, .terminals, .actions(s), .T(s, a), .R(s), .gamma

Adapted from mdp.py (repo root) — pdb artifacts removed, no GridMdp class.
"""

from typing import Tuple, Optional

from grid_utils import orientations, STAY


def action_to_int(a: Tuple[int, int]) -> Optional[int]:
    """Return 0-indexed int representing an action (None → None, STAY → 4)."""
    if a is None:
        return None
    if a == STAY:
        return 4
    return orientations.index(a)


def expected_utility(a, s, V, mdp):
    """Expected value of doing action a in state s given value function V."""
    return sum(p * V[s1] for (p, s1) in mdp.T(s, a))


def policy_evaluation(pi: dict, V: dict, mdp, threshold=0.001):
    """Evaluate a stochastic policy pi in-place on V. pi[s] = {action: prob}."""
    gamma = mdp.gamma
    while True:
        delta = 0
        for s in mdp.states:
            old_v = V[s]
            if s in mdp.terminals:
                V[s] = mdp.R(s)
            else:
                actions = pi[s]
                summed = 0
                for a, prob in actions.items():
                    summed += prob * (mdp.R(s) + gamma * sum(p * V[s1] for (p, s1) in mdp.T(s, a)))
                V[s] = summed
            delta = max(delta, abs(old_v - V[s]))
        if delta < threshold:
            break
    return V


def init_random_policy(mdp) -> dict:
    """Uniform random stochastic policy over all states."""
    return {s: {a: 1 / len(mdp.actions(s)) for a in mdp.actions(s)} for s in mdp.states}


def policy_iteration(mdp, pi: dict = None) -> dict:
    """Policy iteration returning a stochastic policy {state: {action: prob}}.

    Ties (equally good actions) get equal probability.
    """
    V = {s: 0 for s in mdp.states}
    if pi is None:
        pi = init_random_policy(mdp)
    while True:
        V = policy_evaluation(pi, V, mdp)
        unchanged = True
        for s in mdp.states:
            if s in mdp.terminals:
                continue
            action_utils = {a: expected_utility(a, s, V, mdp) for a in mdp.actions(s)}
            max_utility = max(action_utils.values())
            best_actions = [a for a in action_utils if action_utils[a] == max_utility]
            action_probs = {a: (1 / len(best_actions) if a in best_actions else 0)
                           for a in mdp.actions(s)}
            if pi[s] != action_probs:
                pi[s] = action_probs
                unchanged = False
        if unchanged:
            return pi


def q_value(mdp, s, a=None, V: dict = None, pi: dict = None):
    """Q(s, a). If V is not provided, computes it from pi (or runs value iteration)."""
    if not a:
        return mdp.R(s)
    if V is None:
        V = value_function(mdp, pi)
    res = 0
    for prob, s_prime in mdp.T(s, a):
        res += prob * (mdp.R(s) + mdp.gamma * V[s_prime])
    return res


def value_function(mdp, pi=None, epsilon=0.001):
    """Value function for policy pi (stochastic). If pi is None, runs value iteration."""
    V1 = {s: 0 for s in mdp.states}
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    while True:
        V = V1.copy()
        delta = 0
        for s in mdp.states:
            if pi is None:
                V1[s] = R(s) + gamma * max(
                    sum(p * V[s1] for (p, s1) in T(s, a)) for a in mdp.actions(s)
                )
            else:
                if s in mdp.terminals:
                    V1[s] = q_value(mdp, s, None, V)
                else:
                    actions = pi[s]
                    V1[s] = sum(prob * q_value(mdp, s, a, V) for a, prob in actions.items())
            delta = max(delta, abs(V1[s] - V[s]))
        if delta <= epsilon * (1 - gamma) / gamma:
            return V
