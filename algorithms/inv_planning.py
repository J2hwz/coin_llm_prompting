"""
inv_planning.py
---------------
Bayesian Inverse Planning (subgoal inference via soft value iteration).

Algorithm
---------
For each candidate subgoal g, compute a soft-optimal policy toward g using
soft value iteration, then evaluate the likelihood of the observed trajectory
under that policy. A Bayesian posterior over subgoals is formed by normalising
these likelihoods. The predicted subgoal is the argmax of the posterior.

Reference
---------
Baker, Saxe & Tenenbaum (2011). Bayesian theory of mind.
Implementation: jupyter_demos/inverse_planning/bayesian_subgoal_inference_demo.py

Public API
----------
    run_inv_planning(paths, layout, beta=2.0)
        -> (score_grid_display, predicted_json_coords)

Coordinate conventions
----------------------
- JSON coords : (col, row_from_top)  — row 0 = top of grid image
- MDP coords  : (x, y_mdp)           — y=0 = bottom row
- Conversion  : y_mdp = n_rows - 1 - row_top
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

# ── Path setup ────────────────────────────────────────────────────────────────
_ALGO_DIR = Path(__file__).resolve().parent
if str(_ALGO_DIR) not in sys.path:
    sys.path.insert(0, str(_ALGO_DIR))

from algorithms_util import to_mdp, to_json, build_gridmdp_old, json_path_to_mdp_traj

# Cardinal actions in MDP (x, y_mdp) space: W, S, E, N
_ACTIONS = [(-1, 0), (0, -1), (1, 0), (0, 1)]


# ── Soft value iteration ──────────────────────────────────────────────────────

def _soft_value_iteration(mdp, goal, beta, max_iter=600, tol=1e-7):
    """Backward pass from goal using soft Bellman equations.

    V(goal) = 0 (absorbing), step cost = −1 for all other states.
    V(s)    = (1/β) · log Σ_a exp(β · Q(s, a))
    Q(s, a) = −1 + γ · Σ_s' T(s, a, s') · V(s')
    """
    V = {s: 0.0 for s in mdp.states}
    for _ in range(max_iter):
        V_new = {}
        for s in mdp.states:
            if s == goal:
                V_new[s] = 0.0
                continue
            qs = np.array([
                -1.0 + mdp.gamma * sum(p * V[sp]
                                       for p, sp in mdp.transitions[s][a])
                for a in _ACTIONS
            ])
            V_new[s] = (1.0 / beta) * logsumexp(beta * qs)
        delta = max(abs(V_new[s] - V[s]) for s in mdp.states)
        V = V_new
        if delta < tol:
            break

    Q = {}
    for s in mdp.states:
        for a in _ACTIONS:
            if s == goal:
                Q[s, a] = 0.0
            else:
                Q[s, a] = (-1.0 + mdp.gamma *
                           sum(p * V[sp] for p, sp in mdp.transitions[s][a]))
    return V, Q


# ── Likelihood helpers ────────────────────────────────────────────────────────

def _step_log_prob(Q, s, a, beta):
    """log P(a | s) under the Boltzmann policy with Q-function Q."""
    qs = np.array([Q[s, a_] for a_ in _ACTIONS])
    return float(beta * Q[s, a] - logsumexp(beta * qs))


def _traj_log_likelihood(traj, Q_c, Q_g, subgoal, beta):
    """log P(trajectory | subgoal = c).

    Splits at t* = first visit to c.
    Leg 1 (0 … t*−1): scored under Q_c (policy toward subgoal).
    Leg 2 (t* … end−1): scored under Q_g (policy toward terminal).
    Returns −∞ if trajectory never visits c.
    """
    states = [s for s, _ in traj]
    try:
        t_star = states.index(subgoal)
    except ValueError:
        return -np.inf

    ll = 0.0
    for i in range(t_star):
        s, a = traj[i]
        ll += _step_log_prob(Q_c, s, a, beta)
    for i in range(t_star, len(traj) - 1):
        s, a = traj[i]
        if a is None:
            continue
        ll += _step_log_prob(Q_g, s, a, beta)
    return ll


# ── Candidate set and posterior ───────────────────────────────────────────────

def candidate_set(trajectories, starts, terminal):
    """Union of visited states across all trajectories,
    excluding known start positions and the terminal."""
    excluded = set(starts) | {terminal}
    sets = [{s for s, _ in traj if s not in excluded} for traj in trajectories]
    if not sets:
        return frozenset()
    result = sets[0]
    for vs in sets[1:]:
        result = result | vs
    return frozenset(result)


def _candidate_qfunctions(mdp, candidates, terminal, beta):
    """Q_g toward terminal (once) + Q_c toward each candidate (once each).

    Factored out of compute_posterior so compute_posterior_sequential can
    reuse the identical, expensive soft-VI computation without duplicating it.
    """
    _, Q_g = _soft_value_iteration(mdp, terminal, beta)
    Q_c_by_candidate = {c: _soft_value_iteration(mdp, c, beta)[1] for c in candidates}
    return Q_g, Q_c_by_candidate


def _traj_contribution(traj, Q_c, Q_g, subgoal, beta):
    """log-likelihood contribution of one trajectory toward one candidate.

    A trajectory that never visits `subgoal` contributes 0.0 (neutral — no
    evidence either way) rather than -inf. This means a candidate is not
    disqualified just because some pooled trajectory happened not to visit
    it; it remains viable as long as at least one trajectory (anywhere in
    the pool) visits it, which candidate_set already guarantees.
    """
    ll = _traj_log_likelihood(traj, Q_c, Q_g, subgoal, beta)
    return 0.0 if not np.isfinite(ll) else ll


def _normalize_log_liks(log_liks):
    """Log-softmax normalisation with graceful uniform fallback.

    Kept as its own function so compute_posterior and
    compute_posterior_sequential share identical normalisation behaviour.
    """
    finite = {c: v for c, v in log_liks.items() if np.isfinite(v)}
    if not finite:
        n = len(log_liks)
        return {c: 1.0 / n for c in log_liks}

    ll_arr = np.array(list(finite.values()))
    probs  = np.exp(ll_arr - logsumexp(ll_arr))

    posterior = {c: 0.0 for c in log_liks}
    for c, p in zip(finite.keys(), probs):
        posterior[c] = float(p)
    return posterior


def compute_posterior(trajectories, mdp, candidates, terminal, beta):
    """Compute P(c | trajectories) for each candidate c.

    1. Backward pass from terminal (shared across all candidates).
    2. For each c: backward pass from c, sum log-likelihoods (a trajectory
       that never visits c contributes 0.0, not -inf — see
       _traj_contribution).
    3. Log-softmax normalisation (uniform prior).

    Returns dict {candidate → posterior probability}.
    """
    Q_g, Q_c_by_candidate = _candidate_qfunctions(mdp, candidates, terminal, beta)

    log_liks = {
        c: sum(_traj_contribution(t, Q_c_by_candidate[c], Q_g, c, beta) for t in trajectories)
        for c in candidates
    }

    return _normalize_log_liks(log_liks)


def compute_posterior_sequential(trajectories, mdp, candidates, terminal, beta):
    """Incorporate trajectories one at a time; track posterior + entropy history.

    Mathematically identical to compute_posterior(trajectories, mdp, candidates,
    terminal, beta) when given the same trajectory set (order-invariant, since
    it is the same cumulative log-likelihood summation with the same final
    softmax normalisation — intermediate per-trajectory normalisation is purely
    a reporting snapshot and does not feed back into the cumulative
    log-likelihoods used for subsequent trajectories).

    Parameters
    ----------
    trajectories : list of MDP trajectories, in incorporation order.
    mdp, candidates, terminal, beta : same as compute_posterior.

    Returns
    -------
    dict:
        "posterior" : {candidate -> final probability}
        "argmax"    : candidate with highest final probability, or None if
                      `candidates` is empty
        "history"   : list, one entry per trajectory processed, in order:
            {
                "trajectory_index": int,        # 0-based index into `trajectories`
                "num_trajectories_seen": int,   # trajectory_index + 1
                "posterior": {candidate -> probability},  # snapshot after
                                                            # trajectories[0..i]
                "entropy_bits": float,           # Shannon entropy, base 2
            }
    """
    Q_g, Q_c_by_candidate = _candidate_qfunctions(mdp, candidates, terminal, beta)

    cum_log_lik = {c: 0.0 for c in candidates}
    history = []
    for i, traj in enumerate(trajectories):
        for c in candidates:
            cum_log_lik[c] += _traj_contribution(traj, Q_c_by_candidate[c], Q_g, c, beta)
        snapshot = _normalize_log_liks(cum_log_lik)
        history.append({
            "trajectory_index": i,
            "num_trajectories_seen": i + 1,
            "posterior": snapshot,
            "entropy_bits": _shannon_entropy_bits(snapshot),
        })

    final_posterior = history[-1]["posterior"] if history else _normalize_log_liks(cum_log_lik)
    argmax = max(final_posterior, key=final_posterior.get) if final_posterior else None

    return {"posterior": final_posterior, "argmax": argmax, "history": history}


def _shannon_entropy_bits(posterior):
    """-Σ p·log2(p) over a posterior dict; 0-probability entries contribute 0."""
    ps = np.array(list(posterior.values()), dtype=float)
    ps = ps[ps > 0]
    if ps.size == 0:
        return 0.0
    return float(-np.sum(ps * np.log2(ps)))


# ── Grid construction ─────────────────────────────────────────────────────────

# ── Main wrapper ──────────────────────────────────────────────────────────────

def run_inv_planning(paths, layout, beta=2.0):
    """Bayesian subgoal inference via soft value iteration.

    For each candidate cell, evaluates how likely the observed trajectory is
    under a soft-optimal policy aimed at that cell, then normalises to a
    posterior. Works with any number of trajectories (one-shot or aggregated).

    Parameters
    ----------
    paths : list of list of (col, row_top)
        One or more trajectory paths in JSON coordinates.
    layout : dict
        Grid layout dict with keys: grid_layout, goal_pos, coin_pos,
        agent_start_pos.
    beta : float
        Softmax temperature for the soft policy (higher = more deterministic).

    Returns
    -------
    score_grid_display : np.ndarray, shape (n_rows, n_cols)
        Posterior probability at each cell, indexed [row_top, col].
        Wall cells and the terminal (goal) cell are NaN.
    predicted_json_coords : (col, row_top)
        Cell with the highest posterior probability.
    """
    n_rows = len(layout["grid_layout"])
    n_cols = len(layout["grid_layout"][0])
    goal_j = tuple(layout["goal_pos"])
    goal_m = to_mdp(*goal_j, n_rows)

    mdp = build_gridmdp_old(layout)

    trajs_mdp = [
        json_path_to_mdp_traj(p, n_rows, terminal_marker=True)
        for p in paths if len(p) >= 2
    ]
    trajs_mdp = [t for t in trajs_mdp if t]

    if not trajs_mdp:
        return np.full((n_rows, n_cols), np.nan), goal_j

    starts = [t[0][0] for t in trajs_mdp]
    cands  = candidate_set(trajs_mdp, starts, goal_m)

    if not cands:
        excluded = set(starts) | {goal_m}
        cands = frozenset(
            s for t in trajs_mdp for s, _ in t if s not in excluded
        )

    if not cands:
        return np.full((n_rows, n_cols), np.nan), goal_j

    posterior = compute_posterior(trajs_mdp, mdp, cands, goal_m, beta)

    score = np.full((n_rows, n_cols), np.nan)
    for (col, y), prob in posterior.items():
        score[n_rows - 1 - y, col] = prob

    best_m = max(posterior, key=posterior.get)
    return score, to_json(*best_m, n_rows)


# ── CLI glue ──────────────────────────────────────────────────────────────────

def _prepare_candidates(paths, layout, terminal_marker=True):
    """Build MDP + MDP-trajectories + candidate set from JSON-coord paths/layout.

    Small deliberate duplication of the glue logic already inlined in
    run_inv_planning, rather than factoring it out of run_inv_planning itself
    (kept untouched). Avoids a second full soft-VI pass when a caller needs
    both the score grid (run_inv_planning) and the sequential history
    (compute_posterior_sequential) from the same trajectories.

    Returns (mdp, trajs_mdp, candidates, goal_m, goal_j, n_rows, n_cols).
    trajs_mdp / candidates are empty (frozenset()/[]) if no valid trajectories.
    """
    n_rows = len(layout["grid_layout"])
    n_cols = len(layout["grid_layout"][0])
    goal_j = tuple(layout["goal_pos"])
    goal_m = to_mdp(*goal_j, n_rows)
    mdp = build_gridmdp_old(layout)

    trajs_mdp = [
        json_path_to_mdp_traj(p, n_rows, terminal_marker=terminal_marker)
        for p in paths if len(p) >= 2
    ]
    trajs_mdp = [t for t in trajs_mdp if t]

    cands = frozenset()
    if trajs_mdp:
        starts = [t[0][0] for t in trajs_mdp]
        cands = candidate_set(trajs_mdp, starts, goal_m)
        if not cands:
            excluded = set(starts) | {goal_m}
            cands = frozenset(s for t in trajs_mdp for s, _ in t if s not in excluded)

    return mdp, trajs_mdp, cands, goal_m, goal_j, n_rows, n_cols


def _state_key(state):
    return f"{state[0]},{state[1]}"


# ── Smoke-test fixtures ───────────────────────────────────────────────────────

def _corridor_dead_end_fixture():
    """4-row x 7-col grid: a single open corridor with one dead-end branch.

    JSON layout (row 0 = top):
        row0: # # # # # # #
        row1: # _ _ _ _ _ #      <- corridor, cols 1..5 open
        row2: # # # _ # # #      <- only col 3 open: the dead-end (= coin)
        row3: # # # # # # #

    start = (1,1) [json], goal = (5,1) [json], dead-end/coin = (3,2) [json].
    Trajectory: walk the corridor to the junction (3,1), detour one step
    south into the dead end (3,2) and back, then continue to goal (5,1).

    An open (wall-less) grid with a simple detour does not reliably produce a
    unique argmax under the two-phase soft-VI likelihood (many collinear
    points along a monotonic detour tie for maximal likelihood). A single
    true dead-end branch guarantees a strict, non-tied argmax: the dead-end
    cell's entire trajectory is unpenalized under both legs of the split
    (its only non-self-loop action is back out to the junction), while every
    other candidate incurs a real penalty from the same "detour into the dead
    end" step, since that step is not optimal toward any other hypothesis.
    This also mirrors the project's own real coin_placement="dead_end" mode.
    """
    layout = {
        "grid_layout": [
            ["#", "#", "#", "#", "#", "#", "#"],
            ["#", "_", "_", "_", "_", "_", "#"],
            ["#", "#", "#", "_", "#", "#", "#"],
            ["#", "#", "#", "#", "#", "#", "#"],
        ],
        "goal_pos": [5, 1],
        "coin_pos": [3, 2],
        "agent_start_pos": [1, 1],
    }
    paths = [[[1, 1], [2, 1], [3, 1], [3, 2], [3, 1], [4, 1], [5, 1]]]
    true_coin_json = (3, 2)
    return layout, paths, true_coin_json


def _oscillating_path(n_steps, row_top=1, col_lo=1, col_hi=5, start_col=1):
    """Back-and-forth walk within [col_lo, col_hi] at fixed row_top."""
    cols = [start_col]
    direction = 1
    col = start_col
    for _ in range(n_steps):
        nxt = col + direction
        if nxt > col_hi or nxt < col_lo:
            direction *= -1
            nxt = col + direction
        col = nxt
        cols.append(col)
    return [[c, row_top] for c in cols]


# ── Smoke tests (pytest-discoverable and __main__-runnable) ──────────────────

def test_sanity_dead_end_argmax():
    """Argmax must equal the true dead-end coin cell with dominant posterior mass."""
    layout, paths, true_coin_json = _corridor_dead_end_fixture()
    score, predicted = run_inv_planning(paths, layout, beta=2.0)
    assert predicted == true_coin_json, f"expected argmax {true_coin_json}, got {predicted}"

    col, row_top = true_coin_json
    true_prob = score[row_top, col]
    other_probs = [
        score[r, c]
        for r in range(score.shape[0]) for c in range(score.shape[1])
        if not np.isnan(score[r, c]) and (c, r) != (col, row_top)
    ]
    assert true_prob > max(other_probs), (
        f"dead-end cell not strictly dominant: {true_prob} vs max other {max(other_probs)}"
    )

    # cross-check the sequential function agrees on the same single trajectory
    mdp, trajs, cands, goal_m, _, n_rows, _ = _prepare_candidates(paths, layout)
    seq = compute_posterior_sequential(trajs, mdp, cands, goal_m, beta=2.0)
    assert seq["argmax"] == to_mdp(*true_coin_json, n_rows)


def test_order_invariance():
    """Final posterior must not depend on the order trajectories are processed in."""
    layout, _, _ = _corridor_dead_end_fixture()
    paths = [
        [[1, 1], [2, 1], [3, 1], [3, 2], [3, 1], [4, 1], [5, 1]],                 # T1: detour
        [[1, 1], [2, 1], [3, 1], [4, 1], [3, 1], [3, 2], [3, 1], [4, 1], [5, 1]], # T2: overshoot then detour
        [[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]],                                # T3: never visits dead end
    ]
    mdp, trajs, cands, goal_m, _, n_rows, _ = _prepare_candidates(paths, layout)

    posteriors = []
    for perm in itertools.permutations(range(len(trajs))):
        ordered = [trajs[i] for i in perm]
        result = compute_posterior_sequential(ordered, mdp, cands, goal_m, beta=2.0)
        posteriors.append(result["posterior"])

    ref = posteriors[0]
    for other in posteriors[1:]:
        for c in cands:
            assert np.allclose(ref[c], other[c], atol=1e-9)

    batch = compute_posterior(trajs, mdp, cands, goal_m, beta=2.0)
    for c in cands:
        assert np.allclose(ref[c], batch[c], atol=1e-9)


def test_missing_visit_not_disqualifying():
    """A trajectory that never visits a candidate must not zero out its posterior."""
    layout, _, true_coin_json = _corridor_dead_end_fixture()
    detour_path = [[1, 1], [2, 1], [3, 1], [3, 2], [3, 1], [4, 1], [5, 1]]
    straight_path = [[1, 1], [2, 1], [3, 1], [4, 1], [5, 1]]  # never visits the dead end

    mdp, trajs, cands, goal_m, _, n_rows, _ = _prepare_candidates(
        [detour_path, straight_path], layout
    )
    true_coin_mdp = to_mdp(*true_coin_json, n_rows)

    seq = compute_posterior_sequential(trajs, mdp, cands, goal_m, beta=2.0)
    final_posterior = seq["posterior"]
    assert final_posterior[true_coin_mdp] > 0.0, "dead-end candidate was zeroed out"
    assert seq["argmax"] == true_coin_mdp

    # Adding the non-visiting trajectory alone must not change the dead-end
    # candidate's cumulative log-likelihood at all (contributes exactly 0.0).
    Q_g, Q_c_by_candidate = _candidate_qfunctions(mdp, cands, goal_m, 2.0)
    contribution = _traj_contribution(
        trajs[1], Q_c_by_candidate[true_coin_mdp], Q_g, true_coin_mdp, 2.0
    )
    assert contribution == 0.0


def test_wall_bump_steps_collapse_to_next_real_move():
    """A run of consecutive stay-in-place positions must contribute zero
    extra trajectory entries; the next recorded entry is the first step
    where real movement resumes.
    """
    # step1: (0,0)->(1,0) real move | steps2-4: stay at (1,0) (3 wall bumps)
    # | step5: (1,0)->(2,0) real move
    path = [(0, 0), (1, 0), (1, 0), (1, 0), (1, 0), (2, 0)]
    traj = json_path_to_mdp_traj(path, n_rows=3, terminal_marker=False)
    assert len(traj) == 2, f"expected 2 entries (step1 and step5 only), got {len(traj)}"
    assert traj[0] == (to_mdp(0, 0, 3), (1, 0)), f"entry 0 should be step 1's move, got {traj[0]}"
    assert traj[1] == (to_mdp(1, 0, 3), (1, 0)), f"entry 1 should be step 5's move, got {traj[1]}"


def test_no_underflow_many_trajectories():
    """10 length-~50 trajectories must produce finite, normalized posteriors throughout."""
    layout, _, _ = _corridor_dead_end_fixture()
    paths = [_oscillating_path(50) for _ in range(10)]
    mdp, trajs, cands, goal_m, _, n_rows, _ = _prepare_candidates(paths, layout)

    result = compute_posterior_sequential(trajs, mdp, cands, goal_m, beta=2.0)
    assert len(result["history"]) == 10
    for snap in result["history"]:
        probs = np.array(list(snap["posterior"].values()))
        assert np.all(np.isfinite(probs)), "non-finite posterior probability"
        assert np.isclose(probs.sum(), 1.0, atol=1e-8), f"posterior does not sum to 1: {probs.sum()}"
        assert np.isfinite(snap["entropy_bits"])


def _run_smoke_tests():
    """Run built-in test_* functions and report pass/fail. Returns True iff all pass."""
    tests = [
        test_sanity_dead_end_argmax,
        test_order_invariance,
        test_missing_visit_not_disqualifying,
        test_wall_bump_steps_collapse_to_next_real_move,
        test_no_underflow_many_trajectories,
    ]
    all_ok = True
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"[FAIL] {t.__name__}: {e}")
            all_ok = False
        except Exception as e:
            print(f"[ERROR] {t.__name__}: {type(e).__name__}: {e}")
            all_ok = False
        else:
            print(f"[PASS] {t.__name__}")
    return all_ok


# ── CLI entry point ───────────────────────────────────────────────────────────

def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Bayesian inverse planning: sequential posterior over subgoal candidates."
    )
    parser.add_argument("--layout", type=str, default=None,
                         help="Path to JSON layout file (grid_layout, goal_pos, ...).")
    parser.add_argument("--paths", type=str, default=None,
                         help="Path to JSON file: list of paths, each a list of [col, row_top] pairs.")
    parser.add_argument("--beta", type=float, default=2.0,
                         help="Softmax temperature for the soft policy (default: 2.0).")
    parser.add_argument("--output", type=str, default=None,
                         help="Optional path to write result JSON (default: print to stdout).")
    return parser


def main(argv=None):
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.layout is None or args.paths is None:
        ok = _run_smoke_tests()
        sys.exit(0 if ok else 1)

    with open(args.layout) as f:
        layout = json.load(f)
    with open(args.paths) as f:
        paths = json.load(f)

    mdp, trajs, cands, goal_m, goal_j, n_rows, n_cols = _prepare_candidates(paths, layout)

    if not trajs or not cands:
        result = {
            "beta": args.beta,
            "predicted_json_coords": list(goal_j),
            "num_candidates": 0,
            "posterior": {},
            "argmax_mdp_state": None,
            "history": [],
        }
    else:
        seq = compute_posterior_sequential(trajs, mdp, cands, goal_m, args.beta)
        predicted_json = to_json(*seq["argmax"], n_rows) if seq["argmax"] is not None else goal_j
        result = {
            "beta": args.beta,
            "predicted_json_coords": list(predicted_json),
            "num_candidates": len(cands),
            "posterior": {_state_key(c): p for c, p in seq["posterior"].items()},
            "argmax_mdp_state": list(seq["argmax"]) if seq["argmax"] is not None else None,
            "history": [
                {
                    "trajectory_index": h["trajectory_index"],
                    "num_trajectories_seen": h["num_trajectories_seen"],
                    "entropy_bits": h["entropy_bits"],
                    "posterior": {_state_key(c): p for c, p in h["posterior"].items()},
                }
                for h in seq["history"]
            ],
        }

    out_str = json.dumps(result, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(out_str)
    else:
        print(out_str)


if __name__ == "__main__":
    main()
