"""
trex.py
-------
Simplified T-REX (Trajectory-ranked Reward EXtrapolation, Brown et al. 2019).

Given coin-navigation trajectories already collected by coinenv-cli, this module:
  1. Ranks trajectories using observable quality proxies (path efficiency +
     per-trajectory movement entropy).
  2. Samples pairwise clips with the T-REX temporal ordering constraint.
  3. Trains a small PyTorch reward network via Bradley-Terry preference loss.
  4. Extracts a 2-D reward map in the same format as run_maxent(), enabling
     direct side-by-side comparison.

State representation fed to the network: [x_norm, y_norm, coin_flag]
  - x_norm  = col / (n_cols - 1)
  - y_norm  = (n_rows - 1 - row_top) / (n_rows - 1)   (MDP y-axis, 0 = bottom)
  - coin_flag = 1 once coin_pos has been visited in the path, else 0

This file is self-contained within algorithms/ and does not modify any
existing algorithm file.

Public API
----------
    load_all_trajectories(data_dir, grid_id, layout, effort)
        -> (paths, success_flags, traj_ids)

    rank_trajectories(paths, layout, success_flags, w_efficiency, w_entropy)
        -> (ranked_paths, scores, order)

    sample_clip_pairs(ranked_paths, layout, clip_len, n_pairs, seed)
        -> list of (states_i, states_j)

    class RewardNet

    train_trex(ranked_paths, layout, ...)
        -> RewardNet

    extract_reward_map(reward_net, layout)
        -> (score_grid, predicted_json_coords)

    compare_with_maxent(trex_grid, maxent_grid, layout, save_path)
        -> dict with predicted cells, Manhattan distances, Spearman r

Coordinate conventions
----------------------
    JSON coords : (col, row_from_top)  — row 0 = top of grid image
    MDP coords  : (x, y_mdp)          — y_mdp = 0 = bottom row
    score_grid  : np.ndarray[n_rows, n_cols], indexed [row_from_top, col]
"""

import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Path setup ────────────────────────────────────────────────────────────────
_ALGO_DIR = Path(__file__).resolve().parent
if str(_ALGO_DIR) not in sys.path:
    sys.path.insert(0, str(_ALGO_DIR))

# Repo root — needed to import analysis.metrics
_REPO_DIR = _ALGO_DIR.parent
if str(_REPO_DIR) not in sys.path:
    sys.path.insert(0, str(_REPO_DIR))

from analysis.metrics import shannon_entropy  # reused: entropy in bits over ActionDist

# ── Action delta mapping (JSON coordinates) ───────────────────────────────────
# Matches run_algorithms._ACTION_DELTA:
#   "UP"    -> (dcol=0, drow=-1)   ActionID=2
#   "DOWN"  -> (dcol=0, drow=+1)   ActionID=3
#   "LEFT"  -> (dcol=-1, drow=0)   ActionID=0
#   "RIGHT" -> (dcol=+1, drow=0)   ActionID=1
_DELTA_TO_ACTION_ID = {(0, -1): 2, (0, 1): 3, (-1, 0): 0, (1, 0): 1}


# ── Grid helpers ──────────────────────────────────────────────────────────────

def _bfs_distance(layout, start_j, end_j):
    """Wall-aware BFS distance between two JSON-coord cells.

    Manhattan distance (grid_utils.shortest_dist) ignores walls and cannot
    be used here; this BFS operates directly on layout["grid_layout"].

    Parameters
    ----------
    layout : dict
    start_j, end_j : (col, row_from_top)

    Returns
    -------
    int or None
        Number of steps, or None if unreachable.
    """
    from collections import deque

    grid = layout["grid_layout"]
    n_rows = len(grid)
    n_cols = len(grid[0]) if n_rows > 0 else 0
    start = (int(start_j[0]), int(start_j[1]))
    end = (int(end_j[0]), int(end_j[1]))
    if start == end:
        return 0
    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        (c, r), dist = queue.popleft()
        for dc, dr in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            nc, nr = c + dc, r + dr
            if 0 <= nc < n_cols and 0 <= nr < n_rows:
                nb = (nc, nr)
                if nb not in visited and grid[nr][nc] != "#":
                    if nb == end:
                        return dist + 1
                    visited.add(nb)
                    queue.append((nb, dist + 1))
    return None


# ── Per-trajectory metric helpers ─────────────────────────────────────────────

def _path_efficiency(path, optimal_length):
    """Path efficiency for a single trajectory.

    Uses the same formula as the SPL component in
    analysis.analysis_utils.compute_spl: L* / max(L*, L).

    Parameters
    ----------
    path : list of (col, row_top)
    optimal_length : int
        Wall-aware optimal number of steps.

    Returns
    -------
    float in [0, 1]
    """
    actual = len(path) - 1  # number of transitions
    if actual <= 0 or optimal_length <= 0:
        return 0.0
    return optimal_length / max(optimal_length, actual)


def _action_dist_from_path(path):
    """Per-trajectory action frequency distribution for a single path.

    Operates on one trajectory, NOT aggregated over a grid. Counts how many
    times the agent used each direction (UP/DOWN/LEFT/RIGHT) across the whole
    path, then normalises to a probability distribution.

    High entropy -> moves equally in all four directions (unfocused).
    Low entropy  -> mostly one or two directions (purposeful movement).

    This is distinct from compute_empirical_uncertainty_metrics(), which
    pools action counts across ALL trajectories at each visited cell.

    Parameters
    ----------
    path : list of (col, row_top)

    Returns
    -------
    ActionDist : dict[int, float]
        Probability distribution over action IDs {0, 1, 2, 3}.
    """
    counts = Counter()
    for i in range(len(path) - 1):
        dc = path[i + 1][0] - path[i][0]
        dr = path[i + 1][1] - path[i][1]
        a = _DELTA_TO_ACTION_ID.get((dc, dr))
        if a is not None:
            counts[a] += 1
    total = sum(counts.values())
    if total == 0:
        return {a: 0.25 for a in range(4)}  # uniform if no valid moves
    return {a: counts.get(a, 0) / total for a in range(4)}


# ── Trajectory ranking ────────────────────────────────────────────────────────

def rank_trajectories(paths, layout, success_flags=None, w_efficiency=0.6, w_entropy=0.4):
    """Rank trajectories from worst to best using a composite proxy score.

    score(p) = w_efficiency * path_efficiency(p)
             + w_entropy   * (1 - min_max_normalised_entropy(p))

    This is a proxy ranking using observable quality signals; true reward is
    unobserved. Higher score = better trajectory.

    When `success_flags` is provided, unsuccessful trajectories are placed
    below all successful ones regardless of their proxy score (hard group
    separation). Within each group the proxy score ordering is preserved.

    Path efficiency reuses the SPL formula from analysis.analysis_utils.
    Entropy is computed per individual trajectory (not pooled across the
    grid) and then min-max normalised across the batch, consistent with
    normalisation conventions elsewhere in the repo.

    Parameters
    ----------
    paths : list of list of (col, row_top)
    layout : dict
    success_flags : list of bool or None
        If provided, all unsuccessful (False) trajectories rank below all
        successful (True) ones, regardless of proxy score.
    w_efficiency, w_entropy : float
        Weights for the two components (need not sum to 1).

    Returns
    -------
    ranked_paths : list of list of (col, row_top)
        Same paths, sorted ascending (worst first, best last).
    ranked_scores : list of float
        Corresponding composite scores.
    order : list of int
        Original indices in worst→best order; use to map back to traj_ids
        or success_flags after ranking.
    """
    coin_j = tuple(layout["coin_pos"])
    goal_j = tuple(layout["goal_pos"])
    start_j = tuple(layout["agent_start_pos"])

    d_start_coin = _bfs_distance(layout, start_j, coin_j) or 1
    d_coin_goal = _bfs_distance(layout, coin_j, goal_j) or 1
    optimal_length = d_start_coin + d_coin_goal

    efficiencies = [_path_efficiency(p, optimal_length) for p in paths]
    entropies = [shannon_entropy(_action_dist_from_path(p)) for p in paths]

    # Min-max normalise entropy across the batch
    e_min, e_max = min(entropies), max(entropies)
    if e_max > e_min:
        norm_entropies = [(e - e_min) / (e_max - e_min) for e in entropies]
    else:
        norm_entropies = [0.0] * len(entropies)

    scores = [
        w_efficiency * eff + w_entropy * (1.0 - ne)
        for eff, ne in zip(efficiencies, norm_entropies)
    ]

    if success_flags is not None:
        # Hard group separation: all unsuccessful < all successful
        unsuccessful = [(i, scores[i]) for i in range(len(paths)) if not success_flags[i]]
        successful   = [(i, scores[i]) for i in range(len(paths)) if success_flags[i]]
        order = (
            [i for i, _ in sorted(unsuccessful, key=lambda x: x[1])] +
            [i for i, _ in sorted(successful,   key=lambda x: x[1])]
        )
    else:
        order = sorted(range(len(paths)), key=lambda i: scores[i])

    ranked_paths  = [paths[i]  for i in order]
    ranked_scores = [scores[i] for i in order]
    return ranked_paths, ranked_scores, order


# ── State encoding ────────────────────────────────────────────────────────────

def _path_to_states(path, layout):
    """Convert a JSON-coord path to a list of 3-dim state vectors.

    Each state is [x_norm, y_norm, coin_flag]:
      x_norm    = col / (n_cols - 1)
      y_norm    = (n_rows - 1 - row_top) / (n_rows - 1)   (MDP y, 0 = bottom)
      coin_flag = 1.0 from the step the agent first visits coin_pos, else 0.0

    Parameters
    ----------
    path : list of (col, row_top)
    layout : dict

    Returns
    -------
    list of np.ndarray, shape (3,), dtype float32
    """
    grid = layout["grid_layout"]
    n_rows = len(grid)
    n_cols = len(grid[0]) if n_rows > 0 else 1
    coin_j = (int(layout["coin_pos"][0]), int(layout["coin_pos"][1]))

    states = []
    coin_collected = False
    for col, row_top in path:
        if (int(col), int(row_top)) == coin_j:
            coin_collected = True
        x_norm = col / max(1, n_cols - 1)
        y_mdp = n_rows - 1 - row_top
        y_norm = y_mdp / max(1, n_rows - 1)
        states.append(np.array([x_norm, y_norm, float(coin_collected)], dtype=np.float32))
    return states


# ── Clip pair sampling ────────────────────────────────────────────────────────

def sample_clip_pairs(ranked_paths, layout, clip_len=10, n_pairs=5000, seed=42):
    """Sample pairwise partial-trajectory clips following the T-REX procedure.

    For each pair (i, j) where i < j in the ranking (i = worse, j = better):
      t_i ~ Uniform[0, len(p_i) - 1]
      t_j ~ Uniform[t_i, max(t_i, len(p_j) - clip_len)]   (temporal constraint)
      clip_i = p_i[t_i : t_i + clip_len]
      clip_j = p_j[t_j : t_j + clip_len]

    The temporal constraint (t_j >= t_i) ensures that clips from better
    trajectories are sampled no earlier in time than clips from worse ones,
    creating preferences consistent with the temporal reward structure.

    Paths shorter than clip_len yield shorter clips (Python slicing is safe);
    paths shorter than 2 positions are skipped entirely.

    Parameters
    ----------
    ranked_paths : list of list of (col, row_top)
        Worst-to-best ordering from rank_trajectories().
    layout : dict
    clip_len : int
    n_pairs : int
    seed : int

    Returns
    -------
    list of (states_i, states_j)
        Each element is a tuple of two lists of np.ndarray, shape (3,).
    """
    rng = random.Random(seed)
    n = len(ranked_paths)
    if n < 2:
        return []

    pairs = []
    max_attempts = n_pairs * 20
    attempt = 0

    while len(pairs) < n_pairs and attempt < max_attempts:
        attempt += 1

        i = rng.randint(0, n - 2)
        j = rng.randint(i + 1, n - 1)
        p_i, p_j = ranked_paths[i], ranked_paths[j]

        if len(p_i) < 2 or len(p_j) < 2:
            continue

        t_i = rng.randint(0, len(p_i) - 1)

        t_j_lo = t_i
        t_j_hi = max(t_i, len(p_j) - clip_len)
        t_j = rng.randint(t_j_lo, t_j_hi) if t_j_hi > t_j_lo else t_j_lo

        if t_j >= len(p_j):
            continue

        clip_i = p_i[t_i: t_i + clip_len]
        clip_j = p_j[t_j: t_j + clip_len]

        states_i = _path_to_states(clip_i, layout)
        states_j = _path_to_states(clip_j, layout)

        if not states_i or not states_j:
            continue

        pairs.append((states_i, states_j))

    return pairs


# ── Reward network ────────────────────────────────────────────────────────────

class RewardNet(nn.Module):
    """Small MLP: (x_norm, y_norm, coin_flag) -> scalar reward.

    Two hidden layers with ReLU activations. Size is proportional to the
    gridworld state space — no large model needed here.
    """

    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """x: Tensor[batch, 3] -> Tensor[batch]"""
        return self.net(x).squeeze(-1)

    def clip_return(self, states):
        """Predicted cumulative return for a clip (sum of per-state rewards).

        Parameters
        ----------
        states : list of np.ndarray, shape (3,)

        Returns
        -------
        Tensor scalar (differentiable)
        """
        x = torch.tensor(np.stack(states), dtype=torch.float32)
        return self.forward(x).sum()


# ── Training ──────────────────────────────────────────────────────────────────

def train_trex(ranked_paths, layout, clip_len=10, n_pairs=5000, lr=1e-3,
               hidden_dim=64, seed=42, log_every=500):
    """Train a RewardNet via T-REX Bradley-Terry preference loss.

    Implements the preference loss from Brown et al. 2019 (Eq. 1-3):
        L = -log sigmoid(R(clip_j) - R(clip_i))
    where clip_j is from the better-ranked trajectory.

    Parameters
    ----------
    ranked_paths : list of list of (col, row_top)
        Worst-to-best ordering from rank_trajectories().
    layout : dict
    clip_len : int
    n_pairs : int
    lr : float
        Adam learning rate.
    hidden_dim : int
    seed : int
    log_every : int
        Print average loss every this many pairs.

    Returns
    -------
    RewardNet (eval mode)
    """
    torch.manual_seed(seed)
    net = RewardNet(input_dim=3, hidden_dim=hidden_dim)
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)

    pairs = sample_clip_pairs(ranked_paths, layout, clip_len=clip_len,
                               n_pairs=n_pairs, seed=seed)
    if not pairs:
        print("  [T-REX] No valid clip pairs sampled — returning untrained network")
        return net

    print(f"  [T-REX] Training on {len(pairs)} clip pairs  "
          f"(clip_len={clip_len}, lr={lr}, hidden_dim={hidden_dim})")

    net.train()
    running_loss = 0.0

    for k, (states_i, states_j) in enumerate(pairs):
        return_i = net.clip_return(states_i)
        return_j = net.clip_return(states_j)

        # Bradley-Terry preference loss: -log P(j > i) = -log sigmoid(R_j - R_i)
        loss = -F.logsigmoid(return_j - return_i)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        running_loss += loss.item()
        if (k + 1) % log_every == 0:
            avg = running_loss / log_every
            print(f"    pair {k + 1}/{len(pairs)}  avg_loss={avg:.4f}")
            running_loss = 0.0

    net.eval()
    return net


# ── Reward map extraction ─────────────────────────────────────────────────────

def extract_reward_map(reward_net, layout):
    """Query the reward network at every valid cell to produce a 2-D reward map.

    Queries with coin_flag=0 (pre-coin-collection phase), so the map reflects
    reward structure before the intermediate objective is reached. The cell
    with the highest reward is the model's predicted subgoal location.

    Output format matches run_maxent() exactly:
      score_grid[row_from_top, col] = predicted reward
      NaN for wall cells, the start cell, and the goal cell.

    Parameters
    ----------
    reward_net : RewardNet
    layout : dict

    Returns
    -------
    score_grid : np.ndarray, shape (n_rows, n_cols)
    predicted_json_coords : (col, row_from_top)
    """
    grid = layout["grid_layout"]
    n_rows = len(grid)
    n_cols = len(grid[0]) if n_rows > 0 else 0
    goal_j = (int(layout["goal_pos"][0]), int(layout["goal_pos"][1]))
    start_j = (int(layout["agent_start_pos"][0]), int(layout["agent_start_pos"][1]))

    score_grid = np.full((n_rows, n_cols), np.nan)

    reward_net.eval()
    with torch.no_grad():
        for r in range(n_rows):
            for c in range(n_cols):
                if grid[r][c] == "#":
                    continue
                if (c, r) in (goal_j, start_j):
                    continue
                x_norm = c / max(1, n_cols - 1)
                y_mdp = n_rows - 1 - r
                y_norm = y_mdp / max(1, n_rows - 1)
                state = torch.tensor([[x_norm, y_norm, 0.0]], dtype=torch.float32)
                score_grid[r, c] = reward_net(state).item()

    predicted_json = _argmax_json(score_grid, fallback=goal_j)
    return score_grid, predicted_json


def _argmax_json(score_grid, fallback=(0, 0)):
    """Return (col, row_from_top) of the highest non-NaN cell."""
    if np.all(np.isnan(score_grid)):
        return fallback
    flat_idx = np.nanargmax(score_grid)
    r, c = np.unravel_index(flat_idx, score_grid.shape)
    return (int(c), int(r))


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_trajectories(data_dir, grid_id, layout, effort="low"):
    """Load every trajectory file for a grid, regardless of success.

    Unlike load_grid() in run_algorithms.py, which filters to successful
    trajectories only, this function loads all files and returns a success
    flag for each so the caller can use both groups with appropriate weighting.

    Reuses build_path() and is_successful() from run_algorithms to stay
    consistent with the existing parsing logic.

    Parameters
    ----------
    data_dir : str or Path
    grid_id  : int
    layout   : dict  (already loaded; needed for is_successful check)
    effort   : str

    Returns
    -------
    paths         : list of list of (col, row_top)
    success_flags : list of bool   (True = coin collected + goal reached)
    traj_ids      : list of int
    """
    import json as _json
    import re as _re
    from run_algorithms import build_path, is_successful

    data_dir = Path(data_dir)
    traj_files = sorted(data_dir.glob(f"*_grid{grid_id}_coin_{effort}_traj*.json"))

    paths, flags, ids = [], [], []
    for tf in traj_files:
        m = _re.search(r"_traj(\d+)\.json$", tf.name)
        traj_id = int(m.group(1)) if m else -1
        with open(tf) as f:
            data = _json.load(f)
        steps = data.get("steps", [])
        path = build_path(steps)
        if not path:
            continue
        paths.append(path)
        flags.append(is_successful(steps, layout))
        ids.append(traj_id)

    return paths, flags, ids


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _plot_reward_panel(ax, score_grid, layout, predicted_json, title):
    """Draw one reward heatmap panel.

    Mirrors run_algorithms.plot_result so T-REX and MaxEnt maps render
    identically without importing from run_algorithms.
    """
    n_rows = len(layout["grid_layout"])
    n_cols = len(layout["grid_layout"][0])
    grid = layout["grid_layout"]
    coin_j = tuple(layout["coin_pos"])
    goal_j = tuple(layout["goal_pos"])
    start_j = tuple(layout["agent_start_pos"])

    wall_mask = np.array(
        [[grid[r][c] == "#" for c in range(n_cols)] for r in range(n_rows)],
        dtype=bool,
    )
    nan_mask = np.isnan(score_grid)
    masked = np.ma.masked_where(wall_mask | nan_mask, score_grid)

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("#4a4a4a")
    im = ax.imshow(masked, origin="upper", cmap=cmap,
                   interpolation="nearest", aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    for r in range(n_rows + 1):
        ax.axhline(r - 0.5, color="#888", lw=0.3, zorder=1)
    for c in range(n_cols + 1):
        ax.axvline(c - 0.5, color="#888", lw=0.3, zorder=1)

    for (cx, cy), lbl in [(goal_j, "G"), (start_j, "S"), (coin_j, "C")]:
        ax.text(cx, cy, lbl, ha="center", va="center",
                fontsize=7, fontweight="bold", color="black", zorder=5)

    ax.add_patch(mpatches.Circle(
        coin_j, 0.35, color="limegreen", fill=False, lw=1.5, zorder=6
    ))
    ax.plot(predicted_json[0], predicted_json[1], "x",
            color="crimson", ms=8, mew=2, zorder=7)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=6, pad=2)
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)


# ── Comparison ────────────────────────────────────────────────────────────────

def compare_with_maxent(trex_grid, maxent_grid, layout, save_path):
    """Plot T-REX and MaxEnt IRL reward maps side by side and compute metrics.

    Parameters
    ----------
    trex_grid : np.ndarray, shape (n_rows, n_cols)
        From extract_reward_map().
    maxent_grid : np.ndarray, shape (n_rows, n_cols)
        From run_maxent().
    layout : dict
    save_path : str or Path

    Returns
    -------
    dict with keys:
        trex_pred, maxent_pred  : (col, row_from_top)
        trex_dist, maxent_dist  : Manhattan distance to true coin_pos
        spearman_r              : Spearman correlation between the two maps
                                  (NaN if fewer than 3 overlapping valid cells)
    """
    from scipy.stats import spearmanr

    coin_j = (int(layout["coin_pos"][0]), int(layout["coin_pos"][1]))
    goal_j = (int(layout["goal_pos"][0]), int(layout["goal_pos"][1]))

    trex_pred = _argmax_json(trex_grid, fallback=goal_j)
    maxent_pred = _argmax_json(maxent_grid, fallback=goal_j)

    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    trex_dist = _manhattan(trex_pred, coin_j)
    maxent_dist = _manhattan(maxent_pred, coin_j)

    # Spearman correlation on cells valid in both grids
    mask = ~np.isnan(trex_grid) & ~np.isnan(maxent_grid)
    if mask.sum() > 2:
        r, _ = spearmanr(trex_grid[mask], maxent_grid[mask])
        spearman_r = float(r)
    else:
        spearman_r = float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), squeeze=False)
    _plot_reward_panel(axes[0, 0], trex_grid, layout, trex_pred,
                       f"T-REX  d={trex_dist}")
    _plot_reward_panel(axes[0, 1], maxent_grid, layout, maxent_pred,
                       f"MaxEnt IRL  d={maxent_dist}")
    fig.suptitle(
        f"T-REX vs MaxEnt IRL  (coin={coin_j}, Spearman r={spearman_r:.2f})",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    return {
        "trex_pred": trex_pred,
        "maxent_pred": maxent_pred,
        "trex_dist": trex_dist,
        "maxent_dist": maxent_dist,
        "spearman_r": spearman_r,
    }
