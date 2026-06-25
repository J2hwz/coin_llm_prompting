# Plan: Skip Invalid Actions Toggle (trajectory-loading time)

## Context
LLM trajectories contain wall-collision steps where the agent position is unchanged and reward=0. The user wants an optional toggle to discard these steps when loading trajectories into the `/algorithms/` pipeline — not during data collection.

**Key finding**: Most algorithms already implicitly discard wall-bumps during their per-algorithm conversion (by detecting zero-movement). But there is no unified, explicit toggle, and `bnirl` doesn't filter at all. The cleanest injection point is `build_path()` — the single shared function all algorithms use to convert JSON steps into position sequences.

---

## Changes

### File 1: `algorithms/run_algorithms.py`

#### 1a. `build_path()` — lines 100–112

Add `skip_invalid_actions: bool = False`. When enabled, skip any step whose extracted position equals the previous one (= the agent hit a wall and didn't move).

**Before:**
```python
def build_path(steps):
    """Reconstruct (col, row_from_top) path from JSON steps."""
    path = []
    for step in steps:
        pos = _find_symbol(step.get("grid_state", []), "A")
        if pos is None:
            break
        path.append(pos)
    if steps and path:
        last_action = steps[-1].get("agent_action", "").upper()
        dx, dy = _ACTION_DELTA.get(last_action, (0, 0))
        path.append((path[-1][0] + dx, path[-1][1] + dy))
    return path
```

**After:**
```python
def build_path(steps, skip_invalid_actions: bool = False):
    """Reconstruct (col, row_from_top) path from JSON steps."""
    path = []
    for step in steps:
        pos = _find_symbol(step.get("grid_state", []), "A")
        if pos is None:
            break
        if skip_invalid_actions and path and pos == path[-1]:
            continue  # wall bump — agent didn't move
        path.append(pos)
    if steps and path:
        last_action = steps[-1].get("agent_action", "").upper()
        dx, dy = _ACTION_DELTA.get(last_action, (0, 0))
        path.append((path[-1][0] + dx, path[-1][1] + dy))
    return path
```

> Note: `is_successful()` (line 115) also calls `build_path()` but should NOT receive `skip_invalid_actions` — success detection should work on the raw path so we don't accidentally exclude a valid last step.

#### 1b. `load_grid()` — lines 123–144

Add `skip_invalid_actions: bool = False` to the signature and pass it to `build_path()` at line 141.

**Signature change (line 123):**
```python
def load_grid(data_dir, grid_id, effort="low", skip_invalid_actions: bool = False):
```

**Call site change (line 141):**
```python
# Before:
successful_paths.append(build_path(steps))
# After:
successful_paths.append(build_path(steps, skip_invalid_actions=skip_invalid_actions))
```

#### 1c. `run_grid()` — line 205

Add `skip_invalid_actions: bool = False` and pass to `load_grid()` at line 207.

**Signature change (line 205):**
```python
def run_grid(data_dir, grid_id, plots_dir, mode="both", effort="low", skip_invalid_actions: bool = False):
```

**Call site change (line 207):**
```python
# Before:
layout, paths, traj_ids = load_grid(data_dir, grid_id, effort=effort)
# After:
layout, paths, traj_ids = load_grid(data_dir, grid_id, effort=effort, skip_invalid_actions=skip_invalid_actions)
```

#### 1d. `main()` — line 405

Add `skip_invalid_actions: bool = False` and pass to `run_grid()` at line 429.

**Signature change (line 405):**
```python
def main(data_dir, mode="both", effort="low", skip_invalid_actions: bool = False):
```

**Call site change (line 429):**
```python
# Before:
result = run_grid(data_dir, gid, plots_dir, mode=mode, effort=effort)
# After:
result = run_grid(data_dir, gid, plots_dir, mode=mode, effort=effort, skip_invalid_actions=skip_invalid_actions)
```

---

### File 2: `algorithms/trex.py`

#### 2a. `load_all_trajectories()` — lines 538–582

Add `skip_invalid_actions: bool = False` to the signature and pass it to `build_path()` at line 575.

**Signature change (line 538):**
```python
def load_all_trajectories(data_dir, grid_id, layout, effort="low", skip_invalid_actions: bool = False):
```

**Call site change (line 575):**
```python
# Before:
path = build_path(steps)
# After:
path = build_path(steps, skip_invalid_actions=skip_invalid_actions)
```

#### 2b. `run_grid_trex()` — lines 68–70

Add `skip_invalid_actions: bool = False` and pass to both `load_grid()` (line 88) and `load_all_trajectories()` (line 96).

**Signature change (line 68):**
```python
def run_grid_trex(data_dir, grid_id, plots_dir, effort="low",
                  clip_len=10, n_pairs=5000, lr=1e-3, hidden_dim=64,
                  save_model=False, skip_invalid_actions: bool = False):
```

**Call site changes:**
```python
# Line 88 — before:
layout, successful_paths, _ = load_grid(data_dir, grid_id, effort=effort)
# After:
layout, successful_paths, _ = load_grid(data_dir, grid_id, effort=effort, skip_invalid_actions=skip_invalid_actions)

# Lines 96–98 — before:
all_paths, success_flags, all_traj_ids = load_all_trajectories(
    data_dir, grid_id, layout, effort=effort
)
# After:
all_paths, success_flags, all_traj_ids = load_all_trajectories(
    data_dir, grid_id, layout, effort=effort, skip_invalid_actions=skip_invalid_actions
)
```

#### 2c. `main()` — line 153

Add `skip_invalid_actions: bool = False` and pass to `run_grid_trex()` at line 178.

**Signature change (line 153):**
```python
def main(data_dir, effort="low", clip_len=10, n_pairs=5000, lr=1e-3,
         hidden_dim=64, save_model=False, skip_invalid_actions: bool = False):
```

**Call site change (line 178):**
```python
# Before:
res = run_grid_trex(
    data_dir, gid, plots_dir,
    effort=effort, clip_len=clip_len, n_pairs=n_pairs,
    lr=lr, hidden_dim=hidden_dim, save_model=save_model,
)
# After:
res = run_grid_trex(
    data_dir, gid, plots_dir,
    effort=effort, clip_len=clip_len, n_pairs=n_pairs,
    lr=lr, hidden_dim=hidden_dim, save_model=save_model,
    skip_invalid_actions=skip_invalid_actions,
)
```

---

## Summary of All Touch Points

| File | Function | Line | Change |
|---|---|---|---|
| `run_algorithms.py` | `build_path()` | 100 | Add param + skip-duplicate-position logic |
| `run_algorithms.py` | `load_grid()` | 123, 141 | Add param + pass to `build_path()` |
| `run_algorithms.py` | `run_grid()` | 205, 207 | Add param + pass to `load_grid()` |
| `run_algorithms.py` | `main()` | 405, 429 | Add param + pass to `run_grid()` |
| `trex.py` | `load_all_trajectories()` | 538, 575 | Add param + pass to `build_path()` |
| `trex.py` | `run_grid_trex()` | 68, 88, 96 | Add param + pass to both load functions |
| `trex.py` | `main()` | 153, 178 | Add param + pass to `run_grid_trex()` |

No changes to `surprise_v2.py`, `inv_planning.py`, `bnirl.py`, `birl_wrapper.py`, or `maxent_irl.py`.

---

## Verification

```bash
# Run on a known trajectory directory, with and without the flag,
# and compare path lengths — paths should be shorter with flag enabled
# when the LLM agent hit walls during collection.

cd algorithms
python run_algorithms.py <data_dir> --skip-invalid-actions
python run_trex.py <data_dir> --skip-invalid-actions
```
