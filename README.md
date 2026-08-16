# coin-llm-experiments

Experiments running LLMs as agents in gridworld environments, collecting trajectories with token-level logprobs for mechanistic interpretability analysis.

---

## Setup

### Requirements

- [Conda](https://docs.conda.io/en/latest/) (Miniconda or Anaconda)
- Python 3.12+
- A TogetherAI API key

All Python dependencies are declared in `environment.yml`.

### Create the environment

```bash
conda env create -f environment.yml
conda activate coinenv
```

To update after pulling new changes:

```bash
conda env update -f environment.yml --prune
```

### API keys

Create a `.env` file in the project root:

```
TOGETHERAI_API_KEY="<your key here>"
```

Do not commit this file — it is already in `.gitignore`.

### Testing

`algorithms/` has a pytest suite (`algorithms/tests/`, wired up via `testpaths` in `pyproject.toml`):

```bash
pytest
```

---

## Repository structure

```
├── algorithms/               # Subgoal inference algorithms (IRL-based)
│   ├── archive/               # BNIRL, BIRL, T-REX — excluded from the default suite
│   ├── results/               # Cross-condition comparison outputs (CSV, figures, RESULTS.md)
│   └── tests/                 # pytest suite
├── analysis/                  # Offline analysis scripts
│   ├── archive/
│   └── outputs/                # Per-condition R Markdown reports + rendered figures
├── data/                      # Output trajectories (gitignored)
├── docs/                      # Planning notes (fine-tuning, skip-invalid-actions)
├── inspection/                # Manual debug/inspection tools (not pytest tests)
├── scripts/                   # Fine-tuning entry point
├── src/
│   └── coinenv/                # Package source — agents, commands, environment_generator, templates, trajectory_generator
├── tuning_results/            # CV-tuned hyperparameter outputs (per condition)
│   ├── low/
│   └── medium/
├── environment.yml            # Conda environment and dependencies
├── pyproject.toml             # Package metadata and CLI entry point
└── .env                       # API keys (not committed)
```

---

## CLI

The package exposes a `coinenv-cli` command after installation. The entry point is `src/coinenv/commands/cli.py`, which uses `tyro` to dispatch subcommands:

| Subcommand | Description |
|---|---|
| `get_trajectory` | Generate a single trajectory |
| `get_trajectories` | Batch generation across parameter combinations |
| `get_trajectories_multiple_per_grid` | Multiple trajectories on the same grid |
| `get_single_trajectory_coin_env` | Single trajectory with dead-end constraints and optional coin |
| `get_multiple_trajectories_coin_env` | Batch coin environment trajectories |
| `get_single_trajectory_two_coin_env` | Single trajectory with two coins; template controls the agent's objective |
| `get_multiple_trajectories_two_coin_env` | Batch two-coin environment trajectories |
| `augment_from_layouts` | Generate trajectories on ISO-difficulty augmented variants of saved layouts |
| `reshuffle_walls_from_layouts` | Regenerate wall layouts (same complexity, fixed anchors) and collect trajectories |
| `generate_random_trajectories_from_layouts` | Replay existing single-/two-coin layouts with a random-action baseline agent (no LLM calls); writes into the same layout folder |
| `generate_random_trajectories_from_new_grids` | Procedurally generate fresh single-coin grids across a size x complexity sweep and run a random-action baseline agent on each (no LLM calls); persists generated layouts alongside trajectories |
| `generate_random_trajectories_from_new_two_coin_grids` | Procedurally generate fresh two-coin grids across a size x complexity sweep and run a random-action baseline agent on each (no LLM calls); persists generated layouts alongside trajectories |
| `upload_trajectories_dir` | Push saved trajectories to HuggingFace Hub |
| `generate_sft_dataset` | Generate a JSONL Supervised Fine-Tuning dataset from optimal coin-then-goal trajectories |

All output paths are relative to the `data/` folder at the project root — specify just a filename or subfolder name:

```bash
coinenv-cli get_trajectory_deadend_env --coin-placement random --output-path my_run.json
# saves to data/my_run.json
```

Run any subcommand with `--help` to see its arguments:

```bash
coinenv-cli get_trajectory --help
```

---

## Inference pipeline

```
Environment → Text Wrapper → Jinja2 Template → LiteLLM API → Pydantic Parsing → Action → env.step()
```

1. **Maze generation** — `Simple2DNavigationEnv` is wrapped as a text grid (`A`/`G`/`#`/`_`/`*`) via `FullObservabilityTextWrapper` or `FogOfWarTextWrapper`.
2. **Prompt construction** — `LLMAgent` (or a variant: partial-observability, note-taking, chat-history, `CoinAStarAgent`) renders a Jinja2 template from `src/coinenv/templates/` with the text grid.
3. **API call** — `BaseLLMInterface` (`llm_interface.py`) sends the prompt via `litellm`: structured JSON output, up to 5 retries with exponential backoff, per-call and cumulative USD cost tracking.
4. **Response parsing** — the response is validated through a Pydantic `ActionResponse` model and converted to an `Action` enum; Qwen models' `</think>` reasoning tags are stripped automatically.
5. **Step loop** — `trajectory_generator.py` loops reset → observe → prompt → call → parse → `env.step()` until `terminated`/`truncated` or `max_steps`. `AlphaStarAgent` provides an optimal A* baseline with no LLM calls.
6. **Saved output** — each trajectory is saved as JSON with `grid_params`, `model_params`, and `steps[]`, including per-step token-level logprobs for downstream mechanistic interpretability analysis.

Templates cover full/partial observability, one- and two-coin environments, and `avoid`/`deceptive` objective conditions (the deceptive template shows the agent's remaining step budget each turn) — see `src/coinenv/templates/` for the full set.

---

## Subgoal inference algorithms

The `algorithms/` package infers which intermediate cell an agent was heading toward (before the terminal goal), given observed navigation trajectories and a grid layout. Five methods run by default: two visit-frequency baselines (pooled cell-level and per-trajectory), surprise model, Bayesian inverse planning, and maximum causal entropy IRL. BNIRL, BIRL, and T-REX (neural reward learning, requires PyTorch) remain available under `algorithms/archive/` but are excluded from the default suite.

```bash
python algorithms/run_algorithms.py <data_dir> [--mode {individual,pooled,both}]
```

Pass `--skip-invalid-actions` to either entry point to discard wall-collision steps (steps where the agent's position did not change) before they are passed to the algorithms:

```bash
python algorithms/run_algorithms.py <data_dir> --skip-invalid-actions
python algorithms/archive/run_trex.py <data_dir> --skip-invalid-actions
```

See [`algorithms/algorithms.md`](algorithms/algorithms.md) for full documentation, input format, and output schema. The algorithms are compatible with output from `get_multiple_trajectories_coin_env`.

**Hyperparameter tuning and cross-condition comparison:**

```bash
# Cross-validated grid search for inv_planning's beta / surprise_v2's (lambda, gamma, alpha)
python algorithms/tune_params.py <data_dir> [<data_dir> ...] [--model {inv_planning,surprise_v2,both}]
# → tuning_results/<effort>/

# Combine best-available per-algorithm predictions (CV-tuned Surprise v2 + literature-default
# for the rest; Inv. Planning excluded — see module docstring) across conditions
python algorithms/compare_conditions.py [--output-dir algorithms/results]

# Single-grid heatmap + per-algorithm prediction panel across conditions
python algorithms/plot_grid_comparison.py [--grid-size 9] [--density 0.2]
```

---

## Fine-tuning

`generate_sft_dataset` generates a JSONL dataset of optimal coin-then-goal trajectories using `CoinAStarAgent`. Each line is one step formatted as a single-turn chat record matching the prompt format used by `LLMAgent` at inference time, so train and inference distributions are identical:

```json
{"messages": [
  {"role": "user",      "content": "<full rendered template + grid>"},
  {"role": "assistant", "content": "{\"action\": \"RIGHT\"}"}
]}
```

By default it generates 100 episodes per (grid size × complexity) combination across 3 sizes and 4 complexities (1,200 episodes total):

```bash
coinenv-cli generate_sft_dataset \
  --num-episodes-per-combination 100 \
  --grid-sizes 7 9 11 \
  --grid-complexities 0.0 0.2 0.4 0.6 \
  --output-path sft/oracle_combined.jsonl
```

Fine-tune on Together AI using `scripts/finetune_coin_sft.py`:

```bash
python scripts/finetune_coin_sft.py \
  --training-file data/sft/oracle_combined.jsonl \
  --base-model openai/gpt-oss-20b \
  --suffix coinenv-v1
```

The script uploads the JSONL, waits for processing, runs LoRA fine-tuning, polls until completion, and prints the litellm-ready model name. Pass `--dry-run` to validate the upload without starting training.

---

## ISO-difficulty environment transforms

`env_transformations.py` implements structure-preserving grid transformations that change the spatial layout without altering task difficulty (same optimal path length):

| Transform | Description |
|---|---|
| `RotateEnv` | 90° counter-clockwise rotation |
| `ReflectEnv` | Vertical (top/bottom) reflection |
| `TransposeEnv` | Transpose (swap x and y axes) |
| `StartGoalSwap` | Swap agent start and goal positions |

These are used by `augment_from_layouts` to test whether model behaviour generalises across equivalent problem instances.

---

## Analysis

The `analysis/` package computes metrics from saved trajectory JSONs and generates plots. Run from the project root with:

```bash
# Standard navigation trajectories
python -m analysis.full_obs_trajectory_analysis \
  --trajectory-dir data/<run_dir> \
  --output-dir analysis/outputs/<run_name>

# Coin navigation trajectories (phase-aware)
python -m analysis.coin_trajectory_analysis \
  --trajectory-dir data/<run_dir> \
  --output-dir analysis/outputs/<run_name>

# Two-coin trajectories (three-phase optimal policy; collect_one/collect_all
# success criterion inferred from the data directory structure)
python -m analysis.two_coin_trajectory_analysis \
  --trajectory-dir data/two_coins/<run_dir> \
  --output-dir analysis/outputs/<run_name>
```

All three scripts accept `--multi-model` to process multiple model subdirectories in one pass. Formal metric definitions are in [`analysis/coin_trajectory_analysis_metrics.md`](analysis/coin_trajectory_analysis_metrics.md) and [`analysis/coin_metrics_formulas.md`](analysis/coin_metrics_formulas.md).

### Metrics computed

**Success & capability**

| Metric | Description |
|---|---|
| `full_success_rate` | Fraction of trajectories completing all objectives |
| `coin_collected_rate` | Fraction that collected the coin (coin env only) |
| `mean_action_accuracy` | Fraction of steps matching an optimal action |
| `mean_action_accuracy_phase1/2` | Per-phase accuracy toward coin / goal |
| `spl` | Success weighted by (inverse) path length |

**Uncertainty** (computed from empirical action distributions pooled across trajectories — grid-level only)

| Metric | Description |
|---|---|
| `mean_entropy` | Empirical action entropy at each visited state |
| `mean_jsd` | Jensen-Shannon divergence from the optimal policy |
| `ece` | Expected calibration error |
| `mean_optimal_entropy_phase1/2` | Phase-aware pooled uncertainty (coin- vs. goal-directed) |

**Layout & navigation geometry** (coin env, per trajectory)

| Metric | Description |
|---|---|
| `astar_coin_distance` | Optimal distance: start → coin |
| `astar_goal_distance` | Optimal distance: coin → goal |
| `start_to_goal_distance` | Direct optimal distance: start → goal (ignoring coin) |
| `coin_detour_distance` | Min distance from coin to any cell on the optimal start→goal path |

**Behavioural counts** (per trajectory, raw counts)

| Metric | Description |
|---|---|
| `num_actions_up/down/left/right` | Absolute action frequencies |
| `num_steps_front/left_turn/right_turn/back` | Relative direction counts (orientation from previous step) |
| `num_backtracks` | Steps revisiting any previously visited cell |
| `num_backtracks_at_coin` | Steps revisiting the coin cell specifically |

**Spatial preference** (per trajectory)

| Metric | Description |
|---|---|
| `preferred_quadrant` / `preferred_corner` | Most-occupied grid quadrant / corner |
| `quadrant_entropy` | Entropy of occupancy across quadrants |
| `longest_corner_dwell_run` | Longest consecutive run of steps spent in a single corner |

### Outputs

Each run produces:
- **Per-grid CSV** — one row per (grid × effort) combination with all metrics including entropy/JSD/ECE
- **Per-trajectory CSV** (`coin_per_trajectory.csv`) — one row per individual trajectory with all independently computable metrics (excludes entropy/JSD/ECE)
- Size×density summary CSV, distance summary CSV, overall summary JSON
- Figures (PNG + PDF) under `analysis/outputs/<run_name>/`

`analysis/outputs/` also holds per-condition R Markdown reports (one `.Rmd` per experimental condition, plus `overall.Rmd`) that read these CSVs and render the comparison figures used for write-ups.

---

## Key files

| File | Role |
|---|---|
| `src/coinenv/commands/cli.py` | CLI entry point |
| `src/coinenv/commands/get_trajectory/get_trajectory_fn.py` | Orchestrates the full pipeline |
| `src/coinenv/commands/get_trajectory/get_trajectory_utils.py` | Shared helpers: trajectory generation, token processing, HF upload |
| `src/coinenv/commands/get_trajectory/rate_limiter.py` | Token-bucket rate limiter for API throttling |
| `src/coinenv/commands/get_trajectory/compact_json_encoder.py` | JSON encoder that keeps small containers on one line |
| `src/coinenv/agents/llm_agent.py` | Builds prompts, calls API, parses actions |
| `src/coinenv/agents/alpha_start_agent.py` | A* optimal baseline agent |
| `src/coinenv/agents/coin_astar_agent.py` | A* oracle that visits coin before goal (used for SFT data) |
| `scripts/finetune_coin_sft.py` | Upload JSONL and run LoRA fine-tuning on Together AI |
| `src/coinenv/llm_interface.py` | LiteLLM API wrapper with retries and cost tracking |
| `src/coinenv/templates/` | Jinja2 prompt templates |
| `src/coinenv/trajectory_generator/trajectory_generator.py` | Step-by-step environment loop |
| `src/coinenv/datatypes.py` | `Step`, `Trajectory`, `Action` data structures |
| `src/coinenv/environment_generator/custom_minigrid.py` | `Simple2DNavigationEnv`, `CoinNavigationEnv`, and `TwoCoinNavigationEnv` |
| `src/coinenv/environment_generator/env_transformations.py` | ISO-difficulty grid transforms |
| `src/coinenv/environment_generator/utils.py` | Grid utilities (dead-end detection, A* distance, coin position) |
| `analysis/metrics.py` | Pure math utilities: entropy, JSD, KL, calibration, stats |
| `analysis/visualization.py` | Shared matplotlib helpers used by both analysis pipelines |
| `analysis/grid_env_utils.py` | MiniGrid-specific: env-based Dijkstra, cell processing, metadata loading |
| `analysis/analysis_utils.py` | Trajectory data classes, text-grid Dijkstra, logprob parsing |
| `analysis/full_obs_trajectory_analysis.py` | Metrics and plots for standard navigation trajectories |
| `analysis/coin_trajectory_analysis.py` | Phase-aware metrics and plots for coin navigation trajectories |
| `analysis/two_coin_trajectory_analysis.py` | Three-phase metrics and plots for two-coin navigation trajectories |
| `analysis/coin_metrics_formulas.md` | Formal equation reference for coin-trajectory metrics |
| `analysis/plot_condition_trajectories.py` | Renders one success + one failure trajectory per condition on matched grids |
| `analysis/plot_grid_layouts.py` | Renders panel of one-coin grid layouts across sizes/densities, no trajectories |
| `analysis/plot_grid_text_representation.py` | Side-by-side rendered maze and the LLM's text-grid prompt representation |
| `algorithms/tune_params.py` | Cross-validated hyperparameter tuning for `inv_planning`/`surprise_v2` |
| `algorithms/compare_conditions.py` | Cross-condition comparison of subgoal-inference algorithms using best-available (CV-tuned/default) results |
| `algorithms/plot_grid_comparison.py` | Heatmap + per-algorithm prediction grids for one maze across conditions |
| `algorithms/algorithms_util.py`, `grid_utils.py`, `gridworld.py`, `mdp_funcs.py` | Shared MDP/grid helpers used across the algorithms package |

---

## Acknowledgements

Portions of this codebase were generated with [Claude Code](https://claude.com/claude-code); all generated code was manually inspected for correctness before use.

The project was adapted from [SPAR-Telos/reveng](https://github.com/SPAR-Telos/reveng). `algorithms/maxent_irl.py`'s core Maximum Causal Entropy IRL implementation is adapted from [qzed/irl-maxent](https://github.com/qzed/irl-maxent).
