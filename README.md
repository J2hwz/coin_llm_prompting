# coin-llm-experiments

Experiments running LLMs as agents in gridworld environments, collecting trajectories with token-level logprobs for mechanistic interpretability analysis.

Adapted from [SPAR-Telos/reveng](https://github.com/SPAR-Telos/reveng).

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

---

## Repository structure

```
├── algorithms/                      # Subgoal inference algorithms (IRL-based)
│   ├── run_algorithms.py            # Entry point — orchestrates the 5 default algorithms, plots, CSV
│   ├── visit_frequency.py           # Baseline: pooled cell visit frequency
│   ├── trajectory_visit_frequency.py # Baseline: per-trajectory (deduped) visit frequency
│   ├── surprise_v2.py               # Surprise model (terminal-directed)
│   ├── inv_planning.py              # Bayesian inverse planning
│   ├── maxent_irl.py                # Maximum Causal Entropy IRL
│   └── archive/                     # Not in the default suite, kept for reference
│       ├── run_trex.py              #   T-REX reward learning entry point
│       ├── bnirl.py                 #   BNIRL Gibbs sampler
│       ├── birl_wrapper.py          #   BIRL PolicyWalk MCMC
│       └── trex.py                  #   T-REX neural reward learning (requires torch)
├── analysis/                        # Offline analysis scripts
│   ├── metrics.py                   # Pure math: entropy, JSD, KL, calibration, stats
│   ├── visualization.py             # Shared matplotlib helpers (paper-quality figures)
│   ├── grid_env_utils.py            # MiniGrid-specific: optimal actions, cell processing
│   ├── analysis_utils.py            # Trajectory data classes, Dijkstra, logprob parsing
│   ├── full_obs_trajectory_analysis.py  # Metrics + plots for standard navigation
│   ├── coin_trajectory_analysis.py  # Phase-aware metrics + plots for coin envs
│   ├── coin_trajectory_analysis_metrics.md  # Full metrics reference for coin analysis
│   ├── plot_trajectories.py         # Standalone trajectory plotting utilities (supports --effort flag)
│   ├── plot_trajectories_two_coins.py  # Trajectory plotting for two-coin environments
│   └── traj_checking.Rmd            # R Markdown for trajectory inspection
├── data/                            # Output trajectories (gitignored)
├── src/
│   └── coinenv/
│       ├── agents/                  # Agent implementations (LLM, A*, random)
│       ├── commands/                # CLI entry point and subcommands
│       │   └── get_trajectory/      # Trajectory generation logic and utilities
│       ├── environment_generator/   # MiniGrid environment construction
│       │   ├── env_transformations.py   # ISO-difficulty grid transforms
│       │   └── wrappers/            # Text and RGB observation wrappers
│       ├── templates/               # Jinja2 prompt templates
│       ├── trajectory_generator/    # Step loop
│       ├── datatypes.py             # Step, Trajectory, Action data structures
│       └── llm_interface.py         # LiteLLM API wrapper
├── scripts/
│   └── finetune_coin_sft.py         # Fine-tune a model on SFT data via Together AI
├── inspection/                      # Manual debug/inspection tools (not pytest tests)
│   ├── print_prompts.py             # Replay the full LLM prompt for each step of a trajectory
│   ├── inspect_history_prompt.py    # Sanity-check rendered template output with seeded history
│   └── parse_history_trajectory.ipynb  # Notebook: parse a --track-history trajectory JSON
├── environment.yml                  # Conda environment and dependencies
├── pyproject.toml                   # Package metadata and CLI entry point
└── .env                             # API keys (not committed)
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

### 1. Maze generation

`get_trajectory_fn.py` creates a `Simple2DNavigationEnv` with a given `size` and `complexity`, then wraps it with either `FullObservabilityTextWrapper` or `FogOfWarTextWrapper`. The wrapper converts the MiniGrid state into a text grid string with row/column indices and symbols.

| Symbol | Meaning |
|---|---|
| `A` | Agent |
| `G` | Goal |
| `#` | Wall |
| `_` | Empty space |
| `*` | Fog (partial observability only) |

The maze is generated fresh at the start of each trajectory and not stored before inference. Completed trajectories are optionally saved to JSON afterwards.

### 2. Prompt construction

`LLMAgent` takes the text grid observation and renders a Jinja2 template with it, producing a complete prompt. Available templates are in `src/coinenv/templates/`:

| Template | Used for |
|---|---|
| `grid_full_observability.j2` | Full observability, standard navigation |
| `grid_one_coin_control.j2` | Full observability with coin — agent must collect coin then reach goal |
| `grid_two_coins_collect_one.j2` | Two coins present; agent instructed to collect exactly one then reach goal |
| `grid_two_coins_collect_all.j2` | Two coins present; agent instructed to collect both then reach goal |
| `grid_one_coin_avoid.j2` | Coin present; agent instructed to avoid it and reach goal directly |
| `grid_partial_observability.j2` | Partial observability with action history |
| `grid_partial_observability_with_note.j2` | Partial obs + agent note-taking |

Agent variants:

| Class | Description |
|---|---|
| `LLMAgent` | Standard fully observable agent |
| `PartiallyObservableLLMAgent` | Uses fog of war, maintains action history |
| `PartiallyObservableWithNoteLLMAgent` | Can write notes carried across steps |
| `PartiallyObservableWithChatHistoryLLMAgent` | Maintains full chat history with the model |
| `CoinAStarAgent` | Oracle A* agent that visits the coin before the goal; used for SFT data generation |

### 3. API call

`BaseLLMInterface` in `llm_interface.py` sends the prompt via `litellm`:

- Structured JSON output enforced via a Pydantic `ActionResponse` model
- Retry logic: up to 5 attempts with exponential backoff (5–120 second wait)
- Cost tracking: per-call and cumulative USD cost via `litellm.completion_cost()`

### 4. Response parsing

The model returns JSON such as `{"action": "RIGHT"}`, validated through `ActionResponse` and converted to an `Action` enum:

```python
class Action(Enum):
    LEFT  = 0
    RIGHT = 1
    UP    = 2
    DOWN  = 3
```

Qwen models wrap responses in `</think>` reasoning tags before the JSON — this is handled automatically.

### 5. Step loop

`trajectory_generator.py` runs the loop:

```
reset environment
for each step:
    text_obs  ← env wrapped as text grid
    prompt    ← render Jinja2 template with text_obs
    response  ← call LLM API (with retries)
    action    ← parse JSON → Action enum → int (0–3)
    env.step(action) → next_obs, reward, terminated, truncated, info
    record Step(obs, action, reward, metadata, agent_pos)
until terminated or max_steps reached
→ create Trajectory(steps, final_reward, metadata)
→ optionally save to JSON
```

An A\* agent (`AlphaStarAgent`) is available as an optimal baseline — it computes the shortest path and returns optimal actions without any LLM calls.

### 6. Saved output

Each trajectory is saved as a JSON file:

```json
{
  "grid_params": {
    "grid_width": 11,
    "grid_height": 11,
    "grid_complexity": 0.6,
    "fully_observable": true,
    "astar_distance": 14,
    "agent_start_coordinates": [1, 3],
    "goal_coordinates": [9, 7]
  },
  "model_params": {
    "model_id": "gpt-4",
    "temperature": 0.0,
    "top_logprobs": 20
  },
  "steps": [
    {
      "step_id": 0,
      "grid_state": "  0 1 2 ...",
      "agent_action": "RIGHT",
      "output_text": "{\"action\": \"RIGHT\"}",
      "output_tokens": [...],
      "probabilities": {...}
    }
  ]
}
```

Token-level logprobs are stored per step for downstream mechanistic interpretability analysis.

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
```

Both scripts accept `--multi-model` to process multiple model subdirectories in one pass.

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

### Outputs

Each run produces:
- **Per-grid CSV** — one row per (grid × effort) combination with all metrics including entropy/JSD/ECE
- **Per-trajectory CSV** (`coin_per_trajectory.csv`) — one row per individual trajectory with all independently computable metrics (excludes entropy/JSD/ECE)
- Size×density summary CSV, distance summary CSV, overall summary JSON
- Figures (PNG + PDF) under `analysis/outputs/<run_name>/`

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
| `analysis/plot_condition_trajectories.py` | Renders one success + one failure trajectory per condition on matched grids |
| `analysis/plot_grid_layouts.py` | Renders panel of one-coin grid layouts across sizes/densities, no trajectories |
| `analysis/plot_grid_text_representation.py` | Side-by-side rendered maze and the LLM's text-grid prompt representation |
| `algorithms/compare_conditions.py` | Cross-condition comparison of subgoal-inference algorithms using best-available (CV-tuned/default) results |
| `algorithms/plot_grid_comparison.py` | Heatmap + per-algorithm prediction grids for one maze across conditions |
