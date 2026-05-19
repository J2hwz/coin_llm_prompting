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

---

## Repository structure

```
├── data/                            # Output trajectories (gitignored)
├── src/
│   └── coinenv/
│       ├── agents/                  # Agent implementations (LLM, A*, random)
│       ├── commands/                # CLI entry point and subcommands
│       │   └── get_trajectory/      # Trajectory generation logic and utilities
│       ├── environment_generator/   # MiniGrid environment construction
│       │   └── wrappers/            # Text and RGB observation wrappers
│       ├── templates/               # Jinja2 prompt templates
│       ├── trajectory_generator/    # Step loop
│       ├── datatypes.py             # Step, Trajectory, Action data structures
│       └── llm_interface.py         # LiteLLM API wrapper
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
| `get_trajectory_deadend_env` | Single trajectory with dead-end constraints and optional coin |
| `get_trajectories_coin_env` | Batch coin environment trajectories |
| `get_trajectory_key_door_env` | Single trajectory in a rooms/key/door environment |
| `get_trajectories_key_door_env` | Batch rooms/key/door trajectories |
| `upload_trajectories_dir` | Push saved trajectories to HuggingFace Hub |

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
| `grid_partial_observability.j2` | Partial observability with action history |
| `grid_partial_observability_with_note.j2` | Partial obs + agent note-taking |
| `grid_full_observability_coin.j2` | Coin collection variant |
| `grid_full_observability_instrumental_goals.j2` | Multi-goal variant |

Agent variants:

| Class | Description |
|---|---|
| `LLMAgent` | Standard fully observable agent |
| `PartiallyObservableLLMAgent` | Uses fog of war, maintains action history |
| `PartiallyObservableWithNoteLLMAgent` | Can write notes carried across steps |
| `PartiallyObservableWithChatHistoryLLMAgent` | Maintains full chat history with the model |

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
| `src/coinenv/llm_interface.py` | LiteLLM API wrapper with retries and cost tracking |
| `src/coinenv/templates/` | Jinja2 prompt templates |
| `src/coinenv/trajectory_generator/trajectory_generator.py` | Step-by-step environment loop |
| `src/coinenv/datatypes.py` | `Step`, `Trajectory`, `Action` data structures |
| `src/coinenv/environment_generator/` | MiniGrid environment construction and wrappers |
