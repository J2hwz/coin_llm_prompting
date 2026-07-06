# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
conda env create -f environment.yml
conda activate coinenv
```

After pulling changes:
```bash
conda env update -f environment.yml --prune
```

Create a `.env` file in the project root with:
```
TOGETHERAI_API_KEY="<your key>"
```

The package is installed in editable mode via `environment.yml` (`-e .`), so `coinenv-cli` is available after activating the conda env.

## Commands

```bash
# Lint and format
ruff check src/
ruff format src/

# Run tests (none exist yet)
pytest

# CLI — full help
coinenv-cli --help
coinenv-cli <subcommand> --help
```

Pre-commit hooks run `ruff-format` and `ruff` automatically on commit/push.

## Git workflow

Commit and push when explicitly asked (usually end of session), or after any significant structural change. Firstly, update the readme file with corresponding changes.  Then, write informative commit messages that describe what changed and why — not just what files were touched. Push after committing: `git push origin main`.



## Architecture

This codebase runs LLMs as agents in MiniGrid gridworld environments and collects trajectories (with token-level logprobs) for mechanistic interpretability research.

### Inference pipeline

```
Environment → Text Wrapper → Jinja2 Template → LiteLLM API → Pydantic Parsing → Action → env.step()
```

**1. Environment** (`environment_generator/`)
- `Simple2DNavigationEnv` — base maze with configurable `size` and `complexity`
- `CoinNavigationEnv` — extends base with an optional collectible coin (placed externally via `put_obj()` after reset)
- `RoomsMinigridEnv` / `Key2PathMinigridEnv` — multi-room layouts with key/door mechanics
- Text wrappers (`FullObservabilityTextWrapper`, `FogOfWarTextWrapper`) convert MiniGrid state to a character grid: `A`=agent, `G`=goal, `#`=wall, `_`=empty, `*`=fog

**2. Prompt construction** (`agents/llm_agent.py`, `templates/`)
- `LLMAgent` renders a Jinja2 template with the text grid observation
- Agent variants: `PartiallyObservableLLMAgent` (fog + action history), `PartiallyObservableWithNoteLLMAgent` (can write persistent notes), `PartiallyObservableWithChatHistoryLLMAgent` (full chat history)
- Template selection is explicit via `--template-name`; the default for standard navigation is `grid_full_observability.j2`, for dead-end/coin envs it is `grid_one_coin_control.j2`

**3. API call** (`llm_interface.py`)
- Wraps `litellm` with structured JSON output (Pydantic `ActionResponse`), up to 5 retries with exponential backoff, and cumulative USD cost tracking
- Qwen models prepend `</think>` reasoning tags before the JSON — handled automatically

**4. Step loop** (`trajectory_generator/trajectory_generator.py`)
- Runs reset → observe → prompt → call → parse → step until `terminated` or `max_steps`
- Records each `Step(obs, action, reward, metadata, agent_pos)` into a `Trajectory`
- `AlphaStarAgent` (`agents/alpha_start_agent.py`) provides an A* optimal baseline without any LLM calls

**5. CLI** (`commands/cli.py`, `commands/get_trajectory/get_trajectory_fn.py`)
- `tyro` dispatches subcommands; all subcommand functions live in `get_trajectory_fn.py`
- Key subcommands: `get_trajectory`, `get_trajectories`, `get_trajectories_multiple_per_grid`, `get_trajectory_deadend_env`, `get_trajectories_coin_env`, `get_trajectory_key_door_env`, `upload_trajectories_dir`
- Batch commands use `ThreadPoolExecutor` with configurable `--max-workers`; optional token-bucket rate limiting via `--enable-rate-limit`

**6. Output format**
- Each trajectory saved as JSON with `grid_params`, `model_params`, and `steps[]`
- Each step includes `grid_state`, `agent_action`, `output_tokens`, and token-level `probabilities` (logprobs) for downstream analysis
