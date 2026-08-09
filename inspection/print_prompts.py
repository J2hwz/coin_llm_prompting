#!/usr/bin/env python3
"""
Print the full LLM prompt for each step of a trajectory, one at a time.

Usage:
    python inspection/print_prompts.py <trajectory.json> [interval_seconds]

    interval_seconds  seconds to wait between steps (default: 2.0)
                      use 0 to require Enter keypress instead

Example:
    python inspection/print_prompts.py \\
        data/two_coin_tests/collect_one_low_with_history/two_coin_collect_one_batch_test_1/together_ai_openai_gpt-oss-20b_size9_comp0.0_grid0_twocoin_low_traj0.json \\
        3.0
"""

import json
import sys
import time
from pathlib import Path


def find_symbol(grid_lines: list[str], symbol: str) -> tuple[int, int] | None:
    for row in grid_lines:
        parts = row.split()
        if len(parts) < 2:
            continue
        try:
            row_idx = int(parts[0])
        except ValueError:
            continue
        for col_idx, cell in enumerate(parts[1:]):
            if cell == symbol:
                return (col_idx, row_idx)
    return None


def count_coins(grid_lines: list[str]) -> int:
    total = 0
    for row in grid_lines:
        parts = row.split()
        if len(parts) < 2:
            continue
        try:
            int(parts[0])
        except ValueError:
            continue
        total += sum(1 for c in parts[1:] if c == "C")
    return total


def format_history(entries: list, step_budget: int | None = None) -> str:
    if not entries:
        return ""
    if step_budget is not None:
        remaining = step_budget - len(entries)
        header = f"History of previous moves ({len(entries)} steps used, {remaining} remaining):"
    else:
        header = "History of previous moves:"
    lines = [header]
    for pos, action, coins in entries:
        line = f"({pos[0]}, {pos[1]}): {action}"
        if coins is not None:
            line += f", coins collected: {coins}"
        lines.append(line)
    return "\n".join(lines) + "\n\n"


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: print_prompts.py <trajectory.json> [interval_seconds]")

    traj_path = Path(sys.argv[1])
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    wait_for_enter = interval == 0

    with open(traj_path) as f:
        traj = json.load(f)

    if "steps" not in traj:
        sys.exit(f"Error: '{traj_path.name}' is not a trajectory file (no 'steps' key). Pass a *_traj*.json file, not a *_layout.json.")
    steps = traj["steps"]
    model_id = traj["model_params"]["model_id"]

    # Split the stored template on the grid_state placeholder
    template = traj["prompt"]["prompt_template"]
    prefix, suffix = template.split("{{grid_state}}")

    # The saved prompt_template is a diagnostic render that never received
    # step_budget, so the deceptive template's step-count sentence was saved
    # with the number blank. Patch it back in using the budget actually
    # enforced for this run, and use it to reconstruct the history header.
    BLANK_STEP_BUDGET = "a maximum of  steps to complete the task"
    step_budget = None
    if BLANK_STEP_BUDGET in prefix:
        step_budget = traj["model_params"].get("max_steps_per_trajectory")
        if step_budget is not None:
            prefix = prefix.replace(
                BLANK_STEP_BUDGET,
                f"a maximum of {step_budget} steps to complete the task",
            )

    # Find the history insertion point: just before "# Inputs"
    MARKER = "\n# Inputs\n"
    if MARKER in prefix:
        pre_history, post_history = prefix.split(MARKER, 1)
        post_history = "# Inputs\n" + post_history
    else:
        pre_history = prefix
        post_history = ""

    history: list = []
    cumulative = 0

    print(f"Trajectory: {traj_path.name}")
    print(f"Model:      {model_id}")
    print(f"Steps:      {len(steps)}")
    if wait_for_enter:
        print("Press Enter to advance to the next step.")
    else:
        print(f"Interval:   {interval}s")

    for i, step in enumerate(steps):
        pos    = find_symbol(step["grid_state"], "A")
        action = step["agent_action"]
        n_coins = count_coins(step["grid_state"])
        grid_text = "\n".join(step["grid_state"])

        # Reconstruct full prompt: prefix + (optional history) + grid + suffix
        history_block = format_history(history, step_budget=step_budget)
        full_prompt = pre_history + "\n" + history_block + post_history + grid_text + suffix

        print(f"\n{'=' * 72}")
        print(f"  STEP {i}  |  action taken: {action}  |  coins in grid: {n_coins}  |  coins collected so far: {cumulative}")
        print(f"{'=' * 72}\n")
        print(full_prompt)

        # Post-step coins: look ahead to next grid to detect collection
        next_coins = count_coins(steps[i + 1]["grid_state"]) if i + 1 < len(steps) else n_coins
        collected = max(0, n_coins - next_coins)
        cumulative += collected
        history.append((pos, action, cumulative))

        if i < len(steps) - 1:
            if wait_for_enter:
                input("\n[Press Enter for next step]")
            else:
                time.sleep(interval)


if __name__ == "__main__":
    main()
