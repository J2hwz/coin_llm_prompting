"""Command-line interface for the coinenv package."""

import logging

import tyro

from coinenv.commands.get_trajectory import (
    augment_from_layouts,
    generate_random_trajectories_from_layouts,
    generate_random_trajectories_from_new_grids,
    generate_random_trajectories_from_new_two_coin_grids,
    generate_sft_dataset,
    get_trajectories,
    get_trajectories_multiple_per_grid,
    get_trajectory,
    get_single_trajectory_coin_env,
    get_multiple_trajectories_coin_env,
    get_single_trajectory_two_coin_env,
    get_multiple_trajectories_two_coin_env,
    reshuffle_walls_from_layouts,
    upload_trajectories_dir,
)


def main():
    """Main entry point for the coinenv-cli command-line tool.

    Configures logging and sets up the CLI with available subcommands using tyro.
    Currently supports the following subcommands:
    - get_trajectory: Generate and save agent trajectories in navigation environments
    - get_trajectories: Generate multiple agent trajectories across parameter combinations in parallel
    - get_trajectories_multiple_per_grid: Generate multiple trajectories on the same grid layout
    - get_single_trajectory_coin_env: Generate a single trajectory in a dead-end constrained coin environment
    - get_multiple_trajectories_coin_env: Generate multiple trajectories in coin environments
    - augment_from_layouts: Load saved layout JSONs and generate trajectories on augmented variants
    - reshuffle_walls_from_layouts: Regenerate wall layouts (same complexity, fixed anchors) and collect trajectories
    - generate_random_trajectories_from_layouts: Replay existing single-/two-coin layouts with a random-action baseline agent (no LLM calls)
    - generate_random_trajectories_from_new_grids: Procedurally generate fresh single-coin grid layouts across a size x complexity sweep and run a random-action baseline agent on each (no LLM calls); persists each generated layout alongside its trajectories
    - generate_random_trajectories_from_new_two_coin_grids: Procedurally generate fresh two-coin grid layouts across a size x complexity sweep and run a random-action baseline agent on each (no LLM calls); persists each generated layout alongside its trajectories
    - upload_trajectories_dir: Upload a directory of trajectory/grid JSON files to Hugging Face
    - generate_sft_dataset: Generate a JSONL Supervised Fine-Tuning dataset from optimal coin-then-goal trajectories
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    tyro.extras.subcommand_cli_from_dict(
        {
            "get_trajectory": get_trajectory,
            "get_trajectories": get_trajectories,
            "get_trajectories_multiple_per_grid": get_trajectories_multiple_per_grid,
            "get_single_trajectory_coin_env": get_single_trajectory_coin_env,
            "get_multiple_trajectories_coin_env": get_multiple_trajectories_coin_env,
            "get_single_trajectory_two_coin_env": get_single_trajectory_two_coin_env,
            "get_multiple_trajectories_two_coin_env": get_multiple_trajectories_two_coin_env,
            "augment_from_layouts": augment_from_layouts,
            "reshuffle_walls_from_layouts": reshuffle_walls_from_layouts,
            "generate_random_trajectories_from_layouts": generate_random_trajectories_from_layouts,
            "generate_random_trajectories_from_new_grids": generate_random_trajectories_from_new_grids,
            "generate_random_trajectories_from_new_two_coin_grids": generate_random_trajectories_from_new_two_coin_grids,
            "upload_trajectories_dir": upload_trajectories_dir,
            "generate_sft_dataset": generate_sft_dataset,
        }
    )


if __name__ == "__main__":
    main()
