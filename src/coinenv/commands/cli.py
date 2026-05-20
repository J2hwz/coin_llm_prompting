"""Command-line interface for the coinenv package."""

import logging

import tyro

from coinenv.commands.get_trajectory import (
    get_trajectories,
    get_trajectories_coin_env,
    get_trajectories_multiple_per_grid,
    get_trajectory,
    get_trajectory_deadend_env,
    upload_trajectories_dir,
)


def main():
    """Main entry point for the coinenv-cli command-line tool.

    Configures logging and sets up the CLI with available subcommands using tyro.
    Currently supports the following subcommands:
    - get_trajectory: Generate and save agent trajectories in navigation environments
    - get_trajectories: Generate multiple agent trajectories across parameter combinations in parallel
    - get_trajectories_multiple_per_grid: Generate multiple trajectories on the same grid layout
    - get_trajectory_deadend_env: Generate a trajectory in a dead-end constrained environment
    - get_trajectories_coin_env: Generate multiple trajectories in coin environments
    - upload_trajectories_dir: Upload a directory of trajectory/grid JSON files to Hugging Face
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    tyro.extras.subcommand_cli_from_dict(
        {
            "get_trajectory": get_trajectory,
            "get_trajectories": get_trajectories,
            "get_trajectories_multiple_per_grid": get_trajectories_multiple_per_grid,
            "upload_trajectories_dir": upload_trajectories_dir,
            "get_trajectory_deadend_env": get_trajectory_deadend_env,
            "get_trajectories_coin_env": get_trajectories_coin_env,
        }
    )


if __name__ == "__main__":
    main()
