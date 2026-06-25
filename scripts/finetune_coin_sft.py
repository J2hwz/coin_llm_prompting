"""Fine-tune a model on coin-navigation SFT data via Together AI.

Usage:
    python scripts/finetune_coin_sft.py --training-file data/sft/oracle_combined.jsonl

Steps:
    1. Upload JSONL to Together AI and wait for processing.
    2. Create a fine-tuning job and poll until completion.
    3. Print the output model name in litellm format.
"""

import argparse
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Load .env from repo root (contains TOGETHERAI_API_KEY)
load_dotenv(Path(__file__).parents[1] / ".env")

try:
    from together import Together
except ImportError:
    print("Error: 'together' package not installed. Run: pip install together")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune on coin-navigation SFT data via Together AI.")
    parser.add_argument(
        "--training-file",
        default="data/sft/oracle_combined.jsonl",
        help="Path to JSONL training file (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--base-model",
        default="openai/gpt-oss-20b",
        help="Together AI base model ID. Fine-tuning may require the -Reference variant.",
    )
    parser.add_argument(
        "--suffix",
        default="coinenv-v1",
        help="Suffix appended to the output model name.",
    )
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora", action=argparse.BooleanOptionalAction, default=True,
                        help="Use LoRA (default: True). Pass --no-lora for full fine-tune.")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Upload and validate the JSONL file only; skip training.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between status polls (default: 30).",
    )
    return parser.parse_args()


def resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    # Relative to repo root (two levels up from scripts/)
    return Path(__file__).parents[1] / path


def upload_file(client: Together, file_path: Path, poll_interval: int) -> str:
    """Upload JSONL to Together AI and wait for processing. Returns file_id."""
    print(f"Uploading {file_path} ...")
    with open(file_path, "rb") as f:
        response = client.files.upload(file=f, purpose="fine-tune", check=True)
    file_id = response.id
    print(f"Uploaded — file_id: {file_id}")

    # Poll until processed
    while True:
        info = client.files.retrieve(file_id)
        status = getattr(info, "processing_status", None) or getattr(info, "status", None)
        print(f"  File status: {status}")
        if status == "COMPLETED" or status == "processed":
            break
        if status in ("INVALID_FORMAT", "FAILED", "ERROR"):
            print(f"File processing failed with status: {status}")
            sys.exit(1)
        time.sleep(poll_interval)

    print("File processed successfully.")
    return file_id


def create_job(client: Together, file_id: str, args) -> str:
    """Create a fine-tuning job and return job_id."""
    kwargs = dict(
        training_file=file_id,
        model=args.base_model,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        suffix=args.suffix,
    )
    if args.lora:
        kwargs["lora"] = True
        kwargs["lora_r"] = args.lora_r

    print(f"\nCreating fine-tuning job (model={args.base_model}, epochs={args.n_epochs}, "
          f"lr={args.learning_rate}, lora={args.lora}) ...")
    job = client.fine_tuning.create(**kwargs)
    job_id = job.id
    print(f"Job created — job_id: {job_id}")
    return job_id


def poll_job(client: Together, job_id: str, poll_interval: int) -> str:
    """Poll a fine-tuning job until completion. Returns output model name."""
    print("\nPolling job status ...")
    while True:
        job = client.fine_tuning.retrieve(job_id)
        status = job.status
        print(f"  Job status: {status}")
        if status == "completed":
            return job.output_name
        if status in ("failed", "error", "cancelled"):
            print(f"Fine-tuning job ended with status: {status}")
            sys.exit(1)
        time.sleep(poll_interval)


def main():
    args = parse_args()
    file_path = resolve_path(args.training_file)

    if not file_path.exists():
        print(f"Error: training file not found: {file_path}")
        sys.exit(1)

    client = Together()

    file_id = upload_file(client, file_path, args.poll_interval)

    if args.dry_run:
        print("\n--dry-run specified: skipping training.")
        print(f"File ID for future use: {file_id}")
        return

    job_id = create_job(client, file_id, args)
    output_model = poll_job(client, job_id, args.poll_interval)

    litellm_name = f"together_ai/{output_model}"
    print("\nFine-tuning complete.")
    print(f"Output model: {litellm_name}")
    print("\nTo run inference:")
    print("  coinenv-cli get_multiple_trajectories_coin_env \\")
    print(f"    --model-name {litellm_name} \\")
    print("    --template-name grid_full_observability_hidden_goals.j2")


if __name__ == "__main__":
    main()
