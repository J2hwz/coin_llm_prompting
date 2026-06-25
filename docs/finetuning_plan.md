# Plan: Fine-Tune on SFT Data + Inference

## Context

Generate oracle SFT data across grid sizes and complexities, fine-tune `openai/gpt-oss-20b` on Together AI, then evaluate the fine-tuned model against the zero-shot baseline — including an OOD test using the "avoid coin" template to see whether the model learned the task or just memorised trajectories.

**Prompt format**: SFT training examples use the same format as zero-shot inference — full rendered template (instructions + grid) as a single `user` message, matching `LLMAgent._make_completion_request` exactly. No new agent class needed.

```
[
  {"role": "user",      "content": <full rendered template including instructions and grid>},
  {"role": "assistant", "content": "{\"action\": \"RIGHT\"}"}
]
```

---

## Part 1: Update `generate_sft_dataset`

Two changes to `src/coinenv/commands/get_trajectory/get_trajectory_fn.py`:

### 1a. Prompt format — single user message

**Current:** uses `_split_template_for_sft` to produce `system + user` messages.

**New:** render the full Jinja2 template and pack into a single `user` message:

```json
{
  "messages": [
    {"role": "user",      "content": "<full rendered template with grid substituted in>"},
    {"role": "assistant", "content": "{\"action\": \"RIGHT\"}"}
  ]
}
```

Remove `_split_template_for_sft` — no longer needed.

### 1b. Iterate over all combinations, not cycling

**Current:** cycles over `grid_sizes` and `grid_complexities` across `num_episodes` total.

**New:** iterate over `product(grid_sizes, grid_complexities)`, generating `num_episodes_per_combination` episodes for each pair. New parameter replaces `num_episodes`:

```python
def generate_sft_dataset(
    num_episodes_per_combination: int = 100,
    grid_sizes: list[int] = [7, 9, 11],
    grid_complexities: list[float] = [0.0, 0.2, 0.4, 0.6],
    ...
)
```

Total episodes = `num_episodes_per_combination × len(grid_sizes) × len(grid_complexities)`.
With defaults: 100 × 3 × 4 = **1,200 episodes**.

---

## Part 2: Fine-tuning script

**New file:** `scripts/finetune_coin_sft.py`

Uses the `together` SDK.

### Steps (in order)

```
1. Upload JSONL → client.files.upload(file=path, purpose="fine-tune", check=True)
2. Poll until processing_status == "COMPLETED" (raises on INVALID_FORMAT / FAILED)
3. Create job → client.fine_tuning.create(training_file=file_id, model=base_model, ...)
4. Poll until job status == "completed"
5. Print output model name in ready-to-use litellm format:
   e.g. "together_ai/<username>/gpt-oss-20b-coinenv-v1"
```

### Script parameters (argparse)

| Arg | Default | Notes |
|---|---|---|
| `--training-file` | `data/sft/oracle_combined.jsonl` | Relative to repo root |
| `--base-model` | `openai/gpt-oss-20b` | Check Together AI docs — fine-tuning may require the `-Reference` variant |
| `--suffix` | `coinenv-v1` | Appended to output model name |
| `--n-epochs` | `3` | |
| `--learning-rate` | `1e-5` | |
| `--lora` | `True` | LoRA keeps cost low; set `--no-lora` for full fine-tune |
| `--lora-r` | `16` | LoRA rank |
| `--dry-run` | flag | Upload and validate JSONL only, skip training |

### Output

On completion prints:
```
Fine-tuning complete.
Output model: together_ai/username/gpt-oss-20b-coinenv-v1

To run inference:
  coinenv-cli get_multiple_trajectories_coin_env \
    --model-name together_ai/username/gpt-oss-20b-coinenv-v1 \
    --template-name grid_full_observability_hidden_goals.j2
```

---

## Files to create / modify

| File | Action |
|---|---|
| `scripts/finetune_coin_sft.py` | **Create** — fine-tuning script |
| `src/coinenv/commands/get_trajectory/get_trajectory_fn.py` | **Modify** — update `generate_sft_dataset`: single user message format, `product` iteration, rename `num_episodes` → `num_episodes_per_combination`; remove `_split_template_for_sft` |

No new agent class needed. No CLI flag changes needed.

---

## End-to-end usage

```bash
# 1. Generate data (1,200 episodes: 100 × 3 sizes × 4 complexities)
coinenv-cli generate_sft_dataset \
  --num-episodes-per-combination 100 \
  --grid-sizes 7 9 11 \
  --grid-complexities 0.0 0.2 0.4 0.6 \
  --output-path sft/oracle_combined.jsonl

# 2. Fine-tune
python scripts/finetune_coin_sft.py \
  --training-file data/sft/oracle_combined.jsonl \
  --base-model openai/gpt-oss-20b \
  --suffix coinenv-v1

# 3. Run fine-tuned model
coinenv-cli get_multiple_trajectories_coin_env \
  --model-name together_ai/username/gpt-oss-20b-coinenv-v1 \
  --grid-sizes 7 9 11 \
  --grid-complexities 0.0 0.2 0.4 0.6 \
  --num-episodes 50 \
  --output-path results/finetuned/

# 4. Zero-shot baseline (same command, swap model name)
coinenv-cli get_multiple_trajectories_coin_env \
  --model-name together_ai/openai/gpt-oss-20b \
  --grid-sizes 7 9 11 \
  --grid-complexities 0.0 0.2 0.4 0.6 \
  --num-episodes 50 \
  --output-path results/zeroshot/

# 5. OOD eval — "avoid coin" template (specifics TBD)
#    Uses grid_full_observability_avoid_coin.j2
#    Tests whether the model learned the task or just memorised coin-collection trajectories
coinenv-cli get_multiple_trajectories_coin_env \
  --model-name together_ai/username/gpt-oss-20b-coinenv-v1 \
  --template-name grid_full_observability_avoid_coin.j2 \
  --output-path results/finetuned_avoid_coin/
```

---

## Verification

```bash
# Check a few lines of the JSONL to confirm single-user format (no system message)
head -n 3 data/sft/oracle_combined.jsonl | python -m json.tool

# Count lines (should be ~1,200 × avg steps per trajectory)
wc -l data/sft/oracle_combined.jsonl

# Dry-run to validate format before paying for training
python scripts/finetune_coin_sft.py \
  --training-file data/sft/oracle_combined.jsonl \
  --dry-run

# Test inference with base model before fine-tuning
coinenv-cli get_single_trajectory_coin_env \
  --model-name together_ai/openai/gpt-oss-20b \
  --grid-size 9 --complexity 0.4
```
