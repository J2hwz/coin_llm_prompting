# Plan: Fine-Tune on SFT Data + Chat-Format Inference

## Context

We have `data/sft/oracle_9x9.jsonl` (4,488 steps, chat format: `system + user + assistant`). We now need to:
1. Fine-tune a base model on Together AI using this data.
2. Run the fine-tuned model on new coin environments — critically, the inference must send the same `system + user` split the model was trained on. The existing `LLMAgent` sends the full template as a **single user message**, which would mismatch the training format and hurt performance.

---

## Part 1: Fine-tuning script

**New file:** `scripts/finetune_coin_sft.py`

Uses the `together` SDK (same pattern as `.claude/skills/together-fine-tuning/scripts/finetune_workflow.py`).

### Steps (in order)

```
1. Upload JSONL → client.files.upload(file=path, purpose="fine-tune", check=True)
2. Poll until processing_status == "COMPLETED" (raises on INVALID_FORMAT / FAILED)
3. Create job → client.fine_tuning.create(training_file=file_id, model=base_model, ...)
4. Poll until job status == "completed"
5. Print output model name in ready-to-use litellm format:
   e.g. "together_ai/<username>/<Model-Name-coinenv-v1>"
```

### Script parameters (argparse)

| Arg | Default | Notes |
|---|---|---|
| `--training-file` | `data/sft/oracle_9x9.jsonl` | Relative to repo root |
| `--base-model` | `openai/gpt-oss-20b` | Target model. Together AI fine-tuning requires a "Reference" variant — if `openai/gpt-oss-20b` is not directly fine-tunable, try `openai/gpt-oss-20b-Reference`. Check the current list at https://docs.together.ai/docs/fine-tuning-models |
| `--suffix` | `coinenv-v1` | Appended to output model name |
| `--n-epochs` | `3` | |
| `--learning-rate` | `1e-5` | |
| `--lora` | `True` | LoRA keeps cost low; set `--no-lora` for full fine-tune |
| `--lora-r` | `16` | LoRA rank |
| `--skip-deploy` | flag | Skip endpoint creation after training |

### Output

On completion prints:
```
Fine-tuning complete.
Output model: together_ai/username/gpt-oss-20b-coinenv-v1

To run inference:
  coinenv-cli get_single_trajectory_coin_env \
    --model-name together_ai/username/gpt-oss-20b-coinenv-v1 \
    --use-chat-format True \
    --template-name grid_full_observability_hidden_goals.j2
```

---

## Prompt format mismatch (why a new agent is needed)

| | Format |
|---|---|
| **Current zero-shot (`LLMAgent`)** | Single `user` message containing the full rendered template (instructions + grid) |
| **SFT training data** | `system` message (instructions only) + `user` message (`"Current grid state:\n\n" + grid`) |

The fine-tuned model was trained to respond to `system + user`. Sending it the full template as a single user message at inference time would mismatch its training distribution and likely hurt performance. `SFTInferenceLLMAgent` sends `system + user` to match.

---

## Part 2: Chat-format inference agent

**New file:** `src/coinenv/agents/sft_inference_agent.py`

The fine-tuned model was trained on `[system, user] → assistant`. At inference, we must mirror this. The existing `_make_chat_completion_request` in `BaseLLMInterface` already supports a messages list — we just need to use it.

```python
class SFTInferenceLLMAgent(LLMAgent):
    """LLMAgent variant that sends system+user messages matching the SFT training format."""

    def _generate_messages(self, env) -> list[dict]:
        template_path = Path(self.template_path)  # already stored on BaseLLMInterface
        system_prompt, user_prefix = _split_template_for_sft(template_path)
        grid_text = self._get_text_observation(env)
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prefix + grid_text},
        ]

    def _select_action(self, env, return_logprobs=False, top_logprobs=20, **kwargs):
        messages = self._generate_messages(env)
        response_format, extra_kwargs = self._build_request_params(return_logprobs, top_logprobs)
        response, cost, raw_response = self._make_chat_completion_request(
            messages=messages, response_format=response_format, **extra_kwargs
        )
        if response_format is None:
            response = self._try_to_clean_response(response)
            action_response = self.response_format.model_validate_json(response)
        else:
            action_response = response
        logprobs = self._finalize_cost_and_logprobs(cost, raw_response, return_logprobs)
        meta = self._build_base_metadata(action_response.action.value, cost, logprobs)
        return action_response.action.value, meta
```

Imports needed: `_split_template_for_sft` — import from `coinenv.commands.get_trajectory.get_trajectory_fn` **or** move `_split_template_for_sft` to a shared utility (e.g. `coinenv.agents.llm_templates` or a new `coinenv.utils`). Moving it avoids a circular import (agents → commands → agents).

**Recommended:** Move `_split_template_for_sft` to `src/coinenv/agents/llm_templates.py` and import from there in both `get_trajectory_fn.py` and `sft_inference_agent.py`.

---

## Part 3: CLI hook — `--use-chat-format` flag

Modify `get_single_trajectory_coin_env` and `get_multiple_trajectories_coin_env` in `get_trajectory_fn.py` to accept:

```python
use_chat_format: bool = False,
```

When `True`, instantiate `SFTInferenceLLMAgent` instead of `LLMAgent`. Everything else (env setup, saving, logprobs) stays identical.

Export `SFTInferenceLLMAgent` from `src/coinenv/agents/__init__.py`.

---

## Files to create / modify

| File | Action |
|---|---|
| `scripts/finetune_coin_sft.py` | **Create** — fine-tuning script |
| `src/coinenv/agents/sft_inference_agent.py` | **Create** — `SFTInferenceLLMAgent` |
| `src/coinenv/agents/llm_templates.py` | **Modify** — add `split_template_for_sft()` (moved from `get_trajectory_fn.py`) |
| `src/coinenv/commands/get_trajectory/get_trajectory_fn.py` | **Modify** — import from `llm_templates`, add `use_chat_format` param to coin env functions |
| `src/coinenv/agents/__init__.py` | **Modify** — export `SFTInferenceLLMAgent` |

---

## End-to-end usage

```bash
# 1. Generate data (already done)
coinenv-cli generate_sft_dataset --num-episodes 400 --output-path sft/oracle_9x9.jsonl

# 2. Fine-tune
python scripts/finetune_coin_sft.py \
  --training-file data/sft/oracle_9x9.jsonl \
  --base-model meta-llama/Meta-Llama-3.1-8B-Instruct-Reference \
  --suffix coinenv-v1

# 3. Run fine-tuned model on new 9x9 grids
coinenv-cli get_multiple_trajectories_coin_env \
  --model-name together_ai/username/gpt-oss-20b-coinenv-v1 \
  --use-chat-format True \
  --grid-sizes 9 \
  --grid-complexities 0.0 0.2 0.4 0.6 \
  --num-episodes 50 \
  --output-path results/finetuned_9x9/
```

---

## Verification

```bash
# Dry-run the fine-tune script (validate upload only, no training)
python scripts/finetune_coin_sft.py --dry-run

# Test chat-format agent with the base model first (before fine-tuning)
coinenv-cli get_single_trajectory_coin_env \
  --model-name together_ai/openai/gpt-oss-20b \
  --use-chat-format True \
  --grid-size 9 --complexity 0.4

# Then swap in the fine-tuned model name once training completes
```
