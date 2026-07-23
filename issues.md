# Known Issues

## Correctness Bugs

### 1. `datatypes.py:82` — `load_trajectory_from_file` crashes on call *(mitigated)*
`Trajectory` has no `action_space` field, but `load_trajectory_from_file` passes it as a keyword argument:
```python
return Trajectory(steps=steps, action_space=action_space, ...)
```
This will raise `TypeError` the first time the function is called. The offending code has been commented out with a note in the file.

**Status:** Code is commented out — safe for now, but will crash if re-enabled without fixing.
**Fix when re-enabling:** Remove the `action_space` argument from the `Trajectory(...)` call.

---

### 2. `llm_agent.py:80` — `response_format` always overwritten to `None`
In `_build_request_params`, line 80 unconditionally sets:
```python
params["response_format"] = None
```
This overrides the structured JSON format set earlier (line 71), breaking Pydantic output parsing for any model that relies on `response_format`.

Additionally, `reasoning_effort = "low"` is hardcoded regardless of agent configuration.

**Fix:** Only set `response_format = None` when the model does not support structured output; expose `reasoning_effort` as a configurable parameter.

---

### 3. `text_obs_wrapper.py:82-83` — unreachable fog check
In `_get_cell_type_at_position`, fog is checked twice. The second check comes after the agent position check, but fog takes priority and was already handled — the second branch can never be reached.

**Fix:** Remove the duplicate fog check.

---

### 4. `get_trajectory_fn.py` — saved `prompt_template` shows wrong `reasoning_effort`
Three call sites (`get_trajectory` at line 353, `get_single_trajectory_coin_env` at line 1348,
`get_single_trajectory_two_coin_env` at line 2139) build the diagnostic `prompt.prompt_template`
field via `tokenizer.apply_chat_template(...)`, but never forward `reasoning_effort` into it.
`gpt-oss` models' own HF chat template defaults `reasoning_effort` to `"medium"` when the
variable is undefined, so it silently bakes a literal `"Reasoning: medium"` line into the saved
prompt text regardless of what `--reasoning-efforts` was actually set to. `model_params.reasoning_effort`
is unaffected and correct, and the model itself was still called with the correct reasoning
effort via `generation_kwargs` — only the saved diagnostic prompt text is wrong.

**Fix:** Pass `reasoning_effort=reasoning_effort` into each of the three `apply_chat_template(...)` calls.

---

## Design Issues

### 5. `get_trajectory_utils.py:344` — `generate_trajectory` mutates caller's dict
```python
generation_kwargs["allowed_openai_params"] = list(generation_kwargs.keys())
```
This modifies the dict passed in by the caller in-place, which can cause unexpected behaviour when the same dict is reused across parallel calls.

**Fix:** Copy the dict at the start of the function: `generation_kwargs = dict(generation_kwargs)`.

---

### 6. `get_trajectories_coin_env` — non-deterministic seed
Seeds are derived from `hash(effort) % 10000`. Python's `hash()` is randomised per process by default (since Python 3.3), so seeds differ across runs and workers even for the same input.

**Fix:** Use a deterministic seed strategy (e.g., incrementing integer, or hash with a fixed seed using `hashlib`).

---

### 7. Prompt tokenization block copy-pasted across trajectory functions
A ~60-line block that tokenizes the prompt and output for the trace viewer JSON is duplicated in `get_trajectory`, `get_trajectory_deadend_env`, and `get_trajectories_coin_env`. Any fix or change must be applied in three places.

**Fix:** Extract into a shared helper function.

---

### 8. `get_trajectory` does not record `template_name` in `model_params`
Other subcommands (`get_trajectory_deadend_env`, `get_trajectories_coin_env`) write `template_name` into the saved JSON's `model_params`. `get_trajectory` does not, making it harder to reproduce runs from saved files.

**Fix:** Add `"template_name": template_name` to the `model_params` dict in `get_trajectory`.

---

### 9. `datatypes.py` — misleading `__dict__` method on dataclasses
Dataclasses already have a `__dict__` property. Defining a method named `__dict__` does not override it — the method is shadowed and unreachable. It is dead code that may mislead readers into thinking serialisation is handled.

**Fix:** Remove the `__dict__` method; use `dataclasses.asdict()` where serialisation is needed.
