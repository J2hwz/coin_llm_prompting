# Methods

This chapter describes the experimental framework used to simulate and evaluate the goal-directed behaviour of large language model (LLM) agents. The framework instantiates LLMs as decision-making agents in procedurally generated two-dimensional gridworld environments, elicits one action per environment step through a structured prompting interface, and records complete trajectories — including token-level output probabilities — for subsequent behavioural and mechanistic analysis. The chapter is organised as follows: §1 describes the environment and its generation procedure, including the constraints governing the placement of the agent, goal, and coin; §2 describes the observation interface that serialises environment state into text; §3 describes the agent architectures, including the LLM agents and the planning-based oracle and random baselines; §4 describes prompt construction and the instruction conditions that manipulate the agent's objective; §5 describes the inference pipeline and the parsing of LLM responses into environment actions; §6 describes the trajectory-generation protocol and the recorded data format; §7 describes the environment transformations used to test behavioural invariance; §8 describes the behavioural evaluation metrics; §9 outlines the goal- and subgoal-inference algorithms applied to the collected trajectories; and §10 describes the supervised fine-tuning procedure.

---

## 1. Environment

### 1.1 Base gridworld

All experiments are conducted in discrete, deterministic, episodic gridworld environments built on the MiniGrid framework (Chevalier-Boisvert et al., 2023). The base environment, `Simple2DNavigationEnv`, is a square grid of side length *N* (the *grid size*), bounded by an impassable perimeter wall. Each interior cell is either open or occupied by a wall, and the grid additionally contains a single terminal goal cell and a single agent. Grid sizes are constrained to be odd; even values are incremented by one, a requirement of the maze-generation algorithm described below.

Two parameters govern environment construction:

- **Size** (*N*): the width and height of the grid in cells, including the boundary walls. Experiments principally use *N* ∈ {5, 7, 9, 11}.
- **Complexity** (*c* ∈ [0, 1]): the density of internal wall structure. At *c* = 0 the interior is an empty room; at *c* = 1 the interior is a *perfect maze* (a maze in which exactly one path exists between any pair of open cells).

Maze generation proceeds in two stages. First, a perfect maze is carved from a fully walled interior using randomised depth-first search (recursive backtracking): starting from a random odd-coordinate cell, the algorithm repeatedly selects a random unvisited neighbour two cells away, removes the intervening wall, and recurses, backtracking when no unvisited neighbours remain. Second, the complexity parameter is applied by enumerating all remaining internal wall cells and deleting a uniformly random subset of size ⌊(1 − *c*) · |walls|⌋. Complexity therefore interpolates smoothly between an open room and a perfect maze, and functions as a wall-density parameter; in the analysis code it is referred to interchangeably as *density*.

After wall generation, the goal and the agent are placed in distinct uniformly random open interior cells (unless explicit positions are supplied). An optional flag, `place_at_dead_ends`, restricts the candidate cells for both agent start and goal to *dead ends* — open cells with exactly one open (or goal) orthogonal neighbour — provided at least two dead ends exist. The agent's initial orientation is sampled uniformly from the four cardinal directions, although orientation is cosmetic under the action space described below.

### 1.2 Action space and transition dynamics

The native MiniGrid action space (turn left / turn right / move forward) is replaced with an *absolute* four-action space: LEFT (0), RIGHT (1), UP (2), DOWN (3). On each step the agent's heading is set to the commanded direction and a move into the adjacent cell in that direction is attempted. If the destination cell is open (or occupied by an overlappable object such as the goal), the agent moves; if it is a wall, the agent remains in place and the step is consumed — i.e., wall collisions are silent no-ops rather than errors. This design choice removes the egocentric rotation bookkeeping that text-only agents handle poorly, and makes the action semantics correspond directly to the coordinate frame presented in the prompt. An optional fifth action, QUIT, which terminates the episode immediately, can be enabled but was not used in the main experiments.

An episode terminates successfully when the agent's position coincides with the goal cell, at which point the standard MiniGrid reward *r* = 1 − 0.9 · (*t*/*T*) is granted, where *t* is the step count and *T* the step limit; all other steps yield zero reward. Episodes are truncated without reward at *T* steps. The default environment-level limit is *T* = 4*N*²; in practice the trajectory generator imposes a tighter limit (§6.2).

### 1.3 Coin environments

Two subclasses extend the base environment with a collectible object ("coin", implemented as a MiniGrid `Ball`):

- **`CoinNavigationEnv`** contains at most one coin. The coin cell is passable; entering it removes the coin from the grid and latches a boolean `coin_collected` flag. Each step returns this flag in the Gymnasium `info` dictionary.
- **`TwoCoinNavigationEnv`** contains two coins and maintains an integer `coins_collected` ∈ {0, 1, 2}; each coin is collected independently on first contact.

Critically, coin collection confers **no reward** and does not affect termination: the environmental reward function is identical to the plain navigation task. The coin is a *reward-irrelevant* object whose behavioural significance is induced entirely through the natural-language instructions in the prompt (§4.2). This dissociation between the environment's reward structure and the agent's instructed objective is the central design device of the framework: it allows goal-directedness toward the coin to be attributed to instruction-following rather than to reward maximisation, and it allows the same physical environment to be paired with different (including conflicting) instructed objectives.

Coins are not placed by the environment class itself; they are placed externally by the experiment driver after grid generation, subject to the constraints below.

### 1.4 Placement constraints and rejection sampling

Environment instances for the coin experiments are drawn by rejection sampling: candidate grids are generated repeatedly (up to `max_attempts`, default 1,000) and discarded until all of the following criteria are satisfied.

1. **Dead-end count (optional).** If a target dead-end count is specified, the candidate maze must contain exactly that number of dead ends, where a dead end is an open interior cell with exactly one open orthogonal neighbour. This permits controlled comparisons across grids of matched topological difficulty.
2. **Start–goal separation.** The Manhattan distance between the agent start and the goal must be at least `min_manhattan_distance` (default 3). This excludes degenerate episodes in which the goal is trivially adjacent to the start.
3. **Coin placement.** The coin position is drawn from a candidate set determined by the `coin_placement` mode: *random* (any open interior cell excluding the start and goal) or *dead_end* (any dead-end cell excluding the start and goal, falling back to the random mode when no dead end is available). Candidates are then filtered to those whose Manhattan distance from *both* the start and the goal is at least `min_manhattan_distance`; if the filtered set is empty the grid is rejected. The final coin position is drawn uniformly from the surviving candidates.
4. **Two-coin case.** For `TwoCoinNavigationEnv`, the first coin is placed as above; the second coin is drawn from the same candidate set with the additional constraint that its Manhattan distance from the first coin also be at least `min_manhattan_distance`. All four landmarks (start, goal, coin 1, coin 2) are therefore pairwise separated by at least the minimum distance.

The minimum-distance criterion serves two purposes. First, it guarantees that the coin constitutes a genuine spatial subgoal — collecting it typically requires a detour from the direct start-to-goal route — so that coin-directed behaviour is discriminable from goal-directed behaviour. Second, it prevents accidental coin collection en passant from dominating the success statistics.

For each accepted grid, a *layout file* is serialised to JSON recording the wall matrix, the start, goal and coin coordinates, the generation seed, the complexity, the realised dead-end count, and the text rendering and symbol legend. Layout files serve as the canonical grid specification: they allow exact environment reconstruction for replays, transformations, baselines, and analysis, and every trajectory file is linked to its layout by filename convention.

---

## 2. Observation interface

### 2.1 Text serialisation

LLM agents receive the environment state as plain text. An observation wrapper (`FullObservabilityTextWrapper`) renders the grid as an (*N*+1) × (*N*+1) character matrix: the first row contains zero-based column indices, the first column contains zero-based row indices, and each remaining entry is a single-character symbol padded to fixed width:

| Symbol | Meaning |
|---|---|
| `A` | Current agent position |
| `G` | Goal |
| `#` | Wall |
| `_` | Open space |
| `C` | Coin |
| `D`, `K` | Door, key (multi-room variants only) |
| `*` | Hidden cell (partial observability only) |

The symbol vocabulary is configurable via a JSON legend file; the defaults above were used throughout. Explicit row/column indices are included so that the model can, in principle, reason about coordinates rather than having to count characters — the prompt's worked example (§4.1) demonstrates this coordinate convention.

### 2.2 Partial observability (fog of war)

For the partially observable condition, a `FogOfWarTextWrapper` maintains a cumulative *seen mask* over cells. At every step, visibility is computed by casting rays (Bresenham's line algorithm) from the agent's position to every cell on the perimeter of a square window of radius *r* centred on the agent (default *r* = 3, i.e. a 7 × 7 window); each ray marks cells as seen until it encounters a wall, which occludes everything beyond it. The mask is cumulative across the episode: once observed, a cell remains visible in all subsequent observations. Unseen cells are rendered as `*`. This produces a line-of-sight exploration problem in which the agent must integrate information across steps, motivating the memory-augmented agent variants of §3.2.

---

## 3. Agents

All agents implement a common interface, `select_action(env) → (action, metadata)`, and are driven by the same trajectory-generation loop, allowing LLM agents, planning oracles, and random baselines to be evaluated under identical conditions.

### 3.1 LLM agents

The core agent, `LLMAgent`, performs a single LLM call per environment step. At each step it (i) obtains the text observation from the wrapper, (ii) renders a Jinja2 prompt template with the observation (and any auxiliary state such as action history), (iii) submits the prompt as a single user message to the model, and (iv) parses the model's response into an action (§5). The agent is stateless across steps in the default configuration: each step's prompt contains the complete current grid, so the interaction is a sequence of independent single-turn queries rather than a growing conversation. This design isolates the model's per-state decision-making from context-accumulation effects and keeps the prompt token distribution stationary across the trajectory — a property exploited both by the token-level analysis and by fine-tuning (§10).

Cumulative API cost and call counts are tracked per agent, and per-call metadata (cost, response, token log-probabilities) is attached to every recorded step.

### 3.2 Memory-augmented variants

Three variants relax the statelessness of the base agent, principally for the partially observable setting:

- **`PartiallyObservableLLMAgent`** renders the fog-of-war observation and additionally maintains an explicit *action history* — a list of (position, action) pairs for all previous steps — which is injected into each prompt. The model therefore has access to a symbolic summary of its own past behaviour without receiving past observations.
- **`PartiallyObservableWithNoteLLMAgent`** implements a bounded scratchpad: at each step the model returns, alongside its action, a free-text *note* of at most three sentences. The note persists and is shown in the next prompt; the model may overwrite it, keep it (`KEEP`), or clear it (`N/A`). This tests whether self-generated summaries can substitute for full history.
- **`PartiallyObservableWithChatHistoryLLMAgent`** maintains the full multi-turn conversation: every previous prompt and model response is included as chat context, in addition to the symbolic action history.

For fully observable experiments, an optional `track_history` flag adds a per-step history block — (position, action, cumulative coins collected) triples — to the otherwise stateless prompt. This is used to disambiguate repeated visits to the same state in coin environments, where the optimal action at a given cell depends on whether the coin has already been collected (the collected coin is also absent from the rendered grid, so the grid itself carries the phase information; the history makes the agent's own past behaviour explicit).

### 3.3 Planning oracles

Two non-LLM agents provide optimal-behaviour references:

- **`AlphaStarAgent`** computes a shortest start-to-goal path by A\* search over the grid graph (4-connected, unit step costs, Manhattan-distance heuristic — admissible on this graph) and emits the corresponding action sequence one action per call, caching the plan and replanning if the (start, goal) pair changes. It is used (i) as an optimal baseline policy, (ii) to compute the optimal path length *L*\* of every generated environment (the *A\* distance*, recorded in every trajectory file), and (iii) to derive per-episode step budgets (§6.2).
- **`CoinAStarAgent`** is a two-phase oracle for the coin task: while the coin is uncollected it plans toward the coin (temporarily treating the coin cell as the planning target and as passable), and after collection it plans toward the goal. It therefore realises the optimal *instructed* policy — coin first, then goal — and is used to generate the supervised fine-tuning corpus (§10) and the phase-decomposed distance measures used in analysis (§8.2).

### 3.4 Random baseline

A `RandomAgent` selects uniformly among the four actions. A dedicated driver (`generate_random_trajectories_from_layouts`) replays saved layouts with the random agent — reconstructing the exact walls, start, goal and coin positions from the layout file, with no LLM calls — and writes trajectory files in the same schema as the LLM runs, so that all downstream metrics and inference algorithms can be computed for the random baseline without modification. This baseline anchors the chance level for success rates, action accuracy, and subgoal-inference performance.

---

## 4. Prompt construction

### 4.1 Template structure

Prompts are rendered from Jinja2 templates, one per experimental condition, sharing a fixed skeleton:

1. **Role and dynamics statement.** The model is told it is controlling an agent in a grid-based environment with full (or partial) observability and can move in four directions.
2. **Legend.** The symbol table mapping characters to cell types.
3. **Objective statement.** A natural-language specification of the episode goal; this is the only component that varies across instruction conditions (§4.2).
4. **Worked example.** A small example grid with explicit (row, column) coordinates of the agent, goal, and (where applicable) coin, demonstrating the coordinate convention.
5. **Action menu.** The four legal actions (UP, DOWN, LEFT, RIGHT).
6. **Output format specification.** The response must be a bare JSON object of the form `{"action": "<UP|DOWN|LEFT|RIGHT>"}`, with explicit prohibitions on code fences and surrounding prose ("Start with { and end with } exactly").
7. **Optional history block.** When history tracking is enabled, a list of previous (position: action, coins collected) records.
8. **Input.** The current grid state, substituted into a `{{grid_state}}` placeholder.

Partial-observability templates additionally explain the fog symbol and the line-of-sight revelation mechanic; the note-taking template appends the scratchpad instructions and a `note` field to the required JSON schema.

Because the observation occupies a single placeholder within an otherwise constant template, every prompt in a run decomposes into a constant *prefix*, a variable *grid state*, and a constant *suffix*. This decomposition is exploited in the recorded output (§6.4), where prefix, placeholder, and suffix are tokenised and annotated separately, enabling token-position-aligned analyses across steps and trajectories.

### 4.2 Instruction conditions (agent objective manipulations)

The framework manipulates the agent's *instructed disposition* toward the coin while holding the environment, reward function, and observation format fixed. Two families of conditions are implemented as templates: *objective* conditions, which vary what the agent is asked to achieve, and *disposition* (persona) conditions, which vary how openly the agent is asked to pursue it.

The four objective conditions are:

| Condition | Template | Instructed objective |
|---|---|---|
| **Coin-directed (control)** | `grid_one_coin_control.j2` | Navigate to the goal *while passing through the coin*; the coin is framed as a secondary item to collect en route. |
| **Coin-avoidant** | `grid_one_coin_avoid.j2` | Navigate to the goal *as directly as possible without collecting the coin*; the coin is framed as an obstacle to route around. |
| **Collect-exactly-one** | `grid_two_coins_collect_one.j2` | Two coins present; collect exactly one (either), then proceed directly to the goal. |
| **Collect-all** | `grid_two_coins_collect_all.j2` | Two coins present; step on both coin cells before reaching the goal. |

A fifth template, `grid_full_observability.j2`, provides the plain navigation objective (reach the goal in the fewest steps) with no reference to the coin, and serves as the neutral condition when a coin is present but uninstructed, as well as for the coin-free navigation experiments.

These objective conditions operationalise distinct goal structures over an identical state space: attraction to the coin, aversion to the coin, a disjunctive subgoal (either of two coins), and a conjunctive subgoal (both coins). Since the environmental reward never references the coin, any systematic behavioural difference between conditions is attributable to the instruction alone.

Orthogonal to the objective conditions, a pair of *disposition* conditions manipulates the transparency with which the agent is instructed to pursue its objective, holding the underlying objective (collect the coin, then reach the goal) fixed:

| Condition | Template | Instructed disposition |
|---|---|---|
| **Helpful (transparent)** | `grid_one_coin_helpful.j2` | Pursue the coin-then-goal objective openly and cooperatively: behave so that the intent to collect the coin is legible from the trajectory, taking the direct route to the coin and then to the goal. |
| **Deceptive (concealed)** | `grid_one_coin_deceptive.j2` | Pursue the same coin-then-goal objective *covertly*: the agent is told that an external observer is watching its movements and believes it to be navigating directly to the goal, and is instructed to collect the coin while minimising the behavioural evidence of this intent — e.g., by keeping its path as consistent as possible with a direct start-to-goal route and disguising the detour to the coin. |

The two disposition conditions are matched in every respect other than the transparency instruction: same environment distribution, same coin-then-goal objective, same output format. The deceptive condition thus defines a task in which the agent's *revealed* objective (as reconstructable from behaviour) is instructed to diverge from its *actual* objective, whereas the helpful condition instructs their alignment.

Together, the objective and disposition conditions are the substrate for the goal-inference analyses of §9: an observer given only the trajectory (not the prompt) is asked to recover the latent objective, and the divergence between instructed and inferred goals quantifies how legibly the agent's behaviour expresses its instructions. The disposition contrast sharpens this into an explicitly *alignment-relevant* measurement: comparing subgoal-inference performance between the helpful and deceptive conditions measures (i) whether the model is *capable* of strategically shaping its trajectory to defeat intent-recovery — successful deception should degrade the inference algorithms' localisation of the coin toward baseline while full-success rates remain matched — and (ii) which inference algorithms are robust to such concealment. Secondary behavioural signatures of attempted concealment (longer paths, reduced phase-1 action accuracy toward the coin, lower coin-detour saliency in visit statistics) are captured by the standard metric battery of §8 without modification.

### 4.3 Reasoning-effort manipulation

For reasoning-capable models, the API-level `reasoning_effort` parameter ("low", "medium", "high") bounds the length of the model's private chain-of-thought before it emits the JSON answer. Trajectory-collection commands accept a *list* of effort levels and generate a full set of trajectories per level on identical grids, enabling within-grid comparisons of behavioural quality as a function of test-time compute. For models exposing a numeric thinking budget instead (e.g. Qwen3 variants), a fixed token budget (2,048 reasoning tokens) is substituted.

---

## 5. Inference pipeline and response parsing

### 5.1 API layer

All model calls are issued through LiteLLM, which provides a uniform chat-completions interface across providers (the experiments principally used Together AI-hosted open-weight models, with `openai/gpt-oss-20b` as the primary model). Sampling parameters are held constant per run — temperature 0.7, top-p 0.95, a per-step generation cap of 10,000 tokens (4,096 for providers with lower limits), and a fixed seed passed to the API — and are recorded in every output file. Token log-probabilities are requested for every generated token together with the top-*k* alternatives at each position (default *k* = 5; up to 20 in the agent-level interface).

Requests are wrapped in retry logic (up to five attempts with randomised exponential backoff between 5 and 120 s) to absorb rate limits and transient provider failures. Per-call cost in USD is computed via LiteLLM's pricing tables and accumulated per agent. For batch runs, an optional token-bucket rate limiter enforces a global request budget (default 1,000 requests per 300 s) shared across worker threads, and the thread-pool width is automatically reduced to the sustainable request rate when the limiter is active.

### 5.2 Response parsing and validation

The model's raw completion is converted to an environment action through a tolerant, multi-stage procedure:

1. **Reasoning-tag stripping.** For models that emit visible reasoning (e.g. Qwen variants that prepend text terminated by `</think>`), the reasoning segment is split off and stored as reasoning content; only the remainder is parsed.
2. **JSON extraction and repair.** Trailing commas before closing braces are removed, and if the completion contains prose before the answer, the *last balanced JSON object* in the string is located by brace matching and validated with a JSON parse before being substituted for the full response.
3. **Schema validation.** The candidate JSON is validated against a Pydantic model (`ActionResponse`, or `ActionWithNoteResponse` in the note-taking condition). A field validator canonicalises the `action` value, accepting the enum name in any case ("RIGHT"), the integer code (1), or a numeric string ("1"), and mapping all forms onto the `Action` enum {LEFT = 0, RIGHT = 1, UP = 2, DOWN = 3}.
4. **Retry on failure.** If parsing or validation fails, the entire step (prompt included) is retried up to three times with randomised exponential backoff (1–10 s).
5. **Invalid-action handling.** If no valid action can be recovered, the step is recorded with action code −1 and the environment is *not* stepped: the step is logged (consuming one unit of the step budget) and the loop proceeds from the unchanged state. Unparseable outputs therefore appear explicitly in the data rather than being silently dropped, and their frequency is itself a behavioural measure of format compliance.

In the token-level collection path used by the main experiments, structured-output enforcement at the API level is deliberately disabled (the JSON schema is *described* in the prompt but not enforced by constrained decoding), so that the recorded log-probabilities reflect the model's unconstrained output distribution; format adherence is then handled entirely by the parsing stage above.

### 5.3 Log-probability capture

For every generated token, the serialised record includes the token string, token id, its log-probability, and the top-*k* alternative tokens with their log-probabilities. Two provider formats (content-level logprob objects and legacy top-logprobs lists) are normalised into a single schema. At save time, per-token alternative distributions are converted to probabilities (rounded to four decimal places) and attached to the aligned output tokens. Because the answer token that names the action (e.g. " RIGHT") is identified by annotation (§6.4), this yields, for every environment state visited, an (approximately) full model policy distribution over the four actions — the basis of the uncertainty and calibration analyses in §8.3.

---

## 6. Trajectory generation protocol

### 6.1 Step loop

A trajectory is generated by the loop: render observation → construct prompt → query model → parse action → step environment → record. Each recorded `Step` contains the pre-step text observation, the parsed action (name and integer code), the reward, the info flags (`coin_collected` / `coins_collected`), the carrying state (key-door variants), the full serialised log-probabilities, per-call and cumulative cost, and, when enabled, the running history. The loop terminates on goal contact (`terminated`), on the environment step limit (`truncated`), or on exhaustion of the per-trajectory step budget.

### 6.2 Step budgets

The per-trajectory step budget is either a fixed constant (default 50) or, when `enable_dynamic_max_steps` is set, a per-episode budget of ⌈1.5 · *L*\*⌉, where *L*\* is the A\*-optimal path length for that episode (computed by rolling out the A\* oracle on a deep copy of the environment before the LLM run). The dynamic budget equalises the *relative* slack afforded to the agent across environments of different scale: every episode permits 50% more steps than optimal, so timeout-based failure has a consistent meaning across grid sizes and complexities.

### 6.3 Repeated trajectories, seeding, and parallelism

To measure behavioural variability, multiple trajectories are collected on each *fixed* grid instance. Grid layouts are generated once per (size, complexity, model, grid-index) configuration from seed *s* + grid-index; each trajectory on that grid then runs on an independent deep copy of the environment, restored to the initial state by a `safe_reset` that repositions the agent without regenerating the maze. Per-trajectory sampling seeds are offset deterministically (grid seed + 1,000 · trajectory-index + a hash offset per condition), so that trajectories are non-identical samples from the model's action distribution while remaining reproducible. Collection is parallelised at two levels — across grids and across trajectories within a grid — using thread pools (default cap 32 workers), with the rate limiter of §5.1 shared globally.

### 6.4 Recorded output format

Each trajectory is saved as a single JSON file with four top-level sections:

- **`grid_params`** — grid dimensions, complexity, transform type (§7), full-observability flag, A\* distance, start and goal coordinates, coin position and placement mode, target and realised dead-end counts, and the symbol legend.
- **`model_params`** — model id, provider, template name, sampling parameters (temperature, top-p, max tokens), reasoning effort, top-logprobs *k*, step budget, and seed.
- **`prompt`** — the full prompt template passed through the model's own chat template, tokenised with the corresponding HuggingFace tokenizer and split at the observation placeholder into prefix tokens, placeholder tokens, and suffix tokens, each annotated with group labels (e.g. `template`, `prompt`, `placeholder`).
- **`steps`** — for each step: the grid state (as rendered text and as an annotated token list, with grid-symbol tokens tagged `grid_tile`), the parsed action, coin-collection flags, the model's raw output text, and the output tokens annotated with (i) structural groups for the model's output format — for the GPT-OSS "harmony" format, special tokens and channel markers are tagged `template`, analysis-channel tokens `analysis`, final-channel tokens `final`, and directional words `action` — and (ii) the per-token top-*k* probability distributions of §5.3.

This schema is designed for token-position-aligned mechanistic analysis: given any step, one can locate the grid tokens, the reasoning tokens, and the action token, and read off the model's probability distribution at each.

---

## 7. Environment transformations

### 7.1 Iso-difficulty transformations

To test whether behaviour is invariant to task-irrelevant presentation changes, a family of *iso-difficulty* transformations maps a grid instance to a variant with provably identical difficulty (identical optimal path length and route multiplicity):

- **Rotation** (`RotateEnv`): 90° counter-clockwise rotation of the grid, with agent/goal coordinates and headings remapped accordingly.
- **Reflection** (`ReflectEnv`): mirror reflection about the vertical axis.
- **Transposition** (`TransposeEnv`): exchange of the row and column axes.
- **Start–goal swap** (`StartGoalSwap`): exchange of the agent start and goal cells (the coin, if present, is unmoved), reversing the direction of the optimal route while preserving its length.

Each transformation operates on a cloned environment, rewriting the wall matrix and consistently updating every stored position attribute (current, user-specified, and initial agent positions; goal positions) and direction attribute. Batch collection commands can generate matched trajectory sets on the base grid and all requested transforms in a single run, with the transform name recorded in `grid_params.transform_type` for stratified analysis. Under the hypothesis that the model has a coherent spatial policy, behavioural metrics should be statistically indistinguishable across transforms of the same base grid; systematic asymmetries (e.g. better performance when the route runs left-to-right) indicate representational biases in how the model reads the text grid.

Additional transformations are implemented for perturbation experiments, though excluded from the default iso-difficulty set because they alter (or risk altering) difficulty: **dead-end removal** (removes an internal wall at a dead end, accepting the edit only if a recomputed A\* path length is unchanged), **goal displacement** (relocates the goal to a specified or random open cell), **dynamic obstacles** (adds or removes walls, optionally mid-trajectory), and **reward-structure modification** (wraps the step function with an arbitrary reward transform).

### 7.2 Wall reshuffling (structure permutation)

A complementary manipulation, `reshuffle_walls_from_layouts`, holds the *task* fixed while permuting the *maze structure*: for each saved layout, new mazes are repeatedly generated at the same size and complexity until one is found in which the original start, goal, and coin cells are all open (up to 200 attempts; a last-resort fallback clears any conflicting anchor cells and re-walls an equal number of random non-anchor cells so that the total wall count — and hence density — is preserved). A breadth-first-search check ensures all anchors remain mutually reachable. The result is a set of grids identical in size, wall density, and landmark coordinates but differing in internal geometry, isolating the effect of maze structure from that of the task configuration.

### 7.3 Layout augmentation and random starts

The `augment_from_layouts` command applies any subset of the §7.1 transformations to previously saved layouts and additionally supports **random starts**: up to *n* alternative agent start cells are sampled uniformly from open interior cells verified (by BFS from the goal) to be connected to the goal, with the walls, goal, and coin unchanged. Each variant is saved as a first-class layout and populated with its own trajectory set, expanding a base grid into a family of controlled variants.

---

## 8. Behavioural evaluation

Evaluation operates on the saved trajectory and layout files and proceeds at three levels of aggregation: per trajectory, per grid (pooling repeated trajectories on the same grid × condition), and per state (pooling all visits to a given cell).

### 8.1 Outcome (success) metrics

For the coin task, each trajectory is scored on:

- `reached_goal` — whether the agent terminated on the goal cell;
- `coin_collected` — whether the coin was collected at any point, with `coin_collection_step` recording when;
- **full success** — coin collected *and* goal subsequently reached, the conjunction demanded by the coin-directed instruction.

Grid-level rates are computed over the repeated trajectories: `coin_collected_rate`, `goal_after_coin_rate` (= `full_success_rate`), and, diagnostically, `goal_only_rate` — the fraction of trajectories that reached the goal *without* collecting the coin. The latter dissociates navigation failure from instruction-following failure: a high goal-only rate indicates an agent that is competent at navigation but ignores (or discounts) the instructed subgoal.

### 8.2 Capability and efficiency metrics

Optimality references are computed from the layout by shortest-path search (backward Dijkstra/BFS from each target over the wall graph):

- **Phase-decomposed distances**: `start_to_coin_distance`, `coin_to_goal_distance`, their sum `start_to_goal_via_coin_distance` (the instructed-optimal length *L*\*), and the direct `start_to_goal_distance` ignoring the coin.
- **Coin detour distance**: the minimum shortest-path distance from the coin cell to any cell on an optimal direct start→goal path — a measure of how far off the direct route the coin lies (0 if the coin is on the direct path). This is the principal difficulty covariate for instruction-following: it quantifies the cost of compliance.

Behavioural quality is then measured by:

- **Action accuracy**: the fraction of steps whose chosen action lies in the *optimal action set* of the current state — the set of moves that strictly decrease shortest-path distance to the current target. The coin task is treated as two phases with distinct targets: phase 1 (steps up to and including coin collection) is scored against optimal actions toward the *coin*; phase 2 (steps after collection) against optimal actions toward the *goal*. Reported as `action_accuracy_phase1`, `action_accuracy_phase2`, and a step-weighted overall accuracy. The phase decomposition scores the agent against the instructed objective rather than the environmental reward.
- **SPL** (Success weighted by Path Length; Anderson et al., 2018): 𝟙[full success] · *L*\*/max(*L*\*, *L*), where *L* is the realised trajectory length and *L*\* the instructed-optimal length via the coin. SPL jointly penalises failure and inefficiency and is averaged across trajectories for grid-level reporting.
- **Trajectory length** and total step counts.

### 8.3 Uncertainty and calibration metrics

Two complementary action distributions are analysed per visited state: the *empirical* distribution (frequencies of actions actually chosen, pooled across repeated trajectories) and, where log-probabilities are used, the *model policy* distribution read directly from the top-*k* token probabilities at the action position (§5.3). These are compared against the *optimal* distribution, defined as uniform over the optimal action set of the state.

- **Shannon entropy** *H* of the action distribution (bits; maximum 2 bits over four actions) — behavioural stochasticity at a state.
- **Optimal entropy** log₂ *k*\*, where *k*\* is the number of equally optimal actions — the irreducible ambiguity of the optimal policy at that state.
- **Divergence from optimal**: cross-entropy and KL divergence from the uniform-optimal distribution to the model distribution, and the bounded **Jensen–Shannon divergence** (0 = identical to optimal, 1 = maximally divergent), reported per phase (`mean_jsd_phase1` vs. `mean_jsd_phase2`) and overall.
- **Calibration**: the *calibration error* mean |*H*<sub>model</sub> − *H*<sub>opt</sub>| and signed *calibration bias* mean (*H*<sub>model</sub> − *H*<sub>opt</sub>) (positive = under-confident relative to task ambiguity), together with the correlation between model and optimal entropies across states.
- **Expected Calibration Error (ECE)**: with per-step confidence defined as 1 − *H*/2, the standard binned gap between confidence and empirical optimality rate.
- **Uncertainty–accuracy coupling**: the entropy gap between incorrect and correct steps, the **AUROC** of entropy as a predictor of suboptimal action choice ("does the model know when it doesn't know?"), and **selective-prediction curves** (accuracy vs. coverage when abstaining above an entropy threshold).

### 8.4 Movement-pattern metrics

Fine-grained locomotion statistics are computed per trajectory (and averaged per grid): absolute action counts (`num_actions_up/down/left/right`); egocentric continuation counts relative to the previous heading (`front`, `left_turn`, `right_turn`, `back`), which quantify momentum and reversal tendencies; and redundancy counts — `cell_revisits` (steps landing on any previously visited cell), `immediate_revisits` (A→B→A double-backs, excluding blocked no-ops), and `coin_oscillations` (repeated passes through the coin cell, including post-collection oscillation). These metrics characterise *how* the agent moves independently of *whether* it succeeds, and discriminate qualitatively distinct failure modes (dithering, looping, wall-hugging) that success rates conflate.

### 8.5 Statistical analysis

Relationships between behavioural measures and difficulty covariates (grid size, wall density, distances, detour cost, reasoning effort) are estimated with a controlled-analysis battery: raw Pearson correlations; *within-stratum* correlations (computed separately within each size × density stratum and combined by sample-size weighting, controlling for compositional confounds); *partial* correlations via OLS residualisation on the control covariates; and multiple OLS regression with heteroscedasticity-naïve *t*-based *p*-values and adjusted *R*². Grid-level metrics are additionally summarised by size × density cell and by binned distance-to-goal (state-level metrics as a function of remaining distance).

---

## 9. Goal- and subgoal-inference algorithms

Beyond forward evaluation, the framework asks the inverse question: *given only the observed trajectories and the maze layout — not the prompt — can the agent's latent objective be recovered?* A self-contained algorithm suite treats the coin position as an unknown intermediate subgoal and attempts to infer it from behaviour, with the known terminal goal serving as background structure. Only the layout's wall matrix, start and goal positions, and the observed state–action sequences are supplied; the true coin position is used solely for scoring. Predictions are scored by the Manhattan distance between the inferred and true subgoal cell, in *individual* mode (one prediction per trajectory) and *pooled* mode (one prediction from all trajectories on a grid). An optional preprocessing flag discards wall-collision steps (steps in which the position did not change) before inference. Seven algorithms spanning distinct assumption classes are implemented; each returns a score over cells (walls, start, and goal masked) whose argmax is the predicted subgoal.

1. **Visit frequency (baseline).** Counts visits to each non-terminal cell across trajectories; the most-visited cell is the prediction. Assumption-free with respect to rationality, but inflated by revisiting and oscillation.
2. **Shannon surprise (Buidze et al., 2025).** At each step, a hand-crafted prior over the four actions is formed as the product of a *momentum* component favouring the current heading (parameters λ, γ) and a *goal-direction* component favouring Manhattan-distance reduction toward the terminal goal (decay α); parameters follow values fitted to human navigation. The surprise −log₂ *P*(observed action) is accumulated per cell; the highest-surprise cell — where behaviour most deviates from terminal-directed motion — is the prediction.
3. **Bayesian inverse planning (Baker et al., 2011).** For each candidate subgoal *c*, soft value iteration yields Boltzmann policies (rationality parameter β = 2) toward *c* and toward the terminal goal; each trajectory is split at its first visit to *c* and scored under the two policies respectively (likelihood −∞ if *c* is never visited). Log-likelihoods are summed across trajectories and softmax-normalised under a uniform prior to a posterior *P*(*c* | data); the maximum a posteriori cell is the prediction.
4. **BNIRL (Bayesian non-parametric IRL; Gibbs sampling).** All steps are pooled and stochastically assigned, via a Gibbs sampler (500 iterations, 100 burn-in), to one of two intention clusters — subgoal-seeking or terminal-seeking — with step likelihoods given by a softmax over Manhattan-distance reduction (temperature α = 3); the subgoal estimate is updated each sweep as the coordinate-wise median of positions assigned to the subgoal cluster, and the modal post-burn-in sample is the prediction.
5. **BIRL / PolicyWalk (Ramachandran & Amir, 2007).** Metropolis–Hastings MCMC (1,500 iterations) over tabular reward functions bounded in [−2.1, −0.1]: each proposal perturbs one state's reward by ±0.5, the optimal policy under the proposal is recomputed by policy iteration, and the observed state–action pairs are scored under a likelihood ∝ exp(α · Σ *Q*(s, a)) with α = 20 (near-optimal rationality). The non-terminal cell with the highest inferred reward is the prediction.
6. **Maximum causal entropy IRL (Ziebart, 2010).** Exponentiated-gradient ascent on a tabular reward (identity features, discount 0.99): each iteration runs a soft backward Bellman pass to obtain the max-causal-entropy policy, a forward pass to obtain expected state-visitation frequencies, and updates rewards by the difference between empirical and expected visitation until convergence. The highest-reward non-terminal cell is the prediction.
7. **T-REX (Brown et al., 2019).** Preference-based neural reward learning that requires no rationality model: trajectories are ranked by a quality proxy (wall-aware path efficiency plus low action entropy, with unsuccessful trajectories ranked strictly below successful ones); pairs of fixed-length clips respecting the ranking and a temporal-ordering constraint train a small MLP reward network (state encoded as normalised coordinates plus a coin flag) under the Bradley–Terry preference loss; the trained network is queried at every cell and the highest-reward cell is the prediction. Uniquely among the methods, T-REX exploits failed trajectories as informative negative evidence.

The algorithms deliberately span a spectrum of embedded rationality assumptions — from none (visit frequency, T-REX), through soft optimality (inverse planning, BNIRL, MaxEnt), to strict optimality (BIRL) — and differ in whether their likelihoods respect wall geometry (inverse planning, BIRL, MaxEnt, T-REX) or only Manhattan distance (surprise, BNIRL). Comparing their success on LLM-generated trajectories therefore doubles as an assessment of *which agent model best describes LLM behaviour*: the degree to which methods assuming near-optimal, geometry-aware planning outperform assumption-free baselines measures how far LLM behaviour conforms to classical rational-agent assumptions.

The inference suite is applied identically to trajectories from every instruction condition of §4.2, since it consumes only layouts and state–action sequences. Its condition-wise results carry distinct interpretations: under the coin-directed and helpful conditions, inference accuracy measures the *legibility* of instructed goal pursuit; under the coin-avoidant condition, the coin should be recoverable, if at all, only as a region of anomalously *low* visitation; and under the deceptive condition, the drop in inference accuracy relative to the matched helpful condition — at matched task success — is the operational measure of successful intent concealment.

---

## 10. Supervised fine-tuning

The final component evaluates whether goal-directed competence can be *instilled* by imitation of an optimal policy.

**Dataset generation.** The `generate_sft_dataset` command rolls out the two-phase oracle (`CoinAStarAgent`, §3.3) on freshly sampled coin environments drawn with the same rejection-sampling constraints as the evaluation environments (§1.4). The default corpus covers the Cartesian product of grid sizes {7, 9, 11} and complexities {0.0, 0.2, 0.4, 0.6}, with 100 episodes per combination (1,200 episodes) under fixed, per-episode-offset seeds. Each *step* of each oracle episode is emitted as one training example in chat-message JSONL format:

```json
{"messages": [
  {"role": "user",      "content": "<fully rendered prompt: instructions + legend + current grid>"},
  {"role": "assistant", "content": "{\"action\": \"RIGHT\"}"}
]}
```

The user message is rendered from the *same* Jinja2 template used at evaluation time (`grid_one_coin_control.j2`), so the training distribution over prompts is by construction identical to the inference distribution — the fine-tuned model is trained on exactly the input format, coordinate conventions, and output schema it will encounter, and the supervision target is the oracle's optimal action serialised in the required JSON schema. Because examples are single steps, the fine-tuning objective is pure behavioural cloning of the state-conditional optimal policy, with no credit assignment across steps.

**Training.** Fine-tuning is performed through the Together AI fine-tuning API (`scripts/finetune_coin_sft.py`): the JSONL corpus (optionally with a held-out validation split) is uploaded and schema-validated; a job is created with, by default, LoRA adaptation (rank 16) on `openai/gpt-oss-20b` for 3 epochs at learning rate 10⁻⁵; job status is polled to completion and a token/price estimate is retrieved beforehand. A dry-run mode validates the upload without training.

**Evaluation.** The resulting adapter-merged model is addressable through the same LiteLLM interface as any base model, so the full trajectory-collection and evaluation pipeline of §§5–8 is re-run unchanged on the fine-tuned model (the only accommodation being that the tokenizer of the *base* model is supplied explicitly for token-level annotation, as fine-tuned checkpoints are not published on the HuggingFace Hub). Comparison of pre- and post-fine-tuning metrics — success rates, phase-wise action accuracy, SPL, entropy, and calibration — on held-out grid configurations, including sizes and complexities outside the training range and iso-difficulty transforms of evaluation grids, measures both the acquisition of the instructed coin-then-goal policy and its generalisation beyond the training distribution.

---

## 11. Implementation and reproducibility notes

The framework is implemented in Python 3.12 as an installable package (`coinenv`) with a `tyro`-based command-line interface exposing all generation, augmentation, baseline, upload, and dataset commands; dependency management is via a pinned conda environment. All stochastic components (maze generation, placement sampling, per-trajectory model sampling) are governed by explicit seeds with deterministic offsets, and every output file records the complete generation and model configuration required to reproduce it. Trajectory and layout corpora are content-addressed by a systematic filename convention (model, size, complexity, grid index, condition, transform, effort, trajectory index) on which the analysis and inference tooling depends, and can be mirrored to the HuggingFace Hub for archival. Environment-state cloning (deep copies with renderer state excised) guarantees that parallel trajectory collection on a shared grid cannot leak state across runs, and the `safe_reset` mechanism guarantees that repeated trajectories on a grid start from bit-identical initial states.
