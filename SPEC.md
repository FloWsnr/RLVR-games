# Specification

## Mission

Build a trainer-agnostic RLVR task library for executable, verifiable tasks.
The library should make it easy to create scalar task instances that produce
model-facing observations, accept model outputs or tool actions, verify behavior
with executable logic, assign rewards, and record useful trajectories.

The long-term focus is physics and scientific reasoning. The core must also
work for math, coding, logic puzzles, and small games, because those domains are
cheap proving grounds for the same abstractions: deterministic task generation,
clear observations, executable verification, and repeatable rewards.

The library is not a trainer. It should integrate cleanly with large-scale LLM
RL systems by exposing simple dataset, reward-function, environment, and HTTP
adapter surfaces while keeping task state and verification independent of any
particular rollout engine.

## Research Inputs

The current ecosystem points to two dominant trainer-facing contracts:

- Dataset-style RLVR: prompt rows plus reward/verifier functions. This is the
  common path for math, code, and large-batch reasoning training.
- Environment-style RL: scalar sessions with `reset` and step-like interaction.
  This is the path for tool use, games, software tasks, browser/workplace
  agents, and other multi-turn domains.

The most relevant external API pressures are:

- TRL GRPO expects datasets with a `prompt` column, reward functions over
  prompts/completions and row metadata, and optionally an `environment_factory`
  that creates one environment per rollout. Public methods become tools, and
  reward functions can inspect per-rollout environment state.
- OpenRLHF separates single-turn rewards from multi-turn agent execution. Its
  multi-turn API expects `reset` and `step` methods that return observations,
  rewards, done flags, and optional environment feedback while the framework
  owns token-level trajectories, batching, and async policy.
- verl expects data to be exported into trainer-owned tabular formats such as
  parquet with prompt, data source, ability, reward model fields, and extra
  info. Custom reward functions map generated responses plus ground truth and
  extra info into scores. Newer agentic paths add async multi-turn tool loops.
- NeMo Gym splits an environment into agent, model, and resources servers. The
  resources side owns task state, tools, and verification; the agent side owns
  rollout orchestration; the trainer owns concurrency.
- OpenEnv and Gym-style systems still matter as a simple `reset`/`step`
  compatibility target, especially for games and small interactive tasks.

Design implication: the core should describe one executable task episode, not a
batch. Adapters should translate that scalar core into each trainer's preferred
row, reward function, environment class, or server.

## Strong Invariants

These are the parts worth locking early:

- Tasks are scalar. One runtime session represents one completion, rollout, or
  episode.
- Trainer and rollout frameworks own batching, queues, parallelism, model
  inference, weight freshness, token log-probs, and advantage construction.
- Authoritative task instances, canonical state, transition logic, and
  executable verifiers are the source of truth.
- Text, images, tool schemas, and tool outputs are observations over
  authoritative state, not authoritative state themselves.
- Public metadata must stay separate from privileged verification and debug
  metadata.
- Multiple model completions must be possible against the same immutable task
  instance without resampling hidden facts.
- Sessions must record enough verified interaction history for training,
  evaluation, debugging, and offline conversion.
- The core should prefer protocols and plain data over a deep base-class
  hierarchy.
- Backwards compatibility is not a priority while the architecture is still
  being discovered.

## Core Concepts

### Task Instance

A task instance is the immutable payload sampled from a generator or loaded from
records. It contains everything required to replay a task deterministically:

- stable task identity
- task kind and domain
- seed or source record identity
- public inputs that may be rendered to the model
- privileged verifier payload, such as exact answers, physical constants,
  hidden game state, or unit tests
- limits, such as maximum turns, timeout, token budget hint, or action budget
- metadata used for curriculum, filtering, and adapter export

The instance is not a session. Trainers may request many completions for the
same instance. Each completion gets a fresh session initialized from the same
immutable payload.

### Task Backbone

A task backbone is the authoritative executable implementation for one task
family. It owns the domain rules:

- initial canonical state construction
- parser or action decoder
- validity checks
- state transitions
- verifier execution
- reward computation or reward-feature production
- terminal and truncation conditions

The backbone should be the one place where task truth lives. Peripherals such as
renderers, message adapters, dataset exporters, and trainer integrations should
depend on it rather than duplicate task rules.

Backbones may be implemented as small classes, functions, or modules. The core
should require behavior, not inheritance from a single base class.

### Task Session

A task session is the scalar runtime wrapper around one task instance and one
backbone. It exposes the minimal trainer-facing interaction contract.

Current design target:

```python
class TaskSession(Protocol):
    def reset(self, *, seed: int) -> TaskResetResult: ...

    @property
    def turn(self) -> TaskTurn | None: ...

    def submit(self, submission: TaskSubmission) -> TaskStepResult: ...

    @property
    def trajectory(self) -> TaskTrajectory: ...
```

The exact names can change after prototypes, but the semantics should remain:

- `reset` starts a fresh rollout and returns the first model-facing turn.
- `turn` exposes the current observation, expected submission mode, available
  tools or action schema, and public limits.
- `submit` accepts either a final text completion, a parsed action, or a tool
  call payload and returns validity, reward, termination status, public info,
  debug info, and optionally the next turn.
- `trajectory` records the verified interaction history.

Single-step verifier tasks are sessions with one turn and one final submission.
Stateful games, simulations, and tool-use tasks use the same session contract
over multiple turns.

### Renderers

Renderers turn canonical task state and public metadata into observations:

- plain text prompts
- chat messages
- structured tool descriptions
- tool result messages
- images or multimodal content blocks
- compact state views for games or simulations

Renderers should be deterministic for a given state, renderer config, and seed.
They must not own verifier state. If a renderer hides information, that hidden
information must still live in the canonical instance or state.

### Submissions

Submissions are model outputs as seen by the task core. The first prototypes
should support at least:

- raw text completion
- extracted final answer
- tool call name plus structured arguments
- invalid or unparsable submission record

Parsing may happen in an adapter or in the backbone, but the verified trajectory
must record both the raw model output and the interpreted submission.

### Step Results

Each submission should produce a result with these concepts:

- `accepted`: whether the submission was well-formed enough to evaluate or
  apply
- `reward`: scalar reward for the step or episode
- `score`: optional domain score used for filtering or reporting
- `terminal`: whether the task ended successfully or unsuccessfully
- `truncated`: whether limits ended the task before natural termination
- `observation`: next model-facing turn when the task continues
- `public_info`: trainer-safe metadata
- `debug_info`: privileged metadata for local evaluation and debugging
- `events`: trajectory records emitted by this step

Reward is the trainer-facing scalar. A task may also expose interpretable reward
features, but trainer adapters decide how much of that to surface.

### Trajectories

Trajectories are append-only records of one rollout. They should support both
training export and debugging:

- reset event with task identity, seed, renderer, and limits
- observation events with public content hashes or payloads
- raw model output events
- parsed submission or tool call events
- validity and error events
- state transition summaries
- verifier events
- reward, score, terminal, and truncation events

Token ids, log-probs, KL terms, and advantage values are not core trajectory
fields. Trainer adapters may attach trainer-side trace identifiers so task
records can be joined with token-level rollout data later.

## Task Shapes

### Single-Step Verifier Tasks

Single-step tasks are prompt/completion/verifier workloads.

Examples:

- arithmetic
- symbolic or numeric math
- code generation with unit tests
- short physics reasoning problems with computed answers
- logic puzzles with exact solutions

Expected lifecycle:

1. sample or load an immutable task instance
2. render a prompt or chat observation
3. accept one model completion
4. parse and verify the completion
5. return reward, metadata, and a trajectory

This path must export cleanly to TRL, OpenRLHF single-turn reward functions, and
verl-style parquet plus custom reward functions.

### Stateful Tasks

Stateful tasks keep canonical state and may produce multiple turns.

Examples:

- physics puzzles and simulations
- partially observable tasks
- tool-use tasks
- code-editing loops with test feedback
- small games used as architecture probes

Expected lifecycle:

1. sample or load an immutable setup
2. initialize canonical runtime state
3. expose the next model-facing turn
4. accept a model submission or tool call
5. validate and apply a transition or verifier step
6. return reward and either another turn or a terminal/truncated result

This path must export cleanly to environment factories, OpenRLHF multi-turn
agent instances, OpenEnv/Gym wrappers, and NeMo Gym resources servers.

### Games As Probes

Games are not the main mission, but they are useful early test domains because
they force the core to handle state, invalid actions, partial observability,
turn limits, and trajectories without heavy simulation dependencies.

Good first game probes should be deterministic, small, and cheap:

- Nim or take-away games for exact strategy and terminal reward
- Mastermind-style hidden-state deduction with textual feedback
- grid or sliding puzzles with bounded action spaces
- tic-tac-toe only if the goal is interactive state handling, not novelty

The game backbone should be written with the same pattern expected for physics:
canonical state first, renderer second, trainer adapters last.

## Trainer Adapter Requirements

Adapters translate the scalar core into external framework surfaces. They should
be thin, testable, and disposable.

### Dataset And Reward Adapters

The dataset path should export immutable task instances as rows with:

- `id`
- `task_kind`
- `domain` or `ability`
- `prompt` or `messages`
- public metadata needed by the trainer
- opaque task payload pointer or serialized canonical payload
- reward model metadata, such as verifier style and public ground truth when
  safe
- `extra_info` for split, source index, difficulty, and curriculum tags

The reward adapter should reconstruct or look up the immutable task instance,
run the backbone verifier against each completion, and return floats plus
optional logging metrics.

Compatibility targets:

- TRL custom reward functions over prompts, completions, and row metadata
- OpenRLHF custom reward files or remote reward servers
- verl custom reward functions over data source, solution string, ground truth,
  and extra info
- offline SFT/DPO conversion from verified trajectories

### Environment Adapters

The environment path should wrap a `TaskSession` as a trainer-owned scalar
episode:

- construct one session per rollout
- call `reset` with dataset row fields and a deterministic seed
- expose task actions as meaningful tool methods when the trainer uses
  function calling
- route tool calls or text submissions into `submit`
- store final reward and public metrics for the trainer reward function
- preserve invalid action feedback as observations instead of crashing the
  rollout worker

Compatibility targets:

- TRL `environment_factory`
- OpenRLHF `AgentInstanceBase`-style `reset` and `step`
- OpenEnv or Gymnasium-style `reset` and `step`
- NeMo Gym resources server methods such as session seeding, task tools, and
  `verify`

### Message Adapters

Message adapters convert rendered observations into framework-specific message
formats:

- Hugging Face chat-template-compatible message lists
- OpenAI Responses API content blocks for NeMo Gym-style rollouts
- plain prompt strings for simple dataset RLVR
- multimodal blocks for image observations
- tool schemas from typed task actions

The core should not require a tokenizer or chat template. Tokenization belongs
to the trainer.

### HTTP And Service Adapters

Some trainers want reward or environment services. A service adapter may expose
the same core through HTTP, but service concerns must remain peripheral:

- process management
- authentication
- retries
- request batching
- rate limiting
- tracing
- resource cleanup

The underlying task result should remain the same as local execution.

## Task Specs

Task specs should make task setup reproducible and adapter-friendly. A YAML spec
should build a factory for immutable task instances and scalar sessions.

Likely rules:

- use neutral `kind:` dispatch
- keep examples under `config/tasks/<kind>/`
- avoid game-specific or physics-specific top-level schema concepts
- make source, generator, seed policy, renderer, verifier, reward, and limits
  explicit when they affect reproducibility
- separate public prompt metadata from privileged verifier payload
- include adapter export hints only when they do not leak trainer-specific
  behavior into the backbone

Provisional shape:

```yaml
kind: arithmetic.v1
domain: math
source:
  type: procedural
  seed: 17
renderer:
  type: text
verifier:
  type: exact_numeric
reward:
  correct: 1.0
  invalid: 0.0
limits:
  max_turns: 1
exports:
  dataset:
    ability: math
```

Specs are not the authoritative task logic. They configure task factories and
backbones.

## Provisional Repository Shape

Keep the package shallow until real tasks demand more structure:

```text
rlvr_physics/core/
  instances.py    # immutable task instance and payload types
  session.py      # TaskSession protocol and result dataclasses
  trajectory.py   # event log types and helpers
  rendering.py    # observation/content abstractions
  specs.py        # YAML/task spec loading

rlvr_physics/adapters/
  datasets.py     # prompt row and parquet-style export helpers
  trl.py          # TRL reward and environment adapters
  openrlhf.py     # OpenRLHF reward/agent adapters
  verl.py         # verl row/reward adapters
  nemo_gym.py     # NeMo Gym resources/server adapter helpers
  gymnasium.py    # optional reset/step wrapper

rlvr_physics/tasks/
  arithmetic/
  games/
  physics/
  coding/

config/tasks/
  arithmetic/
  games/
  physics/
  coding/
```

Adapters should be added only when an example proves the need. It is fine for
the first implementation to include only generic dataset export and one trainer
adapter.

## First Prototypes

The first implementation should prove the core with three focused tasks. These
are architecture probes, not permanent product scope.

1. Reasoning Gym `countdown`
   - single-step prompt/completion verifier task
   - external procedural dataset integration
   - seeded instance generation through Reasoning Gym metadata
   - multiple valid completions, because many arithmetic expressions can solve
     the same target
   - parser plus executable verifier rather than exact string matching
   - dataset row export with question, answer, metadata, source dataset,
     source index, and difficulty
   - reward adapter test for prompt/completion trainers

   This proves the dataset-style RLVR path and should be the first task
   implemented. The task payload should separate public numbers and target from
   privileged reference expression, source metadata, and verifier details.

2. Seeded 2048
   - single-player multi-step stateful task
   - canonical board, score, turn count, max tile, and spawn history
   - action vocabulary with four moves: up, down, left, right
   - invalid action handling when a move does not change the board
   - seeded stochastic transitions using an RNG stream or, preferably for early
     deterministic tests, a precomputed spawn tape
   - dense score rewards, optional sparse milestone rewards, terminal outcome
     when no legal moves remain, and truncation at a turn budget
   - text renderer first, with image renderer left as a later peripheral
   - environment adapter test for reset/step or tool-call trainers

   This proves the stateful session path. Initial goals should be small, such
   as reaching tile 64, reaching tile 128, surviving a fixed number of turns, or
   maximizing score over a fixed turn budget. Reaching the 2048 tile should not
   be required for early tests.

3. Chess tactics with `python-chess`
   - two-player-rules task using an external rules engine
   - start with mate-in-one before mate-in-two
   - public payload includes FEN, side to move, board rendering, and allowed
     notation
   - submissions may be SAN or UCI moves
   - verifier parses the move, checks legality, applies the move, and checks
     checkmate for mate-in-one
   - mate-in-two verifier should require that the first move has a legal mating
     continuation against every legal opponent reply
   - privileged payload includes puzzle id, solution move set or continuation
     table, mate depth, and source metadata
   - reward features should distinguish parse failure, illegal move, legal
     non-solution, and correct tactic

   This proves external-engine integration, legal action validation, two-player
   turn semantics, notation renderers, and adversarial verification without the
   long-horizon complexity of full chess self-play.

After these three tasks, add a first physics numeric reasoning task:

- deterministic generated parameters
- computed ground truth with tolerances and units
- public prompt and privileged solution separated
- reward features for exactness, units, and invalid parse
- likely first domains: projectile motion, simple circuits, or mechanics
  constraints

Coding verifier tasks should wait until the core needs sandbox or subprocess
boundaries. Do not add broad abstractions before the three initial tasks expose
repeated structure.

## Acceptance Criteria For The Core

The initial core is good enough when:

- the same immutable instance can produce several independent completions
- a single-step task and a multi-turn task share the same session result types
- canonical state is inspectable in debug mode but not leaked through public
  metadata
- a renderer can be swapped without changing verification
- trajectories explain why a reward was assigned
- deterministic tests can replay a task from seed and instance payload
- dataset export can feed prompt/completion reward workflows
- environment export can feed reset/tool/step workflows
- adding a new task mostly requires a backbone, renderer, and spec, not trainer
  code changes

## Non-Goals For Now

- No in-repo trainer, optimizer, inference server, or rollout scheduler.
- No batching or vectorized environment abstraction in task core.
- No tokenizer, log-prob, or advantage logic in task core.
- No heavy HTTP service framework until an adapter needs it.
- No Gym compatibility layer unless a real integration requires it.
- No broad dataset abstraction before record-backed tasks need it.
- No deep base-class hierarchy for every task.
- No commitment to exact public field names before prototypes prove them.

## Open Questions

- What is the smallest stable `TaskSession` protocol after one single-step task
  and one stateful task are implemented?
- Should trajectories be plain dataclasses, typed event logs, JSONL records, or
  a combination?
- Which trainer adapter should be implemented first after generic dataset
  export?
- How should task payloads be serialized for very large hidden state, images, or
  code sandboxes?
- How much tool schema generation belongs in core versus trainer adapters?
- Which physics task should be the first real target: projectile motion,
  circuits, mechanics constraints, or simulation-based puzzles?
- When do dataset utilities become necessary?

## Reference Notes

The local research summary is in
`references/RL-framework-research.md`.
