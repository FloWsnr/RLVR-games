# Specification

## Mission

Build a trainer-agnostic RLVR task library for executable, verifiable tasks.
The library should make it easy to create scalar task instances that produce
model-facing observations, accept model outputs or tool actions, verify behavior
with executable logic, assign rewards, and return useful metadata.

The long-term focus is physics and scientific reasoning. The core must also
work for math, coding, and logic puzzles, because those domains are cheap
proving grounds for the same abstractions: deterministic task generation, clear
observations, executable verification, repeatable rewards, and useful metadata.

The library is not a trainer. It should integrate cleanly with large-scale LLM
RL systems by exposing simple dataset, reward-function, environment, and HTTP
integration surfaces while keeping task state and verification independent of any
particular rollout engine.

## Research Inputs

The current ecosystem points to two dominant trainer-facing contracts:

- Dataset-style RLVR: prompt rows plus reward/verifier functions. This is the
  common path for math, code, and large-batch reasoning training.
- Environment-style RL: scalar sessions with `reset` and step-like interaction.
  This is the path for tool use, software tasks, browser/workplace agents, and
  other multi-turn domains.

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
  compatibility target for small interactive tasks.

Design implication: the core should describe one executable task episode, not a
batch. Future integration layers should translate that scalar core into each
trainer's preferred row, reward function, environment class, or server.

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
- Replay seeds are not public metadata by default. If a task exposes a source
  identity to trainers, that identity must be explicit and safe to reveal.
- Multiple model completions must be possible against the same immutable task
  instance without resampling hidden facts.
- Sessions must return enough public and debug metadata for training,
  evaluation, debugging, and future recorder wrappers.
- The core should prefer protocols and plain data over a deep base-class
  hierarchy.
- Backwards compatibility is not a priority while the architecture is still
  being discovered.

## Core Concepts

### Configured Task

A configured task is the trainer-facing bundle for one task family
configuration. It owns the public setup description and the callables needed to
produce scalar execution:

- a `TaskSpec` describing the configured task family
- an instance builder that creates immutable `TaskInstance` payloads from seeds
  or configured sources
- a session builder that creates fresh scalar `TaskSession` objects from
  immutable instances

Configured tasks are not rollout sessions and do not own batches. Trainer
integrations may use them to build or load instances, create one session per
completion or episode, and keep adapter code independent of concrete task
packages.

### Task Instance

A task instance is the immutable payload sampled from a generator or loaded from
records. It contains everything required to replay a task deterministically:

- stable task identity
- task kind and domain
- replay seed or source record identity
- public inputs that may be rendered to the model
- privileged verifier payload, such as exact answers, physical constants, or
  unit tests
- named rollout budget limits, such as turns, actions, or final answers, plus
  non-consumed hints such as timeout or token budget
- metadata used for curriculum, filtering, and trainer export

The instance is not a session. Trainers may request many completions for the
same instance. Each completion gets a fresh session initialized from the same
immutable payload.

Replay identities such as procedural generation seeds are for deterministic
reconstruction and debugging. They should not appear in the default public
instance view, because a seed may be enough to reconstruct privileged verifier
payloads. Tasks that need public row identities should publish an explicit
trainer-safe source id in metadata.

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
renderers, message converters, dataset exporters, and trainer integrations
should depend on it rather than duplicate task rules.

Backbones may be implemented as small classes, functions, or modules. The core
should require behavior, not inheritance from a single base class.

Task families may depend on shared domain backends when a domain has reusable
authoritative state and executable rules.

For reusable circuit backends, procedural generation should compose declared
motifs through typed port contracts over the canonical part/pin/net/connection
graph. Motifs, not loose generator-added parts, are the unit of diversity.
Generated circuits should contain a connected source-to-load or source-to-probe
path with structural validation before they become task instances. SPICE
netlist export is a deterministic adapter over canonical circuit truth; ngspice
simulation is a separate executable validation layer whose task-facing
parameters should be explicit in task specs or generated circuit payloads.
Generated circuit topology should declare external supply ports without baking in
applied source voltages. Operating conditions such as VCC or VEE voltages are
overlaid by the simulation layer. The initial shared simulation surface returns
operating-point voltages for every canonical circuit net; task-specific
measurement policies decide which of those values are public.

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
```

The exact names can change after prototypes, but the semantics should remain:

- `reset` starts a fresh rollout and returns the first model-facing turn plus
  trainer-safe and privileged reset metadata.
- `turn` exposes the current observation, expected submission mode, available
  tools or action schema, submission encoding format, and public limits.
- `submit` accepts either a final text completion, a parsed action, or a tool
  call payload and returns validity, reward, termination status, public info,
  debug info, and optionally the next turn.

Single-step verifier tasks are sessions with one turn and one final submission.
Stateful simulations and tool-use tasks use the same session contract over
multiple turns.

### Renderers

Renderers turn canonical task state and public metadata into observations:

- plain text prompts
- chat messages
- structured tool descriptions
- tool result messages
- images or multimodal content blocks
- compact state views for simulations or tool state

Image observations emitted to trainers should use PNG payloads. Task renderers
may construct SVG internally, but should rasterize it before creating
model-facing image content.

Renderers should be deterministic for a given state, renderer config, and seed.
They must not own verifier state. If a renderer hides information, that hidden
information must still live in the canonical instance or state.

Substantial model-facing prompt wording should live in task-local prompt files,
with renderers responsible for deterministic formatting, public state insertion,
and content block construction.

Turn renderers should not accumulate prior observations or actions. For
multi-turn tasks, they should render the current public state, current tool
result, or current feedback, while trainer integrations keep the conversation
transcript or rollout trace when a model needs previous turns.

Configured task builders may capture a selected renderer in the session builder;
task specs should still advertise the supported renderer set.

### Submissions

Submissions are model outputs as seen by the task core. The first prototypes
should support at least:

- raw text completion
- extracted final answer
- tool call name plus structured arguments
- invalid or unparsable submission record

Parsing may happen in an integration layer or in the backbone, but the verified
session result should preserve trainer-safe and debug metadata needed to
interpret what happened.

Interactive protocols should expose a canonical submission format separately
from the task action schema. The action schema describes what can be done; the
submission format describes how a model or player must encode it.

Invalid submission handling is task behavior, not only adapter behavior. A
session should make rejected-submission categories and their policy visible in
turn metadata when they affect retries, terminal state, truncation, rollout
budgets, or reward-policy decisions. Invalid-submission policies describe
control effects; scalar rewards for rejected submissions should be produced by
the task reward policy. Budget effects should use the same named public budget
namespace as valid actions, published under ``public_limits["budget_limits"]``
with names such as ``turns``, ``actions``, and task-specific final-answer or
tool budgets. Step metadata should report consumed and remaining budgets using
those same names. Local play surfaces may add transport conveniences around
stdin/stdout, but model-facing invalid submissions should be represented
explicitly by the session when they reach the task core.

### Reset And Step Results

Each reset should produce a result with these concepts:

- `session_id`: stable identifier for the scalar rollout
- `turn`: first model-facing turn
- `public_info`: trainer-safe identity and rollout metadata
- `debug_info`: privileged local reset metadata for evaluation and debugging

Each submission should produce a result with these concepts:

- `accepted`: whether the submission was well-formed enough to evaluate or
  apply
- `reward_result`: structured reward payload containing the scalar reward,
  optional score, and reward metadata
- `terminal`: whether the task ended successfully or unsuccessfully
- `truncated`: whether limits ended the task before natural termination
- `observation`: next model-facing turn when the task continues
- `public_info`: trainer-safe metadata
- `debug_info`: privileged metadata for local evaluation and debugging

Reward is the trainer-facing scalar. A task may also expose interpretable reward
features, but trainer integrations decide how much of that to surface.

Reward policies should return a shared reward result shape containing the scalar
reward, optional domain score, trainer-safe reward metadata, and privileged
debug metadata. Task-specific reward code may live beside a task backbone, but
the returned payload should stay consistent across task families.
For multi-turn tasks, the same task-specific reward policy should handle final
answers, accepted intermediate actions, especially good or bad actions,
invalid submissions, truncations, and other step events rather than embedding
scalar rewards in parser, submission-policy, or renderer code.

### Rollout Recording

The core session API does not own trajectory storage for now. Trainer
frameworks already own rollout traces, batching, token ids, log-probs, KL terms,
and advantage values. Future local evaluation or offline export code may add a
`TrajectoryRecorder` wrapper around `TaskSession`, but recorder concerns should
remain peripheral to the task truth and step contract.

### Public Interaction Play-tests

Every substantial task family should expose a reusable public play-test surface
without reimplementing process IO. The shared play interaction layer owns:

- the JSONL reset and step event protocol
- conversion of `TaskTurn`, `TaskResetResult`, and `TaskStepResult` into public
  JSON-compatible payloads
- public reward metadata from `RewardResult.public_info` on step events, while
  omitting privileged reward debug metadata
- one-submission-per-line stdin/stdout loops
- public-info metadata filtering for blind local play
- text and image observation serialization

Task packages should contribute only a small `PlayableTask` descriptor and any
task-specific parameter parsing needed to build a `ConfiguredTask`. The shared
runner must depend on the scalar `TaskSession` contract, not on task-specific
backbones, actions, or verifier internals.

Play-tests are not trainer integrations. They are local evaluation peripherals
for checking multi-turn interactions, collecting transcripts, estimating task
difficulty, and running fresh blind sessions through the same public surface a
model or benchmark driver would see.

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
5. return reward and metadata

This path should eventually export cleanly to generic prompt rows and trainer
reward functions. Trainer-specific reward surfaces should remain outside the
core API.

### Stateful Tasks

Stateful tasks keep canonical state and may produce multiple turns.

Examples:

- physics puzzles and simulations
- partially observable tasks
- tool-use tasks
- code-editing loops with test feedback

Expected lifecycle:

1. sample or load an immutable setup
2. initialize canonical runtime state
3. expose the next model-facing turn
4. accept a model submission or tool call
5. validate and apply a transition or verifier step
6. return reward and either another turn or a terminal/truncated result

This path should eventually export cleanly to trainer-owned environment
factories. Environment or service surfaces should remain outside the core until
real integrations prove their shape.

## Trainer Integration Requirements

Future trainer integrations should translate the scalar core into external
framework surfaces. They should be thin, testable, and disposable.

### Dataset And Reward Surfaces

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

The reward integration should reconstruct or look up the immutable task instance,
run the backbone verifier against each completion, and return floats plus
optional logging metrics.

Compatibility targets:

- generic prompt rows plus scalar scoring helpers
- trainer reward functions over prompts, completions, and row metadata
- future examples for reward services after the core API stabilizes
- offline SFT/DPO conversion from future recorder outputs or trainer-owned
  rollout records

### Environment Surfaces

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

- trainer-owned environment factories
- future examples for agent interaction and resource-server integrations after
  the core session contract settles

### Message Surfaces

Message converters translate rendered observations into framework-specific message
formats:

- Hugging Face chat-template-compatible message lists
- OpenAI Responses API content blocks for NeMo Gym-style rollouts
- plain prompt strings for simple dataset RLVR
- multimodal blocks for image observations
- tool schemas from typed task actions

The core should not require a tokenizer or chat template. Tokenization belongs
to the trainer.

### HTTP And Service Surfaces

Some trainers want reward or environment services. A service layer may expose
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

Task specs should make task setup reproducible and trainer-friendly. A YAML spec
should build a configured task for immutable task instances and scalar sessions.

Likely rules:

- use neutral `kind:` dispatch
- keep examples under `config/tasks/<kind>/`
- avoid domain-specific top-level schema concepts
- make source, generator, seed policy, renderer, verifier, reward, and rollout
  limit fields explicit when they affect reproducibility
- separate public prompt metadata from privileged verifier payload
- include trainer export hints only when they do not leak trainer-specific
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
  type: arithmetic_event_rewards
  parameters:
    correct_final_answer_reward: 1.0
    invalid_submission_reward: 0.0
budget_limits:
  turns: 1
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
  factory.py      # ConfiguredTask helper for instances and sessions
  instances.py    # immutable task instance types
  payloads.py     # payload freezing, plain-data conversion, and stable hashes
  rewards.py      # shared reward result types
  session.py      # TaskSession protocol and result dataclasses
  rendering.py    # observation/content abstractions
  specs.py        # YAML/task spec loading
  submissions.py  # submission envelopes, parsing, and invalid policies

rlvr_physics/play/
  cli.py          # generic play command entrypoint
  interaction.py  # JSONL protocol and public session loop
  registry.py     # registered playable task descriptors
  task.py         # PlayableTask descriptors and CLI helpers

rlvr_physics/tasks/
  _shared/    # cross-task helpers when repeated task code appears
  arithmetic/
  physics/
  coding/

config/tasks/
  arithmetic/
  physics/
  coding/
```

Trainer integrations should be added only when an example proves the need.

Individual task families may be packages once a single file starts mixing
instance construction, backbones, renderers, verifiers, rewards, and sessions.
Package `__init__.py` files should act as public facades for stable imports,
while internal modules keep authoritative logic separate from peripherals.
Cross-task reuse should go through `rlvr_physics.tasks._shared` only when the
helper is task-implementation support; reusable payload invariants and
validation belong in `rlvr_physics.core`.

## Prototype Expectations

Early task prototypes should validate the scalar core without committing the
project to broad abstractions or trainer-specific adapters. The first useful set
should include both a single-step verifier path and a stateful or tool-use path,
preferably in physics or scientific reasoning once the generic surfaces are
usable.

Each prototype should prove a small number of core behaviors:

- deterministic instance construction from explicit seeds or source records
- computed privileged ground truth separated from public observations
- renderer output derived from canonical state rather than duplicated task logic
- final-answer or tool-action submissions interpreted into structured payloads
- rollout budget limits represented directly on immutable instances and public
  turns as named budget maps
- reward features that explain exactness, invalid submissions, and truncation
- step results with public metadata and privileged debug details for local
  inspection


## Acceptance Criteria For The Core

The initial core is good enough when:

- the same immutable instance can produce several independent completions
- a single-step task and a multi-turn task share the same session result types
- canonical state is inspectable in debug mode but not leaked through public
  metadata
- a renderer can be swapped without changing verification
- step result metadata explains why a reward was assigned
- deterministic tests can replay a task from seed and instance payload
- dataset export can feed prompt/completion reward workflows
- environment export can feed reset/tool/step workflows
- adding a new task mostly requires a backbone, renderer, and spec, not trainer
  code changes

## Non-Goals For Now

- No in-repo trainer, optimizer, inference server, or rollout scheduler.
- No batching or vectorized environment abstraction in task core.
- No tokenizer, log-prob, or advantage logic in task core.
- No heavy HTTP service framework until an integration needs it.
- No Gym compatibility layer unless a real integration requires it.
- No first-class trainer-specific integration while the core API is still moving.
- No broad dataset abstraction before record-backed tasks need it.
- No deep base-class hierarchy for every task.
- No commitment to exact public field names before prototypes prove them.

## Open Questions

- What is the smallest stable `TaskSession` protocol after one single-step task
  and one stateful task are implemented?
- When should a peripheral `TrajectoryRecorder` be added, and should it emit
  plain dataclasses, typed event logs, JSONL records, or a combination?
- Which trainer integration should be implemented first after generic dataset
  export?
- How should task payloads be serialized for very large hidden state, images, or
  code sandboxes?
- How much tool schema generation belongs in core versus trainer integrations?
- Which physics task should be the next target after cart inference: projectile
  motion, mechanics constraints, or simulation-based puzzles?
- When do dataset utilities become necessary?

## Reference Notes

The local research summary is in
`references/RL-framework-research.md`.
