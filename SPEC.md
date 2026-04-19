# Specification

## Mission

Build a trainer-agnostic RLVR task library for executable, verifiable tasks.
The library should define scalar tasks that produce prompts or observations,
accept model outputs, verify them with executable logic, assign rewards, and
record useful trajectories.

The library should be easy to use from large-scale RL training frameworks. See
`references/RL-framework-research.md` for the current background research.

The long-term focus is physics and scientific reasoning tasks. Early tasks may
be simpler domains, such as arithmetic or small games, if they help prove the
core abstractions without adding heavy simulation complexity too early.

## Strong Invariants

These are the parts worth locking early:

- Tasks are scalar. One runtime instance represents one rollout or completion.
- Trainer and rollout frameworks own batching, queues, parallelism, model
  inference, and freshness policy.
- Canonical task payloads, canonical state, and executable verifiers are
  authoritative.
- Text, images, and tool outputs are observations over authoritative state, not
  authoritative state themselves.
- Public task metadata must stay separate from privileged debug metadata.
- Task trajectories should record enough verified interaction history for
  training, evaluation, and debugging.
- Backwards compatibility is not a priority while the architecture is still
  being discovered.

## Current Design Bets

These are working hypotheses, not permanent commitments:

- `TaskSession` is probably the only trainer-facing runtime abstraction.
- A session likely has `reset(seed=...)`, `turn`, `submit(output)`, and
  `trajectory`.
- Task specs should probably build fresh session factories from YAML configs.
- Renderers should probably produce observations, while message adapters turn
  observations into trainer-facing messages.
- Single-step verifier tasks and stateful multi-turn tasks should share one
  backbone.

These bets should be revised if the first real tasks show a simpler shape.

## Task Shapes

### Single-Step Verifier Tasks

Single-step tasks are prompt/completion/verifier workloads.

Examples:

- arithmetic
- question answering with executable checks
- code generation with unit tests
- short physics reasoning problems with computed answers

Expected lifecycle:

1. sample or load a task instance
2. render a prompt or observation
3. accept one model completion
4. parse and verify the completion
5. return reward, metadata, and trajectory record

### Stateful Tasks

Stateful tasks keep canonical state and may produce multiple turns.

Examples:

- physics puzzles and simulations
- partially observable tasks
- tool-use tasks
- small games used as architecture probes

Expected lifecycle:

1. sample or load an initial setup
2. expose the next model-facing turn
3. accept a model submission
4. validate and apply the transition or verifier step
5. return reward and either another turn or a terminal/truncated result

## Capabilities To Enable

The architecture should leave room for:

- hidden state and partial observability
- simulation engines and external verifiers
- deterministic seeded task setup
- stochastic reset events, such as dice rolls or randomized initial states
- text and image observations
- public metadata for trainers and privileged metadata for debugging
- multiple completions against the same immutable task instance

This list is directional. Do not build abstractions for every item before a
task needs them.

## Provisional Runtime Sketch

The current minimal sketch is:

```python
class TaskSession(Protocol):
    def reset(self, *, seed: int) -> TaskResetResult: ...

    @property
    def turn(self) -> TaskTurn | None: ...

    def submit(self, output: str) -> TaskSubmissionResult: ...

    @property
    def trajectory(self) -> TaskTrajectory: ...
```

Likely result concepts:

- task instance identity
- model-facing turn
- parsed submission
- valid-submission flag
- reward
- terminal or truncated status
- public info
- debug info
- trajectory record

The exact dataclass fields should stay flexible until at least one
single-step task and one stateful task are implemented cleanly.

## Provisional Repository Shape

The likely package layout is:

```text
rlvr_physics/core/
  session.py      # task-session protocol and common result types
  verifier.py     # helpers for single-step verifier tasks
  stateful.py     # helpers for stateful tasks, only if useful
  messages.py     # observation to trainer-facing messages
  specs.py        # task specs to session factories

rlvr_physics/tasks/
  arithmetic/
  physics_*/

config/tasks/
  arithmetic/
  physics_*/
```

This layout is a starting point. Keep it shallow until real tasks justify more
structure.

## Task Specs

Task specs should make task setup reproducible. The current bet is that specs
build fresh scalar task-session factories.

Likely rules:

- use neutral `kind:` dispatch
- keep examples under `config/tasks/<kind>/`
- avoid game-specific top-level schema concepts
- make seeds, task source, verifier, reward, renderer, and limits explicit when
  they affect reproducibility

## Open Questions

- What is the smallest useful `TaskSession` protocol after implementing one
  single-step task and one stateful task?
- Should trajectories be plain dataclasses, event logs, or both?
- How much message formatting belongs in core versus trainer adapters?
- Which physics task should be the first real target?
- Do stateful tasks need a reusable helper, or should each task own its loop?
- When do dataset utilities become necessary?

## Non-Goals For Now

- No in-repo async pools or rollout schedulers.
- No trainer-specific inference or optimization code.
- No Gym compatibility layer unless a real integration requires it.
- No broad dataset abstraction before a record-backed task needs it.
- No commitment to exact public field names before the first prototypes prove
  them.
