# Refactor Plan: Unified Verifier Task Backbone

## Goal

Refactor the framework so it has one fundamental executable-task backbone with
two first-class adapters:

- an environment adapter for multi-step `reset()`/`step()` tasks
- a single-step adapter for prompt/completion/verifier tasks

The purpose is not to abandon the existing game engine work. The purpose is to
make games one implementation of a broader scalar verifier-session runtime that
also supports high-throughput RLVR workloads where each sample is a prompt,
completion, and executable reward.

## Review-Driven Changes

An independent review of the first plan changed the sequencing in four
important ways:

- define the task/session/trajectory contracts before touching workflow code
- add task-native session types alongside current workflow classes instead of
  renaming workflow first
- make task-level trajectory semantics part of the first implementation slice
- prove the shape with one non-game verifier before generalizing specs,
  datasets, and public names

The review also identified one missing concept: `TaskInstance`. A scalar
session is one rollout/completion, but high-throughput RLVR often needs many
sampled completions for the same prompt. Those sessions need a shared stable
task identity without sharing mutable session state.

## Design Thesis

The shared primitive should be a scalar task session, not the current full
environment API.

Current environments include concepts that are correct for games but too
specific for prompt-only RLVR:

- legal action enumeration
- current canonical state inspection
- repeated observations after every step
- terminal and truncated episode boundaries
- auto-advanced internal transitions
- action context derived from trajectory step count

Those should remain available for multi-step environments, but they should not
be required for every task. The deeper contract should be:

1. create or reset one scalar task session
2. expose the next model-facing turn, if one exists
3. accept one assistant output
4. verify it against executable task logic
5. return reward, metadata, completion state, and possibly another turn
6. record the interaction in a trajectory

This keeps the environment layer reusable while avoiding fake Gym-style wrappers
around single-step verifier tasks.

## Target Shape

```text
TaskSession backbone
        |
        +-- EnvironmentTaskSession adapter
        |       wraps existing TurnBasedEnv / Environment implementations
        |
        +-- SingleStepVerifierSession adapter
                wraps task sources, prompt renderers, and verifiers
```

The trainer-facing layer should depend on `TaskSession`, not directly on
`TurnBasedEnv`.

The multi-step environment API should remain valuable and stable. It becomes
the stateful domain adapter rather than the only runtime contract.

## Core Concepts

### `TaskInstance`

Represents the immutable task identity and task inputs shared by one or more
scalar sessions.

Suggested fields:

- `task_instance_id: str`
- `task_kind: str`
- `seed: int`
- `prompt_key: str | None`
- `metadata: dict[str, object]`

Why:

- Public RLVR recipes often sample many completions for the same prompt.
- One session should not be reused across those completions because sessions are
  mutable.
- A stable task instance lets rollout code group N completions for one prompt
  while still running one scalar session per completion.

### `TaskTurn`

Represents one model-action opportunity.

Suggested fields:

- `observation: Observation`
- `messages: tuple[ChatMessage, ...]`
- `action_context: ActionContext`

Why:

- This already exists conceptually as `WorkflowTurn`.
- It works for both a game move and a single prompt.
- Trainers need a ready-to-send prompt/message package, not internal engine
  state.

### `TaskResetResult`

Represents the start of one scalar task session.

Suggested fields:

- `task_instance_id: str`
- `turn: TaskTurn | None`
- `info: dict[str, object]`

Why:

- Multi-step environments may terminate during reset due to reset-time events
  or auto-advance.
- Single-step verifier tasks normally produce exactly one initial turn.
- The reset result should expose stable task identity without requiring direct
  access to environment state.

### `TaskSubmissionResult`

Represents the result of submitting one assistant output.

Suggested fields:

- `assistant_output: str`
- `raw_submission: str`
- `parsed_output: object | None`
- `valid_submission: bool`
- `reward: float`
- `terminated: bool`
- `truncated: bool`
- `turn: TaskTurn | None`
- `info: dict[str, object]`
- `debug_info: dict[str, object]`

Suggested computed field:

- `done: bool`

Why:

- Current `WorkflowSubmission` carries a `StepResult`, which assumes an
  environment step.
- Single-step verifier tasks need the same trainer-facing information without
  pretending to have a next environment observation.
- `valid_submission` should mean the submission was parseable and verifiable
  for the task, not that the answer was correct.
- A wrong but well-formed single-step answer should usually be valid with low
  reward, not an invalid action.
- `terminated` and `truncated` preserve the distinction between natural task
  completion and external cutoff or verifier failure.

### `TaskTrajectory`

Represents public-safe and debug interaction history at the task-session level.

Suggested fields:

- `task_instance_id: str`
- `initial_turn: TaskTurn | None`
- `reset_info: dict[str, object]`
- `debug_reset_info: dict[str, object]`
- `submissions: list[TaskSubmissionRecord]`

`TaskSubmissionRecord` should capture:

- assistant output and raw submission
- parsed or verified output when available
- reward
- valid-submission flag
- terminated and truncated flags
- public info
- debug info
- optional environment-specific details such as `StepResult` summaries or
  backend transitions

Why:

- Trajectories are central to the project mission and should not be postponed.
- `EpisodeTrajectory` remains useful for environment internals, but
  single-step verifier tasks should not have to pretend their completion is an
  environment transition.
- Downstream analysis needs one common task-level shape even when
  environment-specific transition details differ.

### `TaskSessionProtocol`

Suggested contract:

```python
class TaskSessionProtocol(Protocol):
    @property
    def done(self) -> bool: ...

    @property
    def task_instance_id(self) -> str: ...

    @property
    def turn(self) -> TaskTurn | None: ...

    @property
    def trajectory(self) -> TaskTrajectory: ...

    @property
    def episode_return(self) -> float: ...

    def reset(self, *, seed: int) -> TaskResetResult: ...

    def submit(self, assistant_output: str) -> TaskSubmissionResult: ...

    def close(self) -> None: ...
```

Why:

- This is the minimal shared interaction lifecycle.
- It preserves scalar sessions: one session is one logical rollout/completion.
- Rollout controllers and trainers can own fan-out, batching, queueing,
  cancellation, and freshness policy.
- Trajectory and task identity are first-class instead of added later.

## Phase 0: Lock Shared Contracts

### Tasks

- Define precise lifecycle semantics before code churn:
  - access before reset
  - submit before reset
  - submit after done
  - reset after done
  - reset while active
  - close idempotency
  - reset failure behavior
- Define `TaskInstance`, `TaskTurn`, `TaskResetResult`,
  `TaskSubmissionResult`, `TaskTrajectory`, and `TaskSessionProtocol`.
- Define `valid_submission` semantics separately from answer correctness.
- Define how `terminated`, `truncated`, and `done` map across both adapters.
- Define how multiple scalar sessions can share one `task_instance_id`.
- Define public `info` versus privileged `debug_info` privacy rules.

### Why

The first plan moved too quickly into renaming workflow types. The core
contracts need to be explicit first because they determine every later layer:
environment adapters, single-step verifiers, async workers, task specs, and
trajectory storage.

This phase should be short but strict. The output is a small set of dataclasses,
protocols, and tests that pin down semantics.

### Acceptance Criteria

- Core session types exist in a task-native module such as
  `rlvr_games/core/session.py`.
- Unit tests cover lifecycle behavior and result invariants.
- No current game implementation needs to change in this phase.

## Phase 1: Add Task-Native Session Types Alongside Workflow

### Tasks

- Add task-native session/result/trajectory types without renaming existing
  `Workflow*` classes yet.
- Keep current `WorkflowSession`, `WorkflowSubmission`, and `StepResult`
  behavior intact while the new API is proven.
- Add conversion helpers only where they reduce duplication.
- Export task-native types from `rlvr_games.core` and the package root.
- Update focused tests for the new types.

### Why

Replacing `WorkflowSubmission.step_result` immediately would ripple through
existing workflow and async code before the new adapters are proven. Adding
task-native types in parallel is safer and keeps the current environment stack
working.

Broad renames should happen late, after both adapters exist and tests prove the
shared contract.

### Acceptance Criteria

- Existing workflow tests continue to pass unchanged.
- New session-type tests pass.
- Trainer-facing code can import task-native types without importing
  `StepResult` or `Environment`.

## Phase 2: Add `EnvironmentTaskSession`

### Tasks

- Implement an adapter that wraps `Environment[Any, Any]`.
- On reset, call `env.reset(seed=seed)` and convert the observation into a
  `TaskTurn` with the existing rollout/message helpers.
- On submit, call `env.step(raw_action)` and map the resulting `StepResult`
  into `TaskSubmissionResult`.
- Map `StepResult.accepted` to `valid_submission` for environment-backed
  sessions.
- Preserve `terminated` and `truncated` exactly from `StepResult`.
- Populate `TaskTrajectory` while retaining a link or optional detail payload
  for the underlying `EpisodeTrajectory`.
- Keep access to the wrapped environment available for debugging and CLI tools.

### Why

This is how all existing games continue to work. The game engine remains the
authoritative source of state, legal actions, transitions, terminality, and
rewards. The new session adapter simply presents that engine through the more
general scalar task contract.

This also establishes the intended layering:

- `TurnBasedEnv` owns stateful episode mechanics.
- `EnvironmentTaskSession` owns trainer-facing session packaging.
- Trainers and rollout controllers use sessions.

### Acceptance Criteria

- All existing game tests pass without changing game behavior.
- Environment-backed sessions expose task-level trajectories.
- The CLI can still run directly against environments.
- New tests prove an environment-backed session can be driven through
  `TaskSessionProtocol`.

## Phase 3: Add Single-Step Verifier Primitives

### Tasks

- Add a `TaskSource` protocol for deterministic task-instance sampling.
- Add a prompt renderer or reuse `Renderer` where the input is a sampled task
  instance.
- Add a `SingleStepVerifier` protocol with a method like
  `verify(task, completion) -> VerificationResult`.
- Add `VerificationResult` with parsed output, valid-submission flag, reward,
  terminated/truncated status, public info, and debug metadata.
- Implement `SingleStepVerifierSession`.
- Add focused tests for reset, submit, double-submit behavior, deterministic
  seeds, reward metadata, debug privacy, and trajectory recording.

### Why

This proves the new mission. Without a real prompt-only verifier path, the
architecture will remain game-shaped despite the new docs.

The single-step path should be intentionally boring:

1. sample or receive a task instance
2. render prompt/messages
3. accept completion
4. verify completion
5. finish

It should not expose fake legal actions, fake multi-turn state, or fake
auto-advance. If those concepts are needed, the task belongs in the environment
adapter instead.

### Acceptance Criteria

- A single-step session can run without constructing `TurnBasedEnv`.
- A submission always produces `done=True` unless verifier failure is modeled
  as truncation.
- A wrong but well-formed answer is valid with low reward.
- A malformed answer is invalid or valid-with-penalty according to explicit
  verifier policy.
- The result includes reward, parsed output when available, public verifier
  metadata, and debug metadata.
- The task can be driven through the same session API as games.

## Phase 4: Add A Minimal Non-Game Reference Task

### Recommended First Task

Add a small procedural arithmetic or symbolic reasoning verifier.

Example:

- sample two integers and an operation
- render a prompt
- parse the final answer from the completion
- reward exact correctness
- record verifier metadata

### Why

This is the lowest-risk way to prove the architecture. Browser, coding, and
tool workflows are more strategically important long term, but they introduce
sandboxing and dependency complexity too early.

A small deterministic verifier will quickly reveal whether the new backbone is
actually game-independent.

### Acceptance Criteria

- The task runs through `TaskSessionProtocol`.
- Multiple scalar sessions can share the same `task_instance_id` and receive
  different completions.
- The task has tests for parsing, verification, rewards, deterministic seeds,
  lifecycle errors, debug privacy, and trajectory recording.

## Phase 5: Refactor Rollout Helpers Around Sessions

### Tasks

- Add generic rollout helpers that accept `TaskSessionProtocol`.
- Keep `prepare_turn(env=...)`, `build_action_context(env=...)`,
  `legal_actions()`, and canonical state inspection as environment-specific
  utilities.
- Move generic message packaging to task-turn construction.
- Make action extraction a session-level concern, not an environment-only
  concern.
- Add tests showing the same rollout loop can drive:
  - a wrapped game environment
  - a single-step verifier session

### Why

The rollout layer is where trainers will integrate. It should not need to know
whether the underlying task is a game, a math verifier, a code checker, or a
tool workflow.

Environment-specific affordances should remain available, but they should be
optional capabilities rather than assumptions.

### Acceptance Criteria

- There is one simple session rollout loop that handles both task shapes.
- Environment-specific utilities still exist for CLI/debug tooling.
- No generic trainer-facing helper requires `env.state`, `env.legal_actions()`,
  `StepResult`, or `Environment`.

## Phase 6: Refactor Async Execution From Env Pool To Session Pool

### Tasks

- Introduce `AsyncSessionPool` that owns one scalar task session per worker.
- Generalize worker commands from `reset`/`step` to `reset`/`submit`.
- Generalize result types from `AsyncResetResult`/`AsyncStepResult` to
  task-session reset/submission results.
- Keep `AsyncEnvPool` as a compatibility wrapper that converts env factories
  into `EnvironmentTaskSession` factories if useful.
- Ensure worker processes close sessions cleanly.
- Add tests for worker exceptions, picklability, lease behavior, and clean
  worker close.

### Why

The research-backed boundary is: task sessions stay scalar; rollout controllers
own concurrency. A process pool is a rollout/controller concern, not a property
of any individual environment.

Generalizing the async layer prevents single-step verifier tasks from needing
their own parallelization path and keeps batching/fan-out outside the verifier
implementation.

### Acceptance Criteria

- Existing async environment tests pass through compatibility wrappers.
- A single-step verifier task can run in the async pool.
- Worker result payloads do not expose environment-only `StepResult` as the
  generic result.

## Phase 7: Generalize Task Specs And Registry

### Tasks

- Refactor registry handlers to build task session factories, not mutable
  sessions.
- Replace schema dispatch based only on `game` with a neutral task key such as
  `kind`, `task_type`, or `domain`.
- Decide whether game specs keep a `game:` field under a migration bridge or
  move directly to the neutral key.
- Keep environment construction functions for CLI/debug paths.
- Add config examples for the first single-step verifier task under a new
  config namespace such as `config/tasks/<domain>/`.
- Add tests for legacy game specs and new neutral non-game specs.

### Why

The current task-spec registry is game-first. That was correct for the old
mission, but it will block prompt-only verifier tasks and future coding/tool
domains.

The registry should answer: “Given this task spec, what scalar task-session
factory should be used?” A factory is important because scalar sessions are
mutable and must not be reused across rollouts or workers.

### Acceptance Criteria

- Existing game YAML specs still load or have a deliberate migration.
- New single-step verifier specs load through the same top-level loader.
- Registry names and errors no longer assume every task is a game.
- Loader APIs that build trainer-facing runtimes return session factories.

## Phase 8: Generalize Dataset Utilities Only When Needed

### Tasks

- Delay broad dataset renaming until a non-game task actually uses
  record-backed sampling.
- When needed, rename or factor `ParquetScenarioDataset` into a more general
  task-record sampler.
- Keep deterministic split/seed sampling behavior.
- Support records that initialize environments and records that represent
  single-step prompts.
- Update chess dataset code to use the generalized sampler through a
  compatibility alias if needed.

### Why

Dataset-first RLVR is a major use case, but generalizing the dataset layer
before a non-game record-backed task exists would be speculative. A procedural
arithmetic reference task may not need Parquet at all.

The dataset layer should be generalized when a real task demonstrates the
needed shape.

### Acceptance Criteria

- Existing chess dataset tests pass.
- A record-backed non-game task can sample records through the same dataset
  layer.
- Dataset manifests remain explicit and reproducible.

## Phase 9: Naming And Public API Cleanup

### Tasks

- Update top-level package docstrings and exports away from game-only language.
- Consider renaming:
  - `GameBackend` to `StatefulTaskBackend` or keep it game-specific
  - `AsyncEnvPool` to `AsyncSessionPool`
  - `WorkflowSession` to `TaskSession`
  - `PreparedTurn` to `TaskTurn`
  - `PLAY_GAME_SPECS` to an environment CLI registry name
- Keep old names as aliases only when migration cost is high.
- Update docs and examples after the code path is proven.

### Why

Naming cleanup should come after behavior is proven. Renaming first creates
large diffs without reducing architectural risk.

Once both adapters are working, names should communicate the actual design:
games are one reference domain, and scalar verifier sessions are the backbone.

### Acceptance Criteria

- Public exports describe executable task sessions, not only game envs.
- README, SPEC, AGENTS, and examples match the implemented architecture.
- Existing users have clear migration paths or intentional breaking changes.

## Non-Goals During This Refactor

- Do not rewrite all games.
- Do not remove `TurnBasedEnv`.
- Do not make single-step verifier tasks pretend to have legal actions.
- Do not put batching inside environments or verifiers.
- Do not generalize the dataset layer before a real non-game record-backed task
  needs it.
- Do not implement browser, code sandboxing, or tool-resource servers before
  the basic session abstraction is proven.

## Key Risks

### Over-Generalizing Too Early

Risk:

The framework could gain abstract interfaces that no real task uses.

Mitigation:

Introduce only the minimal session/result/trajectory types needed to support
existing games plus one single-step verifier task.

### Breaking The Existing Game Stack

Risk:

The refactor could destabilize working games.

Mitigation:

Keep `TurnBasedEnv`, `StepResult`, workflow classes, and game-specific
factories intact until the adapter layer is fully tested.

### Leaking Environment Concepts Into Single-Step Tasks

Risk:

Single-step tasks could be forced to implement fake state, legal actions, or
multi-turn observations.

Mitigation:

Make the universal contract `reset()` plus `submit()`, not `reset()` plus
`step()`.

### Losing Multi-Completion Grouping

Risk:

The framework could make each sampled completion resample the prompt, which
would fight GRPO-style workloads that need many completions per prompt.

Mitigation:

Make `TaskInstance` and `task_instance_id` first-class. Let rollout code create
many scalar sessions from the same immutable task instance.

### Splitting Trajectory Semantics Too Much

Risk:

Environment and verifier trajectories could become incompatible for downstream
analysis.

Mitigation:

Keep a shared task-level record shape for observations, assistant outputs,
rewards, termination/truncation, info, and debug metadata. Allow
environment-specific transition details as optional nested records.

### Debug Metadata Leakage

Risk:

Privileged verifier details could accidentally appear in model-facing messages
or public observation metadata.

Mitigation:

Add privacy tests proving `debug_info` never appears in `TaskTurn.messages`,
`Observation.metadata`, or public `info`.

## Required Tests

- Lifecycle tests for submit before reset, double submit after done, reset after
  done, reset while active, close idempotency, and reset failure behavior.
- Multi-completion tests for N scalar single-step sessions sharing the same
  `task_instance_id` but receiving different completions.
- Environment trajectory tests covering reset events, auto-advance, invalid
  actions, truncation, and optional environment-specific details.
- Single-step trajectory tests covering parse failure, wrong answer, correct
  answer, verifier exception, and verifier timeout/truncation.
- Privacy tests for public info versus debug metadata.
- Async tests for single-step sessions, worker exceptions, picklability of task
  turns, slot lease behavior, and clean worker close.
- Task-spec tests for legacy `game:` specs and new neutral non-game specs,
  including error messages that no longer say every unsupported task is an
  unsupported game.

## Suggested PR Order

1. Lock shared session, result, task-instance, and trajectory contracts.
2. Add task-native types alongside current workflow classes.
3. Implement `EnvironmentTaskSession` around existing environments.
4. Add single-step verifier protocols and `SingleStepVerifierSession`.
5. Add the first minimal non-game verifier task.
6. Refactor rollout helpers to target task sessions.
7. Refactor async env pool into async session pool.
8. Generalize task-spec registry to session factories.
9. Generalize dataset sampler naming and usage only when needed by a real
   record-backed non-game task.
10. Clean up public names and docs.

Each PR should keep tests passing and should preserve the scalar-session rule:
one runtime instance represents one logical task session; concurrency belongs
outside the task implementation.
