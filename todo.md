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
                wraps prompt sources, renderers, and executable verifiers
```

The trainer-facing layer should depend on `TaskSession`, not directly on
`TurnBasedEnv`.

The multi-step environment API should remain valuable and stable. It becomes
the stateful domain adapter rather than the only runtime contract.

## Core Types To Introduce

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

- `turn: TaskTurn | None`
- `info: dict[str, object]`

Why:

- Multi-step environments may terminate during reset due to reset-time events
  or auto-advance.
- Single-step verifier tasks normally produce exactly one initial turn.
- The reset result should not require direct access to environment state.

### `TaskSubmissionResult`

Represents the result of submitting one assistant output.

Suggested fields:

- `assistant_output: str`
- `reward: float`
- `done: bool`
- `turn: TaskTurn | None`
- `accepted: bool`
- `info: dict[str, object]`
- optional `debug_info: dict[str, object]`

Why:

- Current `WorkflowSubmission` carries a `StepResult`, which assumes an
  environment step.
- Single-step verifier tasks need the same trainer-facing information without
  pretending to have a next environment observation.
- This keeps trainer integrations independent of game-specific transition
  details.

### `TaskSessionProtocol`

Suggested contract:

```python
class TaskSessionProtocol(Protocol):
    @property
    def done(self) -> bool: ...

    @property
    def turn(self) -> TaskTurn | None: ...

    @property
    def episode_return(self) -> float: ...

    def reset(self, *, seed: int) -> TaskResetResult: ...

    def submit(self, assistant_output: str) -> TaskSubmissionResult: ...

    def close(self) -> None: ...
```

Why:

- This is the minimal shared interaction lifecycle.
- It preserves scalar sessions: one session is one logical rollout.
- Rollout controllers and trainers can own fan-out, batching, queueing,
  cancellation, and freshness policy.

## Phase 1: Promote Workflow Into The Backbone

### Tasks

- Rename or alias `WorkflowTurn` to `TaskTurn`.
- Rename or alias `WorkflowResetResult` to `TaskResetResult`.
- Replace `WorkflowSubmission.step_result: StepResult` with a neutral
  `TaskSubmissionResult` shape.
- Rename or alias `WorkflowSessionProtocol` to `TaskSessionProtocol`.
- Keep compatibility aliases temporarily if the migration would otherwise be
  noisy.
- Update tests under `tests/test_core/test_workflow.py`.

### Why

`rlvr_games/core/workflow.py` is already closest to the desired trainer-facing
surface. Refactoring this layer first avoids rewriting every game and keeps the
existing `TurnBasedEnv` implementation intact.

The important change is to stop leaking `StepResult` into the generic
submission result. `StepResult` is correct for environments, but it should not
define the universal task API.

### Acceptance Criteria

- Existing game workflows still run unchanged through the compatibility path.
- Trainer-facing code can consume `TaskSubmissionResult` without importing
  `StepResult`.
- `uv run pytest tests/test_core/test_workflow.py` passes.

## Phase 2: Add `EnvironmentTaskSession`

### Tasks

- Implement an adapter that wraps `Environment[Any, Any]`.
- On reset, call `env.reset(seed=seed)` and convert the observation into a
  `TaskTurn` with the existing rollout/message helpers.
- On submit, call `env.step(raw_action)` and map the resulting `StepResult`
  into `TaskSubmissionResult`.
- Keep access to the wrapped environment available for debugging and CLI tools.
- Preserve current `LocalWorkflowSession` behavior either by turning it into
  this adapter or by making it a thin alias.

### Why

This is how all existing games continue to work. The game engine remains the
authoritative source of state, legal actions, transitions, terminality, and
rewards. The new session adapter simply presents that engine through the more
general scalar task contract.

This also establishes the intended layering:

- `TurnBasedEnv` owns stateful episode mechanics.
- `EnvironmentTaskSession` owns trainer-facing packaging.
- Trainers and rollout controllers use sessions.

### Acceptance Criteria

- All existing game tests pass without changing game behavior.
- The CLI can still run against the wrapped environment.
- Existing async pool tests pass or have an explicit migration path.

## Phase 3: Add Single-Step Verifier Primitives

### Tasks

- Add a `SingleStepTaskSource` protocol for deterministic task sampling.
- Add a `SingleStepPromptRenderer` or reuse `Renderer` where the input is a
  sampled task object.
- Add a `SingleStepVerifier` protocol with a method like
  `verify(task, completion) -> VerificationResult`.
- Add `VerificationResult` with reward, accepted flag, info, and optional debug
  metadata.
- Implement `SingleStepVerifierSession`.
- Add focused tests for reset, submit, double-submit behavior, deterministic
  seeds, reward metadata, and trajectory recording.

### Why

This proves the new mission. Without a real prompt-only verifier path, the
architecture will remain game-shaped despite the new docs.

The single-step path should be intentionally boring:

1. sample task
2. render prompt
3. accept completion
4. verify completion
5. finish

It should not expose fake legal actions, fake multi-turn state, or fake
auto-advance. If those concepts are needed, the task belongs in the environment
adapter instead.

### Acceptance Criteria

- A single-step session can run without constructing `TurnBasedEnv`.
- A submission always produces `done=True`.
- The result includes reward and verifier metadata.
- The task can be driven through the same session API as games.

## Phase 4: Generalize Trajectories

### Tasks

- Introduce a task-level trajectory type if `EpisodeTrajectory` becomes awkward
  for single-step verifier tasks.
- Keep `EpisodeTrajectory` for multi-step environments if it remains useful.
- Ensure both task types record:
  - initial prompt or observation
  - assistant outputs
  - parsed or verified outputs when available
  - rewards
  - terminal status
  - public-safe metadata
  - privileged debug metadata
- Decide whether single-step verifier tasks use one `TrajectoryStep` or a new
  `VerificationRecord`.

### Why

Trajectories are central to the project mission. The framework should record
verified interaction history for both task shapes.

For environments, the existing trajectory model is rich and valuable because it
records transitions, reset events, invalid actions, and auto-advance. For
single-step verifier tasks, that same shape may be too env-specific. The first
single-step implementation should reveal whether a wrapper is sufficient or a
new task-level trajectory is cleaner.

### Acceptance Criteria

- Both adapters expose a trajectory through the session layer.
- Offline analysis can distinguish accepted, rejected, terminal, and debug
  metadata for both task types.
- Existing environment trajectory tests remain meaningful.

## Phase 5: Refactor Rollout Helpers Around Sessions

### Tasks

- Make rollout helpers accept `TaskSessionProtocol` where possible.
- Keep environment-specific helpers for legal action inspection and state
  projection.
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
- No generic trainer-facing helper requires `env.state` or
  `env.legal_actions()`.

## Phase 6: Refactor Async Execution From Env Pool To Session Pool

### Tasks

- Introduce `AsyncSessionPool` that owns one scalar task session per worker.
- Generalize worker commands from `reset`/`step` to `reset`/`submit`.
- Generalize result types from `AsyncResetResult`/`AsyncStepResult` to
  task-session reset/submission results.
- Keep `AsyncEnvPool` as a thin compatibility wrapper if useful.
- Ensure worker processes close sessions cleanly.

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

- Replace schema dispatch based only on `game` with a neutral task key such as
  `kind`, `task_type`, or `domain`.
- Decide whether game specs keep a `game:` field under a broader top-level
  shape or migrate directly.
- Refactor registry handlers to build task sessions or task session factories,
  not only environments.
- Keep environment construction functions for CLI/debug paths.
- Add config examples for the first single-step verifier task under a new
  config namespace.

### Why

The current task-spec registry is game-first. That was correct for the old
mission, but it will block prompt-only verifier tasks and future coding/tool
domains.

The registry should answer: “Given this task spec, what scalar task session
factory should be used?” Some specs will build environment-backed sessions.
Others will build single-step verifier sessions.

### Acceptance Criteria

- Existing game YAML specs still load or have a deliberate migration.
- New single-step verifier specs load through the same top-level loader.
- Registry names and errors no longer assume every task is a game.

## Phase 8: Generalize Dataset Utilities

### Tasks

- Rename or factor `ParquetScenarioDataset` into a more general task-record
  sampler.
- Keep deterministic split/seed sampling behavior.
- Support records that initialize environments and records that represent
  single-step prompts.
- Update chess dataset code to use the generalized sampler through a
  compatibility alias if needed.

### Why

Dataset-first RLVR is a major use case. The current Parquet code is useful, but
its naming and framing are scenario-specific.

A generalized sampler lets the framework support:

- chess puzzle scenarios
- math prompt rows
- coding task rows
- generated procedural task records

without duplicating storage and split logic.

### Acceptance Criteria

- Existing chess dataset tests pass.
- A non-game task can sample records through the same dataset layer.
- Dataset manifests remain explicit and reproducible.

## Phase 9: Add A Minimal Non-Game Reference Task

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
- The task runs through `AsyncSessionPool`.
- The task has a task-spec config.
- The task has tests for parsing, verification, rewards, deterministic seeds,
  and trajectory recording.

## Phase 10: Naming And Public API Cleanup

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
- Do not implement browser, code sandboxing, or tool-resource servers before
  the basic session abstraction is proven.

## Key Risks

### Over-Generalizing Too Early

Risk:

The framework could gain abstract interfaces that no real task uses.

Mitigation:

Introduce only the minimal session/result types needed to support existing
games plus one single-step verifier task.

### Breaking The Existing Game Stack

Risk:

The refactor could destabilize working games.

Mitigation:

Keep `TurnBasedEnv` and game-specific factories intact until the adapter layer
is fully tested.

### Leaking Environment Concepts Into Single-Step Tasks

Risk:

Single-step tasks could be forced to implement fake state, legal actions, or
multi-turn observations.

Mitigation:

Make the universal contract `reset()` plus `submit()`, not `reset()` plus
`step()`.

### Splitting Trajectory Semantics Too Much

Risk:

Environment and verifier trajectories could become incompatible for downstream
analysis.

Mitigation:

Keep a shared task-level record shape for observations, assistant outputs,
rewards, done flags, info, and debug metadata. Allow environment-specific
transition details as optional nested records.

## Suggested PR Order

1. Introduce neutral task session/result types and compatibility aliases.
2. Implement `EnvironmentTaskSession` around existing environments.
3. Update local workflow tests and public exports.
4. Add single-step verifier protocols and `SingleStepVerifierSession`.
5. Add the first minimal non-game verifier task.
6. Refactor rollout helpers to target task sessions.
7. Refactor async env pool into async session pool.
8. Generalize task-spec registry.
9. Generalize dataset sampler naming and usage.
10. Clean up public names and docs.

Each PR should keep tests passing and should preserve the scalar-session rule:
one runtime instance represents one logical task session; concurrency belongs
outside the task implementation.
