# Specification

## Mission

Build a trainer-agnostic RLVR task library for executable, verifiable
tasks. The core product is not a game framework and not a rollout scheduler.
It is a clean way to define scalar tasks that produce prompts or observations,
accept model outputs, verify them with executable logic, assign rewards, and
record trajectories.

The framework should work with large-scale RL training frameworks. Check out `references/RL-framework-research.md` for the current state of RL training frameworks and how this library can fit in.

The framework should support two RLVR shapes through one backbone:

- Single-step verifier tasks: prompt, completion, executable verifier, reward.
- Stateful tasks: canonical state, repeated submissions, transitions, terminal
  or truncation logic.

Potential tasks include:
- Single-step: arithmetic, question-answering, code generation with unit tests.
- Stateful: connect4, chess, tool use, browser tasks.
- Physics puzzles and simulations.

End-goal tasks:
Later on, the library will focus on physics and scientific reasoning tasks. However, initially we will start
with simpler tasks to build out the architecture and core abstractions.

This also means we need to enable:
- hidden state, e.g. games like minesweeper or partially observable tasks
- engine support, e.g. physics simulations or chess engines
- task setups, e.g. sampling of initial position for game2048 or rolling a dice in Yahhtzee


## Core ingredients

- Task specs that build fresh task sessions from yaml configs
- The task backbone doing the logic
- task renderers (text, images)


## Greenfield Principles

- `TaskSession` is the only trainer-facing runtime abstraction.
- A task session is scalar. One instance represents one rollout or completion.
- Canonical task payloads, canonical state, and executable verifiers are
  authoritative.
- Text, images, and tool outputs are observations over authoritative state, not
  the state itself.
- Trajectories are first-class for both single-step and stateful tasks.
- Public task metadata must be separated from privileged debug metadata.
- Do not optimize for backwards compatibility while the architecture is still
  being simplified.

## Core Runtime Contract

The target core API should be small:

```python
class TaskSession(Protocol):
    @property
    def task_instance_id(self) -> str: ...

    @property
    def turn(self) -> TaskTurn | None: ...

    @property
    def trajectory(self) -> TaskTrajectory: ...

    @property
    def episode_return(self) -> float: ...

    def reset(self, *, seed: int) -> TaskResetResult: ...

    def submit(self, output: str) -> TaskSubmissionResult: ...

    def close(self) -> None: ...
```

`reset(...)` starts one scalar task session. `turn` is the next model-facing
opportunity. `submit(...)` verifies one assistant output and either produces
another turn or ends the session.


## Core Data Types

### `TaskInstance`

Immutable identity and public task metadata shared by one or more sessions.
This is required for GRPO-style workloads where many completions solve the same
prompt or sampled task.

Required fields:

- `task_instance_id`
- `task_kind`
- `seed`
- `prompt_key`
- public metadata

### `TaskTurn`

One model-action opportunity.

Required fields:

- observation
- chat/messages payload
- action context

The messages are derived from the observation and action context. Renderers
should not know about chat formatting.

### `TaskSubmissionResult`

The result of checking one model output.

Required fields:

- assistant output
- raw verifier submission
- parsed output, if any
- `valid_submission`, meaning parseable/verifiable, not necessarily correct
- reward
- terminated/truncated flags
- next observation and turn, if any
- public info
- debug info

Wrong but well-formed answers should usually be valid with low reward.
Malformed outputs may be invalid or valid-with-penalty, but the policy must be
explicit per task.

### `TaskTrajectory`

Common interaction record for all task shapes.

It should record:

- task instance id
- initial turn
- reset metadata
- ordered submissions
- rewards
- terminal/truncation flags
- public info
- privileged debug info

Stateful tasks may attach additional transition details, but downstream tooling
should be able to consume the common trajectory shape without knowing the
domain.

## Task Families

### Single-Step Verifier Tasks

This is the highest-priority RLVR path.

Target shape:

```python
task = task_source.sample(seed=seed)
observation = prompt_renderer.render(task)
result = verifier.verify(task=task, completion=completion)
```

The session wrapper should expose this as:

1. reset samples or loads a task instance
2. reset returns one `TaskTurn`
3. submit parses and verifies the completion
4. submit returns reward and terminal result
5. trajectory records the prompt, completion, verifier metadata, and reward

### Stateful Tasks

Stateful tasks use the same `TaskSession` contract but may produce multiple
turns.

Target shape:

```python
session.reset(seed=seed)

while session.turn is not None:
    output = agent.act(session.turn.messages)
    result = session.submit(output)
```

Stateful implementations own canonical state, transition validation, optional
internal events, reward assignment, and terminal/truncation logic.

Games, tool workflows, browser tasks, and coding tasks are all stateful tasks
when intermediate interaction matters.

## Repository Shape

The greenfield target layout is:

```text
rlvr_games/core/
  session.py        # TaskSession, turns, results, trajectories
  verifier.py       # single-step verifier helpers
  stateful.py       # reusable stateful task helper
  messages.py       # observation to trainer messages
  specs.py          # neutral task specs to session factories

rlvr_games/tasks/
  arithmetic/
  connect4/
  ...

config/tasks/
  arithmetic/
  connect4/
```

All domains are tasks. Games should live under `rlvr_games/tasks/<domain>/`
once ported. A legacy `rlvr_games/games/` package may exist during migration,
but new architecture should not be designed around it.

## Task Specs

Task specs should build fresh task-session factories, not mutable sessions and
not environment-only objects.