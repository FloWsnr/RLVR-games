# RLVR-games
Executable verifiers for RLVR, starting with games

## Core Idea

This project is a trainer-agnostic RLVR framework for executable, verifiable
tasks. A task may be a single-step prompt/completion/verifier workload or a
multi-step stateful environment. The current implementation is centered on
reusable game environments, but the architecture is intended to generalize to
procedural reasoning, coding, and tool-backed workflows.

- Canonical verifier-owned state is the source of truth for task state, legal
  actions, transitions, terminal conditions, and reward inputs.
- Text, images, and tool outputs are observations over canonical state, not
  the authoritative state themselves.
- Trajectories are first-class data. Each episode records observations, raw
  actions, parsed actions, rewards, terminal flags, public-safe metadata, and
  privileged debug traces over canonical state.
- Bundled games plug into one shared environment loop instead of each domain
  inventing its own runner.
- Games are reference environments, not the sole product identity.

## Direction

The framework direction is shaped by the current RLVR split between two task
contracts:

- single-step verifier tasks for high-throughput prompt to completion to reward
  workloads
- multi-step environments for stateful interaction, tool use, and long-horizon
  behavior

The environment or verifier should describe one logical task session. Rollout
controllers and trainers should own batching, queueing, async overlap, and
freshness policy. That keeps executable task logic reusable across trainer
stacks.

## Current Multi-Step Architecture

Today the architectural center is `TurnBasedEnv` in
`rlvr_games/core/env.py`, with trainer-facing workflow sessions layered on top
in `rlvr_games/core/workflow.py`. Bundled games compose that generic
environment out of a small set of collaborators:

- `Scenario`: creates the initial canonical state for `reset(seed=...)`
- `GameBackend`: parses actions, checks legality, applies transitions, and
  decides terminality
- `Renderer`: turns canonical state into model-facing text and images
- `inspect_canonical_state_fn`: returns a privileged canonical-state summary
  for debugging and tooling
- `RewardFn`: scores accepted environment steps
- `AgentContextProjector` (optional): projects selected public-safe structured
  context, such as opening events, into the agent-facing action context while
  the environment keeps ownership of generic fields like turn index
- `ObservationMessageAdapter`: converts rendered observations plus
  `ActionContext` into trainer-facing chat messages without making the
  renderer itself chat-specific
- `ResetEventPolicy` (optional): applies authoritative reset-time events such
  as dealer actions or chance outcomes before the first observation
- `AutoAdvancePolicy` (optional): applies internal verifier-backed moves such
  as opponent replies until control returns to the agent

The core multi-step loop is:

```python
observation, reset_info = env.reset(seed=seed)

while not env.episode_finished:
    raw_action = agent.act(observation)
    step_result = env.step(raw_action)
    observation = step_result.observation
```

Inside one `step(...)`, the environment does roughly this:

1. parse the raw action against the current canonical state
2. apply the accepted agent action with the game backend
3. optionally auto-advance internal moves until the agent can act again or the
   episode ends
4. score the verified step with the reward function
5. render the next observation
6. record the attempt in the trajectory

That split is deliberate. It keeps the generic episode lifecycle in one place,
while game-specific logic stays inside the backend, scenario, renderer, and
reward components. The longer-term direction is to give prompt-only verifier
tasks an equally first-class path without pushing batching semantics into the
task implementations themselves.

The intended agent-facing surface is the observation plus explicit structured
action context. Canonical inspection through `env.inspect_canonical_state()`
and exact move enumeration through `env.legal_actions()` remain available for
debugging, CLI tooling, and future action-masking experiments, but they are
not injected into the default observation. Any extra agent-visible setup
history should be exposed explicitly through `ActionContext`, not smuggled
through `reset_info` or renderer output.

The same split now applies to trajectory metadata: `reset_info`,
`trajectory.reset_events`, `TrajectoryStep.info`, and `RecordedTransition.info`
stay public-safe, while their `debug_*` counterparts retain privileged
canonical-state traces for offline debugging and analysis.

## Architectural Boundaries

- `rlvr_games/core/` holds the reusable environment abstractions and trajectory
  machinery, rollout helpers, trainer-facing message adapters, and async pool
  support.
- `rlvr_games/games/<game>/` holds the actual game logic, rendering, scenarios,
  rewards, and factories.
  Bundled games currently include chess, connect4, game2048, mastermind,
  minesweeper, and yahtzee.
- `rlvr_games/task_specs/` holds shared YAML task-spec loading, validation, and
  environment construction helpers.
- `config/games/<game>/` holds checked-in example task specs for reproducible
  environment setups.
- `rlvr-games` is a thin interactive play/debug shell over the environments.
- Dataset preprocessing and engine installation live in separate scripts rather
  than bloating the play CLI.

## Install

Sync the Python environment and install a local Stockfish binary for chess
engine-backed rewards:

```bash
uv sync
uv run rlvr-games-install-stockfish
```

The installer downloads the latest official Stockfish release from
https://stockfishchess.org/download/ and places the active binary under
`rlvr_games/games/chess/.stockfish/current/`. If you want to use a different
binary, set `RLVR_GAMES_STOCKFISH_PATH` or pass `--stockfish-path` to the chess
CLI.

Chess dataset preparation is exposed separately through
`uv run rlvr-games-chess-datasets ...`. More CLI coverage can live in a
dedicated CLI document later.

## Interactive Play

The CLI is mainly a thin manual-debugging shell over the environments. For
now, the README only keeps minimal smoke-test examples:

```bash
uv run rlvr-games play chess --seed 0 --reward engine-eval-dense --engine-depth 12 --engine-mate-score 100000
uv run rlvr-games play connect4 --seed 0
uv run rlvr-games play connect4 --seed 0 --reward solver-move-dense --opponent solver
uv run rlvr-games play 2048 --seed 0
uv run rlvr-games play mastermind --seed 0
uv run rlvr-games play minesweeper --seed 0
uv run rlvr-games play yahtzee --seed 0
uv run rlvr-games play connect4 --task-spec config/games/connect4/solver_opponent.yaml --seed 0
```

`--task-spec` lets the CLI load a fully authored environment configuration from
YAML. When a task spec is supplied, conflicting environment overrides such as
`--max-attempts`, `--image-size`, or `--invalid-action-policy` are rejected so
the authored setup stays reproducible.

## YAML Task Specs

Task specs make training and evaluation setups explicit, versioned, and easy to
reuse across the CLI and in-process rollouts. Checked-in examples live under
`config/games/<game>/`.

```yaml
schema_version: 1
id: connect4_solver_opponent
game: connect4

scenario:
  kind: random_position
  rows: 6
  columns: 7
  connect_length: 4
  min_start_moves: 0
  max_start_moves: 0

reward:
  kind: solver_move_dense
  perspective: mover

episode:
  max_attempts: 42
  max_transitions: 84

observation:
  include_images: false
  image_size: 360

control:
  auto_advance:
    kind: solver
```

Load one directly in Python:

```python
from pathlib import Path

from rlvr_games.task_specs import load_environment_from_task_spec_path

task_spec_path = Path("config/games/connect4/solver_opponent.yaml")
env = load_environment_from_task_spec_path(path=task_spec_path)
```

Task specs are validated with Pydantic before the environment is built, and any
relative paths inside the YAML are resolved relative to the task-spec file.
Today, checked-in task specs build game environments. The target architecture is
to have specs build scalar task-session factories, with environment construction
kept for multi-step CLI and debug paths.

## Programmatic Rollouts

The current in-process multi-step surface is still the environment API, but the
trainer-facing helpers now package action context and chat messages alongside
each actionable turn. The target trainer-facing surface is a scalar task session
that can wrap either this environment API or a single-step verifier:

```python
from pathlib import Path

from rlvr_games.core import prepare_turn
from rlvr_games.task_specs import load_environment_from_task_spec_path

env = load_environment_from_task_spec_path(
    path=Path("config/games/connect4/solver_opponent.yaml")
)
observation, reset_info = env.reset(seed=0)

while not env.episode_finished:
    turn = prepare_turn(env=env, observation=observation)
    raw_action = agent.act(
        messages=turn.messages,
        action_context=turn.action_context,
    )
    step_result = env.step(raw_action)
    observation = step_result.observation

trajectory = env.trajectory
```

`ActionContext` always includes the env-owned `turn_index`. Games may add
structured projected data such as `opening_events` through an
`AgentContextProjector`, but that projector only contributes the extra
agent-visible fields rather than constructing the full context itself. The
projector receives detached public reset-event snapshots rather than the full
debug trajectory.

Every bundled game factory installs a default `ObservationMessageAdapter` so
`prepare_turn(...)` and `env.messages_for_observation(...)` return structured
trainer-facing chat messages without baking chat formatting into the renderer.
If you need a different prompt surface, swap in a custom adapter or customize
`DefaultObservationMessagePolicy`.

Game-specific factories, scenarios, renderers, and rewards live under
`rlvr_games/games/<game>/`. The important invariant is the same across games:
the engine-backed canonical state is authoritative, observations are derived
views, and the trajectory records the full verified interaction history.

Observations may contain both text and in-memory images, which makes the same
environment surface usable for text-only and multimodal training loops.

The interactive CLI follows the same split: `state` and `show <key>` read from
observation metadata, while `debug-state`, `debug-show <key>`, and
`debug-legal` are explicit privileged debug commands.

## Async Rollouts

`AsyncEnvPool` provides a process-backed pool for parallel environment stepping.
Each worker owns one live environment and returns results as soon as they are
ready:

```python
from pathlib import Path

from rlvr_games.core.async_env import AsyncEnvPool

task_spec_path = Path("config/games/connect4/solver_opponent.yaml")

with AsyncEnvPool.from_task_spec_paths(
    task_spec_paths=(task_spec_path, task_spec_path),
) as pool:
    pool.reset_all(seeds=(0, 1))

    first_result = pool.recv(timeout_seconds=5.0)
    assert first_result.turn is not None

    pool.step(slot_id=first_result.slot_id, raw_action="4")
    next_result = pool.recv(timeout_seconds=5.0)
```

Reset and step results carry the worker `slot_id`, the per-slot
`episode_index`, the raw env result payload, and an optional `PreparedTurn`
containing the next action context plus trainer-facing messages.

If you want one trainer-facing interface that works the same way for both local
envs and async slots, use workflow sessions:

```python
from rlvr_games import AsyncEnvPool, WorkflowSession

env_session = WorkflowSession(env=env)

with AsyncEnvPool(env_factories=(make_env,)) as pool:
    async_session = pool.session(slot_id=0)

    async_session.reset(seed=0)
    while async_session.turn is not None:
        submission = async_session.submit("4")
        if submission.turn is None:
            break
```

`LocalWorkflowSession` and `AsyncWorkflowSession` share the same
`WorkflowSessionProtocol`: reset an episode, read the current `turn`, submit one
assistant output, and continue until `turn` becomes `None`. Async sessions
lease their pool slot exclusively while they are alive, so raw pool operations
and workflow-session control do not interleave on the same slot.

## Development

Run the full validation stack before finishing:

```bash
uv run ruff check
uv run ruff format
uv run pyright
uv run pytest
```

When you add a new feature or game, update the checked-in examples under
`config/games/` as needed and keep both `README.md` and `AGENTS.md` aligned
with the new user-facing or contributor-facing surfaces.

## License

This project is licensed under the GNU Affero General Public License v3.0 or later.
See [LICENSE](LICENSE).
