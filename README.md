# RLVR-games
Teach LLMs to reason by playing games

## Core Idea

This project is an environment-first RLVR framework. The point is not to turn
games into static question/answer examples. The point is to train and evaluate
an agent through interaction with an executable verifier.

- The game engine is the source of truth for state, legal actions,
  transitions, terminal conditions, and reward inputs.
- Text and images are observations over canonical symbolic state, not the
  authoritative state themselves.
- Trajectories are first-class data. Each episode records observations, raw
  actions, parsed actions, rewards, terminal flags, and transition metadata.
- Games plug into one shared environment loop instead of each game inventing
  its own runner.

## Core Architecture

The architectural center is `TurnBasedEnv` in `rlvr_games/core/env.py`. Each
game composes that generic environment out of a small set of collaborators:

- `Scenario`: creates the initial canonical state for `reset(seed=...)`
- `GameBackend`: parses actions, checks legality, applies transitions, and
  decides terminality
- `Renderer`: turns canonical state into model-facing text and images
- `inspect_state_fn`: returns a debug-oriented state summary for the CLI and
  tooling
- `RewardFn`: scores accepted environment steps
- `AutoAdvancePolicy` (optional): applies internal verifier-backed moves such
  as opponent replies until control returns to the agent

The core loop is:

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
reward components.

## Architectural Boundaries

- `rlvr_games/core/` holds the reusable environment abstractions and trajectory
  machinery.
- `rlvr_games/games/<game>/` holds the actual game logic, rendering, scenarios,
  rewards, and factories.
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
uv run rlvr-games play 2048 --seed 0
```


## Programmatic Rollouts

The main in-process surface is the environment API. Every game follows the
same shape:

```python
observation, reset_info = env.reset(seed=0)

while not env.episode_finished:
    raw_action = agent.act(observation, context)
    step_result = env.step(raw_action)
    observation = step_result.observation

trajectory = env.trajectory
```

Game-specific factories, scenarios, renderers, and rewards live under
`rlvr_games/games/<game>/`. The important invariant is the same across games:
the engine-backed canonical state is authoritative, observations are derived
views, and the trajectory records the full verified interaction history.

Observations may contain both text and in-memory images, which makes the same
environment surface usable for text-only and multimodal training loops.

## Development

Run the test suite with:

```bash
uv run pytest
```
