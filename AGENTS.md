# Repository Guidelines

## Mission

See [SPEC.md](/home/flwi01/coding/RLVR-games/SPEC.md). This repository is a
trainer-agnostic RLVR framework for executable, verifiable tasks. It currently
bundles game environments, but the intended surface includes both single-step
verifier tasks and multi-step environments.

## Core Idea

- Support both dominant RLVR contracts: prompt-only verifier tasks and
  stateful `reset()`/`step()` environments.
- Keep canonical task state and executable verification authoritative.
- Text, images, and tool outputs are observations over canonical state, not
  the authoritative state themselves.
- Trajectories are first-class and should record the verified interaction
  history across both single-step and multi-step tasks.
- Games are bundled reference domains, not the whole product identity.

## Architecture

- `rlvr_games/core/` holds the generic environment abstractions, trajectory
  machinery, rollout helpers, trainer-facing message adapters, async rollout
  helpers, rewards, protocols, and types. New trainer-facing runtime work
  should move toward scalar task-session factories rather than environment-only
  construction.
- `rlvr_games/games/<game>/` holds game-specific backend logic, scenarios,
  rendering, rewards, state types, and factory wiring.
  Bundled games currently include chess, connect4, game2048, mastermind,
  minesweeper, and yahtzee.
- `rlvr_games/datasets/` holds shared offline dataset utilities.
- `rlvr_games/task_specs/` holds shared YAML task-spec parsing, validation,
  registry, and environment-construction helpers.
- `config/games/<game>/` holds checked-in game task specs today. New non-game
  domains should use a neutral namespace such as `config/tasks/<domain>/`.
- `rlvr_games/cli/` is a thin interactive debug shell over the environments.
  Keep it small.
- Offline tooling such as dataset preparation or engine installation should
  live in dedicated scripts rather than bloating the play CLI.

## Design Rules

- Prefer the cleanest design, not the most layered one. Remove abstractions
  that do not carry their weight.
- Do not optimize for backwards compatibility. This is a research codebase.
- Keep canonical state and executable verifier logic authoritative.
- Design scalar task sessions. Batching, queueing, and freshness control belong
  in rollout or trainer layers, not inside each environment implementation.
- Keep domain-specific behavior out of the generic core unless multiple tasks
  clearly need it.
- Put trainer-facing chat formatting in observation message adapters and rollout
  helpers, not in renderers, scenarios, or backends.
- Prefer validated task specs for reproducible environment setups instead of
  growing ad hoc CLI-only configuration paths.
- Prefer abstractions that can support both prompt-only verifier tasks and
  multi-step environments.
- Add new games under `rlvr_games/games/<game>/` with the same separation of
  concerns as the existing games, but prioritize new domains when they stress
  reusable verifier/runtime boundaries better than another game would.
- Keep fixtures and rendered assets near the game that owns them.

## Code Expectations

- Add focused pytest coverage for parsing, illegal actions, rewards, terminal
  behavior, truncation behavior, rendering, and trajectory recording.
- Prefer deterministic tests with explicit seeds.
- When you add or change reusable environment setups, add or update example
  task specs under `config/games/<game>/`. For new non-game domains, add
  examples under a neutral namespace such as `config/tasks/<domain>/`.
- Run the full validation stack before finishing: format & lint (`uv run ruff check`, `uv run ruff format`), static type
  checking (`uv run pyright`), and tests (`uv run pytest`).
- Keep types explicit. Avoid unnecessary optional/default parameters when they
  hide behavior.
- Do not use `from __future__ import annotations`.
- Write numpy-style docstrings for functions and classes.
- Create new worktrees in the `./worktrees/` directory for each task. Name them descriptively.
- Make sure to update the `README.md` and `AGENTS.md` documentation when adding new features, games, or new task domains.

## Git Hygiene

- Keep commits scoped to one behavioral change.
- If you used a git worktree for the task, remove the worktree and branch after
  merge.
