# Repository Guidelines

## Mission

To understand more about the mission and goal of this library, see the SPEC.md file.


## Project Structure & Module Organization

`rlvr_games/core/` contains the shared environment abstractions: reset/step orchestration, protocols, rewards, trajectories, types, and exceptions. `rlvr_games/games/chess/` contains the current verifier-backed game implementation split across actions, backend rules, environment wiring, rendering, scenarios, and state. Add new games under `rlvr_games/games/<game>/` with the same separation. `tests/` holds pytest coverage for core behavior and game behavior. Repository metadata lives in `pyproject.toml`, `pyrightconfig.json`, and `uv.lock`; game fixtures or rendered assets should live near the game that owns them.

## Architecture Overview

This is an environment-first RLVR framework. The game engine is the source of truth for symbolic state, legal actions, transitions, terminal conditions, and rewards. Rendered text and images are observations over canonical state, not authoritative state. Trajectories should record observations, raw actions, parsed actions, rewards, terminal flags, and transition metadata.

## Required Coding Rules

- For large changes, always use git worktrees to isolate different features or bug fixes into separate branches. This keeps the commit history clean and makes it easier to review changes. Remove the worktrees when the feature or fix is merged to avoid clutter.
- Always! think about the optimal, cleanest way to implement a feature. Is the current code structure the best way to support this, or is there a more elegant design? Refactor if needed.
- Don't account for backwards compatibility if you are planning code changes. This is a research codebase, not a production library.
- Use pytest. Add focused tests for action parsing, illegal moves, reward logic, terminal and truncation behavior, renderer output, and trajectory recording. Prefer deterministic scenarios with explicit seeds.
- Use pyright for static type checking. Ensure all new code is fully typed and run pyright to verify.
- Use ruff for linting and formatting. Run ruff check and ruff format before submitting code.
- Avoid default or optional values in function signatures, if they are not strictly necessary. We want to be explicit about all parameters and not have any hidden state or behavior.
- Do not use `from __future__ import annotations`. This is a Python 3.10+ codebase, and we want to keep type annotations straightforward without string literals.
- Write numpy-style docstrings for all functions and classes, including parameters, return values, and raised errors when relevant.

## Build, Test, and Development Commands

- Make sure the uv environment is activated before running any commands.
- `uv sync`: install Python 3.13 dependencies and dev tools from the lockfile.
- `uv run pytest`: run the test suite configured under `tests/`.
- `uv run pyright`: run static type checking; required after changes.
- `uv run ruff check .`: lint the repository.
- `uv run ruff format .`: format Python files before submitting.

## Commit & Pull Request Guidelines

Keep commits scoped to one behavioral change. Check the commit skill to get more detailed instructions.
For merging branches, check out the merge skill for detailed instructions.
