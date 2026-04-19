# Repository Guidelines

## Mission

See [SPEC.md](/home/flwi01/coding/RLVR-physics/SPEC.md). This repository should
be a trainer-agnostic RLVR task library for executable, verifiable tasks built
around immutable task instances, authoritative task backbones, scalar sessions,
renderer peripherals, and trainer adapters.

The initial task probes are Reasoning Gym `countdown`, seeded 2048, chess
tactics with `python-chess`, and then a small physics numeric reasoning task.

## Design Rules

- Prefer the cleanest design, not the most layered one.
- Do not optimize for backwards compatibility while the architecture is being
  simplified.
- Lock invariants before implementation details. Keep exact APIs and layouts
  provisional until real tasks prove them.

## Code Expectations

- Add focused pytest coverage.
- Prefer deterministic tests with explicit seeds.
- Run the full validation stack before finishing: format and lint
  (`uv run ruff check`, `uv run ruff format`), static type checking
  (`uv run pyright`), and tests (`uv run pytest`).
- Keep types explicit. Avoid optional/default parameters when they hide
  behavior.
- Do not use `from __future__ import annotations`.
- Write numpy-style docstrings for functions and classes.
- If asked to use worktrees, create new worktrees in the `./worktrees/`
  directory. Name them descriptively.
- Update `README.md`, `SPEC.md`, and `AGENTS.md` when changing mission,
  architecture, task domains, or public API.

## Git Hygiene

- Keep commits scoped to one behavioral change.
- If you used a git worktree for the task, remove the worktree and branch after
  merge.
