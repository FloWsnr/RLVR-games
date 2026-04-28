# Repository Guidelines

## Mission

See [SPEC.md](/home/flwi01/coding/RLVR-physics/SPEC.md). This repository should
be a trainer-agnostic RLVR task library for executable, verifiable tasks built
around immutable task instances, authoritative task backbones, scalar sessions,
renderer peripherals, and deferred trainer integration surfaces.

## Design Rules

- Prefer the cleanest design, not the most layered one.
- Do not optimize for backwards compatibility while the architecture is being
  simplified.
- Lock invariants before implementation details. Keep exact APIs and layouts
  provisional until real tasks prove them.

## General Code Expectations

- Add focused pytest coverage.
- Prefer deterministic tests with explicit seeds.
- Run the full validation stack before finishing: format and lint
  (`uv run ruff check`, `uv run ruff format`), static type checking
  (`uv run pyright`), and tests (`uv run pytest`).
- Keep types explicit. Avoid optional/default parameters when they hide
  behavior.
- Do not use `from __future__ import annotations`.
- Write numpy-style docstrings for functions and classes.
- Keep task packages split by concern once a task grows beyond a small probe:
  specs, instance construction, authoritative backbones/verifiers, rewards,
  renderers, and sessions should live in separate modules behind a public
  package facade.
- Put cross-task implementation helpers in `rlvr_physics.tasks._shared`; promote
  helpers to `rlvr_physics.core` only when they are core payload/session
  invariants rather than task convenience code.
- If asked to use worktrees, create new worktrees in the `./worktrees/`
  directory. Name them descriptively.
- Update `README.md`, `SPEC.md`, and `AGENTS.md` when changing mission,
  architecture, task domains, or public API.
- Don't export every symbol in `__init__.py` files. Use `__init__.py` only for the public/user-facing API of a package.

## Git Hygiene

- Keep commits scoped to one behavioral change.
- If you used a git worktree for the task, remove the worktree and branch after
  merge.
