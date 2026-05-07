# Repository Guidelines

## Mission and Architecture

Read [SPEC.md](/home/flwi01/coding/RLVR-physics/SPEC.md) to make sure you understand the mission and architecture.
This repository should be a trainer-agnostic RLVR task library for executable, verifiable tasks built
around immutable task instances, authoritative task backbones, scalar sessions,
renderer peripherals, and deferred trainer integration surfaces.

## Design Rules

- Prefer the cleanest design, not the most layered one.
- Do not optimize for backwards compatibility while the architecture is being
  simplified.
- Lock invariants before implementation details. Keep exact APIs and layouts
  provisional until real tasks prove them.

## Dos and Don'ts

- Run the full validation stack before finishing: format and lint
  (`uv run ruff check`, `uv run ruff format`), static type checking
  (`uv run pyright`), and tests (`uv run pytest`).

- Add focused pytest coverage.
- Prefer deterministic tests with explicit seeds.
- Keep types explicit. Avoid optional/default parameters when they hide
  behavior.
- Do not use `from __future__ import annotations`.
- Write numpy-style docstrings for functions and classes.
- If asked to use worktrees, create new worktrees in the `./worktrees/`
  directory. Name them descriptively, starting with `codex/`.
- Don't export every symbol in `__init__.py` files. Use `__init__.py` only for the public/user-facing API of a package.
- If you encounter changes you didn't do yourself, you can usually assume they are from another agent or user. Therefore, don't revert them. If you have questions about them, ask the other agent or user.

- Update `README.md`, `SPEC.md`, and `AGENTS.md` when changing mission,
  architecture, task domains, or public API. However, make sure to not duplicate information across these files. `SPEC.md` should be the source of truth for mission and architecture, while `README.md` should provide a high-level overview and quick start guide. `AGENTS.md` should focus on specific rules and guidelines for agents. Do not add specific architectural details to `AGENTS.md`.

## Git Hygiene

- Keep commits scoped to one behavioral change.
- If you used a git worktree for the task, remove the worktree and branch after
  merge.
