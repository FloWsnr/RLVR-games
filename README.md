# RLVR-physics

Trainer-agnostic executable verifier tasks for RLVR.

This repo is intentionally early-stage. The firm commitments are in
[SPEC.md](SPEC.md): immutable task instances, one authoritative task backbone,
scalar sessions, renderer peripherals, trainer-owned concurrency, executable
verifier logic, and useful trajectories. Exact APIs, dataclass fields, adapters,
and module layout are design bets until the first single-step and stateful task
prototypes prove them.

The first architecture probes are Reasoning Gym `countdown`, seeded 2048, and
chess tactics with `python-chess`, followed by a small physics numeric reasoning
task once the core is proven.

## Current Implementation

The first round now includes a shallow core API in `rlvr_physics.core`:
immutable `TaskInstance` payloads, public/privileged payload separation,
renderer content blocks, scalar session result dataclasses, append-only
trajectories, and Python task spec objects.

Implemented task probes live under `rlvr_physics.tasks.games`:

- `games.countdown.v1`: Reasoning Gym Countdown sampling, text and PNG
  renderers, AST/Fraction expression verification, and a single-step session.
- `games.2048.v1`: deterministic spawn-tape 2048, text and PNG renderers,
  invalid action handling, dense score rewards, and a stateful session.
- `games.chess_tactics.v1`: python-chess mate-in-one tactics, SAN/UCI move
  parsing, text and PNG renderers, and a single-step session.
