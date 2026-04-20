# RLVR-physics

Trainer-agnostic executable verifier tasks for RLVR.

This repo is intentionally early-stage. The firm commitments are in
[SPEC.md](SPEC.md): immutable task instances, one authoritative task backbone,
scalar sessions, renderer peripherals, trainer-owned concurrency, executable
verifier logic, and useful trajectories. Exact APIs, dataclass fields, adapters,
and module layout are design bets until the first single-step and stateful task
prototypes prove them.

The first architecture probes are Reasoning Gym `countdown`, seeded 2048,
chess tactics with `python-chess`, and an interactive physics discovery task
family seeded from a small PhysGym-derived record subset.

## Current Implementation

The first round now includes a shallow core API in `rlvr_physics.core`:
immutable `TaskInstance` payloads, public/privileged payload separation,
renderer content blocks, scalar session result dataclasses, append-only
trajectories, a small task factory protocol, and Python task spec objects.

Implemented game probes live under `rlvr_physics.tasks.games`:

- `games.countdown.v1`: Reasoning Gym Countdown sampling, text and PNG
  renderers, AST/Fraction expression verification, and a single-step session.
- `games.2048.v1`: deterministic spawn-tape 2048, text and PNG renderers,
  invalid action handling, dense score rewards, and a stateful session.
- `games.chess_tactics.v1`: python-chess mate-in-one tactics, SAN/UCI move
  parsing, text and SVG image renderers, and a single-step session.

Implemented physics probes live under `rlvr_physics.tasks.physics`:

- `physics.discovery.v1`: interactive equation discovery over curated
  PhysGym-derived scalar laws, text observations, JSON experiment and
  hypothesis actions, L1-L4-style prior modes, hidden numeric verification, and
  experiment-cost rewards.

The first adapter helpers live under `rlvr_physics.adapters`:

- `datasets`: generic prompt rows, task instance registries, and scalar scoring.
- `multiturn`: shared scalar-session environment wrappers for step rewards.
- `trl`: Hugging Face Dataset conversion, reward callables, and tool-calling
  environments.

## Development

Install the full development environment with:

```bash
uv sync --group dev
```

The dev group includes the current trainer-facing dependency surface: TRL plus
its Hugging Face dataset, transformer, accelerate, and tool-response parsing
dependencies. OpenRLHF, verl, FlashAttention, and parquet-specific dependencies
are not part of the development environment while the core API is still being
simplified.
