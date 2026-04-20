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

The first adapter helpers live under `rlvr_physics.adapters`:

- `datasets`: generic prompt rows, task registries, and scalar reward scoring.
- `trl`: Hugging Face Dataset conversion and TRL-style reward callables.
- `verl`: verl parquet-style row conversion and reward callables.

## Development

Install the full development environment with:

```bash
uv sync --group dev
```

The dev group includes trainer-facing dependencies used for adapter probes:
`trl`, `verl`, and OpenRLHF from the `main` branch of
`https://github.com/OpenRLHF/OpenRLHF.git`.

FlashAttention needs special handling. OpenRLHF requires `flash-attn==2.8.3`,
but installing `flash-attn` from PyPI can fall back to a large CUDA source build
that is slow, memory-hungry, and sensitive to the exact Python, PyTorch, CUDA,
and platform ABI combination. This project currently develops on Linux x86_64
with Python 3.13, PyTorch 2.9, and CUDA 12.8, so the dev dependency is pinned to
the public `flash-attn==2.8.3` version while `tool.uv.sources` points uv at the
matching prebuilt wheel:

```text
flash_attn-2.8.3+cu128torch2.9-cp313-cp313-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
```

The installed package reports the public version as `2.8.3`; the local version
suffix records the wheel build's CUDA and PyTorch target. The dependency marker
limits that direct wheel to Linux x86_64 on Python 3.13. Other platforms may
need their own wheel source or a separate build strategy.
