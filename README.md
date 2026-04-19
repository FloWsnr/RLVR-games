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
