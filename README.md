# RLVR-physics

Trainer-agnostic executable verifier tasks for RLVR.

This repo is intentionally early-stage. The firm commitments are in
[SPEC.md](SPEC.md): immutable task instances, one authoritative task backbone,
scalar sessions, renderer peripherals, trainer-owned concurrency, executable
verifier logic, and useful trajectories. Exact APIs, dataclass fields,
integration surfaces, and module layout are design bets until the first
single-step and stateful task
prototypes prove them.

## Current Implementation

The first round now includes a shallow core API in `rlvr_physics.core`:
immutable `TaskInstance` payloads, public/privileged payload separation,
renderer content blocks, scalar session result dataclasses, append-only
trajectories, a small task factory protocol, and Python task spec objects.

No concrete task families are currently packaged. New task implementations
should live behind public package facades and split specs, instance
construction, renderers, verifier/rules logic, and sessions once they grow
beyond a small probe.

No trainer integration helpers are currently packaged. Dataset, reward,
environment, and service integrations should be added only after concrete tasks
prove the scalar core API.

## Development

Install the full development environment with:

```bash
uv sync --group dev
```
