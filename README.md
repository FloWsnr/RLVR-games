# RLVR-physics

Trainer-agnostic executable verifier tasks for RLVR.

This repo is intentionally early-stage. The firm commitments are in
[SPEC.md](SPEC.md): immutable task instances, one authoritative task backbone,
scalar sessions, renderer peripherals, trainer-owned concurrency, executable
verifier logic, and useful result metadata.

## Structure

```text
rlvr_physics/
  core/
    factory.py      # configured task helper
    instances.py    # immutable task instance payloads
    payloads.py     # freezing, plain-data conversion, stable hashes
    rendering.py    # observation and content block dataclasses
    rewards.py      # shared reward result payload
    session.py      # scalar session protocol and result dataclasses
    specs.py        # Python task spec dataclasses
  tasks/
    physics/
      cart_inference/
        backbone.py   # authoritative constant-acceleration rules
        instances.py  # deterministic instance construction
        renderers.py  # deterministic text and SVG image observations
        rewards.py    # task-specific reward assignment
        sessions.py   # scalar runtime session
        specs.py      # public task configuration and spec helpers
        task.py       # configured task builder

tests/
  core/
  tasks/physics/cart_inference/
```

## Development

Install the full development environment with:

```bash
uv sync --group dev
```

Run the validation stack with:

```bash
uv run ruff check
uv run ruff format
uv run pyright
uv run pytest
```
