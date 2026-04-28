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
  play/
    cli.py          # generic play command entrypoint
    interaction.py  # public JSONL interaction protocol
    registry.py     # registered playable task descriptors
    task.py         # reusable play-test descriptors and CLI helpers
  tasks/
    _shared/
      rendering.py  # cross-task SVG rasterization helpers
    physics/
      cart_inference/
        backbone.py   # authoritative constant-acceleration rules
        instances.py  # deterministic instance construction
        prompting.py  # task-local prompt resource loading
        prompts/      # model-facing prompt text templates
        play.py       # cart PlayableTask descriptor
        renderers.py  # deterministic text and PNG image observations
        rewards.py    # task-specific reward assignment
        sessions.py   # scalar runtime session
        specs.py      # public task configuration and spec helpers
        task.py       # configured task builder

tests/
  core/
  play/
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

## Play-test interactions

Tasks can expose a small `PlayableTask` descriptor and use the shared JSONL
interaction protocol in `rlvr_physics.play.interaction`. The protocol prints
a public reset event, then reads one model submission per line from stdin until
the rollout is terminal or truncated. This gives each task the same local
play-test surface for interaction checks and difficulty probes.

The generic play command selects the task by name or alias:

```bash
uv run play cart_inference --instance-seed 123 --session-seed 456
```

The process prints a public reset event, then reads one action per line from
stdin until the rollout is terminal or truncated. Supported action examples:

```json
{"action": "measure_position", "time": 5}
{"action": "final_answer", "x": 3.2}
```

The runner omits privileged debug fields and the private instance seed from its
stdout protocol.

Public task parameters can be overridden with repeated `--parameter KEY=JSON`
arguments when a task's `PlayableTask` builder supports them.

List registered playable tasks with:

```bash
uv run play --list
```
