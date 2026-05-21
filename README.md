# RLVR-physics

Trainer-agnostic executable verifier tasks for RLVR.

This repo is intentionally early-stage. The firm commitments are in
[SPEC.md](SPEC.md): immutable task instances, one authoritative task backbone,
scalar sessions, renderer peripherals, trainer-owned concurrency, executable
verifier logic, and useful result metadata.

Public task views omit replay seeds by default. Use stable task IDs or
explicitly safe source metadata for trainer-facing joins.

The shared physics circuit backend builds canonical circuits from reusable
motifs with typed source, sink, supply, ground, and probe ports. Procedural
circuit generation composes a config-controlled motif count into connected
topologies with declared external supply ports and an operating-point
simulation spec. Task-level operating conditions add applied source voltages
during ngspice simulation, which returns voltages for all canonical circuit
nets while keeping replay seeds out of public circuit metadata.
Circuit-specific schematic, SVG, and PNG rendering are intentionally absent for
now; future task views can add observation peripherals over the canonical
circuit state when a task needs them.

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
    submissions.py  # submission envelopes, parsing, and invalid policies
  play/
    cli.py          # generic play command entrypoint
    interaction.py  # public JSONL interaction protocol
    registry.py     # registered playable task descriptors
    task.py         # reusable play-test descriptors and CLI helpers
  tasks/
    physics/
      cart_inference/
        backbone.py   # authoritative constant-acceleration rules
        instances.py  # deterministic instance construction
        prompting.py  # task-local prompt resource loading
        prompts/      # model-facing prompt text templates
        play.py       # cart PlayableTask descriptor
        renderers.py  # deterministic text observations
        rewards.py    # task-specific reward assignment
        sessions.py   # scalar runtime session
        specs.py      # public task configuration and spec helpers
        task.py       # configured task builder
      task2/
        ...

tests/
  core/
  play/
  tasks/physics/cart_inference/
  tasks/physics/task2/
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
stdin until the rollout is terminal or truncated. Cart inference actions use the
canonical JSON envelope shown in each turn's `submission_format`. Supported
examples:

```json
{"action": "measure_position", "arguments": {"time": 5}}
{"action": "final_answer", "arguments": {"x": 3.2}}
```

Each turn publishes consumed rollout budgets under
`public_limits.budget_limits`. Cart inference uses `turns`, `actions`, and
`final_answers`. Rejected-submission policies are exposed as
`invalid_submission_policies`, and step metadata reports `budget_usage` and
`budget_remaining` with the same budget names. Step events also include
`reward_info` from the task reward policy. Scalar rewards, including accepted
action rewards and rejected-submission rewards, are assigned by each task-local
rewards module.

The runner omits privileged debug fields and the private instance seed from its
stdout protocol.

Public task parameters can be overridden with repeated `--parameter KEY=JSON`
arguments when a task's `PlayableTask` builder supports them.

List registered playable tasks with:

```bash
uv run play --list
```
