# RLVR-physics

Trainer-agnostic executable verifier tasks for RLVR.

This repo is intentionally early-stage. The firm commitments are in
[SPEC.md](SPEC.md): immutable task instances, one authoritative task backbone,
scalar sessions, renderer peripherals, trainer-owned concurrency, executable
verifier logic, and useful result metadata.

Public task views omit replay seeds by default. Use stable task IDs or
explicitly safe source metadata for trainer-facing joins.

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
      circuit_diagnosis/
        backbone.py   # canonical circuit graph, fault overlays, MNA verifier
        instances.py  # curated deterministic circuit/fault construction
        prompting.py  # task-local prompt resource loading
        prompts/      # model-facing prompt text templates
        play.py       # circuit PlayableTask descriptor
        renderers.py  # text netlist and PNG schematic observations
        rewards.py    # task-specific reward assignment
        sessions.py   # scalar probe/repair session
        specs.py      # public task configuration and spec helpers
        task.py       # configured task builder

tests/
  core/
  play/
  tasks/physics/cart_inference/
  tasks/physics/circuit_diagnosis/
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
uv run play circuit_diagnosis --renderer circuit_diagnosis.image --instance-seed 4 --session-seed 99
```

The process prints a public reset event, then reads one action per line from
stdin until the rollout is terminal or truncated. Cart actions use the canonical
JSON envelope shown in each turn's `submission_format`. Supported examples:

```json
{"action": "measure_position", "arguments": {"time": 5}}
{"action": "final_answer", "arguments": {"x": 3.2}}
```

Circuit diagnosis uses the same JSON action envelope. It exposes probe,
repair, and final-answer actions:

```json
{"action": "set_source", "arguments": {"node_plus": "VIN", "node_minus": "GND", "voltage_V": 5.0}}
{"action": "measure_voltage", "arguments": {"node_a": "OUT", "node_b": "GND"}}
{"action": "replace_component", "arguments": {"component": "R1", "kind": "resistor", "value_ohm": 1000}}
{"action": "final_answer", "arguments": {"faults": ["R1_open"], "repairs": ["replace_R1_1000_ohm"]}}
```

Each turn publishes consumed rollout budgets under
`public_limits.budget_limits`. Cart uses `turns`, `actions`, and
`final_answers`; circuit diagnosis uses `turns`, `probe_actions`,
`repair_actions`, and `final_answers`. Rejected-submission policies are exposed as
`invalid_submission_policies`, and step metadata reports `budget_usage` and
`budget_remaining` with the same budget names. Step events also include
`reward_info` from the task reward policy. Scalar rewards, including
intermediate accepted-action rewards and rejected-submission rewards, are
assigned by each task-local rewards module.

The runner omits privileged debug fields and the private instance seed from its
stdout protocol.

Public task parameters can be overridden with repeated `--parameter KEY=JSON`
arguments when a task's `PlayableTask` builder supports them.

List registered playable tasks with:

```bash
uv run play --list
```
