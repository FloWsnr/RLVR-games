"""Tests for task specs."""

from rlvr_physics.core.specs import (
    RendererSpec,
    RewardSpec,
    SourceSpec,
    TaskSpec,
    VerifierSpec,
)
from rlvr_physics.core.instances import TaskLimits


def test_task_spec_freezes_nested_metadata() -> None:
    spec = TaskSpec(
        kind="example.v1",
        domain="tests",
        source=SourceSpec(source_type="procedural", seed=3, parameters={"size": 10}),
        renderers=(RendererSpec(renderer_type="text"),),
        verifier=VerifierSpec(verifier_type="exact"),
        reward=RewardSpec(reward_type="binary", parameters={"correct": 1.0}),
        limits=TaskLimits(max_turns=1),
        metadata={"exports": {"dataset": {"ability": "example"}}},
    )

    assert spec.source.parameters["size"] == 10
    assert spec.metadata["exports"] == {"dataset": {"ability": "example"}}
