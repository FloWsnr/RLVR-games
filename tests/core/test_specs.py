"""Tests for task specs."""

from typing import MutableMapping, cast

import pytest

from rlvr_physics.core.specs import (
    RendererSpec,
    RewardSpec,
    SourceSpec,
    TaskSpec,
    VerifierSpec,
)


def test_task_spec_groups_public_task_components_and_freezes_payloads() -> None:
    spec = TaskSpec(
        kind="example.v1",
        domain="tests",
        source=SourceSpec(
            source_type="procedural",
            seed=3,
            parameters={"size": 10, "bounds": {"min": 0}},
        ),
        renderers=(RendererSpec(renderer_type="text", parameters={"width": 80}),),
        verifier=VerifierSpec(verifier_type="exact", parameters={"field": "x"}),
        reward=RewardSpec(reward_type="binary", parameters={"correct": 1.0}),
        max_turns=1,
        metadata={"exports": {"dataset": {"ability": "example"}}},
    )

    assert spec.source.source_type == "procedural"
    assert spec.source.parameters["size"] == 10
    assert spec.renderers[0].renderer_type == "text"
    assert spec.verifier.verifier_type == "exact"
    assert spec.reward.reward_type == "binary"
    assert spec.max_turns == 1
    assert spec.metadata["exports"] == {"dataset": {"ability": "example"}}

    source_parameters = cast(MutableMapping[str, object], spec.source.parameters)
    source_bounds = cast(MutableMapping[str, object], spec.source.parameters["bounds"])
    renderer_parameters = cast(
        MutableMapping[str, object], spec.renderers[0].parameters
    )
    verifier_parameters = cast(MutableMapping[str, object], spec.verifier.parameters)
    reward_parameters = cast(MutableMapping[str, object], spec.reward.parameters)
    metadata = cast(MutableMapping[str, object], spec.metadata)

    with pytest.raises(TypeError):
        source_parameters["size"] = 11
    with pytest.raises(TypeError):
        source_bounds["max"] = 5
    with pytest.raises(TypeError):
        renderer_parameters["width"] = 100
    with pytest.raises(TypeError):
        verifier_parameters["field"] = "y"
    with pytest.raises(TypeError):
        reward_parameters["correct"] = 0.5
    with pytest.raises(TypeError):
        metadata["split"] = "train"
