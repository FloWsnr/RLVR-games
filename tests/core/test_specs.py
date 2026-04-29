"""Tests for task specs."""

from typing import Mapping, MutableMapping, cast

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
        budget_limits={"turns": 1, "final_answers": 1},
        metadata={"exports": {"dataset": {"ability": "example"}}},
    )

    assert spec.source.source_type == "procedural"
    assert spec.source.parameters["size"] == 10
    assert spec.renderers[0].renderer_type == "text"
    assert spec.verifier.verifier_type == "exact"
    assert spec.reward.reward_type == "binary"
    assert spec.budget_limits["turns"] == 1
    assert spec.budget_limits["final_answers"] == 1
    assert spec.metadata["exports"] == {"dataset": {"ability": "example"}}

    source_parameters = cast(MutableMapping[str, object], spec.source.parameters)
    source_bounds = cast(MutableMapping[str, object], spec.source.parameters["bounds"])
    renderer_parameters = cast(
        MutableMapping[str, object], spec.renderers[0].parameters
    )
    verifier_parameters = cast(MutableMapping[str, object], spec.verifier.parameters)
    reward_parameters = cast(MutableMapping[str, object], spec.reward.parameters)
    budget_limits = cast(MutableMapping[str, object], spec.budget_limits)
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
        budget_limits["final_answers"] = 2
    with pytest.raises(TypeError):
        metadata["split"] = "train"


def test_task_spec_rejects_invalid_budget_names() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        TaskSpec(
            kind="example.v1",
            domain="tests",
            source=SourceSpec(source_type="procedural", seed=3),
            renderers=(RendererSpec(renderer_type="text"),),
            verifier=VerifierSpec(verifier_type="exact"),
            reward=RewardSpec(reward_type="binary", parameters={}),
            budget_limits=cast(Mapping[str, int], {1: 1}),
        )
