"""Tests for circuit diagnosis instance construction."""

from collections.abc import Mapping

from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.payloads import (
    state_from_instance,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.repairs import (
    nominal_replacement_for_component,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.simulation import (
    simulate_circuit,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.verification import (
    evaluate_target_checks,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.instances import (
    build_circuit_diagnosis_instance,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.specs import DEFAULT_CONFIG


def test_instance_generation_is_deterministic_and_private_seed_is_not_public() -> None:
    first = build_circuit_diagnosis_instance(seed=123, config=DEFAULT_CONFIG)
    second = build_circuit_diagnosis_instance(seed=123, config=DEFAULT_CONFIG)

    assert first.content_hash() == second.content_hash()
    assert first.public_view()["task_id"] == first.task_id
    assert "seed" not in first.public_view()
    public_payload = first.public_view()["payload"]
    assert isinstance(public_payload, Mapping)
    assert "faults" not in public_payload
    circuit_payload = public_payload["circuit"]
    assert isinstance(circuit_payload, Mapping)
    assert "node_positions" not in circuit_payload
    schematic_payload = public_payload["schematic"]
    assert isinstance(schematic_payload, Mapping)
    assert "node_positions" in schematic_payload
    private_faults = first.privileged_payload["faults"]
    assert isinstance(private_faults, tuple)
    assert len(private_faults) in {1, 2}


def test_generated_instances_have_failing_faults_and_restorable_repairs() -> None:
    for seed in range(1, 14):
        instance = build_circuit_diagnosis_instance(seed=seed, config=DEFAULT_CONFIG)
        state = state_from_instance(instance)
        definition = state.truth.public_definition
        faults = state.truth.hidden_faults
        faulty = simulate_circuit(state.truth, {}, definition.target_source)
        repairs = {
            fault.component_id: nominal_replacement_for_component(
                definition.component(fault.component_id)
            )
            for fault in faults
        }
        repaired = simulate_circuit(state.truth, repairs, definition.target_source)

        assert not all(
            result.passed for result in evaluate_target_checks(definition, faulty)
        )
        assert all(
            result.passed for result in evaluate_target_checks(definition, repaired)
        )
