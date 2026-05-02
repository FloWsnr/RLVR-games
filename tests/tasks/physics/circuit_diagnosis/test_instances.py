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
from rlvr_physics.tasks.physics.circuit_diagnosis.generation import (
    build_generated_circuit,
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
    assert "schematic" not in public_payload
    generation_payload = public_payload["generation"]
    assert isinstance(generation_payload, Mapping)
    assert generation_payload["n_components"] == DEFAULT_CONFIG.component_count
    private_faults = first.privileged_payload["faults"]
    assert isinstance(private_faults, tuple)
    assert len(private_faults) == 1
    assert "generation_debug" in first.privileged_payload


def test_public_payload_exposes_answer_vocabulary_without_selecting_answer() -> None:
    instance = build_circuit_diagnosis_instance(seed=1, config=DEFAULT_CONFIG)
    public_payload = instance.public_payload
    options = public_payload["diagnosis_options"]
    private_faults = instance.privileged_payload["faults"]
    generation_debug = instance.privileged_payload["generation_debug"]

    assert isinstance(options, Mapping)
    assert isinstance(private_faults, tuple)
    assert isinstance(generation_debug, Mapping)
    expected_fault = private_faults[0]
    assert isinstance(expected_fault, Mapping)
    fault_ids = options["fault_ids"]
    repair_codes = options["repair_codes"]
    fault_options = options["faults"]
    repair_options = options["repairs"]

    assert isinstance(fault_ids, tuple)
    assert isinstance(repair_codes, tuple)
    assert isinstance(fault_options, tuple)
    assert isinstance(repair_options, tuple)
    assert expected_fault["fault_id"] in fault_ids
    assert expected_fault["repair_code"] in repair_codes
    assert "hidden_fault" not in options
    assert len(fault_ids) == generation_debug["distinguishable_faults"]
    assert len(fault_ids) < DEFAULT_CONFIG.component_count * 4
    fault_repair_codes: set[str] = set()
    for option in fault_options:
        assert isinstance(option, Mapping)
        repair_code = option["repair_code"]
        assert isinstance(repair_code, str)
        fault_repair_codes.add(repair_code)
    assert set(repair_codes) == fault_repair_codes


def test_generator_does_not_carry_graphical_layout_state() -> None:
    generated = build_generated_circuit(seed=123, config=DEFAULT_CONFIG)

    assert not hasattr(generated, "node_positions")
    assert not hasattr(generated, "component_routes")


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


def test_generated_instances_have_exact_component_counts() -> None:
    for seed in range(1, 10):
        instance = build_circuit_diagnosis_instance(seed=seed, config=DEFAULT_CONFIG)
        state = state_from_instance(instance)
        definition = state.truth.public_definition

        assert len(definition.components) == DEFAULT_CONFIG.component_count
        assert all(component.kind == "resistor" for component in definition.components)


def test_generation_debug_reports_validated_fault_difficulty() -> None:
    instance = build_circuit_diagnosis_instance(seed=4, config=DEFAULT_CONFIG)
    debug = instance.privileged_payload["generation_debug"]

    assert isinstance(debug, Mapping)
    assert debug["optimal_measurement_depth"] >= (
        DEFAULT_CONFIG.min_diagnosis_measurements
    )
    assert debug["optimal_measurement_depth"] <= (
        DEFAULT_CONFIG.max_diagnosis_measurements
    )
    assert debug["distinguishable_faults"] >= 2
    assert "hidden_fault" in debug
