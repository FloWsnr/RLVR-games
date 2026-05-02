"""Tests for the circuit diagnosis scalar session."""

from dataclasses import replace
import json
from typing import Any
from typing import Mapping

from rlvr_physics.core.submissions import TaskSubmission
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.payloads import (
    state_from_instance,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.sessions import (
    CircuitDiagnosisSession,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.specs import (
    CIRCUIT_TEXT_RENDERER,
    DEFAULT_CONFIG,
)
from tests.tasks.physics.circuit_diagnosis.conftest import (
    CIRCUIT_INSTANCE_SEED,
    CIRCUIT_SESSION_SEED,
    CircuitDiagnosisFixture,
)


def test_session_accepts_probe_and_reports_public_measurement(
    circuit_task_fixture: CircuitDiagnosisFixture,
) -> None:
    result = circuit_task_fixture.session.submit(
        TaskSubmission.action(
            '{"action": "set_source", "arguments": '
            '{"node_plus": "VIN", "node_minus": "GND", "voltage_V": 5}}'
        )
    )

    assert result.accepted
    assert result.public_info["accepted_action"] == "set_source"
    assert result.public_info["budget_usage"] == {
        "turns": 1,
        "probe_actions": 1,
        "repair_actions": 0,
        "final_answers": 0,
    }

    measurement = circuit_task_fixture.session.submit(
        TaskSubmission.action(
            '{"action": "measure_voltage", "arguments": '
            '{"node_a": "VIN", "node_b": "GND"}}'
        )
    )

    assert measurement.accepted
    assert isinstance(measurement.public_info["measurement"], Mapping)
    assert "faults" not in measurement.public_info


def test_repair_action_schema_exposes_kind_specific_required_fields(
    circuit_task_fixture: CircuitDiagnosisFixture,
) -> None:
    schema = circuit_task_fixture.reset.turn.action_schema
    actions = schema["actions"]
    assert isinstance(actions, Mapping)
    repair_schema = actions["replace_component"]
    assert isinstance(repair_schema, Mapping)
    arguments = repair_schema["arguments"]
    assert isinstance(arguments, Mapping)
    kind_parameters = arguments["kind_parameters"]
    assert isinstance(kind_parameters, Mapping)

    resistor_schema = kind_parameters["resistor"]
    assert isinstance(resistor_schema, Mapping)
    assert resistor_schema["required"] == ("value_ohm",)


def test_final_answer_schema_and_public_info_include_answer_vocabulary(
    circuit_task_fixture: CircuitDiagnosisFixture,
) -> None:
    reset = circuit_task_fixture.reset
    schema = reset.turn.action_schema
    actions = schema["actions"]
    assert isinstance(actions, Mapping)
    final_schema = actions["final_answer"]
    assert isinstance(final_schema, Mapping)
    arguments = final_schema["arguments"]
    assert isinstance(arguments, Mapping)
    faults = arguments["faults"]
    repairs = arguments["repairs"]
    assert isinstance(faults, Mapping)
    assert isinstance(repairs, Mapping)

    assert "R1_wrong_low" in faults["allowed"]
    assert "replace_R1_470_ohm" in repairs["allowed"]
    assert isinstance(reset.public_info["diagnosis_options"], Mapping)
    assert isinstance(reset.turn.public_info["diagnosis_options"], Mapping)


def test_submission_format_examples_are_valid_for_generated_circuit(
    circuit_task_fixture: CircuitDiagnosisFixture,
) -> None:
    reset = circuit_task_fixture.reset
    state = state_from_instance(circuit_task_fixture.instance)
    definition = state.truth.public_definition
    examples = reset.turn.submission_format["examples"]

    assert isinstance(examples, tuple)
    source_example = examples[0]
    voltage_example = examples[1]
    repair_example = examples[3]
    assert isinstance(source_example, Mapping)
    assert isinstance(voltage_example, Mapping)
    assert isinstance(repair_example, Mapping)
    source_arguments = source_example["arguments"]
    voltage_arguments = voltage_example["arguments"]
    repair_arguments = repair_example["arguments"]
    assert isinstance(source_arguments, Mapping)
    assert isinstance(voltage_arguments, Mapping)
    assert isinstance(repair_arguments, Mapping)

    assert source_arguments["node_plus"] == "VIN"
    assert source_arguments["node_minus"] == "GND"
    assert voltage_arguments["node_a"] == "VIN"
    assert voltage_arguments["node_b"] == "GND"
    repaired_component = definition.component(str(repair_arguments["component"]))
    assert repair_arguments["value_ohm"] == repaired_component.parameters["value_ohm"]


def test_initial_feedback_uses_public_fault_count_range() -> None:
    """Initial feedback should match the public fault-count range."""

    from rlvr_physics.tasks.physics.circuit_diagnosis.instances import (
        build_circuit_diagnosis_instance,
    )

    config = replace(DEFAULT_CONFIG, min_fault_count=1, max_fault_count=1)
    instance = build_circuit_diagnosis_instance(seed=55, config=config)
    session = CircuitDiagnosisSession(instance, CIRCUIT_TEXT_RENDERER, config.reward)

    reset = session.reset(seed=CIRCUIT_SESSION_SEED)

    assert "One hidden fault is present in the physical circuit" in (
        reset.turn.observation.text()
    )
    assert "One or two hidden faults" not in reset.turn.observation.text()


def test_repair_rejects_non_nominal_replacement_value() -> None:
    from rlvr_physics.tasks.physics.circuit_diagnosis.instances import (
        build_circuit_diagnosis_instance,
    )

    instance = build_circuit_diagnosis_instance(seed=55, config=DEFAULT_CONFIG)
    session = CircuitDiagnosisSession(
        instance, CIRCUIT_TEXT_RENDERER, DEFAULT_CONFIG.reward
    )
    session.reset(seed=CIRCUIT_SESSION_SEED)

    result = session.submit(
        TaskSubmission.action(
            json.dumps(
                {
                    "action": "replace_component",
                    "arguments": {
                        "component": "R2",
                        "kind": "resistor",
                        "value_ohm": 3000,
                    },
                }
            )
        )
    )

    assert not result.accepted
    assert not result.truncated
    assert result.public_info["invalid_submission_category"] == "invalid_repair_action"
    assert result.public_info["budget_usage"] == {
        "turns": 1,
        "probe_actions": 0,
        "repair_actions": 0,
        "final_answers": 0,
    }


def test_session_repairs_hidden_faults_and_rewards_final_verification() -> None:
    from rlvr_physics.tasks.physics.circuit_diagnosis.instances import (
        build_circuit_diagnosis_instance,
    )

    instance = build_circuit_diagnosis_instance(
        seed=CIRCUIT_INSTANCE_SEED, config=DEFAULT_CONFIG
    )
    session = CircuitDiagnosisSession(
        instance, CIRCUIT_TEXT_RENDERER, DEFAULT_CONFIG.reward
    )
    session.reset(seed=CIRCUIT_SESSION_SEED)
    state = state_from_instance(instance)
    definition = state.truth.public_definition
    faults = state.truth.hidden_faults

    for fault in faults:
        component = definition.component(fault.component_id)
        repair_result = session.submit(
            TaskSubmission.action(json.dumps(_repair_action(component)))
        )
        assert repair_result.accepted

    final = {
        "action": "final_answer",
        "arguments": {
            "faults": [fault.fault_id for fault in faults],
            "repairs": [fault.repair_code for fault in faults],
        },
    }
    result = session.submit(TaskSubmission.action(json.dumps(final)))

    assert result.accepted
    assert result.terminal
    assert result.reward == 1.0
    assert result.public_info["target_restored"] is True
    assert result.public_info["diagnosis_correct"] is True
    assert "expected_faults" not in result.public_info
    assert result.debug_info["expected_faults"] == tuple(
        fault.fault_id for fault in faults
    )


def test_final_answer_without_repairs_can_match_diagnosis_but_fail_behavior() -> None:
    from rlvr_physics.tasks.physics.circuit_diagnosis.instances import (
        build_circuit_diagnosis_instance,
    )

    instance = build_circuit_diagnosis_instance(
        seed=CIRCUIT_INSTANCE_SEED, config=DEFAULT_CONFIG
    )
    session = CircuitDiagnosisSession(
        instance, CIRCUIT_TEXT_RENDERER, DEFAULT_CONFIG.reward
    )
    session.reset(seed=CIRCUIT_SESSION_SEED)
    state = state_from_instance(instance)
    faults = state.truth.hidden_faults
    final = {
        "action": "final_answer",
        "arguments": {
            "faults": [fault.fault_id for fault in faults],
            "repairs": [fault.repair_code for fault in faults],
        },
    }

    result = session.submit(TaskSubmission.action(json.dumps(final)))

    assert result.accepted
    assert result.terminal
    assert result.reward == 0.0
    assert result.public_info["target_restored"] is False
    assert result.public_info["diagnosis_correct"] is True


def test_final_answer_rejects_duplicate_labels() -> None:
    from rlvr_physics.tasks.physics.circuit_diagnosis.instances import (
        build_circuit_diagnosis_instance,
    )

    instance = build_circuit_diagnosis_instance(
        seed=CIRCUIT_INSTANCE_SEED, config=DEFAULT_CONFIG
    )
    session = CircuitDiagnosisSession(
        instance, CIRCUIT_TEXT_RENDERER, DEFAULT_CONFIG.reward
    )
    session.reset(seed=CIRCUIT_SESSION_SEED)
    state = state_from_instance(instance)
    fault = state.truth.hidden_faults[0]
    final = {
        "action": "final_answer",
        "arguments": {
            "faults": [fault.fault_id, fault.fault_id],
            "repairs": [fault.repair_code, fault.repair_code],
        },
    }

    result = session.submit(TaskSubmission.action(json.dumps(final)))

    assert not result.accepted
    assert result.terminal
    assert result.public_info["invalid_submission_category"] == (
        "invalid_final_answer_arguments"
    )
    assert "duplicates" in str(result.public_info["reason"])


def test_final_answer_rejects_out_of_vocabulary_labels() -> None:
    from rlvr_physics.tasks.physics.circuit_diagnosis.instances import (
        build_circuit_diagnosis_instance,
    )

    instance = build_circuit_diagnosis_instance(
        seed=CIRCUIT_INSTANCE_SEED, config=DEFAULT_CONFIG
    )
    session = CircuitDiagnosisSession(
        instance, CIRCUIT_TEXT_RENDERER, DEFAULT_CONFIG.reward
    )
    session.reset(seed=CIRCUIT_SESSION_SEED)
    final = {
        "action": "final_answer",
        "arguments": {
            "faults": ["not_a_fault"],
            "repairs": ["not_a_repair"],
        },
    }

    result = session.submit(TaskSubmission.action(json.dumps(final)))

    assert not result.accepted
    assert result.terminal
    assert result.public_info["invalid_submission_category"] == (
        "invalid_final_answer_arguments"
    )
    assert "unsupported label" in str(result.public_info["reason"])


def test_session_reset_debug_contains_faults_but_public_metadata_omits_them(
    circuit_task_fixture: CircuitDiagnosisFixture,
) -> None:
    reset = circuit_task_fixture.reset

    assert "faults" not in reset.public_info
    assert "instance_hash" not in reset.public_info
    assert "faults" in reset.debug_info
    assert "rollout_seed" not in reset.public_info
    assert reset.debug_info["rollout_seed"] == CIRCUIT_SESSION_SEED


def _repair_action(component: Any) -> dict[str, object]:
    """Return a nominal repair action for a public component."""

    component_id = component.component_id
    kind = component.kind
    parameters = component.parameters
    arguments: dict[str, object] = {"component": component_id, "kind": kind}
    if kind == "resistor":
        arguments["value_ohm"] = parameters["value_ohm"]
    elif kind == "capacitor":
        arguments["value_F"] = parameters["value_F"]
    elif kind == "diode":
        arguments["forward_drop_V"] = parameters["forward_drop_V"]
    elif kind == "switch":
        arguments["closed"] = parameters["closed"]
    elif kind == "voltage_source":
        arguments["voltage_V"] = parameters["voltage_V"]
    else:
        raise AssertionError(f"unsupported repair kind: {kind}")
    return {"action": "replace_component", "arguments": arguments}
