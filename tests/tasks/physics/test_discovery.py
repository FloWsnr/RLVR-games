"""Tests for interactive physics discovery tasks."""

from importlib import resources

import pytest

from rlvr_physics.core.factory import ConfiguredTaskFactory
from rlvr_physics.core.instances import require_mapping, require_str
from rlvr_physics.core.session import TaskSubmission
from rlvr_physics.tasks.physics.discovery import (
    PHYSICS_DISCOVERY_RECORDS_FILE,
    PhysicsDiscoverySession,
    evaluate_physics_hypothesis,
    make_physics_discovery_instance,
    physics_discovery_records,
    physics_discovery_task_spec,
)


def test_discovery_records_load_from_packaged_json() -> None:
    data_file = resources.files("rlvr_physics.tasks.physics.data").joinpath(
        PHYSICS_DISCOVERY_RECORDS_FILE
    )
    records = physics_discovery_records()

    assert data_file.is_file()
    assert len(records) == 6
    assert tuple(record.source_id for record in records) == (
        285,
        134,
        478,
        72,
        256,
        458,
    )


def test_discovery_records_and_anonymous_prior_mode_are_deterministic() -> None:
    first = make_physics_discovery_instance(
        source_id=285,
        seed=7,
        prior_mode="no_description_anonymous",
        sample_quota=3,
        hypothesis_quota=1,
    )
    second = make_physics_discovery_instance(
        source_id=285,
        seed=7,
        prior_mode="no_description_anonymous",
        sample_quota=3,
        hypothesis_quota=1,
    )

    input_variables = require_mapping(
        first.public_payload["input_variables"], "input_variables"
    )
    output_variable = require_mapping(
        first.public_payload["output_variable"], "output_variable"
    )
    equation = require_str(first.privileged_payload["equation"], "equation")

    assert len(physics_discovery_records()) == 6
    assert first.task_id == second.task_id
    assert tuple(input_variables.keys()) == (
        "var_1",
        "var_2",
        "var_3",
    )
    assert first.public_payload["problem"] == "Unknown context."
    assert dict(output_variable) == {"var_obs": "Some variable."}
    assert "m" not in equation
    assert "var_1" in equation


def test_discovery_session_runs_experiment_and_updates_observation() -> None:
    instance = make_physics_discovery_instance(
        source_id=285,
        seed=7,
        prior_mode="default",
        sample_quota=2,
        hypothesis_quota=1,
    )
    session = PhysicsDiscoverySession(instance, "text")
    reset = session.reset(seed=1)

    result = session.submit(
        TaskSubmission.action(
            '{"action": "run_experiment", "inputs": {"m": 2.0, "R": 3.0, "omega": 4.0}}'
        )
    )

    assert result.accepted
    assert result.reward == -0.01
    assert not result.done
    assert result.public_info["output"] == 480.0
    assert result.public_info["samples_used"] == 1
    assert result.observation is session.turn
    assert result.observation is not None
    one_of = result.observation.action_schema["oneOf"]
    assert isinstance(one_of, tuple)
    run_experiment_schema = require_mapping(one_of[0], "run_experiment_schema")
    properties = require_mapping(run_experiment_schema["properties"], "properties")
    inputs_schema = require_mapping(properties["inputs"], "inputs_schema")
    input_properties = require_mapping(inputs_schema["properties"], "input_properties")
    m_schema = require_mapping(input_properties["m"], "m_schema")
    assert inputs_schema["required"] == ("m", "R", "omega")
    assert inputs_schema["additionalProperties"] is False
    assert m_schema["minimum"] == 0.5
    assert m_schema["maximum"] == 5.0
    assert result.observation is not reset.turn
    assert "F_max=480" in result.observation.observation.text()


def test_discovery_correct_final_text_hypothesis_scores_one() -> None:
    instance = make_physics_discovery_instance(
        source_id=285,
        seed=11,
        prior_mode="default",
        sample_quota=2,
        hypothesis_quota=1,
    )
    session = PhysicsDiscoverySession(instance, "text")
    session.reset(seed=1)

    result = session.submit(TaskSubmission.final_text("F_max = 5 * m * omega**2 * R"))

    assert result.accepted
    assert result.reward == 1.0
    assert result.score == 1.0
    assert result.terminal
    assert not result.truncated
    assert session.turn is None
    assert result.public_info["reason"] == "correct_hypothesis"
    assert result.debug_info["true_equation"] == "5 * m * (omega**2) * R"


def test_discovery_wrong_hypothesis_can_continue_until_budget() -> None:
    instance = make_physics_discovery_instance(
        source_id=134,
        seed=11,
        prior_mode="default",
        sample_quota=1,
        hypothesis_quota=2,
    )
    session = PhysicsDiscoverySession(instance, "text")
    session.reset(seed=1)

    wrong = session.submit(
        TaskSubmission.action(
            '{"action": "submit_hypothesis", "equation": "v1 + v2 + L"}'
        )
    )
    correct = session.submit(
        TaskSubmission.action(
            '{"action": "submit_hypothesis", "equation": "(v1 * v2) / L"}'
        )
    )

    assert wrong.accepted
    assert 0.0 <= wrong.reward < 1.0
    assert not wrong.done
    assert wrong.observation is not None
    assert "score=" in wrong.observation.observation.text()
    assert correct.accepted
    assert correct.terminal
    assert correct.reward == 1.0


def test_discovery_invalid_experiment_inputs_are_rejected() -> None:
    instance = make_physics_discovery_instance(
        source_id=72,
        seed=5,
        prior_mode="default",
        sample_quota=1,
        hypothesis_quota=1,
    )
    session = PhysicsDiscoverySession(instance, "text")
    session.reset(seed=1)

    result = session.submit(
        TaskSubmission.action(
            '{"action": "run_experiment", "inputs": {"V_0": 1.0, "extra": 2.0}}'
        )
    )

    assert not result.accepted
    assert result.reward == -0.05
    assert result.public_info["reason"] == "invalid_experiment"
    assert session.turn is not None


def test_discovery_factory_spec_and_numeric_evaluator() -> None:
    instance = make_physics_discovery_instance(
        source_id=478,
        seed=3,
        prior_mode="no_context",
        sample_quota=2,
        hypothesis_quota=1,
    )
    spec = physics_discovery_task_spec(
        seed=3,
        sample_quota=2,
        hypothesis_quota=1,
        prior_mode="no_context",
    )
    factory = ConfiguredTaskFactory(
        spec=spec,
        session_builder=lambda task_instance: PhysicsDiscoverySession(
            task_instance, "text"
        ),
    )
    session = factory.create_session(instance)
    reset = session.reset(seed=1)
    evaluation = evaluate_physics_hypothesis(instance, "(Q * E_0) / epsilon_r")

    assert factory.spec.kind == instance.kind
    assert "Unknown context." in reset.turn.observation.text()
    assert evaluation.accepted
    assert evaluation.correct
    assert evaluation.score == 1.0


def test_discovery_task_spec_rejects_invalid_quotas() -> None:
    with pytest.raises(ValueError, match="sample_quota"):
        physics_discovery_task_spec(
            seed=1,
            sample_quota=0,
            hypothesis_quota=1,
            prior_mode="default",
        )

    with pytest.raises(ValueError, match="hypothesis_quota"):
        physics_discovery_task_spec(
            seed=1,
            sample_quota=1,
            hypothesis_quota=-1,
            prior_mode="default",
        )
