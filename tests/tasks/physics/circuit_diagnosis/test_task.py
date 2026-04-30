"""Tests for configured circuit diagnosis tasks."""

from rlvr_physics.tasks.physics.circuit_diagnosis.sessions import (
    CircuitDiagnosisSession,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.specs import (
    CIRCUIT_DIAGNOSIS_KIND,
    CIRCUIT_TEXT_RENDERER,
    DEFAULT_CONFIG,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.task import circuit_diagnosis_task


def test_configured_task_builds_instances_and_sessions() -> None:
    task = circuit_diagnosis_task(DEFAULT_CONFIG, renderer_type=CIRCUIT_TEXT_RENDERER)

    instance = task.build_instance(seed=123)
    session = task.create_session(instance)

    assert task.spec.kind == CIRCUIT_DIAGNOSIS_KIND
    assert instance.kind == CIRCUIT_DIAGNOSIS_KIND
    assert isinstance(session, CircuitDiagnosisSession)
