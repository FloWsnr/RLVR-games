"""Shared fixtures for circuit diagnosis task tests."""

from dataclasses import dataclass

import pytest

from rlvr_physics.core.factory import ConfiguredTask
from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.core.session import TaskResetResult
from rlvr_physics.tasks.physics.circuit_diagnosis.instances import (
    build_circuit_diagnosis_instance,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.sessions import (
    CircuitDiagnosisSession,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.specs import (
    DEFAULT_CONFIG,
    CircuitDiagnosisConfig,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.task import circuit_diagnosis_task

CIRCUIT_INSTANCE_SEED = 4
CIRCUIT_SESSION_SEED = 99


@dataclass(frozen=True)
class CircuitDiagnosisFixture:
    """Reusable circuit task setup for task-module tests."""

    config: CircuitDiagnosisConfig
    task: ConfiguredTask
    instance: TaskInstance
    session: CircuitDiagnosisSession
    reset: TaskResetResult
    renderer_name: str


@pytest.fixture
def circuit_config() -> CircuitDiagnosisConfig:
    """Return the default circuit diagnosis configuration."""

    return DEFAULT_CONFIG


@pytest.fixture
def circuit_task(circuit_config: CircuitDiagnosisConfig) -> ConfiguredTask:
    """Return a configured circuit diagnosis task."""

    return circuit_diagnosis_task(circuit_config)


@pytest.fixture
def circuit_instance(circuit_config: CircuitDiagnosisConfig) -> TaskInstance:
    """Return a deterministic circuit diagnosis instance."""

    return build_circuit_diagnosis_instance(
        seed=CIRCUIT_INSTANCE_SEED, config=circuit_config
    )


@pytest.fixture
def circuit_task_fixture(
    circuit_config: CircuitDiagnosisConfig,
    circuit_task: ConfiguredTask,
) -> CircuitDiagnosisFixture:
    """Return a configured circuit task with reset session and renderer info."""

    instance = circuit_task.build_instance(seed=CIRCUIT_INSTANCE_SEED)
    session = circuit_task.create_session(instance)
    if not isinstance(session, CircuitDiagnosisSession):
        raise TypeError("circuit_task must create CircuitDiagnosisSession")
    reset = session.reset(seed=CIRCUIT_SESSION_SEED)
    return CircuitDiagnosisFixture(
        config=circuit_config,
        task=circuit_task,
        instance=instance,
        session=session,
        reset=reset,
        renderer_name=reset.turn.observation.renderer_name,
    )
