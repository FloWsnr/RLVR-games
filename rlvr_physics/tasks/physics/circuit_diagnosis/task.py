"""Configured task builder for the circuit diagnosis task."""

from rlvr_physics.core.factory import ConfiguredTask
from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.tasks.physics.circuit_diagnosis.instances import (
    build_circuit_diagnosis_instance,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.sessions import (
    CircuitDiagnosisSession,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.specs import (
    CIRCUIT_TEXT_RENDERER,
    CircuitDiagnosisConfig,
    circuit_diagnosis_spec,
    validate_circuit_renderer_type,
)


def circuit_diagnosis_task(
    config: CircuitDiagnosisConfig, renderer_type: str = CIRCUIT_TEXT_RENDERER
) -> ConfiguredTask:
    """Build a configured circuit diagnosis task.

    Parameters
    ----------
    config:
        Public generation, rollout, and verifier configuration.
    renderer_type:
        Renderer identifier captured by sessions created from this task.

    Returns
    -------
    ConfiguredTask
        Configured task that builds circuit instances and sessions from the
        same configuration.
    """

    validate_circuit_renderer_type(renderer_type)

    def build_instance(seed: int) -> TaskInstance:
        """Build a circuit diagnosis instance from the configured task."""

        return build_circuit_diagnosis_instance(seed, config)

    def build_session(instance: TaskInstance) -> CircuitDiagnosisSession:
        """Build a circuit diagnosis session with the configured renderer."""

        return CircuitDiagnosisSession(instance, renderer_type, config.reward)

    return ConfiguredTask(
        spec=circuit_diagnosis_spec(config),
        instance_builder=build_instance,
        session_builder=build_session,
    )
