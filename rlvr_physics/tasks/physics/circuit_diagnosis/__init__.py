"""DC circuit diagnosis task."""

from rlvr_physics.tasks.physics.circuit_diagnosis.instances import (
    build_circuit_diagnosis_instance,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.play import CIRCUIT_PLAYABLE
from rlvr_physics.tasks.physics.circuit_diagnosis.rewards import (
    DEFAULT_REWARD_CONFIG,
    CircuitRewardConfig,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.sessions import (
    CircuitDiagnosisSession,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.specs import (
    CIRCUIT_DIAGNOSIS_DOMAIN,
    CIRCUIT_DIAGNOSIS_KIND,
    CIRCUIT_TEXT_RENDERER,
    DEFAULT_CONFIG,
    CircuitDiagnosisConfig,
    circuit_diagnosis_spec,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.task import circuit_diagnosis_task

__all__ = [
    "CIRCUIT_DIAGNOSIS_DOMAIN",
    "CIRCUIT_DIAGNOSIS_KIND",
    "CIRCUIT_PLAYABLE",
    "CIRCUIT_TEXT_RENDERER",
    "DEFAULT_CONFIG",
    "DEFAULT_REWARD_CONFIG",
    "CircuitDiagnosisConfig",
    "CircuitDiagnosisSession",
    "CircuitRewardConfig",
    "build_circuit_diagnosis_instance",
    "circuit_diagnosis_spec",
    "circuit_diagnosis_task",
]
