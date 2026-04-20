"""Interactive physics equation discovery task package."""

from rlvr_physics.tasks.physics.discovery.constants import (
    PHYSICS_DISCOVERY_KIND,
    PHYSICS_DISCOVERY_PRIOR_MODES,
    PHYSICS_DISCOVERY_RECORDS_FILE,
    PHYSICS_DISCOVERY_SOURCE,
    PHYSICS_DOMAIN,
)
from rlvr_physics.tasks.physics.discovery.evaluation import evaluate_physics_hypothesis
from rlvr_physics.tasks.physics.discovery.instances import (
    make_physics_discovery_instance,
)
from rlvr_physics.tasks.physics.discovery.records import physics_discovery_records
from rlvr_physics.tasks.physics.discovery.renderers import render_physics_discovery_text
from rlvr_physics.tasks.physics.discovery.session import PhysicsDiscoverySession
from rlvr_physics.tasks.physics.discovery.spec import physics_discovery_task_spec
from rlvr_physics.tasks.physics.discovery.types import (
    ExperimentObservation,
    HypothesisAttempt,
    HypothesisEvaluation,
    ParsedDiscoveryAction,
    PhysicsDiscoveryRecord,
)

__all__ = [
    "PHYSICS_DISCOVERY_KIND",
    "PHYSICS_DISCOVERY_PRIOR_MODES",
    "PHYSICS_DISCOVERY_RECORDS_FILE",
    "PHYSICS_DISCOVERY_SOURCE",
    "PHYSICS_DOMAIN",
    "ExperimentObservation",
    "HypothesisAttempt",
    "HypothesisEvaluation",
    "ParsedDiscoveryAction",
    "PhysicsDiscoveryRecord",
    "PhysicsDiscoverySession",
    "evaluate_physics_hypothesis",
    "make_physics_discovery_instance",
    "physics_discovery_records",
    "physics_discovery_task_spec",
    "render_physics_discovery_text",
]
