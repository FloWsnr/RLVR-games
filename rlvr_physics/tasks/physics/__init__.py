"""Physics and scientific reasoning task families."""

from rlvr_physics.tasks.physics.discovery import (
    PHYSICS_DISCOVERY_KIND,
    PHYSICS_DISCOVERY_PRIOR_MODES,
    PHYSICS_DISCOVERY_RECORDS_FILE,
    PhysicsDiscoveryRecord,
    PhysicsDiscoverySession,
    evaluate_physics_hypothesis,
    make_physics_discovery_instance,
    physics_discovery_records,
    physics_discovery_task_spec,
    render_physics_discovery_text,
)

__all__ = [
    "PHYSICS_DISCOVERY_KIND",
    "PHYSICS_DISCOVERY_PRIOR_MODES",
    "PHYSICS_DISCOVERY_RECORDS_FILE",
    "PhysicsDiscoveryRecord",
    "PhysicsDiscoverySession",
    "evaluate_physics_hypothesis",
    "make_physics_discovery_instance",
    "physics_discovery_records",
    "physics_discovery_task_spec",
    "render_physics_discovery_text",
]
