"""Reusable circuit backend for physics task families."""

from rlvr_physics.tasks.physics.circuits.parts import (
    default_catalog,
    default_part_catalog,
    require_part_spec,
)
from rlvr_physics.tasks.physics.circuits.erc import (
    CheckIssue,
    CheckReport,
    IssueSeverity,
    check_circuit,
)
from rlvr_physics.tasks.physics.circuits.generation import (
    CircuitGenerationError,
    CircuitSupplyPort,
    GeneratedCircuit,
    GeneratorConfig,
    generate_circuit,
)
from rlvr_physics.tasks.physics.circuits.motifs import (
    CircuitMotif,
    InstantiatedMotif,
    MotifPort,
    MotifPortRole,
    MotifSignalKind,
    default_motif_weights,
    default_motifs,
)
from rlvr_physics.tasks.physics.circuits.model import (
    Circuit,
    CircuitBuilder,
    CircuitTopologyError,
    ComponentFamily,
    Connection,
    PartInstance,
    PartSpec,
    PinKind,
    PinSpec,
    is_ground_net,
)

__all__ = [
    "CheckIssue",
    "CheckReport",
    "CircuitGenerationError",
    "CircuitMotif",
    "CircuitSupplyPort",
    "Circuit",
    "CircuitBuilder",
    "CircuitTopologyError",
    "ComponentFamily",
    "Connection",
    "GeneratedCircuit",
    "GeneratorConfig",
    "IssueSeverity",
    "InstantiatedMotif",
    "MotifPort",
    "MotifPortRole",
    "MotifSignalKind",
    "PartInstance",
    "PartSpec",
    "PinKind",
    "PinSpec",
    "check_circuit",
    "default_catalog",
    "default_motif_weights",
    "default_motifs",
    "default_part_catalog",
    "generate_circuit",
    "is_ground_net",
    "require_part_spec",
]
