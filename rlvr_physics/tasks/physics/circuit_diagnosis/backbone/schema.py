"""Canonical circuit diagnosis schema objects."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

from rlvr_physics.core.payloads import freeze_mapping
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.errors import (
    SubmissionParseError,
)


@dataclass(frozen=True)
class SourceSetting:
    """A bench or target voltage source setting.

    Parameters
    ----------
    node_plus:
        Positive source terminal node.
    node_minus:
        Negative source terminal node.
    voltage_V:
        Source voltage in volts.
    """

    node_plus: str
    node_minus: str
    voltage_V: float


@dataclass(frozen=True)
class CircuitComponent:
    """One public nominal circuit component.

    Parameters
    ----------
    component_id:
        Public component label shown to the model.
    kind:
        Component kind such as ``resistor`` or ``diode``.
    node_a:
        First public terminal. For diodes this is the nominal anode.
    node_b:
        Second public terminal. For diodes this is the nominal cathode.
    parameters:
        Public nominal component parameters.
    """

    component_id: str
    kind: str
    node_a: str
    node_b: str
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        """Freeze component parameters."""

        object.__setattr__(self, "parameters", freeze_mapping(self.parameters))


@dataclass(frozen=True)
class TargetCheck:
    """One public post-repair behavior check.

    Parameters
    ----------
    check_id:
        Public label for the behavior check.
    kind:
        Check kind such as ``voltage_between`` or ``current_range``.
    parameters:
        Public check parameters.
    """

    check_id: str
    kind: str
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        """Freeze target-check parameters."""

        object.__setattr__(self, "parameters", freeze_mapping(self.parameters))


@dataclass(frozen=True)
class CircuitDefinition:
    """Public nominal circuit graph for one circuit task.

    Parameters
    ----------
    circuit_id:
        Stable template-local circuit identifier.
    description:
        Short public description of the nominal circuit purpose.
    nodes:
        Public node labels, including the ground node.
    ground_node:
        Public ground node label.
    components:
        Public nominal component list.
    target_source:
        Source setting used for final target verification, when external.
    target_checks:
        Public behavior checks used after repair.
    """

    circuit_id: str
    description: str
    nodes: tuple[str, ...]
    ground_node: str
    components: tuple[CircuitComponent, ...]
    target_source: SourceSetting | None
    target_checks: tuple[TargetCheck, ...]

    def __post_init__(self) -> None:
        """Validate and freeze public circuit metadata."""

        if self.ground_node not in self.nodes:
            raise ValueError("ground_node must appear in nodes")
        component_ids = [component.component_id for component in self.components]
        if len(set(component_ids)) != len(component_ids):
            raise ValueError("component IDs must be unique")
        for component in self.components:
            if component.node_a not in self.nodes or component.node_b not in self.nodes:
                raise ValueError(f"component references unknown node: {component}")
        if self.target_source is not None:
            validate_source_nodes(self, self.target_source)

    def component(self, component_id: str) -> CircuitComponent:
        """Return a public component by ID.

        Parameters
        ----------
        component_id:
            Public component label.

        Returns
        -------
        CircuitComponent
            Matching component.

        Raises
        ------
        SubmissionParseError
            Raised when no component has that ID.
        """

        for component in self.components:
            if component.component_id == component_id:
                return component
        raise SubmissionParseError(f"unknown component: {component_id}")


@dataclass(frozen=True)
class FaultSpec:
    """Privileged hidden fault overlay.

    Parameters
    ----------
    fault_id:
        Canonical privileged fault label.
    component_id:
        Component affected by the fault.
    fault_type:
        Fault transformation type.
    parameters:
        Privileged fault parameters.
    repair_code:
        Canonical repair label used for diagnosis metadata.
    """

    fault_id: str
    component_id: str
    fault_type: str
    parameters: Mapping[str, object]
    repair_code: str

    def __post_init__(self) -> None:
        """Freeze fault parameters."""

        object.__setattr__(self, "parameters", freeze_mapping(self.parameters))


@dataclass(frozen=True)
class ReplacementSpec:
    """Session-local repair overlay for one component.

    Parameters
    ----------
    component_id:
        Replaced component ID.
    kind:
        Replacement component kind.
    parameters:
        Replacement parameters from the accepted repair action.
    """

    component_id: str
    kind: str
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        """Freeze replacement parameters."""

        object.__setattr__(self, "parameters", freeze_mapping(self.parameters))


@dataclass(frozen=True)
class CircuitPublicView:
    """Trainer-safe circuit view exposed to renderers.

    Parameters
    ----------
    definition:
        Public nominal schematic and target behavior checks.
    fault_count_range:
        Public lower and upper bounds for the number of hidden faults.
    """

    definition: CircuitDefinition
    fault_count_range: tuple[int, int]


@dataclass(frozen=True)
class CircuitTruth:
    """Privileged physical circuit truth for one task instance.

    Parameters
    ----------
    public_definition:
        Public nominal schematic used as the basis for observations.
    hidden_faults:
        Privileged physical fault overlays applied before simulation.
    fault_count_range:
        Public lower and upper bounds for the number of hidden faults.
    """

    public_definition: CircuitDefinition
    hidden_faults: tuple[FaultSpec, ...]
    fault_count_range: tuple[int, int]

    def __post_init__(self) -> None:
        """Validate hidden faults against the public schematic."""

        min_fault_count, max_fault_count = self.fault_count_range
        if min_fault_count < 0 or max_fault_count < 0:
            raise ValueError("fault_count_range values must be non-negative")
        if min_fault_count > max_fault_count:
            raise ValueError("fault_count_range minimum cannot exceed maximum")
        fault_count = len(self.hidden_faults)
        if fault_count < min_fault_count or fault_count > max_fault_count:
            raise ValueError("hidden fault count must fall within fault_count_range")
        fault_component_ids = [fault.component_id for fault in self.hidden_faults]
        if len(set(fault_component_ids)) != len(fault_component_ids):
            raise ValueError("hidden faults must target unique components")
        public_component_ids = {
            component.component_id for component in self.public_definition.components
        }
        for fault in self.hidden_faults:
            if fault.component_id not in public_component_ids:
                raise ValueError(
                    f"hidden fault references unknown component: {fault.component_id}"
                )

    @property
    def public_view(self) -> CircuitPublicView:
        """Return a renderer-safe view of this circuit truth."""

        return CircuitPublicView(
            definition=self.public_definition,
            fault_count_range=self.fault_count_range,
        )


@dataclass(frozen=True)
class CircuitDiagnosisState:
    """Authoritative immutable state for one circuit diagnosis instance.

    Parameters
    ----------
    truth:
        Privileged physical circuit truth for this immutable instance.
    """

    truth: CircuitTruth

    @property
    def public_view(self) -> CircuitPublicView:
        """Return the trainer-safe public circuit view."""

        return self.truth.public_view


def validate_source_nodes(definition: CircuitDefinition, source: SourceSetting) -> None:
    """Validate source terminal nodes."""

    validate_public_node(definition, source.node_plus)
    validate_public_node(definition, source.node_minus)
    if source.node_plus == source.node_minus:
        raise SubmissionParseError("source terminals must be distinct")
    if not isfinite(source.voltage_V):
        raise SubmissionParseError("voltage_V must be finite")


def validate_public_node(definition: CircuitDefinition, node_name: str) -> None:
    """Validate one public node label."""

    if node_name not in definition.nodes:
        raise SubmissionParseError(f"unknown node: {node_name}")
