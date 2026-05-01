"""Payload conversion for circuit diagnosis backbone data."""

from collections.abc import Mapping
from typing import cast

from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.schema import (
    CircuitComponent,
    CircuitDefinition,
    CircuitDiagnosisState,
    CircuitTruth,
    FaultSpec,
    ReplacementSpec,
    SourceSetting,
    TargetCheck,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.values import (
    float_field,
    int_field,
    mapping_field,
    sequence_field,
    str_field,
    str_sequence,
)


def state_from_instance(instance: TaskInstance) -> CircuitDiagnosisState:
    """Build authoritative circuit state from an immutable instance."""

    definition = circuit_definition_from_mapping(
        mapping_field(instance.public_payload, "circuit")
    )
    fault_values = sequence_field(instance.privileged_payload, "faults")
    faults = tuple(
        fault_from_mapping(cast(Mapping[str, object], fault_value))
        for fault_value in fault_values
    )
    fault_range = mapping_field(instance.public_payload, "fault_count_range")
    truth = CircuitTruth(
        public_definition=definition,
        hidden_faults=faults,
        fault_count_range=(
            int_field(fault_range, "min"),
            int_field(fault_range, "max"),
        ),
    )
    return CircuitDiagnosisState(truth=truth)


def circuit_definition_from_mapping(
    values: Mapping[str, object],
) -> CircuitDefinition:
    """Build a public circuit definition from plain instance data."""

    source_value = values.get("target_source")
    target_source = None
    if isinstance(source_value, Mapping):
        target_source = source_from_mapping(cast(Mapping[str, object], source_value))
    components = tuple(
        component_from_mapping(cast(Mapping[str, object], component_value))
        for component_value in sequence_field(values, "components")
    )
    target_checks = tuple(
        target_check_from_mapping(cast(Mapping[str, object], check_value))
        for check_value in sequence_field(values, "target_checks")
    )
    return CircuitDefinition(
        circuit_id=str_field(values, "circuit_id"),
        description=str_field(values, "description"),
        nodes=tuple(str_sequence(values, "nodes")),
        ground_node=str_field(values, "ground_node"),
        components=components,
        target_source=target_source,
        target_checks=target_checks,
    )


def source_from_mapping(values: Mapping[str, object]) -> SourceSetting:
    """Build a source setting from plain instance data."""

    return SourceSetting(
        node_plus=str_field(values, "node_plus"),
        node_minus=str_field(values, "node_minus"),
        voltage_V=float_field(values, "voltage_V"),
    )


def component_from_mapping(values: Mapping[str, object]) -> CircuitComponent:
    """Build a public circuit component from plain instance data."""

    return CircuitComponent(
        component_id=str_field(values, "component_id"),
        kind=str_field(values, "kind"),
        node_a=str_field(values, "node_a"),
        node_b=str_field(values, "node_b"),
        parameters=mapping_field(values, "parameters"),
    )


def target_check_from_mapping(values: Mapping[str, object]) -> TargetCheck:
    """Build a public target check from plain instance data."""

    return TargetCheck(
        check_id=str_field(values, "check_id"),
        kind=str_field(values, "kind"),
        parameters=mapping_field(values, "parameters"),
    )


def fault_from_mapping(values: Mapping[str, object]) -> FaultSpec:
    """Build a privileged fault spec from plain instance data."""

    return FaultSpec(
        fault_id=str_field(values, "fault_id"),
        component_id=str_field(values, "component_id"),
        fault_type=str_field(values, "fault_type"),
        parameters=mapping_field(values, "parameters"),
        repair_code=str_field(values, "repair_code"),
    )


def circuit_definition_payload(definition: CircuitDefinition) -> dict[str, object]:
    """Return plain public payload data for a circuit definition."""

    return {
        "circuit_id": definition.circuit_id,
        "description": definition.description,
        "nodes": list(definition.nodes),
        "ground_node": definition.ground_node,
        "components": [
            component_payload(component) for component in definition.components
        ],
        "target_source": (
            None
            if definition.target_source is None
            else source_payload(definition.target_source)
        ),
        "target_checks": [
            target_check_payload(check) for check in definition.target_checks
        ],
    }


def source_payload(source: SourceSetting) -> dict[str, object]:
    """Return plain public payload data for a source setting."""

    return {
        "node_plus": source.node_plus,
        "node_minus": source.node_minus,
        "voltage_V": source.voltage_V,
    }


def component_payload(component: CircuitComponent) -> dict[str, object]:
    """Return plain public payload data for a component."""

    return {
        "component_id": component.component_id,
        "kind": component.kind,
        "node_a": component.node_a,
        "node_b": component.node_b,
        "parameters": dict(component.parameters),
    }


def target_check_payload(check: TargetCheck) -> dict[str, object]:
    """Return plain public payload data for a target check."""

    return {
        "check_id": check.check_id,
        "kind": check.kind,
        "parameters": dict(check.parameters),
    }


def fault_payload(fault: FaultSpec) -> dict[str, object]:
    """Return plain privileged payload data for a fault."""

    return {
        "fault_id": fault.fault_id,
        "component_id": fault.component_id,
        "fault_type": fault.fault_type,
        "parameters": dict(fault.parameters),
        "repair_code": fault.repair_code,
    }


def replacement_payload(replacement: ReplacementSpec) -> dict[str, object]:
    """Return plain public/debug payload data for a repair overlay."""

    return {
        "component_id": replacement.component_id,
        "kind": replacement.kind,
        "parameters": dict(replacement.parameters),
    }
