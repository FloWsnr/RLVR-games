"""Physical circuit construction from public definitions and hidden state."""

from collections.abc import Mapping
from dataclasses import dataclass

from rlvr_physics.core.payloads import freeze_mapping
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.errors import (
    CircuitSimulationError,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.schema import (
    CircuitComponent,
    CircuitTruth,
    FaultSpec,
    ReplacementSpec,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.values import (
    bool_parameter,
    float_parameter,
)


@dataclass(frozen=True)
class PhysicalComponent:
    """One component in the hidden physical circuit."""

    component_id: str
    public_kind: str
    effective_kind: str
    node_a: str
    node_b: str
    parameters: Mapping[str, object]
    measurement_sign: float

    def __post_init__(self) -> None:
        """Freeze physical component parameters."""

        object.__setattr__(self, "parameters", freeze_mapping(self.parameters))


def physical_components(
    truth: CircuitTruth,
    repairs: Mapping[str, ReplacementSpec],
) -> tuple[PhysicalComponent, ...]:
    """Return hidden physical components after faults and repairs."""

    fault_by_component = {fault.component_id: fault for fault in truth.hidden_faults}
    physical: list[PhysicalComponent] = []
    for component in truth.public_definition.components:
        replacement = repairs.get(component.component_id)
        if replacement is not None:
            physical.append(_component_from_replacement(component, replacement))
            continue
        fault = fault_by_component.get(component.component_id)
        if fault is None:
            physical.append(_component_from_nominal(component))
        else:
            physical.append(_component_from_fault(component, fault))
    return tuple(physical)


def _component_from_nominal(component: CircuitComponent) -> PhysicalComponent:
    """Return a physical component matching the nominal public component."""

    return PhysicalComponent(
        component_id=component.component_id,
        public_kind=component.kind,
        effective_kind=component.kind,
        node_a=component.node_a,
        node_b=component.node_b,
        parameters=component.parameters,
        measurement_sign=1.0,
    )


def _component_from_replacement(
    nominal: CircuitComponent, replacement: ReplacementSpec
) -> PhysicalComponent:
    """Return a physical component from a session repair overlay."""

    if replacement.kind != nominal.kind:
        raise CircuitSimulationError(
            f"replacement kind mismatch for {nominal.component_id}"
        )
    return PhysicalComponent(
        component_id=nominal.component_id,
        public_kind=nominal.kind,
        effective_kind=nominal.kind,
        node_a=nominal.node_a,
        node_b=nominal.node_b,
        parameters=replacement.parameters,
        measurement_sign=1.0,
    )


def _component_from_fault(
    component: CircuitComponent, fault: FaultSpec
) -> PhysicalComponent:
    """Return a hidden physical component after one fault."""

    if fault.fault_type == "open_resistor":
        _require_kind(component, "resistor", fault)
        return _faulted_component(component, "open", component.parameters, 1.0)
    if fault.fault_type == "shorted_resistor":
        _require_kind(component, "resistor", fault)
        return _faulted_component(component, "short", component.parameters, 1.0)
    if fault.fault_type == "wrong_value":
        _require_kind(component, "resistor", fault)
        return _faulted_component(
            component,
            component.kind,
            {"value_ohm": float_parameter(fault.parameters, "value_ohm")},
            1.0,
        )
    if fault.fault_type == "shorted_capacitor":
        _require_kind(component, "capacitor", fault)
        return _faulted_component(component, "short", component.parameters, 1.0)
    if fault.fault_type == "reversed_diode":
        _require_kind(component, "diode", fault)
        return PhysicalComponent(
            component_id=component.component_id,
            public_kind=component.kind,
            effective_kind=component.kind,
            node_a=component.node_b,
            node_b=component.node_a,
            parameters=component.parameters,
            measurement_sign=-1.0,
        )
    if fault.fault_type == "broken_switch":
        _require_kind(component, "switch", fault)
        return _faulted_component(
            component,
            component.kind,
            {"closed": bool_parameter(fault.parameters, "closed")},
            1.0,
        )
    if fault.fault_type == "internal_source_resistance":
        _require_kind(component, "voltage_source", fault)
        parameters = dict(component.parameters)
        parameters["internal_resistance_ohm"] = float_parameter(
            fault.parameters, "internal_resistance_ohm"
        )
        return _faulted_component(component, component.kind, parameters, 1.0)
    raise CircuitSimulationError(f"unsupported fault type: {fault.fault_type}")


def _faulted_component(
    component: CircuitComponent,
    effective_kind: str,
    parameters: Mapping[str, object],
    measurement_sign: float,
) -> PhysicalComponent:
    """Build a hidden physical component from fault details."""

    return PhysicalComponent(
        component_id=component.component_id,
        public_kind=component.kind,
        effective_kind=effective_kind,
        node_a=component.node_a,
        node_b=component.node_b,
        parameters=parameters,
        measurement_sign=measurement_sign,
    )


def _require_kind(component: CircuitComponent, kind: str, fault: FaultSpec) -> None:
    """Validate that a fault targets the required component kind."""

    if component.kind != kind:
        raise CircuitSimulationError(
            f"fault {fault.fault_id} requires {kind}, got {component.kind}"
        )
