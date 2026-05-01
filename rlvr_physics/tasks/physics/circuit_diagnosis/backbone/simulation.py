"""DC circuit simulation for the circuit diagnosis backbone."""

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product

from rlvr_physics.core.payloads import freeze_mapping
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.constants import (
    DIODE_STATE_TOLERANCE,
    MIN_RESISTANCE_OHM,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.errors import (
    CircuitSimulationError,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.physical import (
    PhysicalComponent,
    physical_components,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.schema import (
    CircuitTruth,
    ReplacementSpec,
    SourceSetting,
    validate_source_nodes,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.solver import (
    LinearElement,
    SolvedLinearCircuit,
    solve_linear_circuit,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.values import (
    bool_parameter,
    float_parameter,
    optional_float_parameter,
)


@dataclass(frozen=True)
class SimulationResult:
    """Solved DC operating point for a hidden physical circuit."""

    node_voltages_V: Mapping[str, float]
    component_currents_A: Mapping[str, float]
    component_powers_W: Mapping[str, float]
    diode_states: Mapping[str, bool]

    def __post_init__(self) -> None:
        """Freeze simulation result mappings."""

        object.__setattr__(
            self, "node_voltages_V", freeze_mapping(self.node_voltages_V)
        )
        object.__setattr__(
            self, "component_currents_A", freeze_mapping(self.component_currents_A)
        )
        object.__setattr__(
            self, "component_powers_W", freeze_mapping(self.component_powers_W)
        )
        object.__setattr__(self, "diode_states", freeze_mapping(self.diode_states))


def simulate_circuit(
    truth: CircuitTruth,
    repairs: Mapping[str, ReplacementSpec],
    source_setting: SourceSetting | None,
) -> SimulationResult:
    """Simulate one physical DC circuit with hidden faults and repairs.

    Parameters
    ----------
    truth:
        Privileged physical circuit truth.
    repairs:
        Session-local repair overlays.
    source_setting:
        Optional external source setting for this solve.

    Returns
    -------
    SimulationResult
        Solved node voltages, component currents, powers, and diode states.
    """

    definition = truth.public_definition
    if source_setting is not None:
        validate_source_nodes(definition, source_setting)
    components = physical_components(truth, repairs)
    diode_components = tuple(
        component for component in components if component.effective_kind == "diode"
    )
    for diode_states in _candidate_diode_states(diode_components):
        elements, internal_nodes = _linear_elements(
            components=components,
            source_setting=source_setting,
            diode_states=diode_states,
        )
        try:
            solved = solve_linear_circuit(
                definition=definition,
                internal_nodes=internal_nodes,
                elements=elements,
            )
        except CircuitSimulationError:
            continue
        if _diode_state_is_valid(solved, diode_components, diode_states):
            return _simulation_result_from_solution(
                components=components,
                solved=solved,
                diode_states=diode_states,
            )
    raise CircuitSimulationError("no valid DC operating point")


def node_voltage(
    simulation: SimulationResult | SolvedLinearCircuit, node_name: str
) -> float:
    """Return one solved node voltage."""

    voltage = simulation.node_voltages_V.get(node_name)
    if voltage is None:
        raise CircuitSimulationError(f"node voltage unavailable: {node_name}")
    return voltage


def component_current(simulation: SimulationResult, component_id: str) -> float:
    """Return one solved component current."""

    current = simulation.component_currents_A.get(component_id)
    if current is None:
        raise CircuitSimulationError(f"component current unavailable: {component_id}")
    return current


def component_power(simulation: SimulationResult, component_id: str) -> float:
    """Return one solved component power."""

    power = simulation.component_powers_W.get(component_id)
    if power is None:
        raise CircuitSimulationError(f"component power unavailable: {component_id}")
    return power


def _linear_elements(
    components: tuple[PhysicalComponent, ...],
    source_setting: SourceSetting | None,
    diode_states: Mapping[str, bool],
) -> tuple[tuple[LinearElement, ...], tuple[str, ...]]:
    """Build linear elements for one diode-state assignment."""

    elements: list[LinearElement] = []
    internal_nodes: list[str] = []
    if source_setting is not None:
        elements.append(
            LinearElement(
                element_type="voltage_source",
                component_id="__bench_source__",
                node_a=source_setting.node_plus,
                node_b=source_setting.node_minus,
                value=source_setting.voltage_V,
                measurement_sign=1.0,
            )
        )
    for component in components:
        if component.effective_kind == "open":
            continue
        if component.effective_kind == "resistor":
            elements.append(_resistor_element(component))
        elif component.effective_kind == "capacitor":
            continue
        elif component.effective_kind == "short":
            elements.append(_zero_volt_element(component))
        elif component.effective_kind == "switch":
            if bool_parameter(component.parameters, "closed"):
                elements.append(_zero_volt_element(component))
        elif component.effective_kind == "diode":
            if diode_states[component.component_id]:
                elements.append(
                    LinearElement(
                        element_type="voltage_source",
                        component_id=component.component_id,
                        node_a=component.node_a,
                        node_b=component.node_b,
                        value=float_parameter(component.parameters, "forward_drop_V"),
                        measurement_sign=component.measurement_sign,
                    )
                )
        elif component.effective_kind == "voltage_source":
            source_elements, source_nodes = _voltage_source_elements(component)
            elements.extend(source_elements)
            internal_nodes.extend(source_nodes)
        elif component.effective_kind == "current_source":
            elements.append(
                LinearElement(
                    element_type="current_source",
                    component_id=component.component_id,
                    node_a=component.node_a,
                    node_b=component.node_b,
                    value=float_parameter(component.parameters, "current_A"),
                    measurement_sign=component.measurement_sign,
                )
            )
        else:
            raise CircuitSimulationError(
                f"unsupported component kind: {component.effective_kind}"
            )
    return tuple(elements), tuple(internal_nodes)


def _resistor_element(component: PhysicalComponent) -> LinearElement:
    """Return a resistor linear element."""

    resistance = float_parameter(component.parameters, "value_ohm")
    if resistance < MIN_RESISTANCE_OHM:
        raise CircuitSimulationError(
            f"resistance too small for {component.component_id}"
        )
    return LinearElement(
        element_type="resistor",
        component_id=component.component_id,
        node_a=component.node_a,
        node_b=component.node_b,
        value=resistance,
        measurement_sign=component.measurement_sign,
    )


def _zero_volt_element(component: PhysicalComponent) -> LinearElement:
    """Return an ideal zero-volt branch element."""

    return LinearElement(
        element_type="voltage_source",
        component_id=component.component_id,
        node_a=component.node_a,
        node_b=component.node_b,
        value=0.0,
        measurement_sign=component.measurement_sign,
    )


def _voltage_source_elements(
    component: PhysicalComponent,
) -> tuple[tuple[LinearElement, ...], tuple[str, ...]]:
    """Return elements for an ideal or internally resisted voltage source."""

    voltage = float_parameter(component.parameters, "voltage_V")
    internal_resistance = optional_float_parameter(
        component.parameters, "internal_resistance_ohm", 0.0
    )
    if internal_resistance < 0.0:
        raise CircuitSimulationError("internal source resistance cannot be negative")
    if internal_resistance < MIN_RESISTANCE_OHM:
        return (
            (
                LinearElement(
                    element_type="voltage_source",
                    component_id=component.component_id,
                    node_a=component.node_a,
                    node_b=component.node_b,
                    value=voltage,
                    measurement_sign=component.measurement_sign,
                ),
            ),
            (),
        )
    internal_node = f"__{component.component_id}_internal_plus"
    return (
        (
            LinearElement(
                element_type="voltage_source",
                component_id=f"{component.component_id}__ideal",
                node_a=internal_node,
                node_b=component.node_b,
                value=voltage,
                measurement_sign=1.0,
            ),
            LinearElement(
                element_type="resistor",
                component_id=f"{component.component_id}__internal_resistance",
                node_a=internal_node,
                node_b=component.node_a,
                value=internal_resistance,
                measurement_sign=1.0,
            ),
        ),
        (internal_node,),
    )


def _candidate_diode_states(
    diode_components: tuple[PhysicalComponent, ...],
) -> tuple[Mapping[str, bool], ...]:
    """Return deterministic diode on/off state candidates."""

    if len(diode_components) == 0:
        return ({},)
    candidates: list[Mapping[str, bool]] = []
    for states in product((False, True), repeat=len(diode_components)):
        candidates.append(
            {
                component.component_id: states[index]
                for index, component in enumerate(diode_components)
            }
        )
    return tuple(candidates)


def _diode_state_is_valid(
    solved: SolvedLinearCircuit,
    diode_components: tuple[PhysicalComponent, ...],
    diode_states: Mapping[str, bool],
) -> bool:
    """Return whether diode inequalities match the assumed state."""

    for component in diode_components:
        voltage = node_voltage(solved, component.node_a) - node_voltage(
            solved, component.node_b
        )
        drop = float_parameter(component.parameters, "forward_drop_V")
        is_on = diode_states[component.component_id]
        if is_on:
            branch_current = solved.branch_currents_A.get(component.component_id, 0.0)
            if branch_current < -DIODE_STATE_TOLERANCE:
                return False
            if abs(voltage - drop) > 1.0e-5:
                return False
        else:
            if voltage > drop + DIODE_STATE_TOLERANCE:
                return False
    return True


def _simulation_result_from_solution(
    components: tuple[PhysicalComponent, ...],
    solved: SolvedLinearCircuit,
    diode_states: Mapping[str, bool],
) -> SimulationResult:
    """Build public-component currents and powers from a raw solution."""

    currents: dict[str, float] = {}
    powers: dict[str, float] = {}
    for component in components:
        current = _component_current_from_solution(component, solved, diode_states)
        voltage = node_voltage(solved, component.node_a) - node_voltage(
            solved, component.node_b
        )
        currents[component.component_id] = current
        powers[component.component_id] = abs(voltage * current)
    public_node_voltages = {
        node: voltage
        for node, voltage in solved.node_voltages_V.items()
        if not node.startswith("__")
    }
    return SimulationResult(
        node_voltages_V=public_node_voltages,
        component_currents_A=currents,
        component_powers_W=powers,
        diode_states=diode_states,
    )


def _component_current_from_solution(
    component: PhysicalComponent,
    solved: SolvedLinearCircuit,
    diode_states: Mapping[str, bool],
) -> float:
    """Return component current in the public nominal orientation."""

    if component.effective_kind == "open":
        return 0.0
    if component.effective_kind == "resistor":
        resistance = float_parameter(component.parameters, "value_ohm")
        current = (
            node_voltage(solved, component.node_a)
            - node_voltage(solved, component.node_b)
        ) / resistance
        return component.measurement_sign * current
    if component.effective_kind == "capacitor":
        return 0.0
    if component.effective_kind in {"short", "switch"}:
        if component.effective_kind == "switch" and not bool_parameter(
            component.parameters, "closed"
        ):
            return 0.0
        return component.measurement_sign * solved.branch_currents_A.get(
            component.component_id, 0.0
        )
    if component.effective_kind == "diode":
        if not diode_states[component.component_id]:
            return 0.0
        return component.measurement_sign * solved.branch_currents_A.get(
            component.component_id, 0.0
        )
    if component.effective_kind == "voltage_source":
        internal_resistance = optional_float_parameter(
            component.parameters, "internal_resistance_ohm", 0.0
        )
        if internal_resistance >= MIN_RESISTANCE_OHM:
            internal_node = f"__{component.component_id}_internal_plus"
            current = (
                node_voltage(solved, component.node_a)
                - node_voltage(solved, internal_node)
            ) / internal_resistance
            return component.measurement_sign * current
        return component.measurement_sign * solved.branch_currents_A.get(
            component.component_id, 0.0
        )
    if component.effective_kind == "current_source":
        return component.measurement_sign * float_parameter(
            component.parameters, "current_A"
        )
    raise CircuitSimulationError(
        f"unsupported component kind: {component.effective_kind}"
    )
