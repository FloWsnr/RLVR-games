"""Authoritative circuit diagnosis backbone and DC verifier."""

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from math import isfinite
from types import MappingProxyType
from typing import cast

from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.core.payloads import freeze_mapping
from rlvr_physics.core.submissions import (
    ACTION_ARGUMENTS_FIELD,
    ACTION_NAME_FIELD,
    ParsedAction,
    TaskSubmission,
    parse_action_submission as parse_core_action_submission,
)

SET_SOURCE_ACTION = "set_source"
MEASURE_VOLTAGE_ACTION = "measure_voltage"
MEASURE_CURRENT_ACTION = "measure_current"
REPLACE_COMPONENT_ACTION = "replace_component"
FINAL_ANSWER_ACTION = "final_answer"
ACTION_SUBMISSION_PARSE_ERROR = (
    "could not parse action submission; expected one JSON line with fields "
    f'"{ACTION_NAME_FIELD}" and "{ACTION_ARGUMENTS_FIELD}"'
)
GROUND_NODE = "GND"
MAX_SOURCE_ABS_VOLTAGE = 24.0
MIN_RESISTANCE_OHM = 1.0e-6
DIODE_STATE_TOLERANCE = 1.0e-7
CHECK_TOLERANCE = 1.0e-9


class SubmissionParseError(ValueError):
    """Raised when a model submission cannot be interpreted for this task."""


class CircuitSimulationError(RuntimeError):
    """Raised when a circuit realization cannot be solved."""


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
    """Public nominal circuit graph used by backbone and renderers.

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
            _validate_source_nodes(self, self.target_source)

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
class CircuitDiagnosisState:
    """Authoritative immutable state for one circuit diagnosis instance."""

    definition: CircuitDefinition
    faults: tuple[FaultSpec, ...]
    fault_count_range: tuple[int, int]


@dataclass(frozen=True)
class VoltageMeasurement:
    """Result of one voltage probe."""

    node_a: str
    node_b: str
    voltage_V: float


@dataclass(frozen=True)
class CurrentMeasurement:
    """Result of one component current probe."""

    component_id: str
    current_A: float


@dataclass(frozen=True)
class TargetCheckResult:
    """Result of one post-repair behavior check."""

    check_id: str
    kind: str
    passed: bool
    value: float
    lower_bound: float | None
    upper_bound: float | None


@dataclass(frozen=True)
class FinalCircuitEvaluation:
    """Verifier evaluation for the final repair state."""

    target_restored: bool
    diagnosis_correct: bool
    submitted_faults: tuple[str, ...]
    submitted_repairs: tuple[str, ...]
    expected_faults: tuple[str, ...]
    expected_repairs: tuple[str, ...]
    check_results: tuple[TargetCheckResult, ...]
    simulation_error: str | None


@dataclass(frozen=True)
class SimulationResult:
    """Solved DC operating point for a realized circuit."""

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


@dataclass(frozen=True)
class _RealizedComponent:
    """One component after hidden fault and repair overlays."""

    component_id: str
    public_kind: str
    effective_kind: str
    node_a: str
    node_b: str
    parameters: Mapping[str, object]
    measurement_sign: float

    def __post_init__(self) -> None:
        """Freeze realized component parameters."""

        object.__setattr__(self, "parameters", freeze_mapping(self.parameters))


@dataclass(frozen=True)
class _LinearElement:
    """One stamped linear circuit element."""

    element_type: str
    component_id: str
    node_a: str
    node_b: str
    value: float
    measurement_sign: float


@dataclass(frozen=True)
class _SolvedLinearCircuit:
    """Raw MNA solution with branch currents."""

    node_voltages_V: dict[str, float]
    branch_currents_A: dict[str, float]


class CircuitDiagnosisBackbone:
    """Authoritative executable backbone for one circuit diagnosis rollout."""

    def __init__(self, instance: TaskInstance) -> None:
        """Initialize canonical state and repair/source overlays."""

        self.instance = instance
        self._state = state_from_instance(instance)
        self._source_setting: SourceSetting | None = None
        self._repairs: dict[str, ReplacementSpec] = {}

    @property
    def state(self) -> CircuitDiagnosisState:
        """Return the immutable authoritative circuit state."""

        return self._state

    @property
    def source_setting(self) -> SourceSetting | None:
        """Return the current bench source setting, if one is connected."""

        return self._source_setting

    @property
    def repairs(self) -> Mapping[str, ReplacementSpec]:
        """Return accepted repair overlays keyed by component ID."""

        return MappingProxyType(dict(self._repairs))

    def reset_rollout(self) -> None:
        """Reset rollout-local source and repair state."""

        self._source_setting = None
        self._repairs = {}

    def parse_action(self, submission: TaskSubmission) -> ParsedAction:
        """Parse a submission as a structured circuit action."""

        return parse_action_submission(submission)

    def set_source_from_action(self, action: ParsedAction) -> SourceSetting:
        """Apply a bench source action and return the new setting."""

        if action.name != SET_SOURCE_ACTION:
            raise SubmissionParseError(f"expected action: {SET_SOURCE_ACTION}")
        source = SourceSetting(
            node_plus=_required_str_argument(action, "node_plus"),
            node_minus=_required_str_argument(action, "node_minus"),
            voltage_V=_required_numeric_argument(action, "voltage_V"),
        )
        _validate_source_nodes(self._state.definition, source)
        if abs(source.voltage_V) > MAX_SOURCE_ABS_VOLTAGE:
            raise SubmissionParseError(
                f"voltage_V must be between {-MAX_SOURCE_ABS_VOLTAGE:g} and "
                f"{MAX_SOURCE_ABS_VOLTAGE:g}"
            )
        self._source_setting = source
        return source

    def measure_voltage_from_action(self, action: ParsedAction) -> VoltageMeasurement:
        """Apply a voltage probe action."""

        if action.name != MEASURE_VOLTAGE_ACTION:
            raise SubmissionParseError(f"expected action: {MEASURE_VOLTAGE_ACTION}")
        node_a = _required_str_argument(action, "node_a")
        node_b = _required_str_argument(action, "node_b")
        _validate_public_node(self._state.definition, node_a)
        _validate_public_node(self._state.definition, node_b)
        result = self.simulate_current_state()
        voltage = _node_voltage(result, node_a) - _node_voltage(result, node_b)
        return VoltageMeasurement(node_a=node_a, node_b=node_b, voltage_V=voltage)

    def measure_current_from_action(self, action: ParsedAction) -> CurrentMeasurement:
        """Apply a component current probe action."""

        if action.name != MEASURE_CURRENT_ACTION:
            raise SubmissionParseError(f"expected action: {MEASURE_CURRENT_ACTION}")
        component_id = _required_str_argument(action, "component")
        self._state.definition.component(component_id)
        result = self.simulate_current_state()
        current = result.component_currents_A.get(component_id)
        if current is None:
            raise CircuitSimulationError(
                f"component current unavailable: {component_id}"
            )
        return CurrentMeasurement(component_id=component_id, current_A=current)

    def replace_component_from_action(self, action: ParsedAction) -> ReplacementSpec:
        """Apply a component replacement action."""

        if action.name != REPLACE_COMPONENT_ACTION:
            raise SubmissionParseError(f"expected action: {REPLACE_COMPONENT_ACTION}")
        component_id = _required_str_argument(action, "component")
        requested_kind = _required_str_argument(action, "kind")
        nominal = self._state.definition.component(component_id)
        if requested_kind != nominal.kind:
            raise SubmissionParseError(
                f"replacement kind for {component_id} must be {nominal.kind}"
            )
        replacement = _replacement_from_action(nominal, action)
        _validate_nominal_replacement(nominal, replacement)
        self._repairs[component_id] = replacement
        return replacement

    def final_answer_from_action(
        self, action: ParsedAction
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Extract final diagnosis strings from a parsed action."""

        if action.name != FINAL_ANSWER_ACTION:
            raise SubmissionParseError(f"expected action: {FINAL_ANSWER_ACTION}")
        faults = _string_tuple_argument(
            action, plural_name="faults", singular_name="fault"
        )
        repairs = _string_tuple_argument(
            action, plural_name="repairs", singular_name="repair"
        )
        return faults, repairs

    def evaluate_final_answer(
        self,
        submitted_faults: tuple[str, ...],
        submitted_repairs: tuple[str, ...],
    ) -> FinalCircuitEvaluation:
        """Evaluate the current repair state and submitted diagnosis."""

        expected_faults = tuple(fault.fault_id for fault in self._state.faults)
        expected_repairs = tuple(fault.repair_code for fault in self._state.faults)
        diagnosis_correct = set(submitted_faults) == set(expected_faults) and set(
            submitted_repairs
        ) == set(expected_repairs)
        try:
            simulation = simulate_circuit(
                definition=self._state.definition,
                faults=self._state.faults,
                repairs=self._repairs,
                source_setting=self._state.definition.target_source,
            )
            check_results = evaluate_target_checks(self._state.definition, simulation)
            simulation_error = None
        except CircuitSimulationError as error:
            check_results = ()
            simulation_error = str(error)
        target_restored = (
            simulation_error is None
            and len(check_results) > 0
            and all(result.passed for result in check_results)
        )
        return FinalCircuitEvaluation(
            target_restored=target_restored,
            diagnosis_correct=diagnosis_correct,
            submitted_faults=submitted_faults,
            submitted_repairs=submitted_repairs,
            expected_faults=expected_faults,
            expected_repairs=expected_repairs,
            check_results=check_results,
            simulation_error=simulation_error,
        )

    def simulate_current_state(self) -> SimulationResult:
        """Simulate the current faulty circuit, repairs, and bench source."""

        return simulate_circuit(
            definition=self._state.definition,
            faults=self._state.faults,
            repairs=self._repairs,
            source_setting=self._source_setting,
        )


def state_from_instance(instance: TaskInstance) -> CircuitDiagnosisState:
    """Build authoritative circuit state from an immutable instance."""

    definition = circuit_definition_from_mapping(
        _mapping_field(instance.public_payload, "circuit")
    )
    fault_values = _sequence_field(instance.privileged_payload, "faults")
    faults = tuple(
        fault_from_mapping(cast(Mapping[str, object], fault_value))
        for fault_value in fault_values
    )
    fault_range = _mapping_field(instance.public_payload, "fault_count_range")
    return CircuitDiagnosisState(
        definition=definition,
        faults=faults,
        fault_count_range=(
            _int_field(fault_range, "min"),
            _int_field(fault_range, "max"),
        ),
    )


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
        for component_value in _sequence_field(values, "components")
    )
    target_checks = tuple(
        target_check_from_mapping(cast(Mapping[str, object], check_value))
        for check_value in _sequence_field(values, "target_checks")
    )
    return CircuitDefinition(
        circuit_id=_str_field(values, "circuit_id"),
        description=_str_field(values, "description"),
        nodes=tuple(_str_sequence(values, "nodes")),
        ground_node=_str_field(values, "ground_node"),
        components=components,
        target_source=target_source,
        target_checks=target_checks,
    )


def source_from_mapping(values: Mapping[str, object]) -> SourceSetting:
    """Build a source setting from plain instance data."""

    return SourceSetting(
        node_plus=_str_field(values, "node_plus"),
        node_minus=_str_field(values, "node_minus"),
        voltage_V=_float_field(values, "voltage_V"),
    )


def component_from_mapping(values: Mapping[str, object]) -> CircuitComponent:
    """Build a public circuit component from plain instance data."""

    return CircuitComponent(
        component_id=_str_field(values, "component_id"),
        kind=_str_field(values, "kind"),
        node_a=_str_field(values, "node_a"),
        node_b=_str_field(values, "node_b"),
        parameters=_mapping_field(values, "parameters"),
    )


def target_check_from_mapping(values: Mapping[str, object]) -> TargetCheck:
    """Build a public target check from plain instance data."""

    return TargetCheck(
        check_id=_str_field(values, "check_id"),
        kind=_str_field(values, "kind"),
        parameters=_mapping_field(values, "parameters"),
    )


def fault_from_mapping(values: Mapping[str, object]) -> FaultSpec:
    """Build a privileged fault spec from plain instance data."""

    return FaultSpec(
        fault_id=_str_field(values, "fault_id"),
        component_id=_str_field(values, "component_id"),
        fault_type=_str_field(values, "fault_type"),
        parameters=_mapping_field(values, "parameters"),
        repair_code=_str_field(values, "repair_code"),
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


def parse_action_submission(submission: TaskSubmission) -> ParsedAction:
    """Parse a model submission as a structured action."""

    parsed_action = parse_core_action_submission(submission)
    if parsed_action is not None:
        return parsed_action
    raise SubmissionParseError(ACTION_SUBMISSION_PARSE_ERROR)


def simulate_circuit(
    definition: CircuitDefinition,
    faults: tuple[FaultSpec, ...],
    repairs: Mapping[str, ReplacementSpec],
    source_setting: SourceSetting | None,
) -> SimulationResult:
    """Simulate one realized DC circuit with hidden faults and repairs.

    Parameters
    ----------
    definition:
        Public nominal circuit graph.
    faults:
        Privileged hidden fault overlays.
    repairs:
        Session-local repair overlays.
    source_setting:
        Optional external source setting for this solve.

    Returns
    -------
    SimulationResult
        Solved node voltages, component currents, powers, and diode states.
    """

    if source_setting is not None:
        _validate_source_nodes(definition, source_setting)
    components = _realized_components(definition, faults, repairs)
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
            solved = _solve_linear_circuit(
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


def evaluate_target_checks(
    definition: CircuitDefinition, simulation: SimulationResult
) -> tuple[TargetCheckResult, ...]:
    """Evaluate public post-repair target behavior checks."""

    results: list[TargetCheckResult] = []
    for check in definition.target_checks:
        if check.kind == "voltage_between":
            node_a = _str_parameter(check.parameters, "node_a")
            node_b = _str_parameter(check.parameters, "node_b")
            value = _node_voltage(simulation, node_a) - _node_voltage(
                simulation, node_b
            )
            lower_bound = _float_parameter(check.parameters, "min_V")
            upper_bound = _float_parameter(check.parameters, "max_V")
        elif check.kind == "current_range":
            component_id = _str_parameter(check.parameters, "component")
            value = _component_current(simulation, component_id)
            lower_bound = _float_parameter(check.parameters, "min_A")
            upper_bound = _float_parameter(check.parameters, "max_A")
        elif check.kind == "power_max":
            component_id = _str_parameter(check.parameters, "component")
            value = _component_power(simulation, component_id)
            lower_bound = None
            upper_bound = _float_parameter(check.parameters, "max_W")
        else:
            raise CircuitSimulationError(f"unsupported target check: {check.kind}")
        passed = _value_passes_bounds(value, lower_bound, upper_bound)
        results.append(
            TargetCheckResult(
                check_id=check.check_id,
                kind=check.kind,
                passed=passed,
                value=value,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
        )
    return tuple(results)


def nominal_replacement_for_component(component: CircuitComponent) -> ReplacementSpec:
    """Return the nominal repair overlay for a public component."""

    return ReplacementSpec(
        component_id=component.component_id,
        kind=component.kind,
        parameters=_nominal_replacement_parameters(component),
    )


def canonical_repair_code(component: CircuitComponent) -> str:
    """Return the canonical repair code for a component."""

    if component.kind == "resistor":
        return (
            f"replace_{component.component_id}_"
            f"{_format_code_number(_float_parameter(component.parameters, 'value_ohm'))}"
            "_ohm"
        )
    if component.kind == "capacitor":
        return (
            f"replace_{component.component_id}_"
            f"{_format_code_number(_float_parameter(component.parameters, 'value_F'))}"
            "_F"
        )
    if component.kind == "switch":
        closed = _bool_parameter(component.parameters, "closed")
        state = "closed" if closed else "open"
        return f"replace_{component.component_id}_{state}"
    if component.kind == "voltage_source":
        return (
            f"replace_{component.component_id}_"
            f"{_format_code_number(_float_parameter(component.parameters, 'voltage_V'))}"
            "_V"
        )
    return f"replace_{component.component_id}_{component.kind}"


def target_check_public_payloads(
    check_results: tuple[TargetCheckResult, ...],
) -> tuple[dict[str, object], ...]:
    """Return trainer-safe final check result payloads."""

    return tuple(
        {
            "check_id": result.check_id,
            "kind": result.kind,
            "passed": result.passed,
        }
        for result in check_results
    )


def target_check_debug_payloads(
    check_results: tuple[TargetCheckResult, ...],
) -> tuple[dict[str, object], ...]:
    """Return privileged final check result payloads."""

    return tuple(
        {
            "check_id": result.check_id,
            "kind": result.kind,
            "passed": result.passed,
            "value": result.value,
            "lower_bound": result.lower_bound,
            "upper_bound": result.upper_bound,
        }
        for result in check_results
    )


def _realized_components(
    definition: CircuitDefinition,
    faults: tuple[FaultSpec, ...],
    repairs: Mapping[str, ReplacementSpec],
) -> tuple[_RealizedComponent, ...]:
    """Return components after applying hidden faults and repairs."""

    fault_by_component = {fault.component_id: fault for fault in faults}
    realized: list[_RealizedComponent] = []
    for component in definition.components:
        replacement = repairs.get(component.component_id)
        if replacement is not None:
            realized.append(_component_from_replacement(component, replacement))
            continue
        fault = fault_by_component.get(component.component_id)
        if fault is None:
            realized.append(_component_from_nominal(component))
        else:
            realized.append(_component_from_fault(component, fault))
    return tuple(realized)


def _component_from_nominal(component: CircuitComponent) -> _RealizedComponent:
    """Return a realized component matching the nominal public component."""

    return _RealizedComponent(
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
) -> _RealizedComponent:
    """Return a realized component from a repair overlay."""

    if replacement.kind != nominal.kind:
        raise CircuitSimulationError(
            f"replacement kind mismatch for {nominal.component_id}"
        )
    return _RealizedComponent(
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
) -> _RealizedComponent:
    """Return a realized component after one hidden fault."""

    if fault.fault_type == "open_resistor":
        _require_kind(component, "resistor", fault)
        return _faulted_component(component, "open", component.parameters, 1.0)
    if fault.fault_type == "wrong_value":
        _require_kind(component, "resistor", fault)
        return _faulted_component(
            component,
            component.kind,
            {"value_ohm": _float_parameter(fault.parameters, "value_ohm")},
            1.0,
        )
    if fault.fault_type == "shorted_capacitor":
        _require_kind(component, "capacitor", fault)
        return _faulted_component(component, "short", component.parameters, 1.0)
    if fault.fault_type == "reversed_diode":
        _require_kind(component, "diode", fault)
        return _RealizedComponent(
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
            {"closed": _bool_parameter(fault.parameters, "closed")},
            1.0,
        )
    if fault.fault_type == "internal_source_resistance":
        _require_kind(component, "voltage_source", fault)
        parameters = dict(component.parameters)
        parameters["internal_resistance_ohm"] = _float_parameter(
            fault.parameters, "internal_resistance_ohm"
        )
        return _faulted_component(component, component.kind, parameters, 1.0)
    raise CircuitSimulationError(f"unsupported fault type: {fault.fault_type}")


def _faulted_component(
    component: CircuitComponent,
    effective_kind: str,
    parameters: Mapping[str, object],
    measurement_sign: float,
) -> _RealizedComponent:
    """Build a realized component from fault details."""

    return _RealizedComponent(
        component_id=component.component_id,
        public_kind=component.kind,
        effective_kind=effective_kind,
        node_a=component.node_a,
        node_b=component.node_b,
        parameters=parameters,
        measurement_sign=measurement_sign,
    )


def _linear_elements(
    components: tuple[_RealizedComponent, ...],
    source_setting: SourceSetting | None,
    diode_states: Mapping[str, bool],
) -> tuple[tuple[_LinearElement, ...], tuple[str, ...]]:
    """Build linear elements for one diode-state assignment."""

    elements: list[_LinearElement] = []
    internal_nodes: list[str] = []
    if source_setting is not None:
        elements.append(
            _LinearElement(
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
            if _bool_parameter(component.parameters, "closed"):
                elements.append(_zero_volt_element(component))
        elif component.effective_kind == "diode":
            if diode_states[component.component_id]:
                elements.append(
                    _LinearElement(
                        element_type="voltage_source",
                        component_id=component.component_id,
                        node_a=component.node_a,
                        node_b=component.node_b,
                        value=_float_parameter(component.parameters, "forward_drop_V"),
                        measurement_sign=component.measurement_sign,
                    )
                )
        elif component.effective_kind == "voltage_source":
            source_elements, source_nodes = _voltage_source_elements(component)
            elements.extend(source_elements)
            internal_nodes.extend(source_nodes)
        elif component.effective_kind == "current_source":
            elements.append(
                _LinearElement(
                    element_type="current_source",
                    component_id=component.component_id,
                    node_a=component.node_a,
                    node_b=component.node_b,
                    value=_float_parameter(component.parameters, "current_A"),
                    measurement_sign=component.measurement_sign,
                )
            )
        else:
            raise CircuitSimulationError(
                f"unsupported component kind: {component.effective_kind}"
            )
    return tuple(elements), tuple(internal_nodes)


def _resistor_element(component: _RealizedComponent) -> _LinearElement:
    """Return a resistor linear element."""

    resistance = _float_parameter(component.parameters, "value_ohm")
    if resistance < MIN_RESISTANCE_OHM:
        raise CircuitSimulationError(
            f"resistance too small for {component.component_id}"
        )
    return _LinearElement(
        element_type="resistor",
        component_id=component.component_id,
        node_a=component.node_a,
        node_b=component.node_b,
        value=resistance,
        measurement_sign=component.measurement_sign,
    )


def _zero_volt_element(component: _RealizedComponent) -> _LinearElement:
    """Return an ideal zero-volt branch element."""

    return _LinearElement(
        element_type="voltage_source",
        component_id=component.component_id,
        node_a=component.node_a,
        node_b=component.node_b,
        value=0.0,
        measurement_sign=component.measurement_sign,
    )


def _voltage_source_elements(
    component: _RealizedComponent,
) -> tuple[tuple[_LinearElement, ...], tuple[str, ...]]:
    """Return elements for an ideal or internally resisted voltage source."""

    voltage = _float_parameter(component.parameters, "voltage_V")
    internal_resistance = _optional_float_parameter(
        component.parameters, "internal_resistance_ohm", 0.0
    )
    if internal_resistance < 0.0:
        raise CircuitSimulationError("internal source resistance cannot be negative")
    if internal_resistance < MIN_RESISTANCE_OHM:
        return (
            (
                _LinearElement(
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
            _LinearElement(
                element_type="voltage_source",
                component_id=f"{component.component_id}__ideal",
                node_a=internal_node,
                node_b=component.node_b,
                value=voltage,
                measurement_sign=1.0,
            ),
            _LinearElement(
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


def _solve_linear_circuit(
    definition: CircuitDefinition,
    internal_nodes: tuple[str, ...],
    elements: tuple[_LinearElement, ...],
) -> _SolvedLinearCircuit:
    """Solve a linear circuit using dense modified nodal analysis."""

    node_names = tuple(
        node
        for node in (*definition.nodes, *internal_nodes)
        if node != definition.ground_node
    )
    node_index = {node: index for index, node in enumerate(node_names)}
    voltage_elements = tuple(
        element for element in elements if element.element_type == "voltage_source"
    )
    size = len(node_names) + len(voltage_elements)
    if size == 0:
        return _SolvedLinearCircuit(
            node_voltages_V={definition.ground_node: 0.0},
            branch_currents_A={},
        )
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    rhs = [0.0 for _ in range(size)]

    for element in elements:
        if element.element_type == "resistor":
            _stamp_resistor(matrix, node_index, element)
        elif element.element_type == "current_source":
            _stamp_current_source(rhs, node_index, element)

    for offset, element in enumerate(voltage_elements):
        branch_index = len(node_names) + offset
        _stamp_voltage_source(matrix, rhs, node_index, branch_index, element)

    solution = _solve_dense_system(matrix, rhs)
    node_voltages = {definition.ground_node: 0.0}
    for node, index in node_index.items():
        node_voltages[node] = solution[index]
    branch_currents = {
        element.component_id: solution[len(node_names) + offset]
        for offset, element in enumerate(voltage_elements)
    }
    return _SolvedLinearCircuit(
        node_voltages_V=node_voltages,
        branch_currents_A=branch_currents,
    )


def _stamp_resistor(
    matrix: list[list[float]],
    node_index: Mapping[str, int],
    element: _LinearElement,
) -> None:
    """Stamp one resistor into an MNA matrix."""

    conductance = 1.0 / element.value
    a_index = node_index.get(element.node_a)
    b_index = node_index.get(element.node_b)
    if a_index is not None:
        matrix[a_index][a_index] += conductance
    if b_index is not None:
        matrix[b_index][b_index] += conductance
    if a_index is not None and b_index is not None:
        matrix[a_index][b_index] -= conductance
        matrix[b_index][a_index] -= conductance


def _stamp_current_source(
    rhs: list[float],
    node_index: Mapping[str, int],
    element: _LinearElement,
) -> None:
    """Stamp one current source into an MNA right-hand side."""

    current = element.value
    a_index = node_index.get(element.node_a)
    b_index = node_index.get(element.node_b)
    if a_index is not None:
        rhs[a_index] -= current
    if b_index is not None:
        rhs[b_index] += current


def _stamp_voltage_source(
    matrix: list[list[float]],
    rhs: list[float],
    node_index: Mapping[str, int],
    branch_index: int,
    element: _LinearElement,
) -> None:
    """Stamp one ideal voltage source into an MNA matrix."""

    a_index = node_index.get(element.node_a)
    b_index = node_index.get(element.node_b)
    if a_index is not None:
        matrix[a_index][branch_index] += 1.0
        matrix[branch_index][a_index] += 1.0
    if b_index is not None:
        matrix[b_index][branch_index] -= 1.0
        matrix[branch_index][b_index] -= 1.0
    rhs[branch_index] = element.value


def _solve_dense_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve a dense linear system with partial-pivot Gaussian elimination."""

    size = len(rhs)
    augmented = [row[:] + [rhs[index]] for index, row in enumerate(matrix)]
    for pivot_col in range(size):
        pivot_row = max(
            range(pivot_col, size), key=lambda row: abs(augmented[row][pivot_col])
        )
        pivot_value = augmented[pivot_row][pivot_col]
        if abs(pivot_value) < 1.0e-12:
            raise CircuitSimulationError("singular circuit matrix")
        if pivot_row != pivot_col:
            augmented[pivot_col], augmented[pivot_row] = (
                augmented[pivot_row],
                augmented[pivot_col],
            )
        pivot_value = augmented[pivot_col][pivot_col]
        for column in range(pivot_col, size + 1):
            augmented[pivot_col][column] /= pivot_value
        for row in range(size):
            if row == pivot_col:
                continue
            factor = augmented[row][pivot_col]
            if factor == 0.0:
                continue
            for column in range(pivot_col, size + 1):
                augmented[row][column] -= factor * augmented[pivot_col][column]
    return [augmented[row][size] for row in range(size)]


def _candidate_diode_states(
    diode_components: tuple[_RealizedComponent, ...],
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
    solved: _SolvedLinearCircuit,
    diode_components: tuple[_RealizedComponent, ...],
    diode_states: Mapping[str, bool],
) -> bool:
    """Return whether diode inequalities match the assumed state."""

    for component in diode_components:
        voltage = _node_voltage(solved, component.node_a) - _node_voltage(
            solved, component.node_b
        )
        drop = _float_parameter(component.parameters, "forward_drop_V")
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
    components: tuple[_RealizedComponent, ...],
    solved: _SolvedLinearCircuit,
    diode_states: Mapping[str, bool],
) -> SimulationResult:
    """Build public-component currents and powers from a raw solution."""

    currents: dict[str, float] = {}
    powers: dict[str, float] = {}
    for component in components:
        current = _component_current_from_solution(component, solved, diode_states)
        voltage = _node_voltage(solved, component.node_a) - _node_voltage(
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
    component: _RealizedComponent,
    solved: _SolvedLinearCircuit,
    diode_states: Mapping[str, bool],
) -> float:
    """Return component current in the public nominal orientation."""

    if component.effective_kind == "open":
        return 0.0
    if component.effective_kind == "resistor":
        resistance = _float_parameter(component.parameters, "value_ohm")
        current = (
            _node_voltage(solved, component.node_a)
            - _node_voltage(solved, component.node_b)
        ) / resistance
        return component.measurement_sign * current
    if component.effective_kind == "capacitor":
        return 0.0
    if component.effective_kind in {"short", "switch"}:
        if component.effective_kind == "switch" and not _bool_parameter(
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
        internal_resistance = _optional_float_parameter(
            component.parameters, "internal_resistance_ohm", 0.0
        )
        if internal_resistance >= MIN_RESISTANCE_OHM:
            internal_node = f"__{component.component_id}_internal_plus"
            current = (
                _node_voltage(solved, component.node_a)
                - _node_voltage(solved, internal_node)
            ) / internal_resistance
            return component.measurement_sign * current
        return component.measurement_sign * solved.branch_currents_A.get(
            component.component_id, 0.0
        )
    if component.effective_kind == "current_source":
        return component.measurement_sign * _float_parameter(
            component.parameters, "current_A"
        )
    raise CircuitSimulationError(
        f"unsupported component kind: {component.effective_kind}"
    )


def _replacement_from_action(
    nominal: CircuitComponent, action: ParsedAction
) -> ReplacementSpec:
    """Build a replacement spec from action arguments."""

    if nominal.kind == "resistor":
        return ReplacementSpec(
            component_id=nominal.component_id,
            kind=nominal.kind,
            parameters={
                "value_ohm": _positive_numeric_argument(action, "value_ohm"),
            },
        )
    if nominal.kind == "capacitor":
        return ReplacementSpec(
            component_id=nominal.component_id,
            kind=nominal.kind,
            parameters={"value_F": _positive_numeric_argument(action, "value_F")},
        )
    if nominal.kind == "diode":
        return ReplacementSpec(
            component_id=nominal.component_id,
            kind=nominal.kind,
            parameters={
                "forward_drop_V": _positive_numeric_argument(action, "forward_drop_V"),
            },
        )
    if nominal.kind == "switch":
        value = action.arguments.get("closed")
        if not isinstance(value, bool):
            raise SubmissionParseError("closed must be a boolean")
        return ReplacementSpec(
            component_id=nominal.component_id,
            kind=nominal.kind,
            parameters={"closed": value},
        )
    if nominal.kind == "voltage_source":
        return ReplacementSpec(
            component_id=nominal.component_id,
            kind=nominal.kind,
            parameters={
                "voltage_V": _required_numeric_argument(action, "voltage_V"),
                "internal_resistance_ohm": 0.0,
            },
        )
    if nominal.kind == "current_source":
        return ReplacementSpec(
            component_id=nominal.component_id,
            kind=nominal.kind,
            parameters={"current_A": _required_numeric_argument(action, "current_A")},
        )
    raise SubmissionParseError(f"unsupported replacement kind: {nominal.kind}")


def _validate_nominal_replacement(
    nominal: CircuitComponent, replacement: ReplacementSpec
) -> None:
    """Reject replacements that do not match the nominal schematic component."""

    nominal_parameters = _nominal_replacement_parameters(nominal)
    if replacement.kind != nominal.kind:
        raise SubmissionParseError(
            f"replacement kind for {nominal.component_id} must be {nominal.kind}"
        )
    for name, expected_value in nominal_parameters.items():
        submitted_value = replacement.parameters.get(name)
        if not _parameter_values_match(submitted_value, expected_value):
            raise SubmissionParseError(
                f"{name} for {nominal.component_id} must match nominal value "
                f"{_format_code_number(float(expected_value))}"
                if isinstance(expected_value, int | float)
                and not isinstance(expected_value, bool)
                else f"{name} for {nominal.component_id} must match nominal value"
            )


def _parameter_values_match(submitted_value: object, expected_value: object) -> bool:
    """Return whether a repair parameter exactly matches the nominal value."""

    if isinstance(expected_value, bool):
        return submitted_value is expected_value
    if isinstance(expected_value, int | float) and not isinstance(expected_value, bool):
        if isinstance(submitted_value, bool) or not isinstance(
            submitted_value, int | float
        ):
            return False
        return abs(float(submitted_value) - float(expected_value)) <= 1.0e-9
    return submitted_value == expected_value


def _nominal_replacement_parameters(component: CircuitComponent) -> dict[str, object]:
    """Return the repair parameters that restore one nominal component."""

    if component.kind == "resistor":
        return {"value_ohm": _float_parameter(component.parameters, "value_ohm")}
    if component.kind == "capacitor":
        return {"value_F": _float_parameter(component.parameters, "value_F")}
    if component.kind == "diode":
        return {
            "forward_drop_V": _float_parameter(component.parameters, "forward_drop_V")
        }
    if component.kind == "switch":
        return {"closed": _bool_parameter(component.parameters, "closed")}
    if component.kind == "voltage_source":
        return {
            "voltage_V": _float_parameter(component.parameters, "voltage_V"),
            "internal_resistance_ohm": 0.0,
        }
    if component.kind == "current_source":
        return {"current_A": _float_parameter(component.parameters, "current_A")}
    raise ValueError(f"unsupported component kind: {component.kind}")


def _value_passes_bounds(
    value: float, lower_bound: float | None, upper_bound: float | None
) -> bool:
    """Return whether a value passes optional inclusive bounds."""

    if lower_bound is not None and value < lower_bound - CHECK_TOLERANCE:
        return False
    if upper_bound is not None and value > upper_bound + CHECK_TOLERANCE:
        return False
    return True


def _node_voltage(
    simulation: SimulationResult | _SolvedLinearCircuit, node_name: str
) -> float:
    """Return one solved node voltage."""

    voltage = simulation.node_voltages_V.get(node_name)
    if voltage is None:
        raise CircuitSimulationError(f"node voltage unavailable: {node_name}")
    return voltage


def _component_current(simulation: SimulationResult, component_id: str) -> float:
    """Return one solved component current."""

    current = simulation.component_currents_A.get(component_id)
    if current is None:
        raise CircuitSimulationError(f"component current unavailable: {component_id}")
    return current


def _component_power(simulation: SimulationResult, component_id: str) -> float:
    """Return one solved component power."""

    power = simulation.component_powers_W.get(component_id)
    if power is None:
        raise CircuitSimulationError(f"component power unavailable: {component_id}")
    return power


def _require_kind(component: CircuitComponent, kind: str, fault: FaultSpec) -> None:
    """Validate that a fault targets the required component kind."""

    if component.kind != kind:
        raise CircuitSimulationError(
            f"fault {fault.fault_id} requires {kind}, got {component.kind}"
        )


def _validate_source_nodes(
    definition: CircuitDefinition, source: SourceSetting
) -> None:
    """Validate source terminal nodes."""

    _validate_public_node(definition, source.node_plus)
    _validate_public_node(definition, source.node_minus)
    if source.node_plus == source.node_minus:
        raise SubmissionParseError("source terminals must be distinct")
    if not isfinite(source.voltage_V):
        raise SubmissionParseError("voltage_V must be finite")


def _validate_public_node(definition: CircuitDefinition, node_name: str) -> None:
    """Validate one public node label."""

    if node_name not in definition.nodes:
        raise SubmissionParseError(f"unknown node: {node_name}")


def _required_str_argument(action: ParsedAction, name: str) -> str:
    """Read one required string action argument."""

    value = action.arguments.get(name)
    if isinstance(value, str) and value != "":
        return value
    raise SubmissionParseError(f"{name} must be a non-empty string")


def _required_numeric_argument(action: ParsedAction, name: str) -> float:
    """Read one required numeric action argument."""

    value = action.arguments.get(name)
    if value is None:
        raise SubmissionParseError(f"missing argument: {name}")
    if isinstance(value, bool):
        raise SubmissionParseError(f"{name} must be numeric")
    if isinstance(value, int | float):
        numeric_value = float(value)
    else:
        raise SubmissionParseError(f"{name} must be numeric")
    if not isfinite(numeric_value):
        raise SubmissionParseError(f"{name} must be finite")
    return numeric_value


def _positive_numeric_argument(action: ParsedAction, name: str) -> float:
    """Read one positive numeric action argument."""

    value = _required_numeric_argument(action, name)
    if value <= 0.0:
        raise SubmissionParseError(f"{name} must be positive")
    return value


def _string_tuple_argument(
    action: ParsedAction, plural_name: str, singular_name: str
) -> tuple[str, ...]:
    """Read a plural list or singular string final-answer argument."""

    plural_value = action.arguments.get(plural_name)
    if isinstance(plural_value, list | tuple):
        values: list[str] = []
        for item in plural_value:
            if not isinstance(item, str) or item == "":
                raise SubmissionParseError(f"{plural_name} entries must be strings")
            values.append(item)
        return tuple(values)
    singular_value = action.arguments.get(singular_name)
    if isinstance(singular_value, str) and singular_value != "":
        return (singular_value,)
    raise SubmissionParseError(
        f"missing argument: {plural_name} must be a list of strings"
    )


def _str_sequence(values: Mapping[str, object], name: str) -> tuple[str, ...]:
    """Return a required string sequence field from a mapping."""

    raw_values = _sequence_field(values, name)
    strings: list[str] = []
    for value in raw_values:
        if not isinstance(value, str) or value == "":
            raise TypeError(f"{name} entries must be non-empty strings")
        strings.append(value)
    return tuple(strings)


def _sequence_field(values: Mapping[str, object], name: str) -> tuple[object, ...]:
    """Return a required sequence field from a mapping."""

    value = values[name]
    if isinstance(value, list | tuple):
        return tuple(value)
    raise TypeError(f"{name} must be a sequence")


def _mapping_field(values: Mapping[str, object], name: str) -> Mapping[str, object]:
    """Return a required mapping field from a mapping."""

    value = values[name]
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"{name} must be a mapping")


def _str_field(values: Mapping[str, object], name: str) -> str:
    """Return a required string field from a mapping."""

    value = values[name]
    if isinstance(value, str) and value != "":
        return value
    raise TypeError(f"{name} must be a non-empty string")


def _int_field(values: Mapping[str, object], name: str) -> int:
    """Return a required integer field from a mapping."""

    value = values[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _float_field(values: Mapping[str, object], name: str) -> float:
    """Return a required numeric field from a mapping."""

    return _numeric_value(values[name], name)


def _str_parameter(values: Mapping[str, object], name: str) -> str:
    """Return a required string parameter."""

    return _str_field(values, name)


def _float_parameter(values: Mapping[str, object], name: str) -> float:
    """Return a required numeric parameter."""

    return _float_field(values, name)


def _optional_float_parameter(
    values: Mapping[str, object], name: str, fallback: float
) -> float:
    """Return an optional numeric parameter."""

    value = values.get(name)
    if value is None:
        return fallback
    return _numeric_value(value, name)


def _bool_parameter(values: Mapping[str, object], name: str) -> bool:
    """Return a required boolean parameter."""

    value = values[name]
    if isinstance(value, bool):
        return value
    raise TypeError(f"{name} must be a boolean")


def _numeric_value(value: object, name: str) -> float:
    """Return a finite numeric value."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    if isinstance(value, int | float):
        numeric_value = float(value)
        if isfinite(numeric_value):
            return numeric_value
        raise TypeError(f"{name} must be finite")
    raise TypeError(f"{name} must be numeric")


def _format_code_number(value: float) -> str:
    """Return a stable compact number for repair labels."""

    return f"{value:g}"
