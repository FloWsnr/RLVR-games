"""Stateful rollout backbone for circuit diagnosis."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.core.submissions import ParsedAction, TaskSubmission
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.actions import (
    parse_action_submission,
    required_numeric_argument,
    required_str_argument,
    string_tuple_argument,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.constants import (
    FINAL_ANSWER_ACTION,
    MAX_SOURCE_ABS_VOLTAGE,
    MEASURE_CURRENT_ACTION,
    MEASURE_VOLTAGE_ACTION,
    REPLACE_COMPONENT_ACTION,
    SET_SOURCE_ACTION,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.errors import (
    CircuitSimulationError,
    SubmissionParseError,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.payloads import (
    state_from_instance,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.repairs import (
    replacement_from_action,
    validate_nominal_replacement,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.schema import (
    CircuitDiagnosisState,
    ReplacementSpec,
    SourceSetting,
    validate_public_node,
    validate_source_nodes,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.simulation import (
    SimulationResult,
    node_voltage,
    simulate_circuit,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.verification import (
    FinalCircuitEvaluation,
    evaluate_target_checks,
)


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
        definition = self._state.truth.public_definition
        source = SourceSetting(
            node_plus=required_str_argument(action, "node_plus"),
            node_minus=required_str_argument(action, "node_minus"),
            voltage_V=required_numeric_argument(action, "voltage_V"),
        )
        validate_source_nodes(definition, source)
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
        definition = self._state.truth.public_definition
        node_a = required_str_argument(action, "node_a")
        node_b = required_str_argument(action, "node_b")
        validate_public_node(definition, node_a)
        validate_public_node(definition, node_b)
        result = self.simulate_current_state()
        voltage = node_voltage(result, node_a) - node_voltage(result, node_b)
        return VoltageMeasurement(node_a=node_a, node_b=node_b, voltage_V=voltage)

    def measure_current_from_action(self, action: ParsedAction) -> CurrentMeasurement:
        """Apply a component current probe action."""

        if action.name != MEASURE_CURRENT_ACTION:
            raise SubmissionParseError(f"expected action: {MEASURE_CURRENT_ACTION}")
        definition = self._state.truth.public_definition
        component_id = required_str_argument(action, "component")
        definition.component(component_id)
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
        definition = self._state.truth.public_definition
        component_id = required_str_argument(action, "component")
        requested_kind = required_str_argument(action, "kind")
        nominal = definition.component(component_id)
        if requested_kind != nominal.kind:
            raise SubmissionParseError(
                f"replacement kind for {component_id} must be {nominal.kind}"
            )
        replacement = replacement_from_action(nominal, action)
        validate_nominal_replacement(nominal, replacement)
        self._repairs[component_id] = replacement
        return replacement

    def final_answer_from_action(
        self, action: ParsedAction
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Extract final diagnosis strings from a parsed action."""

        if action.name != FINAL_ANSWER_ACTION:
            raise SubmissionParseError(f"expected action: {FINAL_ANSWER_ACTION}")
        faults = string_tuple_argument(
            action, plural_name="faults", singular_name="fault"
        )
        repairs = string_tuple_argument(
            action, plural_name="repairs", singular_name="repair"
        )
        return faults, repairs

    def evaluate_final_answer(
        self,
        submitted_faults: tuple[str, ...],
        submitted_repairs: tuple[str, ...],
    ) -> FinalCircuitEvaluation:
        """Evaluate the current repair state and submitted diagnosis."""

        truth = self._state.truth
        definition = truth.public_definition
        expected_faults = tuple(fault.fault_id for fault in truth.hidden_faults)
        expected_repairs = tuple(fault.repair_code for fault in truth.hidden_faults)
        diagnosis_correct = set(submitted_faults) == set(expected_faults) and set(
            submitted_repairs
        ) == set(expected_repairs)
        try:
            simulation = simulate_circuit(
                truth=truth,
                repairs=self._repairs,
                source_setting=definition.target_source,
            )
            check_results = evaluate_target_checks(definition, simulation)
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
            truth=self._state.truth,
            repairs=self._repairs,
            source_setting=self._source_setting,
        )
