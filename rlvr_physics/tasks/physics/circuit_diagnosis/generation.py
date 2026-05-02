"""Procedural circuit generation for circuit diagnosis instances."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import isfinite
from random import Random

from rlvr_physics.core.payloads import freeze_mapping
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.constants import (
    GROUND_NODE,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.repairs import (
    canonical_repair_code,
    nominal_replacement_for_component,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.schema import (
    CircuitComponent,
    CircuitDefinition,
    CircuitTruth,
    FaultSpec,
    SourceSetting,
    TargetCheck,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.simulation import (
    SimulationResult,
    simulate_circuit,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.solver import (
    LinearElement,
    diagnose_linear_circuit,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.values import (
    float_parameter,
    format_code_number,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.verification import (
    evaluate_target_checks,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.specs import (
    CircuitDiagnosisConfig,
)

VIN_NODE = "VIN"
SOURCE_VOLTAGE_V = 5.0
_IMPOSSIBLE_DEPTH = 10_000


@dataclass(frozen=True)
class GeneratedCircuit:
    """Generated and validated circuit candidate.

    Parameters
    ----------
    definition:
        Public nominal circuit definition derived from the canonical graph.
    hidden_fault:
        Privileged fault chosen after task validation.
    eligible_faults:
        Public final-answer fault candidates that passed task validation and
        may be sampled as the hidden fault.
    public_metrics:
        Trainer-safe generation metrics.
    debug_metrics:
        Privileged validation and difficulty metrics.
    """

    definition: CircuitDefinition
    hidden_fault: FaultSpec
    eligible_faults: tuple[FaultSpec, ...]
    public_metrics: Mapping[str, object]
    debug_metrics: Mapping[str, object]

    def __post_init__(self) -> None:
        """Freeze generated payload mappings."""

        object.__setattr__(self, "public_metrics", freeze_mapping(self.public_metrics))
        object.__setattr__(self, "debug_metrics", freeze_mapping(self.debug_metrics))


@dataclass(frozen=True)
class _ResistorStub:
    """Generated resistor before value assignment."""

    component_id: str
    node_a: str
    node_b: str


@dataclass(frozen=True)
class _GeneratedTopology:
    """Canonical generated topology before physical values."""

    components: tuple[_ResistorStub, ...]
    nets: tuple[str, ...]


@dataclass(frozen=True)
class _MeasurementSpec:
    """One canonical diagnosis measurement dimension."""

    kind: str
    target: str
    reference: str | None


@dataclass(frozen=True)
class _FaultSignature:
    """Fault simulation signature over canonical measurement dimensions."""

    fault: FaultSpec
    values: tuple[float, ...]


class _GenerationReject(Exception):
    """Rejected generated candidate with a stable reason string."""


class _TopologyBuilder:
    """Mutable helper that allocates generated nets and components."""

    def __init__(self) -> None:
        """Initialize an empty generated topology builder."""

        self._net_index = 0
        self._component_index = 0
        self._components: list[_ResistorStub] = []
        self._internal_nets: list[str] = []

    @property
    def components(self) -> tuple[_ResistorStub, ...]:
        """Return allocated components in public ID order."""

        return tuple(self._components)

    @property
    def nets(self) -> tuple[str, ...]:
        """Return public terminal and internal net names."""

        return (VIN_NODE, *self._internal_nets, GROUND_NODE)

    def new_net(self) -> str:
        """Allocate one internal net ID."""

        self._net_index += 1
        net = f"N{self._net_index}"
        self._internal_nets.append(net)
        return net

    def new_resistor(self, node_a: str, node_b: str) -> None:
        """Allocate one resistor leaf between two nets."""

        self._component_index += 1
        component_id = f"R{self._component_index}"
        self._components.append(
            _ResistorStub(component_id=component_id, node_a=node_a, node_b=node_b)
        )


def build_generated_circuit(
    seed: int, config: CircuitDiagnosisConfig
) -> GeneratedCircuit:
    """Build a validated procedural circuit.

    Parameters
    ----------
    seed:
        Deterministic generator seed.
    config:
        Circuit diagnosis generator and rollout configuration.

    Returns
    -------
    GeneratedCircuit
        Validated generated circuit ready to become a task instance.
    """

    rng = Random(seed)
    reject_counts: dict[str, int] = {}
    for attempt_index in range(config.generator_attempt_limit):
        try:
            return _build_candidate(
                rng=rng,
                config=config,
                attempt_count=attempt_index + 1,
                reject_counts=reject_counts,
            )
        except _GenerationReject as error:
            reason = str(error)
            reject_counts[reason] = reject_counts.get(reason, 0) + 1
    raise RuntimeError(
        "could not sample a valid procedural circuit diagnosis instance: "
        f"{reject_counts}"
    )


def diagnosis_options_payload(
    definition: CircuitDefinition, eligible_faults: tuple[FaultSpec, ...]
) -> dict[str, object]:
    """Return public final-answer vocabulary for a circuit definition.

    Parameters
    ----------
    definition:
        Public nominal circuit definition.
    eligible_faults:
        Fault candidates that passed generator validation and may be sampled as
        the hidden fault.

    Returns
    -------
    dict[str, object]
        Public fault and repair option payloads. These list the allowed labels
        without identifying which fault is hidden in the physical circuit.
    """

    fault_options = tuple(_fault_option_payload(fault) for fault in eligible_faults)
    eligible_component_ids = {fault.component_id for fault in eligible_faults}
    repair_options = tuple(
        _repair_option_payload(component)
        for component in definition.components
        if component.component_id in eligible_component_ids
    )
    return {
        "fault_ids": tuple(str(option["fault_id"]) for option in fault_options),
        "repair_codes": tuple(str(option["repair_code"]) for option in repair_options),
        "faults": fault_options,
        "repairs": repair_options,
    }


def _build_candidate(
    rng: Random,
    config: CircuitDiagnosisConfig,
    attempt_count: int,
    reject_counts: Mapping[str, int],
) -> GeneratedCircuit:
    """Build one candidate or reject it with a stable reason."""

    topology = _sample_topology(config.component_count, rng)
    _validate_topology(topology, config.component_count)
    definition = _definition_from_topology(topology, rng)
    diagnostics = diagnose_linear_circuit(
        definition=definition,
        internal_nodes=(),
        elements=_nominal_linear_elements(definition),
    )
    if not diagnostics.full_rank:
        raise _GenerationReject("singular_mna")
    if diagnostics.condition_number > config.max_mna_condition_number:
        raise _GenerationReject("ill_conditioned_mna")

    nominal = _simulate_nominal(definition)
    _validate_nominal_operating_point(definition, nominal, config)
    definition = replace(
        definition,
        target_checks=_target_checks_from_nominal(definition, nominal, config),
    )
    nominal_signature, measurement_specs = _measurement_signature(definition, nominal)
    fault_signatures = _validated_fault_signatures(
        definition=definition,
        candidates=_candidate_faults(definition),
        nominal_signature=nominal_signature,
        config=config,
    )
    if len(fault_signatures) < 2:
        raise _GenerationReject("too_few_distinguishable_faults")

    difficulty = _minimum_measurement_depth(
        signatures=fault_signatures,
        measurement_indexes=tuple(range(len(measurement_specs))),
        tolerance=config.min_observable_delta,
    )
    if difficulty == _IMPOSSIBLE_DEPTH:
        raise _GenerationReject("indistinguishable_fault_set")
    if difficulty < config.min_diagnosis_measurements:
        raise _GenerationReject("diagnosis_too_easy")
    if difficulty > config.max_diagnosis_measurements:
        raise _GenerationReject("diagnosis_too_hard")

    hidden_signature = rng.choice(fault_signatures)
    eligible_faults = tuple(signature.fault for signature in fault_signatures)
    public_metrics = {
        "generator": "procedural_passive_resistor_network_v1",
        "n_components": len(definition.components),
        "n_nets": len(definition.nodes),
        "diagnosis_measurement_depth": difficulty,
    }
    debug_metrics = {
        "attempt_count": attempt_count,
        "rejections": dict(reject_counts),
        "cycle_rank": _cycle_rank(definition),
        "mna_condition_number": diagnostics.condition_number,
        "mna_rank": diagnostics.rank,
        "mna_size": diagnostics.size,
        "candidate_faults": len(_candidate_faults(definition)),
        "distinguishable_faults": len(fault_signatures),
        "measurement_count": len(measurement_specs),
        "measurement_specs": [_measurement_payload(spec) for spec in measurement_specs],
        "optimal_measurement_depth": difficulty,
        "hidden_fault": hidden_signature.fault.fault_id,
    }
    return GeneratedCircuit(
        definition=definition,
        hidden_fault=hidden_signature.fault,
        eligible_faults=eligible_faults,
        public_metrics=public_metrics,
        debug_metrics=debug_metrics,
    )


def _sample_topology(component_count: int, rng: Random) -> _GeneratedTopology:
    """Sample a passive two-terminal topology with exact component count."""

    builder = _TopologyBuilder()
    _gen_passive_two_terminal(
        node_a=VIN_NODE,
        node_b=GROUND_NODE,
        component_count=component_count,
        rng=rng,
        builder=builder,
    )
    return _GeneratedTopology(components=builder.components, nets=builder.nets)


def _gen_passive_two_terminal(
    node_a: str,
    node_b: str,
    component_count: int,
    rng: Random,
    builder: _TopologyBuilder,
) -> None:
    """Generate a two-terminal passive resistor subnetwork."""

    if component_count == 1:
        builder.new_resistor(node_a, node_b)
        return None

    choices = ["series", "parallel"]
    if component_count >= 5:
        choices.append("bridge")
    rule = rng.choice(choices)

    if rule == "series":
        left_count = rng.randint(1, component_count - 1)
        right_count = component_count - left_count
        join_net = builder.new_net()
        _gen_passive_two_terminal(node_a, join_net, left_count, rng, builder)
        _gen_passive_two_terminal(join_net, node_b, right_count, rng, builder)
        return None

    if rule == "parallel":
        upper_count = rng.randint(1, component_count - 1)
        lower_count = component_count - upper_count
        _gen_passive_two_terminal(node_a, node_b, upper_count, rng, builder)
        _gen_passive_two_terminal(node_a, node_b, lower_count, rng, builder)
        return None

    parts = _positive_partition(component_count, 5, rng)
    upper_mid = builder.new_net()
    lower_mid = builder.new_net()
    _gen_passive_two_terminal(node_a, upper_mid, parts[0], rng, builder)
    _gen_passive_two_terminal(upper_mid, node_b, parts[1], rng, builder)
    _gen_passive_two_terminal(node_a, lower_mid, parts[2], rng, builder)
    _gen_passive_two_terminal(lower_mid, node_b, parts[3], rng, builder)
    _gen_passive_two_terminal(upper_mid, lower_mid, parts[4], rng, builder)


def _positive_partition(total: int, parts: int, rng: Random) -> tuple[int, ...]:
    """Return a random positive integer partition."""

    cuts = sorted(rng.sample(range(1, total), parts - 1))
    values: list[int] = []
    previous = 0
    for cut in cuts:
        values.append(cut - previous)
        previous = cut
    values.append(total - previous)
    return tuple(values)


def _validate_topology(topology: _GeneratedTopology, component_count: int) -> None:
    """Validate generated graph topology before numerical solving."""

    if topology.nets.count(GROUND_NODE) != 1:
        raise _GenerationReject("invalid_ground_count")
    if len(topology.components) != component_count:
        raise _GenerationReject("wrong_component_count")
    net_set = set(topology.nets)
    if len(net_set) != len(topology.nets):
        raise _GenerationReject("duplicate_net")
    incidents = {net: 0 for net in topology.nets}
    graph = {net: set[str]() for net in topology.nets}
    for component in topology.components:
        if component.node_a not in net_set or component.node_b not in net_set:
            raise _GenerationReject("component_references_unknown_net")
        incidents[component.node_a] += 1
        incidents[component.node_b] += 1
        graph[component.node_a].add(component.node_b)
        graph[component.node_b].add(component.node_a)
    for net, incident_count in incidents.items():
        if net not in {VIN_NODE, GROUND_NODE} and incident_count < 2:
            raise _GenerationReject("floating_internal_net")
    reachable = _reachable_nets(graph, VIN_NODE)
    if reachable != net_set:
        raise _GenerationReject("floating_component_island")
    if GROUND_NODE not in reachable:
        raise _GenerationReject("ground_unreachable")


def _definition_from_topology(
    topology: _GeneratedTopology, rng: Random
) -> CircuitDefinition:
    """Assign resistor values and build a public circuit definition."""

    components = tuple(_resistor_from_stub(stub, rng) for stub in topology.components)
    return CircuitDefinition(
        circuit_id="procedural_passive_resistor_network",
        description=(
            "A generated passive resistor network should match its nominal "
            "5 V DC operating point after repair."
        ),
        nodes=topology.nets,
        ground_node=GROUND_NODE,
        components=components,
        target_source=SourceSetting(
            node_plus=VIN_NODE,
            node_minus=GROUND_NODE,
            voltage_V=SOURCE_VOLTAGE_V,
        ),
        target_checks=(),
    )


def _resistor_from_stub(stub: _ResistorStub, rng: Random) -> CircuitComponent:
    """Return a valued resistor component from a generated stub."""

    value = rng.choice(
        (
            220.0,
            330.0,
            470.0,
            680.0,
            1000.0,
            1500.0,
            2200.0,
            3300.0,
            4700.0,
            6800.0,
            10000.0,
        )
    )
    return CircuitComponent(
        component_id=stub.component_id,
        kind="resistor",
        node_a=stub.node_a,
        node_b=stub.node_b,
        parameters={"value_ohm": value, "max_power_W": 0.25},
    )


def _nominal_linear_elements(
    definition: CircuitDefinition,
) -> tuple[LinearElement, ...]:
    """Return linear elements for the generated nominal circuit."""

    elements: list[LinearElement] = []
    if definition.target_source is not None:
        source = definition.target_source
        elements.append(
            LinearElement(
                element_type="voltage_source",
                component_id="__bench_source__",
                node_a=source.node_plus,
                node_b=source.node_minus,
                value=source.voltage_V,
                measurement_sign=1.0,
            )
        )
    for component in definition.components:
        elements.append(
            LinearElement(
                element_type="resistor",
                component_id=component.component_id,
                node_a=component.node_a,
                node_b=component.node_b,
                value=float_parameter(component.parameters, "value_ohm"),
                measurement_sign=1.0,
            )
        )
    return tuple(elements)


def _simulate_nominal(definition: CircuitDefinition) -> SimulationResult:
    """Return the nominal generated circuit operating point."""

    try:
        return simulate_circuit(
            CircuitTruth(
                public_definition=definition,
                hidden_faults=(),
                fault_count_range=(0, 0),
            ),
            {},
            definition.target_source,
        )
    except Exception as error:
        raise _GenerationReject("nominal_simulation_failed") from error


def _validate_nominal_operating_point(
    definition: CircuitDefinition,
    nominal: SimulationResult,
    config: CircuitDiagnosisConfig,
) -> None:
    """Reject numerically valid but unhelpful nominal circuits."""

    for value in (
        *nominal.node_voltages_V.values(),
        *nominal.component_currents_A.values(),
        *nominal.component_powers_W.values(),
    ):
        if not isfinite(value):
            raise _GenerationReject("nonfinite_nominal_solution")
    for component_id, power in nominal.component_powers_W.items():
        component = definition.component(component_id)
        max_power_value = component.parameters.get("max_power_W", 0.25)
        if isinstance(max_power_value, bool) or not isinstance(
            max_power_value, int | float
        ):
            raise _GenerationReject("invalid_power_limit")
        max_power = float(max_power_value)
        if power > max_power:
            raise _GenerationReject("unsafe_power")
    source = definition.target_source
    if source is None:
        raise _GenerationReject("missing_target_source")
    useful_nodes = [
        node
        for node, voltage in nominal.node_voltages_V.items()
        if node not in {source.node_plus, source.node_minus}
        and voltage > config.min_observable_delta
        and voltage < source.voltage_V - config.min_observable_delta
    ]
    if len(useful_nodes) == 0:
        raise _GenerationReject("boring_node_voltages")


def _target_checks_from_nominal(
    definition: CircuitDefinition,
    nominal: SimulationResult,
    config: CircuitDiagnosisConfig,
) -> tuple[TargetCheck, ...]:
    """Return public target behavior checks from the nominal solution."""

    checks: list[TargetCheck] = []
    for node in definition.nodes:
        if node in {VIN_NODE, GROUND_NODE}:
            continue
        checks.append(
            _voltage_check(
                check_id=f"{node}_voltage",
                node_a=node,
                node_b=GROUND_NODE,
                nominal_voltage=nominal.node_voltages_V[node],
                config=config,
            )
        )
    for component in definition.components:
        checks.append(
            _current_check(
                check_id=f"{component.component_id}_current",
                component_id=component.component_id,
                nominal_current=nominal.component_currents_A[component.component_id],
                config=config,
            )
        )
    return tuple(checks)


def _candidate_faults(definition: CircuitDefinition) -> tuple[FaultSpec, ...]:
    """Return deterministic resistor fault candidates for a definition."""

    faults: list[FaultSpec] = []
    for component in definition.components:
        value = float_parameter(component.parameters, "value_ohm")
        faults.extend(
            (
                _fault(
                    component, "open_resistor", f"{component.component_id}_open", {}
                ),
                _fault(
                    component,
                    "shorted_resistor",
                    f"{component.component_id}_short",
                    {},
                ),
                _fault(
                    component,
                    "wrong_value",
                    f"{component.component_id}_wrong_high",
                    {"value_ohm": value * 3.0},
                ),
                _fault(
                    component,
                    "wrong_value",
                    f"{component.component_id}_wrong_low",
                    {"value_ohm": max(value / 3.0, 10.0)},
                ),
            )
        )
    return tuple(faults)


def _fault(
    component: CircuitComponent,
    fault_type: str,
    fault_id: str,
    parameters: Mapping[str, object],
) -> FaultSpec:
    """Return one privileged generated fault candidate."""

    return FaultSpec(
        fault_id=fault_id,
        component_id=component.component_id,
        fault_type=fault_type,
        parameters=parameters,
        repair_code=canonical_repair_code(component),
    )


def _fault_option_payload(fault: FaultSpec) -> dict[str, object]:
    """Return public option metadata for one possible fault."""

    payload: dict[str, object] = {
        "fault_id": fault.fault_id,
        "component": fault.component_id,
        "fault_type": fault.fault_type,
        "repair_code": fault.repair_code,
        "description": _fault_description(fault),
    }
    if len(fault.parameters) > 0:
        payload["parameters"] = dict(fault.parameters)
    return payload


def _repair_option_payload(component: CircuitComponent) -> dict[str, object]:
    """Return public option metadata for one possible nominal repair."""

    replacement = nominal_replacement_for_component(component)
    parameters = dict(replacement.parameters)
    return {
        "repair_code": canonical_repair_code(component),
        "component": component.component_id,
        "kind": replacement.kind,
        "parameters": parameters,
        "action": {
            "action": "replace_component",
            "arguments": {
                "component": component.component_id,
                "kind": replacement.kind,
                **parameters,
            },
        },
    }


def _fault_description(fault: FaultSpec) -> str:
    """Return a concise public description for one possible fault."""

    if fault.fault_type == "open_resistor":
        return f"{fault.component_id} is open circuit"
    if fault.fault_type == "shorted_resistor":
        return f"{fault.component_id} is shorted"
    if fault.fault_type == "wrong_value":
        value = float_parameter(fault.parameters, "value_ohm")
        return f"{fault.component_id} resistance is {format_code_number(value)} ohm"
    return f"{fault.component_id} has fault type {fault.fault_type}"


def _validated_fault_signatures(
    definition: CircuitDefinition,
    candidates: tuple[FaultSpec, ...],
    nominal_signature: tuple[float, ...],
    config: CircuitDiagnosisConfig,
) -> tuple[_FaultSignature, ...]:
    """Return fault signatures that satisfy task validity checks."""

    signatures: list[_FaultSignature] = []
    for fault in candidates:
        try:
            truth = _truth_for_fault(definition, fault)
            faulty = simulate_circuit(truth, {}, definition.target_source)
            if all(
                result.passed for result in evaluate_target_checks(definition, faulty)
            ):
                continue
            repairs = {
                fault.component_id: nominal_replacement_for_component(
                    definition.component(fault.component_id)
                )
            }
            repaired = simulate_circuit(truth, repairs, definition.target_source)
            if not all(
                result.passed for result in evaluate_target_checks(definition, repaired)
            ):
                continue
            signature, _ = _measurement_signature(definition, faulty)
        except Exception:
            continue
        if _max_signature_delta(signature, nominal_signature) <= (
            config.min_observable_delta
        ):
            continue
        signatures.append(_FaultSignature(fault=fault, values=signature))
    return tuple(signatures)


def _truth_for_fault(definition: CircuitDefinition, fault: FaultSpec) -> CircuitTruth:
    """Return privileged truth for one generated fault."""

    return CircuitTruth(
        public_definition=definition,
        hidden_faults=(fault,),
        fault_count_range=(1, 1),
    )


def _measurement_signature(
    definition: CircuitDefinition,
    simulation: SimulationResult,
) -> tuple[tuple[float, ...], tuple[_MeasurementSpec, ...]]:
    """Return canonical measurement values and their public specs."""

    values: list[float] = []
    specs: list[_MeasurementSpec] = []
    for node in definition.nodes:
        if node == GROUND_NODE:
            continue
        values.append(simulation.node_voltages_V[node])
        specs.append(
            _MeasurementSpec(kind="voltage", target=node, reference=GROUND_NODE)
        )
    for component in definition.components:
        values.append(simulation.component_currents_A[component.component_id])
        specs.append(
            _MeasurementSpec(
                kind="current", target=component.component_id, reference=None
            )
        )
    return tuple(values), tuple(specs)


def _minimum_measurement_depth(
    signatures: tuple[_FaultSignature, ...],
    measurement_indexes: tuple[int, ...],
    tolerance: float,
) -> int:
    """Return minimum worst-case measurements needed to identify candidates."""

    indexes = tuple(range(len(signatures)))
    memo: dict[tuple[tuple[int, ...], tuple[int, ...]], int] = {}
    return _minimum_depth_recursive(
        signatures=signatures,
        candidate_indexes=indexes,
        measurement_indexes=measurement_indexes,
        tolerance=tolerance,
        memo=memo,
    )


def _minimum_depth_recursive(
    signatures: tuple[_FaultSignature, ...],
    candidate_indexes: tuple[int, ...],
    measurement_indexes: tuple[int, ...],
    tolerance: float,
    memo: dict[tuple[tuple[int, ...], tuple[int, ...]], int],
) -> int:
    """Recursively score a finite measurement decision tree."""

    if len(candidate_indexes) <= 1:
        return 0
    if len(measurement_indexes) == 0:
        return _IMPOSSIBLE_DEPTH
    key = (candidate_indexes, measurement_indexes)
    cached = memo.get(key)
    if cached is not None:
        return cached

    best_depth = _IMPOSSIBLE_DEPTH
    for measurement_index in measurement_indexes:
        partitions = _partition_by_measurement(
            signatures, candidate_indexes, measurement_index, tolerance
        )
        if len(partitions) <= 1:
            continue
        remaining = tuple(
            index for index in measurement_indexes if index != measurement_index
        )
        branch_depths = [
            _minimum_depth_recursive(
                signatures=signatures,
                candidate_indexes=partition,
                measurement_indexes=remaining,
                tolerance=tolerance,
                memo=memo,
            )
            for partition in partitions
        ]
        worst_branch = max(branch_depths)
        if worst_branch != _IMPOSSIBLE_DEPTH:
            best_depth = min(best_depth, 1 + worst_branch)

    memo[key] = best_depth
    return best_depth


def _partition_by_measurement(
    signatures: tuple[_FaultSignature, ...],
    candidate_indexes: tuple[int, ...],
    measurement_index: int,
    tolerance: float,
) -> tuple[tuple[int, ...], ...]:
    """Partition candidates by one noisy measurement dimension."""

    representatives: list[float] = []
    groups: list[list[int]] = []
    for candidate_index in candidate_indexes:
        value = signatures[candidate_index].values[measurement_index]
        placed = False
        for group_index, representative in enumerate(representatives):
            if abs(value - representative) <= tolerance:
                groups[group_index].append(candidate_index)
                placed = True
                break
        if not placed:
            representatives.append(value)
            groups.append([candidate_index])
    return tuple(tuple(group) for group in groups)


def _voltage_check(
    check_id: str,
    node_a: str,
    node_b: str,
    nominal_voltage: float,
    config: CircuitDiagnosisConfig,
) -> TargetCheck:
    """Return a public voltage range check around a nominal value."""

    tolerance = _target_tolerance(nominal_voltage, config)
    return TargetCheck(
        check_id=check_id,
        kind="voltage_between",
        parameters={
            "node_a": node_a,
            "node_b": node_b,
            "min_V": nominal_voltage - tolerance,
            "max_V": nominal_voltage + tolerance,
        },
    )


def _current_check(
    check_id: str,
    component_id: str,
    nominal_current: float,
    config: CircuitDiagnosisConfig,
) -> TargetCheck:
    """Return a public current range check around a nominal value."""

    tolerance = max(abs(nominal_current) * config.target_tolerance_fraction, 1.0e-4)
    return TargetCheck(
        check_id=check_id,
        kind="current_range",
        parameters={
            "component": component_id,
            "min_A": nominal_current - tolerance,
            "max_A": nominal_current + tolerance,
        },
    )


def _target_tolerance(value: float, config: CircuitDiagnosisConfig) -> float:
    """Return an absolute target tolerance for a nominal value."""

    return max(
        abs(value) * config.target_tolerance_fraction,
        config.target_tolerance_abs,
    )


def _reachable_nets(graph: Mapping[str, set[str]], start: str) -> set[str]:
    """Return all nets reachable from a starting net."""

    seen = {start}
    frontier = [start]
    while len(frontier) > 0:
        current = frontier.pop()
        for neighbor in graph[current]:
            if neighbor in seen:
                continue
            seen.add(neighbor)
            frontier.append(neighbor)
    return seen


def _cycle_rank(definition: CircuitDefinition) -> int:
    """Return graph cycle rank for a connected two-terminal network."""

    return len(definition.components) - len(definition.nodes) + 1


def _measurement_payload(spec: _MeasurementSpec) -> dict[str, object]:
    """Return debug payload data for one measurement dimension."""

    payload: dict[str, object] = {"kind": spec.kind, "target": spec.target}
    if spec.reference is not None:
        payload["reference"] = spec.reference
    return payload


def _max_signature_delta(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    """Return max absolute difference between two signatures."""

    return max(abs(first[index] - second[index]) for index in range(len(first)))
