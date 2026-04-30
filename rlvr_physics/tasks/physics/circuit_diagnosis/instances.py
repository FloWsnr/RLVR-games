"""Immutable instance construction for the circuit diagnosis task."""

from collections.abc import Callable, Mapping
from dataclasses import replace
from random import Random

from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.core.payloads import stable_hash
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone import (
    GROUND_NODE,
    CircuitComponent,
    CircuitDefinition,
    FaultSpec,
    SourceSetting,
    TargetCheck,
    canonical_repair_code,
    circuit_definition_payload,
    evaluate_target_checks,
    fault_payload,
    nominal_replacement_for_component,
    simulate_circuit,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.budgets import (
    circuit_budget_limits,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.specs import (
    CIRCUIT_DIAGNOSIS_DOMAIN,
    CIRCUIT_DIAGNOSIS_KIND,
    CircuitDiagnosisConfig,
    config_parameters,
    validate_circuit_diagnosis_config,
)

NodePositions = Mapping[str, tuple[float, float]]
TemplateCandidate = tuple[CircuitDefinition, NodePositions, tuple[FaultSpec, ...]]


def build_circuit_diagnosis_instance(
    seed: int, config: CircuitDiagnosisConfig
) -> TaskInstance:
    """Build one deterministic circuit diagnosis task instance.

    Parameters
    ----------
    seed:
        Generator seed for reproducible public and privileged payloads.
    config:
        Public generation, rollout, and verifier configuration.

    Returns
    -------
    TaskInstance
        Immutable scalar task instance ready for session creation.
    """

    validate_circuit_diagnosis_config(config)
    rng = Random(seed)
    builders = _template_builders()
    start_index = rng.randrange(len(builders))
    for offset in range(len(builders)):
        template_name, builder = builders[(start_index + offset) % len(builders)]
        definition, node_positions, fault_options = builder(rng, config)
        max_fault_count = min(config.max_fault_count, len(fault_options))
        min_fault_count = min(config.min_fault_count, max_fault_count)
        for _ in range(80):
            fault_count = rng.randint(min_fault_count, max_fault_count)
            shuffled_options = list(fault_options)
            rng.shuffle(shuffled_options)
            faults = tuple(shuffled_options[:fault_count])
            if _fault_components_are_unique(faults) and _candidate_is_valid(
                definition, faults
            ):
                return _instance_from_candidate(
                    seed=seed,
                    config=config,
                    template_name=template_name,
                    definition=definition,
                    node_positions=node_positions,
                    faults=faults,
                )
    raise RuntimeError("could not sample a valid circuit diagnosis instance")


def _instance_from_candidate(
    seed: int,
    config: CircuitDiagnosisConfig,
    template_name: str,
    definition: CircuitDefinition,
    node_positions: NodePositions,
    faults: tuple[FaultSpec, ...],
) -> TaskInstance:
    """Build a task instance from a validated candidate."""

    public_payload: dict[str, object] = {
        "circuit": circuit_definition_payload(definition),
        "schematic": {"node_positions": _node_positions_payload(node_positions)},
        "fault_count_range": {
            "min": config.min_fault_count,
            "max": config.max_fault_count,
        },
        "required_answer": {
            "action": "final_answer",
            "fields": ("faults", "repairs"),
        },
    }
    privileged_payload: dict[str, object] = {
        "faults": [fault_payload(fault) for fault in faults],
        "template_name": template_name,
    }
    task_hash = stable_hash(
        {
            "kind": CIRCUIT_DIAGNOSIS_KIND,
            "seed": seed,
            "config": config_parameters(config),
            "public_payload": public_payload,
            "privileged_payload": privileged_payload,
        }
    )[:16]
    return TaskInstance(
        task_id=f"circuit-diagnosis-v1-{task_hash}",
        kind=CIRCUIT_DIAGNOSIS_KIND,
        domain=CIRCUIT_DIAGNOSIS_DOMAIN,
        seed=seed,
        public_payload=public_payload,
        privileged_payload=privileged_payload,
        budget_limits=circuit_budget_limits(
            turn_budget=config.turn_budget,
            probe_budget=config.probe_budget,
            repair_budget=config.repair_budget,
            final_answer_budget=config.final_answer_budget,
        ),
        timeout_seconds=config.timeout_seconds,
        token_budget=config.token_budget,
        metadata={
            "task_family": "circuit_diagnosis",
            "template": template_name,
            "difficulty": "diagnostic",
        },
    )


def _template_builders() -> tuple[
    tuple[str, Callable[[Random, CircuitDiagnosisConfig], TemplateCandidate]],
    ...,
]:
    """Return curated circuit template builders."""

    return (
        ("resistor_divider", _resistor_divider_template),
        ("led_limiter", _led_limiter_template),
        ("rc_dc_node", _rc_dc_node_template),
        ("switched_load", _switched_load_template),
        ("internal_source", _internal_source_template),
        ("bridge_balance", _bridge_balance_template),
    )


def _resistor_divider_template(
    rng: Random, config: CircuitDiagnosisConfig
) -> TemplateCandidate:
    """Return a resistor divider diagnosis template."""

    r1 = rng.choice((1000.0, 1500.0, 2200.0))
    r2 = rng.choice((1000.0, 2200.0, 3300.0))
    source = SourceSetting(node_plus="VIN", node_minus=GROUND_NODE, voltage_V=5.0)
    base = CircuitDefinition(
        circuit_id="resistor_divider",
        description="A two-resistor divider should produce a stable OUT voltage.",
        nodes=("VIN", "OUT", GROUND_NODE),
        ground_node=GROUND_NODE,
        components=(
            _resistor("R1", "VIN", "OUT", r1),
            _resistor("R2", "OUT", GROUND_NODE, r2),
        ),
        target_source=source,
        target_checks=(),
    )
    node_positions = {
        "VIN": (160.0, 140.0),
        "OUT": (420.0, 140.0),
        GROUND_NODE: (420.0, 430.0),
    }
    nominal = simulate_circuit(base, (), {}, source)
    out_voltage = nominal.node_voltages_V["OUT"]
    definition = replace(
        base,
        target_checks=(
            _voltage_check("out_voltage", "OUT", GROUND_NODE, out_voltage, config),
        ),
    )
    return (
        definition,
        node_positions,
        (
            _fault(definition, "R1", "open_resistor", {}),
            _fault(definition, "R2", "open_resistor", {}),
            _fault(definition, "R1", "wrong_value", {"value_ohm": r1 * 3.0}),
            _fault(definition, "R2", "wrong_value", {"value_ohm": r2 * 0.35}),
        ),
    )


def _led_limiter_template(
    rng: Random, config: CircuitDiagnosisConfig
) -> TemplateCandidate:
    """Return an LED current limiter diagnosis template."""

    _ = rng
    source = SourceSetting(node_plus="VIN", node_minus=GROUND_NODE, voltage_V=5.0)
    base = CircuitDefinition(
        circuit_id="led_limiter",
        description="A resistor should limit LED current from a 5 V source.",
        nodes=("VIN", "LED_A", GROUND_NODE),
        ground_node=GROUND_NODE,
        components=(
            _resistor("R1", "VIN", "LED_A", 330.0, max_power_W=0.08),
            _diode("D1", "LED_A", GROUND_NODE, 2.0, max_current_A=0.02),
        ),
        target_source=source,
        target_checks=(),
    )
    node_positions = {
        "VIN": (150.0, 170.0),
        "LED_A": (430.0, 170.0),
        GROUND_NODE: (650.0, 390.0),
    }
    nominal = simulate_circuit(base, (), {}, source)
    diode_current = nominal.component_currents_A["D1"]
    definition = replace(
        base,
        target_checks=(
            _current_check("led_current", "D1", diode_current, config),
            TargetCheck(
                check_id="r1_power_limit",
                kind="power_max",
                parameters={"component": "R1", "max_W": 0.08},
            ),
        ),
    )
    return (
        definition,
        node_positions,
        (
            _fault(definition, "R1", "open_resistor", {}),
            _fault(definition, "R1", "wrong_value", {"value_ohm": 1000.0}),
            _fault(definition, "D1", "reversed_diode", {}),
        ),
    )


def _rc_dc_node_template(
    rng: Random, config: CircuitDiagnosisConfig
) -> TemplateCandidate:
    """Return an RC DC-node diagnosis template."""

    _ = rng
    source = SourceSetting(node_plus="VIN", node_minus=GROUND_NODE, voltage_V=5.0)
    base = CircuitDefinition(
        circuit_id="rc_dc_node",
        description="A capacitor should be open at DC while a divider sets OUT.",
        nodes=("VIN", "OUT", GROUND_NODE),
        ground_node=GROUND_NODE,
        components=(
            _resistor("R1", "VIN", "OUT", 1000.0),
            _resistor("R2", "OUT", GROUND_NODE, 2000.0),
            _capacitor("C1", "OUT", GROUND_NODE, 1.0e-6),
        ),
        target_source=source,
        target_checks=(),
    )
    node_positions = {
        "VIN": (150.0, 140.0),
        "OUT": (430.0, 140.0),
        GROUND_NODE: (430.0, 430.0),
    }
    nominal = simulate_circuit(base, (), {}, source)
    out_voltage = nominal.node_voltages_V["OUT"]
    definition = replace(
        base,
        target_checks=(
            _voltage_check("out_voltage", "OUT", GROUND_NODE, out_voltage, config),
        ),
    )
    return (
        definition,
        node_positions,
        (
            _fault(definition, "C1", "shorted_capacitor", {}),
            _fault(definition, "R1", "open_resistor", {}),
            _fault(definition, "R2", "wrong_value", {"value_ohm": 750.0}),
        ),
    )


def _switched_load_template(
    rng: Random, config: CircuitDiagnosisConfig
) -> TemplateCandidate:
    """Return a switched load diagnosis template."""

    _ = rng
    source = SourceSetting(node_plus="VIN", node_minus=GROUND_NODE, voltage_V=5.0)
    base = CircuitDefinition(
        circuit_id="switched_load",
        description="A closed switch should connect the source to the load.",
        nodes=("VIN", "LOAD", GROUND_NODE),
        ground_node=GROUND_NODE,
        components=(
            _switch("SW1", "VIN", "LOAD", closed=True),
            _resistor("RLOAD", "LOAD", GROUND_NODE, 1000.0, max_power_W=0.05),
        ),
        target_source=source,
        target_checks=(),
    )
    node_positions = {
        "VIN": (150.0, 150.0),
        "LOAD": (430.0, 150.0),
        GROUND_NODE: (430.0, 430.0),
    }
    nominal = simulate_circuit(base, (), {}, source)
    load_current = nominal.component_currents_A["RLOAD"]
    definition = replace(
        base,
        target_checks=(
            _current_check("load_current", "RLOAD", load_current, config),
            TargetCheck(
                check_id="load_power_limit",
                kind="power_max",
                parameters={"component": "RLOAD", "max_W": 0.05},
            ),
        ),
    )
    return (
        definition,
        node_positions,
        (
            _fault(definition, "SW1", "broken_switch", {"closed": False}),
            _fault(definition, "RLOAD", "open_resistor", {}),
            _fault(definition, "RLOAD", "wrong_value", {"value_ohm": 3300.0}),
        ),
    )


def _internal_source_template(
    rng: Random, config: CircuitDiagnosisConfig
) -> TemplateCandidate:
    """Return an internal source resistance diagnosis template."""

    _ = rng
    base = CircuitDefinition(
        circuit_id="internal_source",
        description="An internal 9 V source should drive the load directly.",
        nodes=("VCC", GROUND_NODE),
        ground_node=GROUND_NODE,
        components=(
            _voltage_source("VS1", "VCC", GROUND_NODE, 9.0),
            _resistor("RLOAD", "VCC", GROUND_NODE, 1000.0, max_power_W=0.12),
        ),
        target_source=None,
        target_checks=(),
    )
    node_positions = {
        "VCC": (280.0, 160.0),
        GROUND_NODE: (280.0, 430.0),
    }
    nominal = simulate_circuit(base, (), {}, None)
    load_current = nominal.component_currents_A["RLOAD"]
    definition = replace(
        base,
        target_checks=(
            _current_check("load_current", "RLOAD", load_current, config),
            _voltage_check("vcc_voltage", "VCC", GROUND_NODE, 9.0, config),
        ),
    )
    return (
        definition,
        node_positions,
        (
            _fault(
                definition,
                "VS1",
                "internal_source_resistance",
                {"internal_resistance_ohm": 680.0},
            ),
            _fault(definition, "RLOAD", "open_resistor", {}),
            _fault(definition, "RLOAD", "wrong_value", {"value_ohm": 2200.0}),
        ),
    )


def _bridge_balance_template(
    rng: Random, config: CircuitDiagnosisConfig
) -> TemplateCandidate:
    """Return a bridge balance diagnosis template."""

    _ = rng
    source = SourceSetting(node_plus="VIN", node_minus=GROUND_NODE, voltage_V=5.0)
    base = CircuitDefinition(
        circuit_id="bridge_balance",
        description="Two divider legs should hold nodes A and B at the same voltage.",
        nodes=("VIN", "A", "B", GROUND_NODE),
        ground_node=GROUND_NODE,
        components=(
            _resistor("R1", "VIN", "A", 1000.0),
            _resistor("R2", "A", GROUND_NODE, 2000.0),
            _resistor("R3", "VIN", "B", 1500.0),
            _resistor("R4", "B", GROUND_NODE, 3000.0),
        ),
        target_source=source,
        target_checks=(),
    )
    node_positions = {
        "VIN": (150.0, 130.0),
        "A": (380.0, 200.0),
        "B": (630.0, 200.0),
        GROUND_NODE: (500.0, 440.0),
    }
    nominal = simulate_circuit(base, (), {}, source)
    bridge_voltage = nominal.node_voltages_V["A"] - nominal.node_voltages_V["B"]
    definition = replace(
        base,
        target_checks=(
            _voltage_check("bridge_balance", "A", "B", bridge_voltage, config),
        ),
    )
    return (
        definition,
        node_positions,
        (
            _fault(definition, "R2", "wrong_value", {"value_ohm": 1000.0}),
            _fault(definition, "R4", "wrong_value", {"value_ohm": 1500.0}),
            _fault(definition, "R1", "open_resistor", {}),
            _fault(definition, "R3", "open_resistor", {}),
        ),
    )


def _candidate_is_valid(
    definition: CircuitDefinition, faults: tuple[FaultSpec, ...]
) -> bool:
    """Return whether a sampled fault set satisfies task invariants."""

    try:
        nominal = simulate_circuit(definition, (), {}, definition.target_source)
        if not all(
            result.passed for result in evaluate_target_checks(definition, nominal)
        ):
            return False
        faulty = simulate_circuit(definition, faults, {}, definition.target_source)
        if all(result.passed for result in evaluate_target_checks(definition, faulty)):
            return False
        repairs = {
            fault.component_id: nominal_replacement_for_component(
                definition.component(fault.component_id)
            )
            for fault in faults
        }
        repaired = simulate_circuit(
            definition, faults, repairs, definition.target_source
        )
        if not all(
            result.passed for result in evaluate_target_checks(definition, repaired)
        ):
            return False
        if len(faults) > 1:
            for fault in faults:
                partial_repairs = {
                    fault.component_id: nominal_replacement_for_component(
                        definition.component(fault.component_id)
                    )
                }
                partial = simulate_circuit(
                    definition, faults, partial_repairs, definition.target_source
                )
                if all(
                    result.passed
                    for result in evaluate_target_checks(definition, partial)
                ):
                    return False
    except Exception:
        return False
    return True


def _fault_components_are_unique(faults: tuple[FaultSpec, ...]) -> bool:
    """Return whether no sampled faults target the same component."""

    component_ids = [fault.component_id for fault in faults]
    return len(component_ids) == len(set(component_ids))


def _node_positions_payload(node_positions: NodePositions) -> dict[str, list[float]]:
    """Return renderer-facing schematic positions as plain payload data."""

    return {
        node: [position[0], position[1]]
        for node, position in sorted(node_positions.items())
    }


def _resistor(
    component_id: str,
    node_a: str,
    node_b: str,
    value_ohm: float,
    max_power_W: float = 0.25,
) -> CircuitComponent:
    """Return a public resistor component."""

    return CircuitComponent(
        component_id=component_id,
        kind="resistor",
        node_a=node_a,
        node_b=node_b,
        parameters={"value_ohm": value_ohm, "max_power_W": max_power_W},
    )


def _capacitor(
    component_id: str, node_a: str, node_b: str, value_F: float
) -> CircuitComponent:
    """Return a public capacitor component."""

    return CircuitComponent(
        component_id=component_id,
        kind="capacitor",
        node_a=node_a,
        node_b=node_b,
        parameters={"value_F": value_F},
    )


def _diode(
    component_id: str,
    node_a: str,
    node_b: str,
    forward_drop_V: float,
    max_current_A: float,
) -> CircuitComponent:
    """Return a public diode component."""

    return CircuitComponent(
        component_id=component_id,
        kind="diode",
        node_a=node_a,
        node_b=node_b,
        parameters={
            "forward_drop_V": forward_drop_V,
            "max_current_A": max_current_A,
        },
    )


def _switch(
    component_id: str, node_a: str, node_b: str, closed: bool
) -> CircuitComponent:
    """Return a public switch component."""

    return CircuitComponent(
        component_id=component_id,
        kind="switch",
        node_a=node_a,
        node_b=node_b,
        parameters={"closed": closed},
    )


def _voltage_source(
    component_id: str, node_a: str, node_b: str, voltage_V: float
) -> CircuitComponent:
    """Return a public voltage source component."""

    return CircuitComponent(
        component_id=component_id,
        kind="voltage_source",
        node_a=node_a,
        node_b=node_b,
        parameters={"voltage_V": voltage_V, "internal_resistance_ohm": 0.0},
    )


def _fault(
    definition: CircuitDefinition,
    component_id: str,
    fault_type: str,
    parameters: dict[str, object],
) -> FaultSpec:
    """Return a privileged fault spec for a component."""

    component = definition.component(component_id)
    fault_id = _fault_id(component_id, fault_type)
    return FaultSpec(
        fault_id=fault_id,
        component_id=component_id,
        fault_type=fault_type,
        parameters=parameters,
        repair_code=canonical_repair_code(component),
    )


def _fault_id(component_id: str, fault_type: str) -> str:
    """Return a compact canonical fault label."""

    if fault_type == "open_resistor":
        return f"{component_id}_open"
    if fault_type == "wrong_value":
        return f"{component_id}_wrong_value"
    if fault_type == "shorted_capacitor":
        return f"{component_id}_shorted"
    if fault_type == "reversed_diode":
        return f"{component_id}_reversed"
    if fault_type == "broken_switch":
        return f"{component_id}_broken_open"
    if fault_type == "internal_source_resistance":
        return f"{component_id}_internal_resistance"
    return f"{component_id}_{fault_type}"


def _voltage_check(
    check_id: str,
    node_a: str,
    node_b: str,
    nominal_voltage: float,
    config: CircuitDiagnosisConfig,
) -> TargetCheck:
    """Return a target voltage range check around a nominal value."""

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
    """Return a target current range check around a nominal value."""

    tolerance = _current_tolerance(nominal_current, config)
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
    """Return a template target tolerance for a nominal value."""

    return max(
        abs(value) * config.target_tolerance_fraction, config.target_tolerance_abs
    )


def _current_tolerance(value: float, config: CircuitDiagnosisConfig) -> float:
    """Return a template target tolerance for a current value."""

    return max(abs(value) * config.target_tolerance_fraction, 1.0e-4)
