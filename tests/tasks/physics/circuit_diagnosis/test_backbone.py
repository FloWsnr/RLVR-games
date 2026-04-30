"""Tests for circuit diagnosis simulation and verification."""

from dataclasses import replace

from rlvr_physics.tasks.physics.circuit_diagnosis.backbone import (
    GROUND_NODE,
    CircuitComponent,
    CircuitDefinition,
    FaultSpec,
    SourceSetting,
    canonical_repair_code,
    evaluate_target_checks,
    nominal_replacement_for_component,
    simulate_circuit,
    state_from_instance,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.instances import (
    build_circuit_diagnosis_instance,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.specs import DEFAULT_CONFIG


def test_solver_computes_resistor_divider_voltage() -> None:
    definition = _divider_definition()

    result = simulate_circuit(definition, (), {}, definition.target_source)

    assert result.node_voltages_V["OUT"] == 2.5
    assert result.component_currents_A["R1"] == 0.0025
    assert result.component_currents_A["R2"] == 0.0025


def test_hidden_faults_fail_and_nominal_repairs_restore_target() -> None:
    instance = build_circuit_diagnosis_instance(seed=4, config=DEFAULT_CONFIG)
    state = state_from_instance(instance)
    faulty = simulate_circuit(
        state.definition, state.faults, {}, state.definition.target_source
    )
    repairs = {
        fault.component_id: nominal_replacement_for_component(
            state.definition.component(fault.component_id)
        )
        for fault in state.faults
    }
    repaired = simulate_circuit(
        state.definition, state.faults, repairs, state.definition.target_source
    )

    assert not all(
        result.passed for result in evaluate_target_checks(state.definition, faulty)
    )
    assert all(
        result.passed for result in evaluate_target_checks(state.definition, repaired)
    )


def test_shorted_capacitor_and_reversed_diode_are_simulated() -> None:
    rc_definition = _rc_definition()
    capacitor = rc_definition.component("C1")
    cap_fault = FaultSpec(
        fault_id="C1_shorted",
        component_id="C1",
        fault_type="shorted_capacitor",
        parameters={},
        repair_code=canonical_repair_code(capacitor),
    )
    faulty_rc = simulate_circuit(
        rc_definition, (cap_fault,), {}, rc_definition.target_source
    )
    repaired_rc = simulate_circuit(
        rc_definition,
        (cap_fault,),
        {"C1": nominal_replacement_for_component(capacitor)},
        rc_definition.target_source,
    )

    assert faulty_rc.node_voltages_V["OUT"] == 0.0
    assert repaired_rc.node_voltages_V["OUT"] > 3.0

    led_definition = _led_definition()
    diode = led_definition.component("D1")
    diode_fault = FaultSpec(
        fault_id="D1_reversed",
        component_id="D1",
        fault_type="reversed_diode",
        parameters={},
        repair_code=canonical_repair_code(diode),
    )
    faulty_led = simulate_circuit(
        led_definition, (diode_fault,), {}, led_definition.target_source
    )
    repaired_led = simulate_circuit(
        led_definition,
        (diode_fault,),
        {"D1": nominal_replacement_for_component(diode)},
        led_definition.target_source,
    )

    assert faulty_led.component_currents_A["D1"] == 0.0
    assert repaired_led.component_currents_A["D1"] > 0.008


def test_broken_switch_and_internal_source_resistance_are_simulated() -> None:
    switch_definition = _switch_definition()
    switch = switch_definition.component("SW1")
    switch_fault = FaultSpec(
        fault_id="SW1_broken_open",
        component_id="SW1",
        fault_type="broken_switch",
        parameters={"closed": False},
        repair_code=canonical_repair_code(switch),
    )
    faulty_switch = simulate_circuit(
        switch_definition, (switch_fault,), {}, switch_definition.target_source
    )
    repaired_switch = simulate_circuit(
        switch_definition,
        (switch_fault,),
        {"SW1": nominal_replacement_for_component(switch)},
        switch_definition.target_source,
    )

    assert faulty_switch.component_currents_A["RLOAD"] == 0.0
    assert repaired_switch.component_currents_A["RLOAD"] == 0.005

    source_definition = _internal_source_definition()
    source = source_definition.component("VS1")
    source_fault = FaultSpec(
        fault_id="VS1_internal_resistance",
        component_id="VS1",
        fault_type="internal_source_resistance",
        parameters={"internal_resistance_ohm": 500.0},
        repair_code=canonical_repair_code(source),
    )
    faulty_source = simulate_circuit(source_definition, (source_fault,), {}, None)
    repaired_source = simulate_circuit(
        source_definition,
        (source_fault,),
        {"VS1": nominal_replacement_for_component(source)},
        None,
    )

    assert faulty_source.node_voltages_V["VCC"] == 6.0
    assert repaired_source.node_voltages_V["VCC"] == 9.0


def _divider_definition() -> CircuitDefinition:
    """Return a simple divider circuit."""

    source = SourceSetting(node_plus="VIN", node_minus=GROUND_NODE, voltage_V=5.0)
    return CircuitDefinition(
        circuit_id="test_divider",
        description="Divider",
        nodes=("VIN", "OUT", GROUND_NODE),
        ground_node=GROUND_NODE,
        components=(
            _resistor("R1", "VIN", "OUT", 1000.0),
            _resistor("R2", "OUT", GROUND_NODE, 1000.0),
        ),
        target_source=source,
        target_checks=(),
    )


def _rc_definition() -> CircuitDefinition:
    """Return a divider with a DC-open capacitor."""

    definition = _divider_definition()
    return replace(
        definition,
        components=(
            _resistor("R1", "VIN", "OUT", 1000.0),
            _resistor("R2", "OUT", GROUND_NODE, 2000.0),
            CircuitComponent(
                component_id="C1",
                kind="capacitor",
                node_a="OUT",
                node_b=GROUND_NODE,
                parameters={"value_F": 1.0e-6},
            ),
        ),
    )


def _led_definition() -> CircuitDefinition:
    """Return a simple LED limiter circuit."""

    source = SourceSetting(node_plus="VIN", node_minus=GROUND_NODE, voltage_V=5.0)
    return CircuitDefinition(
        circuit_id="test_led",
        description="LED",
        nodes=("VIN", "LED_A", GROUND_NODE),
        ground_node=GROUND_NODE,
        components=(
            _resistor("R1", "VIN", "LED_A", 330.0),
            CircuitComponent(
                component_id="D1",
                kind="diode",
                node_a="LED_A",
                node_b=GROUND_NODE,
                parameters={"forward_drop_V": 2.0},
            ),
        ),
        target_source=source,
        target_checks=(),
    )


def _switch_definition() -> CircuitDefinition:
    """Return a switched-load circuit."""

    source = SourceSetting(node_plus="VIN", node_minus=GROUND_NODE, voltage_V=5.0)
    return CircuitDefinition(
        circuit_id="test_switch",
        description="Switch",
        nodes=("VIN", "LOAD", GROUND_NODE),
        ground_node=GROUND_NODE,
        components=(
            CircuitComponent(
                component_id="SW1",
                kind="switch",
                node_a="VIN",
                node_b="LOAD",
                parameters={"closed": True},
            ),
            _resistor("RLOAD", "LOAD", GROUND_NODE, 1000.0),
        ),
        target_source=source,
        target_checks=(),
    )


def _internal_source_definition() -> CircuitDefinition:
    """Return a voltage-source load circuit."""

    return CircuitDefinition(
        circuit_id="test_internal_source",
        description="Source",
        nodes=("VCC", GROUND_NODE),
        ground_node=GROUND_NODE,
        components=(
            CircuitComponent(
                component_id="VS1",
                kind="voltage_source",
                node_a="VCC",
                node_b=GROUND_NODE,
                parameters={"voltage_V": 9.0, "internal_resistance_ohm": 0.0},
            ),
            _resistor("RLOAD", "VCC", GROUND_NODE, 1000.0),
        ),
        target_source=None,
        target_checks=(),
    )


def _resistor(
    component_id: str, node_a: str, node_b: str, value_ohm: float
) -> CircuitComponent:
    """Return a resistor component."""

    return CircuitComponent(
        component_id=component_id,
        kind="resistor",
        node_a=node_a,
        node_b=node_b,
        parameters={"value_ohm": value_ohm},
    )
