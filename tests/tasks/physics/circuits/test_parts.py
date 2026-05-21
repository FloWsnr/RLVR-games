"""Tests for built-in component catalog coverage."""

import importlib.util

import rlvr_physics.tasks.physics.circuits as circuits
from rlvr_physics.tasks.physics.circuits import default_part_catalog


def test_part_catalog_covers_planned_common_component_classes() -> None:
    catalog = default_part_catalog()
    required_kinds = {
        "ground",
        "power_rail",
        "test_point",
        "resistor",
        "variable_resistor",
        "capacitor",
        "polarized_capacitor",
        "inductor",
        "inductor_looped",
        "crystal",
        "lamp",
        "motor",
        "voltage_source_dc",
        "battery",
        "voltage_source_ac",
        "current_source_dc",
        "vcvs",
        "vccs",
        "diode",
        "led",
        "zener",
        "photodiode",
        "bjt_npn",
        "mosfet_n",
        "mosfet_p",
        "jfet_n",
        "jfet_p",
        "pullup_resistor",
        "pulldown_resistor",
        "ideal_switch",
        "pushbutton_switch",
        "controlled_switch",
        "relay",
        "op_amp",
        "comparator",
        "instrumentation_amplifier",
        "timer_555",
        "counter_4bit",
        "transformer",
        "connector_2",
        "voltmeter",
        "ammeter",
        "and_gate",
        "nand_gate",
        "or_gate",
        "xor_gate",
        "not_gate",
        "generic_ic",
    }

    assert required_kinds <= set(catalog)
    assert catalog["bjt_npn"].ref_prefix == "Q"
    assert catalog["mosfet_n"].ref_prefix == "Q"
    assert catalog["jfet_n"].ref_prefix == "J"


def test_obsolete_circuit_backends_are_not_public() -> None:
    """Verify deleted circuit backends are not advertised through the public API."""

    assert not hasattr(circuits, "AnalysisSupport")
    assert not hasattr(circuits, "SpiceSpec")
    assert not hasattr(circuits, "solve_dc_linear")
    assert not hasattr(circuits, "LinearDcResult")
    assert not hasattr(circuits, "UnsupportedCircuitError")
    assert not hasattr(circuits, "export_spice")
    assert not hasattr(circuits, "simulate_spice")
    assert not hasattr(circuits, "default_spice_simulator_config")
    assert not hasattr(circuits, "simulation_spec_with_supply_voltages")
    assert (
        importlib.util.find_spec("rlvr_physics.tasks.physics.circuits.solver") is None
    )
    assert importlib.util.find_spec("rlvr_physics.tasks.physics.circuits.spice") is None
    assert (
        importlib.util.find_spec("rlvr_physics.tasks.physics.circuits.spice_export")
        is None
    )
    assert (
        importlib.util.find_spec("rlvr_physics.tasks.physics.circuits.spice_sim")
        is None
    )


def test_circuit_rendering_public_surface_is_removed() -> None:
    """Verify circuit schematics are not part of the shared backend API."""

    catalog = default_part_catalog()

    assert not hasattr(circuits, "PinSide")
    assert not hasattr(circuits, "plan_layout")
    assert not hasattr(circuits, "draw_svg")
    assert not hasattr(circuits, "draw_png")
    assert not hasattr(circuits, "to_png")
    assert (
        importlib.util.find_spec("rlvr_physics.tasks.physics.circuits.layout") is None
    )
    assert importlib.util.find_spec("rlvr_physics.tasks.physics.circuits.svg") is None
    assert (
        importlib.util.find_spec("rlvr_physics.tasks.physics.circuits.symbol_assets")
        is None
    )
    assert (
        importlib.util.find_spec("rlvr_physics.tasks.physics.circuits.assets") is None
    )
    assert all(not hasattr(spec, "icon") for spec in catalog.values())
    assert all(
        not hasattr(pin, "side") for spec in catalog.values() for pin in spec.pins
    )
    assert all(not hasattr(spec, "spice") for spec in catalog.values())
    assert all(not hasattr(spec, "analysis_support") for spec in catalog.values())
