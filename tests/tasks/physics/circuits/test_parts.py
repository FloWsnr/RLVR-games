"""Tests for built-in component catalog coverage."""

from rlvr_physics.tasks.physics.circuits import (
    AnalysisSupport,
    PinSide,
    default_part_catalog,
)


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
        "bjt_pnp",
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
        "dip_20_ic",
    }

    assert required_kinds <= set(catalog)
    assert AnalysisSupport.LINEAR_DC in catalog["vcvs"].analysis_support
    assert AnalysisSupport.LINEAR_DC in catalog["vccs"].analysis_support
    assert AnalysisSupport.SPICE_EXPORT in catalog["jfet_n"].analysis_support
    assert catalog["bjt_npn"].ref_prefix == "Q"
    assert catalog["mosfet_n"].ref_prefix == "Q"
    assert catalog["jfet_n"].ref_prefix == "J"
    assert AnalysisSupport.SPICE_EXPORT in catalog["connector_2"].analysis_support
    assert catalog["dip_20_ic"].pin("20").side == PinSide.RIGHT
    assert tuple(pin.name for pin in catalog["dip_20_ic"].pins[-10:]) == (
        "20",
        "19",
        "18",
        "17",
        "16",
        "15",
        "14",
        "13",
        "12",
        "11",
    )


def test_spice_export_support_requires_spice_semantics() -> None:
    catalog = default_part_catalog()

    for spec in catalog.values():
        if spec.kind == "ground":
            continue
        if AnalysisSupport.SPICE_EXPORT in spec.analysis_support:
            assert spec.spice is not None, spec.kind


def test_polarity_specific_device_pins_face_supply_rails() -> None:
    catalog = default_part_catalog()

    assert catalog["bjt_pnp"].pin("e").side == PinSide.TOP
    assert catalog["bjt_pnp"].pin("c").side == PinSide.BOTTOM
    assert catalog["mosfet_p"].pin("s").side == PinSide.TOP
    assert catalog["mosfet_p"].pin("d").side == PinSide.BOTTOM
    assert catalog["jfet_p"].pin("s").side == PinSide.TOP
    assert catalog["jfet_p"].pin("d").side == PinSide.BOTTOM
