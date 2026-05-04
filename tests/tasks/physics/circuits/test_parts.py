"""Tests for built-in component catalog coverage."""

from rlvr_physics.tasks.physics.circuits import AnalysisSupport, default_part_catalog


def test_part_catalog_covers_planned_common_component_classes() -> None:
    catalog = default_part_catalog()
    required_kinds = {
        "ground",
        "resistor",
        "capacitor",
        "inductor",
        "lamp",
        "motor",
        "voltage_source_dc",
        "current_source_dc",
        "vcvs",
        "vccs",
        "diode",
        "led",
        "zener",
        "bjt_npn",
        "bjt_pnp",
        "mosfet_n",
        "mosfet_p",
        "jfet_n",
        "jfet_p",
        "pullup_resistor",
        "pulldown_resistor",
        "ideal_switch",
        "relay",
        "op_amp",
        "comparator",
        "transformer",
        "connector_2",
        "voltmeter",
        "ammeter",
        "and_gate",
        "or_gate",
        "not_gate",
        "generic_ic",
    }

    assert required_kinds <= set(catalog)
    assert AnalysisSupport.LINEAR_DC in catalog["vcvs"].analysis_support
    assert AnalysisSupport.LINEAR_DC in catalog["vccs"].analysis_support
    assert AnalysisSupport.SPICE_EXPORT in catalog["jfet_n"].analysis_support
    assert catalog["bjt_npn"].ref_prefix == "Q"
    assert catalog["mosfet_n"].ref_prefix == "Q"
    assert catalog["jfet_n"].ref_prefix == "J"
    assert AnalysisSupport.SPICE_EXPORT in catalog["connector_2"].analysis_support


def test_spice_export_support_requires_spice_semantics() -> None:
    catalog = default_part_catalog()

    for spec in catalog.values():
        if spec.kind == "ground":
            continue
        if AnalysisSupport.SPICE_EXPORT in spec.analysis_support:
            assert spec.spice is not None, spec.kind
