"""Tests for SPICE export."""

import pytest

from rlvr_physics.tasks.physics.circuits import (
    CircuitBuilder,
    SpiceAnalysis,
    SpiceAnalysisKind,
    dc_sweep_analysis,
    default_catalog,
    export_spice,
    operating_point_analysis,
    transient_analysis,
)
from tests.tasks.physics.circuits.test_model import divider_circuit


def test_export_spice_for_voltage_divider() -> None:
    netlist = export_spice(
        divider_circuit(), default_catalog(), operating_point_analysis()
    )

    assert "* RLVR-physics circuit: divider" in netlist.text
    assert "R1 VIN MID 1000" in netlist.text
    assert "R2 MID 0 2000" in netlist.text
    assert "V1 VIN 0 DC 12" in netlist.text
    assert netlist.text.endswith(".op\n.end\n")


def test_export_spice_supports_dc_sweep_and_transient_cards() -> None:
    dc_netlist = export_spice(
        divider_circuit(),
        default_catalog(),
        dc_sweep_analysis("V1", 0.0, 5.0, 1.0),
    )
    transient_netlist = export_spice(
        divider_circuit(),
        default_catalog(),
        transient_analysis(1e-6, 1e-3),
    )

    assert ".dc V1 0 5 1" in dc_netlist.text
    assert ".tran 1e-06 0.001" in transient_netlist.text


def test_export_spice_normalizes_dc_sweep_source_ref() -> None:
    catalog = default_catalog()
    builder = CircuitBuilder("custom-source-ref", catalog)
    builder.add_part("SRC1", "voltage_source_dc", "5V", {"voltage_v": 5.0}, {})
    builder.add_part("GND1", "ground", "0", {}, {})
    builder.add_part("R1", "resistor", "1k", {"resistance_ohm": 1000.0}, {})
    builder.connect("SRC1", "p", "VIN")
    builder.connect("SRC1", "n", "0")
    builder.connect("GND1", "0", "0")
    builder.connect("R1", "1", "VIN")
    builder.connect("R1", "2", "0")

    netlist = export_spice(
        builder.freeze(),
        catalog,
        dc_sweep_analysis("SRC1", 0.0, 5.0, 1.0),
    )

    assert "VSRC1 VIN 0 DC 5" in netlist.text
    assert ".dc VSRC1 0 5 1" in netlist.text
    assert ".dc SRC1 0 5 1" not in netlist.text


def test_export_spice_preserves_connector_as_noop_subcircuit() -> None:
    catalog = default_catalog()
    builder = CircuitBuilder("connector", catalog)
    builder.add_part("GND1", "ground", "0", {}, {})
    builder.add_part("J1", "connector_2", "J", {}, {})
    builder.connect("GND1", "0", "0")
    builder.connect("J1", "1", "A")
    builder.connect("J1", "2", "0")

    netlist = export_spice(builder.freeze(), catalog, operating_point_analysis())

    assert "XJ1 A 0 RLVR_CONNECTOR_2" in netlist.text
    assert ".subckt RLVR_CONNECTOR_2 1 2\n.ends RLVR_CONNECTOR_2" in netlist.text


@pytest.mark.parametrize(
    ("source_ref", "start", "stop", "step"),
    (
        ("", 0.0, 1.0, 0.1),
        ("V1", 0.0, 1.0, 0.0),
        ("V1", 0.0, 1.0, -0.1),
        ("V1", 1.0, 0.0, 0.1),
        ("V1", 0.0, float("inf"), 0.1),
    ),
)
def test_dc_sweep_analysis_rejects_invalid_values(
    source_ref: str, start: float, stop: float, step: float
) -> None:
    with pytest.raises(ValueError):
        dc_sweep_analysis(source_ref, start, stop, step)


@pytest.mark.parametrize(
    ("step", "stop_time"),
    (
        (0.0, 1e-3),
        (-1e-6, 1e-3),
        (1e-6, 0.0),
        (2e-3, 1e-3),
        (float("nan"), 1e-3),
    ),
)
def test_transient_analysis_rejects_invalid_values(
    step: float, stop_time: float
) -> None:
    with pytest.raises(ValueError):
        transient_analysis(step, stop_time)


def test_export_spice_revalidates_analysis_instances() -> None:
    bad_analysis = SpiceAnalysis(
        kind=SpiceAnalysisKind.TRANSIENT,
        source_ref=None,
        start=None,
        stop=None,
        step=0.0,
        stop_time=1e-3,
    )

    with pytest.raises(ValueError, match="transient step"):
        export_spice(divider_circuit(), default_catalog(), bad_analysis)


def test_export_spice_for_controlled_source_and_jfet() -> None:
    catalog = default_catalog()
    builder = CircuitBuilder("controlled", catalog)
    builder.add_part("V1", "voltage_source_dc", "2V", {"voltage_v": 2.0}, {})
    builder.add_part("GND1", "ground", "0", {}, {})
    builder.add_part("E1", "vcvs", "gain=3", {"gain": 3.0}, {})
    builder.add_part("J1", "jfet_n", "NJFET", {}, {})
    builder.add_part("R1", "resistor", "1k", {"resistance_ohm": 1000.0}, {})
    builder.connect("V1", "p", "IN")
    builder.connect("V1", "n", "0")
    builder.connect("GND1", "0", "0")
    builder.connect("E1", "p", "OUT")
    builder.connect("E1", "n", "0")
    builder.connect("E1", "cp", "IN")
    builder.connect("E1", "cn", "0")
    builder.connect("J1", "d", "OUT")
    builder.connect("J1", "g", "IN")
    builder.connect("J1", "s", "0")
    builder.connect("R1", "1", "OUT")
    builder.connect("R1", "2", "0")

    netlist = export_spice(builder.freeze(), catalog, operating_point_analysis())

    assert "E1 OUT 0 IN 0 3" in netlist.text
    assert "J1 OUT IN 0 J_NJFET_RLVR" in netlist.text
    assert ".model J_NJFET_RLVR NJF" in netlist.text
