"""Tests for SPICE export."""

from pathlib import Path
import shutil
import subprocess

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


def test_export_spice_handles_new_asset_backed_parts() -> None:
    netlist_text = _asset_backed_parts_netlist_text()

    assert "VBT1 VIN 0 DC 9" in netlist_text
    assert "RV1 VIN MID 5000" in netlist_text
    assert "CP1 MID 0 1e-05" in netlist_text
    assert "L1 VIN 0 0.001" in netlist_text
    assert "RS1 VIN SW 1e+12" in netlist_text
    assert "XPWR1 VIN RLVR_POWER_RAIL" in netlist_text
    assert "XTP1 MID RLVR_TEST_POINT" in netlist_text
    assert ".subckt RLVR_POWER_RAIL net\n.ends RLVR_POWER_RAIL" in netlist_text
    assert ".subckt RLVR_TEST_POINT net\n.ends RLVR_TEST_POINT" in netlist_text
    assert "RLEAK" not in netlist_text


def test_ngspice_accepts_noop_visual_helper_subcircuits(tmp_path: Path) -> None:
    ngspice = shutil.which("ngspice")
    if ngspice is None:
        pytest.skip("ngspice executable is not available")
    netlist_path = tmp_path / "asset_backed_parts.cir"
    netlist_path.write_text(_asset_backed_parts_netlist_text(), encoding="utf-8")

    completed = subprocess.run(
        (ngspice, "-b", str(netlist_path)),
        check=False,
        capture_output=True,
        text=True,
        timeout=20.0,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def _asset_backed_parts_netlist_text() -> str:
    """Return SPICE text for the asset-backed helper part smoke circuit."""

    catalog = default_catalog()
    builder = CircuitBuilder("asset-backed-parts", catalog)
    builder.add_part("GND1", "ground", "0", {}, {})
    builder.add_part("BT1", "battery", "9V", {"voltage_v": 9.0}, {})
    builder.add_part("RV1", "variable_resistor", "5k", {"resistance_ohm": 5000.0}, {})
    builder.add_part(
        "CP1",
        "polarized_capacitor",
        "10u",
        {"capacitance_f": 1.0e-5},
        {},
    )
    builder.add_part("L1", "inductor_looped", "1m", {"inductance_h": 1.0e-3}, {})
    builder.add_part(
        "S1",
        "pushbutton_switch",
        "open",
        {"state_resistance_ohm": 1.0e12},
        {},
    )
    builder.add_part("PWR1", "power_rail", "VCC", {}, {})
    builder.add_part("TP1", "test_point", "", {}, {})
    builder.connect("GND1", "0", "0")
    builder.connect("BT1", "p", "VIN")
    builder.connect("BT1", "n", "0")
    builder.connect("RV1", "1", "VIN")
    builder.connect("RV1", "2", "MID")
    builder.connect("CP1", "p", "MID")
    builder.connect("CP1", "n", "0")
    builder.connect("L1", "1", "VIN")
    builder.connect("L1", "2", "0")
    builder.connect("S1", "1", "VIN")
    builder.connect("S1", "2", "SW")
    builder.connect("PWR1", "net", "VIN")
    builder.connect("TP1", "net", "MID")

    return export_spice(builder.freeze(), catalog, operating_point_analysis()).text


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
