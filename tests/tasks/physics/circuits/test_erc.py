"""Tests for electrical rule checking."""

from rlvr_physics.tasks.physics.circuits import (
    AnalysisSupport,
    CircuitBuilder,
    IssueSeverity,
    check_circuit,
    default_catalog,
)
from tests.tasks.physics.circuits.test_model import divider_circuit


def test_erc_accepts_divider_without_errors() -> None:
    report = check_circuit(
        divider_circuit(), default_catalog(), AnalysisSupport.SPICE_EXPORT
    )

    assert report.is_valid
    assert not report.errors


def test_erc_accepts_parallel_current_sources() -> None:
    catalog = default_catalog()
    builder = CircuitBuilder("parallel-current-sources", catalog)
    builder.add_part("GND1", "ground", "0", {}, {})
    builder.add_part("I1", "current_source_dc", "1mA", {"current_a": 0.001}, {})
    builder.add_part("I2", "current_source_dc", "2mA", {"current_a": 0.002}, {})
    builder.add_part("R1", "resistor", "1k", {"resistance_ohm": 1000.0}, {})
    builder.connect("GND1", "0", "0")
    for ref in ("I1", "I2"):
        builder.connect(ref, "p", "LOAD")
        builder.connect(ref, "n", "0")
    builder.connect("R1", "1", "LOAD")
    builder.connect("R1", "2", "0")

    report = check_circuit(builder.freeze(), catalog, AnalysisSupport.SPICE_EXPORT)

    assert report.is_valid
    assert not any(issue.code == "pin_conflict" for issue in report.errors)


def test_erc_accepts_multiple_ground_parts_on_ground_net() -> None:
    catalog = default_catalog()
    builder = CircuitBuilder("repeated-ground-parts", catalog)
    builder.add_part("V1", "voltage_source_dc", "5V", {"voltage_v": 5.0}, {})
    builder.add_part("GND1", "ground", "0", {}, {})
    builder.add_part("GND2", "ground", "0", {}, {})
    builder.add_part("R1", "resistor", "1k", {"resistance_ohm": 1000.0}, {})
    builder.connect("V1", "p", "VIN")
    builder.connect("V1", "n", "0")
    builder.connect("GND1", "0", "0")
    builder.connect("GND2", "0", "0")
    builder.connect("R1", "1", "VIN")
    builder.connect("R1", "2", "0")

    report = check_circuit(builder.freeze(), catalog, AnalysisSupport.SPICE_EXPORT)

    assert report.is_valid
    assert not any(issue.code == "pin_conflict" for issue in report.errors)


def test_erc_reports_missing_reference_node() -> None:
    builder = CircuitBuilder("floating", default_catalog())
    builder.add_part("R1", "resistor", "1k", {"resistance_ohm": 1000.0}, {})
    builder.add_part("R2", "resistor", "1k", {"resistance_ohm": 1000.0}, {})
    builder.connect("R1", "1", "A")
    builder.connect("R1", "2", "B")
    builder.connect("R2", "1", "B")
    builder.connect("R2", "2", "A")

    report = check_circuit(
        builder.freeze(), default_catalog(), AnalysisSupport.SPICE_EXPORT
    )

    assert any(issue.code == "missing_reference_node" for issue in report.errors)


def test_erc_reports_output_conflict() -> None:
    builder = CircuitBuilder("logic-conflict", default_catalog())
    builder.add_part("V1", "voltage_source_dc", "5V", {"voltage_v": 5.0}, {})
    builder.add_part("GND1", "ground", "0", {}, {})
    builder.add_part("U1", "not_gate", "NOT", {}, {})
    builder.add_part("U2", "not_gate", "NOT", {}, {})
    builder.connect("V1", "p", "VCC")
    builder.connect("V1", "n", "0")
    builder.connect("GND1", "0", "0")
    for ref in ("U1", "U2"):
        builder.connect(ref, "in1", "A")
        builder.connect(ref, "out", "Y")
        builder.connect(ref, "vcc", "VCC")
        builder.connect(ref, "gnd", "0")

    report = check_circuit(
        builder.freeze(), default_catalog(), AnalysisSupport.SPICE_EXPORT
    )

    assert any(issue.code == "pin_conflict" for issue in report.errors)
    assert any(issue.severity is IssueSeverity.ERROR for issue in report.issues)


def test_erc_does_not_report_output_self_drive_as_excessive() -> None:
    catalog = default_catalog()
    builder = CircuitBuilder("logic-chain", catalog)
    builder.add_part("V1", "voltage_source_dc", "5V", {"voltage_v": 5.0}, {})
    builder.add_part("GND1", "ground", "0", {}, {})
    builder.add_part("U1", "not_gate", "NOT", {}, {})
    builder.add_part("U2", "not_gate", "NOT", {}, {})
    builder.add_part("R1", "resistor", "1k", {"resistance_ohm": 1000.0}, {})
    builder.connect("V1", "p", "VCC")
    builder.connect("V1", "n", "0")
    builder.connect("GND1", "0", "0")
    builder.connect("U1", "in1", "VCC")
    builder.connect("U1", "out", "N1")
    builder.connect("U1", "vcc", "VCC")
    builder.connect("U1", "gnd", "0")
    builder.connect("U2", "in1", "N1")
    builder.connect("U2", "out", "N2")
    builder.connect("U2", "vcc", "VCC")
    builder.connect("U2", "gnd", "0")
    builder.connect("R1", "1", "N2")
    builder.connect("R1", "2", "0")

    report = check_circuit(builder.freeze(), catalog, AnalysisSupport.SPICE_EXPORT)

    assert report.is_valid
    assert not any(issue.code == "excessive_drive" for issue in report.warnings)


def test_erc_warns_when_open_collector_is_directly_power_driven() -> None:
    catalog = default_catalog()
    builder = CircuitBuilder("open-collector-overdrive", catalog)
    builder.add_part("V1", "voltage_source_dc", "5V", {"voltage_v": 5.0}, {})
    builder.add_part("GND1", "ground", "0", {}, {})
    builder.add_part("U555", "timer_555", "555", {}, {})
    builder.add_part("R1", "resistor", "1k", {"resistance_ohm": 1000.0}, {})
    builder.connect("V1", "p", "VCC")
    builder.connect("V1", "n", "0")
    builder.connect("GND1", "0", "0")
    builder.connect("U555", "gnd", "0")
    builder.connect("U555", "vcc", "VCC")
    builder.connect("U555", "reset", "VCC")
    builder.connect("U555", "ctrl", "0")
    builder.connect("U555", "disch", "VCC")
    builder.connect("U555", "thresh", "0")
    builder.connect("U555", "trig", "0")
    builder.connect("U555", "out", "OUT")
    builder.connect("R1", "1", "OUT")
    builder.connect("R1", "2", "0")

    report = check_circuit(builder.freeze(), catalog, AnalysisSupport.SPICE_EXPORT)

    assert report.is_valid
    assert any(
        issue.code == "excessive_drive" and issue.pins == ("U555.disch",)
        for issue in report.warnings
    )


def test_erc_warns_about_unsupported_transient_analysis() -> None:
    builder = CircuitBuilder("crystal", default_catalog())
    builder.add_part("V1", "voltage_source_dc", "5V", {"voltage_v": 5.0}, {})
    builder.add_part("GND1", "ground", "0", {}, {})
    builder.add_part("XTAL1", "crystal", "XTAL", {}, {})
    builder.connect("V1", "p", "VIN")
    builder.connect("V1", "n", "0")
    builder.connect("GND1", "0", "0")
    builder.connect("XTAL1", "1", "VIN")
    builder.connect("XTAL1", "2", "0")

    report = check_circuit(
        builder.freeze(), default_catalog(), AnalysisSupport.TRANSIENT_EXPORT
    )

    assert report.is_valid
    assert any(issue.code == "unsupported_analysis" for issue in report.warnings)
