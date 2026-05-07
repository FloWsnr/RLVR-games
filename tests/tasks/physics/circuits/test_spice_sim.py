"""Tests for ngspice-backed circuit simulation."""

from pathlib import Path
import shutil
import subprocess

import pytest

import rlvr_physics.tasks.physics.circuits.spice_sim as spice_sim_module
from rlvr_physics.tasks.physics.circuits import (
    Circuit,
    CircuitBuilder,
    SpiceSimulationSpec,
    SpiceSimulatorConfig,
    SpiceVoltageSource,
    dc_sweep_analysis,
    default_catalog,
    default_spice_simulator_config,
    operating_point_analysis,
    simulate_spice,
)
from tests.tasks.physics.circuits.test_model import divider_circuit


def test_simulate_spice_operating_point_returns_all_node_voltages() -> None:
    if shutil.which("ngspice") is None:
        pytest.skip("ngspice executable is not available")
    result = simulate_spice(
        divider_circuit(),
        default_catalog(),
        SpiceSimulationSpec(analysis=operating_point_analysis(), voltage_sources=()),
        default_spice_simulator_config(),
    )

    assert result.ok, result.issues
    assert set(result.values) == {"0", "MID", "VIN"}
    assert result.values["0"] == pytest.approx(0.0)
    assert result.values["VIN"] == pytest.approx(12.0)
    assert result.values["MID"] == pytest.approx(8.0)


def test_simulate_spice_parses_successful_output_without_ngspice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(
        args: tuple[str, str, str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        """Write deterministic ngspice-like output for the simulator."""

        assert args[:2] == ("fake-ngspice", "-b")
        assert check is False
        assert capture_output is True
        assert text is True
        assert timeout == 1.0
        netlist_text = Path(args[2]).read_text(encoding="utf-8")
        wrdata_line = next(
            line for line in netlist_text.splitlines() if line.startswith("wrdata ")
        )
        assert wrdata_line.endswith("v(MID) v(VIN)")
        node_voltage_path = Path(wrdata_line.split()[1])
        node_voltage_path.write_text(
            "index v(MID) v(VIN)\n0 8 12\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(spice_sim_module.subprocess, "run", fake_run)

    result = simulate_spice(
        divider_circuit(),
        default_catalog(),
        SpiceSimulationSpec(analysis=operating_point_analysis(), voltage_sources=()),
        SpiceSimulatorConfig(ngspice_command="fake-ngspice", timeout_s=1.0),
    )

    assert result.ok, result.issues
    assert result.stdout == "ok"
    assert set(result.values) == {"0", "MID", "VIN"}
    assert result.values["0"] == pytest.approx(0.0)
    assert result.values["MID"] == pytest.approx(8.0)
    assert result.values["VIN"] == pytest.approx(12.0)


def test_simulate_spice_applies_external_voltage_source_without_ngspice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(
        args: tuple[str, str, str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        """Write deterministic ngspice-like output for an external supply."""

        assert check is False
        assert capture_output is True
        assert text is True
        assert timeout == 1.0
        netlist_text = Path(args[2]).read_text(encoding="utf-8")
        assert "VOP_VCC VCC 0 DC 5" in netlist_text
        assert netlist_text.index("VOP_VCC VCC 0 DC 5") < netlist_text.index(".op")
        wrdata_line = next(
            line for line in netlist_text.splitlines() if line.startswith("wrdata ")
        )
        assert wrdata_line.endswith("v(VCC)")
        node_voltage_path = Path(wrdata_line.split()[1])
        node_voltage_path.write_text(
            "index v(VCC)\n0 5\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(spice_sim_module.subprocess, "run", fake_run)

    result = simulate_spice(
        _supply_port_load_circuit(),
        default_catalog(),
        SpiceSimulationSpec(
            analysis=operating_point_analysis(),
            voltage_sources=(SpiceVoltageSource("VCC", "VCC", "0", 5.0),),
        ),
        SpiceSimulatorConfig(ngspice_command="fake-ngspice", timeout_s=1.0),
    )

    assert result.ok, result.issues
    assert result.values["0"] == pytest.approx(0.0)
    assert result.values["VCC"] == pytest.approx(5.0)


def test_simulate_spice_rejects_voltage_source_across_ground_aliases() -> None:
    with pytest.raises(ValueError, match="same SPICE node"):
        simulate_spice(
            _ground_alias_circuit(),
            default_catalog(),
            SpiceSimulationSpec(
                analysis=operating_point_analysis(),
                voltage_sources=(SpiceVoltageSource("bad", "GND", "0", 5.0),),
            ),
            SpiceSimulatorConfig(ngspice_command="ngspice", timeout_s=1.0),
        )


def test_comparator_model_is_rail_bounded_in_operating_point() -> None:
    if shutil.which("ngspice") is None:
        pytest.skip("ngspice executable is not available")
    result = simulate_spice(
        _comparator_threshold_circuit(input_v=1.0, reference_v=0.5),
        default_catalog(),
        SpiceSimulationSpec(analysis=operating_point_analysis(), voltage_sources=()),
        default_spice_simulator_config(),
    )

    assert result.ok, result.issues
    assert 0.0 <= result.values["OUT"] <= 5.0
    assert result.values["OUT"] > 4.8


def test_comparator_model_handles_low_operating_point_output() -> None:
    if shutil.which("ngspice") is None:
        pytest.skip("ngspice executable is not available")
    result = simulate_spice(
        _comparator_threshold_circuit(input_v=0.5, reference_v=1.0),
        default_catalog(),
        SpiceSimulationSpec(analysis=operating_point_analysis(), voltage_sources=()),
        default_spice_simulator_config(),
    )

    assert result.ok, result.issues
    assert 0.0 <= result.values["OUT"] <= 5.0
    assert result.values["OUT"] < 0.2


def test_op_amp_model_handles_closed_loop_operating_point_gain() -> None:
    if shutil.which("ngspice") is None:
        pytest.skip("ngspice executable is not available")
    result = simulate_spice(
        _non_inverting_op_amp_circuit(),
        default_catalog(),
        SpiceSimulationSpec(analysis=operating_point_analysis(), voltage_sources=()),
        default_spice_simulator_config(),
    )

    assert result.ok, result.issues
    assert result.values["OUT"] == pytest.approx(1.0, abs=0.05)


def test_logic_gate_model_is_rail_bounded_in_operating_point() -> None:
    if shutil.which("ngspice") is None:
        pytest.skip("ngspice executable is not available")
    result = simulate_spice(
        _logic_gate_circuit(),
        default_catalog(),
        SpiceSimulationSpec(analysis=operating_point_analysis(), voltage_sources=()),
        default_spice_simulator_config(),
    )

    assert result.ok, result.issues
    assert 0.0 <= result.values["AND_OUT"] <= 5.0
    assert 0.0 <= result.values["OR_OUT"] <= 5.0
    assert result.values["AND_OUT"] < 0.2
    assert result.values["OR_OUT"] > 4.8


def test_simulate_spice_rejects_non_operating_point_analysis() -> None:
    with pytest.raises(ValueError, match="operating-point"):
        simulate_spice(
            divider_circuit(),
            default_catalog(),
            SpiceSimulationSpec(
                analysis=dc_sweep_analysis("V1", 0.0, 5.0, 1.0),
                voltage_sources=(),
            ),
            SpiceSimulatorConfig(ngspice_command="ngspice", timeout_s=1.0),
        )


def test_simulate_spice_reports_missing_ngspice() -> None:
    result = simulate_spice(
        divider_circuit(),
        default_catalog(),
        SpiceSimulationSpec(analysis=operating_point_analysis(), voltage_sources=()),
        SpiceSimulatorConfig(
            ngspice_command="/definitely/missing/ngspice",
            timeout_s=1.0,
        ),
    )

    assert not result.ok
    assert tuple(issue.code for issue in result.issues) == ("ngspice_not_found",)


def _supply_port_load_circuit() -> Circuit:
    """Build a load connected to an external VCC supply port."""

    builder = CircuitBuilder("external-supply-load", default_catalog())
    builder.add_part("GND1", "ground", "0", {}, {})
    builder.add_part("R1", "resistor", "1k", {"resistance_ohm": 1000.0}, {})
    builder.connect("GND1", "0", "0")
    builder.connect("R1", "1", "VCC")
    builder.connect("R1", "2", "0")
    return builder.freeze()


def _ground_alias_circuit() -> Circuit:
    """Build a circuit containing two canonical ground aliases."""

    builder = CircuitBuilder("ground-aliases", default_catalog())
    builder.add_part("GND1", "ground", "0", {}, {})
    builder.add_part("GND2", "ground", "0", {}, {})
    builder.connect("GND1", "0", "0")
    builder.connect("GND2", "0", "GND")
    return builder.freeze()


def _comparator_threshold_circuit(input_v: float, reference_v: float) -> Circuit:
    """Build a comparator threshold circuit."""

    builder = CircuitBuilder("comparator-threshold", default_catalog())
    builder.add_part("GND1", "ground", "0", {}, {})
    builder.add_part("VCC1", "voltage_source_dc", "5V", {"voltage_v": 5.0}, {})
    builder.add_part("VIN1", "voltage_source_dc", "VIN", {"voltage_v": input_v}, {})
    builder.add_part(
        "VREF1",
        "voltage_source_dc",
        "VREF",
        {"voltage_v": reference_v},
        {},
    )
    builder.add_part("U1", "comparator", "cmp", {}, {})
    builder.add_part("RLOAD1", "resistor", "10k", {"resistance_ohm": 10000.0}, {})
    builder.connect("GND1", "0", "0")
    builder.connect("VCC1", "p", "VCC")
    builder.connect("VCC1", "n", "0")
    builder.connect("VIN1", "p", "VIN")
    builder.connect("VIN1", "n", "0")
    builder.connect("VREF1", "p", "VREF")
    builder.connect("VREF1", "n", "0")
    builder.connect("U1", "noninv", "VIN")
    builder.connect("U1", "inv", "VREF")
    builder.connect("U1", "vpos", "VCC")
    builder.connect("U1", "vneg", "0")
    builder.connect("U1", "out", "OUT")
    builder.connect("RLOAD1", "1", "OUT")
    builder.connect("RLOAD1", "2", "0")
    return builder.freeze()


def _non_inverting_op_amp_circuit() -> Circuit:
    """Build a closed-loop non-inverting op-amp gain circuit."""

    builder = CircuitBuilder("op-amp-closed-loop", default_catalog())
    builder.add_part("GND1", "ground", "0", {}, {})
    builder.add_part("VCC1", "voltage_source_dc", "5V", {"voltage_v": 5.0}, {})
    builder.add_part("VIN1", "voltage_source_dc", "0.5V", {"voltage_v": 0.5}, {})
    builder.add_part("U1", "op_amp", "op", {}, {})
    builder.add_part("RF1", "resistor", "10k", {"resistance_ohm": 10000.0}, {})
    builder.add_part("RG1", "resistor", "10k", {"resistance_ohm": 10000.0}, {})
    builder.connect("GND1", "0", "0")
    builder.connect("VCC1", "p", "VCC")
    builder.connect("VCC1", "n", "0")
    builder.connect("VIN1", "p", "VIN")
    builder.connect("VIN1", "n", "0")
    builder.connect("U1", "noninv", "VIN")
    builder.connect("U1", "inv", "NFB")
    builder.connect("U1", "vpos", "VCC")
    builder.connect("U1", "vneg", "0")
    builder.connect("U1", "out", "OUT")
    builder.connect("RF1", "1", "OUT")
    builder.connect("RF1", "2", "NFB")
    builder.connect("RG1", "1", "NFB")
    builder.connect("RG1", "2", "0")
    return builder.freeze()


def _logic_gate_circuit() -> Circuit:
    """Build basic logic gates with one high and one low input."""

    builder = CircuitBuilder("logic-gates", default_catalog())
    builder.add_part("GND1", "ground", "0", {}, {})
    builder.add_part("VCC1", "voltage_source_dc", "5V", {"voltage_v": 5.0}, {})
    builder.add_part("VLOW1", "voltage_source_dc", "0V", {"voltage_v": 0.0}, {})
    builder.add_part("AND1", "and_gate", "AND", {}, {})
    builder.add_part("OR1", "or_gate", "OR", {}, {})
    builder.add_part("RAND1", "resistor", "10k", {"resistance_ohm": 10000.0}, {})
    builder.add_part("ROR1", "resistor", "10k", {"resistance_ohm": 10000.0}, {})
    builder.connect("GND1", "0", "0")
    builder.connect("VCC1", "p", "HI")
    builder.connect("VCC1", "n", "0")
    builder.connect("VLOW1", "p", "LO")
    builder.connect("VLOW1", "n", "0")
    builder.connect("AND1", "in1", "HI")
    builder.connect("AND1", "in2", "LO")
    builder.connect("AND1", "vcc", "HI")
    builder.connect("AND1", "gnd", "0")
    builder.connect("AND1", "out", "AND_OUT")
    builder.connect("OR1", "in1", "HI")
    builder.connect("OR1", "in2", "LO")
    builder.connect("OR1", "vcc", "HI")
    builder.connect("OR1", "gnd", "0")
    builder.connect("OR1", "out", "OR_OUT")
    builder.connect("RAND1", "1", "AND_OUT")
    builder.connect("RAND1", "2", "0")
    builder.connect("ROR1", "1", "OR_OUT")
    builder.connect("ROR1", "2", "0")
    return builder.freeze()
