"""Tests for ngspice-backed circuit simulation."""

from pathlib import Path
import shutil
import subprocess

import pytest

import rlvr_physics.tasks.physics.circuits.spice_sim as spice_sim_module
from rlvr_physics.tasks.physics.circuits import (
    SpiceSimulationSpec,
    SpiceSimulatorConfig,
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
        SpiceSimulationSpec(analysis=operating_point_analysis()),
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
        SpiceSimulationSpec(analysis=operating_point_analysis()),
        SpiceSimulatorConfig(ngspice_command="fake-ngspice", timeout_s=1.0),
    )

    assert result.ok, result.issues
    assert result.stdout == "ok"
    assert set(result.values) == {"0", "MID", "VIN"}
    assert result.values["0"] == pytest.approx(0.0)
    assert result.values["MID"] == pytest.approx(8.0)
    assert result.values["VIN"] == pytest.approx(12.0)


def test_simulate_spice_rejects_non_operating_point_analysis() -> None:
    with pytest.raises(ValueError, match="operating-point"):
        simulate_spice(
            divider_circuit(),
            default_catalog(),
            SpiceSimulationSpec(analysis=dc_sweep_analysis("V1", 0.0, 5.0, 1.0)),
            SpiceSimulatorConfig(ngspice_command="ngspice", timeout_s=1.0),
        )


def test_simulate_spice_reports_missing_ngspice() -> None:
    result = simulate_spice(
        divider_circuit(),
        default_catalog(),
        SpiceSimulationSpec(analysis=operating_point_analysis()),
        SpiceSimulatorConfig(
            ngspice_command="/definitely/missing/ngspice",
            timeout_s=1.0,
        ),
    )

    assert not result.ok
    assert tuple(issue.code for issue in result.issues) == ("ngspice_not_found",)
