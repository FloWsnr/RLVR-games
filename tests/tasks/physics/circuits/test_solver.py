"""Tests for the small linear DC solver."""

import pytest

from rlvr_physics.tasks.physics.circuits import (
    CircuitBuilder,
    UnsupportedCircuitError,
    default_catalog,
    solve_dc_linear,
)
from tests.tasks.physics.circuits.test_model import divider_circuit


def test_linear_dc_solver_solves_voltage_divider() -> None:
    result = solve_dc_linear(divider_circuit(), default_catalog())

    assert result.node_voltages["0"] == 0.0
    assert result.node_voltages["VIN"] == pytest.approx(12.0)
    assert result.node_voltages["MID"] == pytest.approx(8.0)
    assert result.voltage_source_currents["V1"] == pytest.approx(-0.004)


def test_linear_dc_solver_rejects_nonlinear_part() -> None:
    builder = CircuitBuilder("diode", default_catalog())
    builder.add_part("V1", "voltage_source_dc", "5V", {"voltage_v": 5.0}, {})
    builder.add_part("GND1", "ground", "0", {}, {})
    builder.add_part("D1", "diode", "D", {}, {})
    builder.connect("V1", "p", "VIN")
    builder.connect("V1", "n", "0")
    builder.connect("GND1", "0", "0")
    builder.connect("D1", "a", "VIN")
    builder.connect("D1", "k", "0")

    with pytest.raises(UnsupportedCircuitError, match="unsupported linear DC part"):
        solve_dc_linear(builder.freeze(), default_catalog())


def test_linear_dc_solver_solves_voltage_controlled_source() -> None:
    catalog = default_catalog()
    builder = CircuitBuilder("vcvs", catalog)
    builder.add_part("V1", "voltage_source_dc", "2V", {"voltage_v": 2.0}, {})
    builder.add_part("GND1", "ground", "0", {}, {})
    builder.add_part("E1", "vcvs", "gain=3", {"gain": 3.0}, {})
    builder.add_part("R1", "resistor", "1k", {"resistance_ohm": 1000.0}, {})
    builder.connect("V1", "p", "IN")
    builder.connect("V1", "n", "0")
    builder.connect("GND1", "0", "0")
    builder.connect("E1", "p", "OUT")
    builder.connect("E1", "n", "0")
    builder.connect("E1", "cp", "IN")
    builder.connect("E1", "cn", "0")
    builder.connect("R1", "1", "OUT")
    builder.connect("R1", "2", "0")

    result = solve_dc_linear(builder.freeze(), catalog)

    assert result.node_voltages["IN"] == pytest.approx(2.0)
    assert result.node_voltages["OUT"] == pytest.approx(6.0)
    assert result.voltage_source_currents["E1"] == pytest.approx(-0.006)
