"""Small dependency-free linear DC solver for simple circuit sanity checks."""

from dataclasses import dataclass
from typing import Mapping

from rlvr_physics.tasks.physics.circuits.model import (
    Circuit,
    PartInstance,
    PartSpec,
    is_ground_net,
)


class UnsupportedCircuitError(RuntimeError):
    """Raised when a circuit is outside the v1 numeric solver scope."""


@dataclass(frozen=True)
class LinearDcResult:
    """Linear DC operating point result.

    Parameters
    ----------
    node_voltages:
        Voltage for every named circuit net.
    voltage_source_currents:
        Solved current through each independent voltage source.
    """

    node_voltages: Mapping[str, float]
    voltage_source_currents: Mapping[str, float]


def solve_dc_linear(
    circuit: Circuit, catalog: Mapping[str, PartSpec]
) -> LinearDcResult:
    """Solve a simple linear DC operating point with MNA.

    Parameters
    ----------
    circuit:
        Circuit to solve.
    catalog:
        Component catalog.

    Returns
    -------
    LinearDcResult
        Node voltages and voltage-source branch currents.

    Raises
    ------
    UnsupportedCircuitError
        Raised when the circuit uses unsupported component kinds.
    """

    ground_nets = [net for net in circuit.nets if is_ground_net(net)]
    if not ground_nets:
        raise UnsupportedCircuitError("linear DC solve requires a ground net")

    supported = {
        "ground",
        "resistor",
        "lamp",
        "motor",
        "pullup_resistor",
        "pulldown_resistor",
        "voltage_source_dc",
        "current_source_dc",
        "vcvs",
        "vccs",
        "ideal_switch",
    }
    for part in circuit.parts:
        if part.kind not in supported:
            raise UnsupportedCircuitError(f"unsupported linear DC part: {part.ref}")

    node_names = tuple(net for net in circuit.nets if not is_ground_net(net))
    node_index = {net: idx for idx, net in enumerate(node_names)}
    voltage_sources = tuple(
        part for part in circuit.parts if part.kind in {"voltage_source_dc", "vcvs"}
    )
    size = len(node_names) + len(voltage_sources)
    if size == 0:
        return LinearDcResult(
            node_voltages={net: 0.0 for net in circuit.nets}, voltage_source_currents={}
        )

    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    rhs = [0.0 for _ in range(size)]

    for part in circuit.parts:
        if part.kind in {
            "resistor",
            "lamp",
            "motor",
            "pullup_resistor",
            "pulldown_resistor",
        }:
            _stamp_resistor(circuit, node_index, matrix, part)
        elif part.kind == "ideal_switch":
            _stamp_switch(circuit, node_index, matrix, part)
        elif part.kind == "current_source_dc":
            _stamp_current_source(circuit, node_index, rhs, part)
        elif part.kind == "vccs":
            _stamp_vccs(circuit, node_index, matrix, part)

    for source_number, part in enumerate(voltage_sources):
        if part.kind == "voltage_source_dc":
            _stamp_voltage_source(
                circuit,
                node_index,
                matrix,
                rhs,
                part,
                len(node_names) + source_number,
            )
        else:
            _stamp_vcvs(
                circuit,
                node_index,
                matrix,
                part,
                len(node_names) + source_number,
            )

    solution = _solve_linear_system(matrix, rhs)
    node_voltages = {net: solution[idx] for net, idx in node_index.items()}
    for net in circuit.nets:
        if is_ground_net(net):
            node_voltages[net] = 0.0
    source_currents = {
        part.ref: solution[len(node_names) + idx]
        for idx, part in enumerate(voltage_sources)
    }
    return LinearDcResult(
        node_voltages=dict(sorted(node_voltages.items())),
        voltage_source_currents=source_currents,
    )


def _stamp_resistor(
    circuit: Circuit,
    node_index: Mapping[str, int],
    matrix: list[list[float]],
    part: PartInstance,
) -> None:
    """Stamp one resistor-like component."""

    resistance = _positive_float(part, "resistance_ohm")
    conductance = 1.0 / resistance
    pin_a, pin_b = _resistive_pins(part)
    net_a = _required_net(circuit, part.ref, pin_a)
    net_b = _required_net(circuit, part.ref, pin_b)
    _stamp_conductance(node_index, matrix, net_a, net_b, conductance)


def _stamp_switch(
    circuit: Circuit,
    node_index: Mapping[str, int],
    matrix: list[list[float]],
    part: PartInstance,
) -> None:
    """Stamp an ideal switch as an explicit resistance."""

    resistance = _positive_float(part, "state_resistance_ohm")
    net_a = _required_net(circuit, part.ref, "1")
    net_b = _required_net(circuit, part.ref, "2")
    _stamp_conductance(node_index, matrix, net_a, net_b, 1.0 / resistance)


def _stamp_current_source(
    circuit: Circuit,
    node_index: Mapping[str, int],
    rhs: list[float],
    part: PartInstance,
) -> None:
    """Stamp an independent current source from p to n."""

    current = _float_parameter(part, "current_a")
    pos = _required_net(circuit, part.ref, "p")
    neg = _required_net(circuit, part.ref, "n")
    if not is_ground_net(pos):
        rhs[node_index[pos]] -= current
    if not is_ground_net(neg):
        rhs[node_index[neg]] += current


def _stamp_voltage_source(
    circuit: Circuit,
    node_index: Mapping[str, int],
    matrix: list[list[float]],
    rhs: list[float],
    part: PartInstance,
    source_index: int,
) -> None:
    """Stamp an independent voltage source."""

    voltage = _float_parameter(part, "voltage_v")
    pos = _required_net(circuit, part.ref, "p")
    neg = _required_net(circuit, part.ref, "n")
    if not is_ground_net(pos):
        idx = node_index[pos]
        matrix[idx][source_index] += 1.0
        matrix[source_index][idx] += 1.0
    if not is_ground_net(neg):
        idx = node_index[neg]
        matrix[idx][source_index] -= 1.0
        matrix[source_index][idx] -= 1.0
    rhs[source_index] += voltage


def _stamp_vcvs(
    circuit: Circuit,
    node_index: Mapping[str, int],
    matrix: list[list[float]],
    part: PartInstance,
    source_index: int,
) -> None:
    """Stamp a voltage-controlled voltage source."""

    gain = _float_parameter(part, "gain")
    pos = _required_net(circuit, part.ref, "p")
    neg = _required_net(circuit, part.ref, "n")
    ctrl_pos = _required_net(circuit, part.ref, "cp")
    ctrl_neg = _required_net(circuit, part.ref, "cn")
    if not is_ground_net(pos):
        idx = node_index[pos]
        matrix[idx][source_index] += 1.0
        matrix[source_index][idx] += 1.0
    if not is_ground_net(neg):
        idx = node_index[neg]
        matrix[idx][source_index] -= 1.0
        matrix[source_index][idx] -= 1.0
    if not is_ground_net(ctrl_pos):
        matrix[source_index][node_index[ctrl_pos]] -= gain
    if not is_ground_net(ctrl_neg):
        matrix[source_index][node_index[ctrl_neg]] += gain


def _stamp_vccs(
    circuit: Circuit,
    node_index: Mapping[str, int],
    matrix: list[list[float]],
    part: PartInstance,
) -> None:
    """Stamp a voltage-controlled current source."""

    gain = _float_parameter(part, "gain")
    pos = _required_net(circuit, part.ref, "p")
    neg = _required_net(circuit, part.ref, "n")
    ctrl_pos = _required_net(circuit, part.ref, "cp")
    ctrl_neg = _required_net(circuit, part.ref, "cn")
    _add_matrix(matrix, node_index, pos, ctrl_pos, gain)
    _add_matrix(matrix, node_index, pos, ctrl_neg, -gain)
    _add_matrix(matrix, node_index, neg, ctrl_pos, -gain)
    _add_matrix(matrix, node_index, neg, ctrl_neg, gain)


def _stamp_conductance(
    node_index: Mapping[str, int],
    matrix: list[list[float]],
    net_a: str,
    net_b: str,
    conductance: float,
) -> None:
    """Stamp a conductance between two nets."""

    if not is_ground_net(net_a):
        idx_a = node_index[net_a]
        matrix[idx_a][idx_a] += conductance
    if not is_ground_net(net_b):
        idx_b = node_index[net_b]
        matrix[idx_b][idx_b] += conductance
    if not is_ground_net(net_a) and not is_ground_net(net_b):
        idx_a = node_index[net_a]
        idx_b = node_index[net_b]
        matrix[idx_a][idx_b] -= conductance
        matrix[idx_b][idx_a] -= conductance


def _add_matrix(
    matrix: list[list[float]],
    node_index: Mapping[str, int],
    row_net: str,
    col_net: str,
    value: float,
) -> None:
    """Add a matrix entry unless a row or column is ground."""

    if is_ground_net(row_net) or is_ground_net(col_net):
        return
    matrix[node_index[row_net]][node_index[col_net]] += value


def _solve_linear_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve a dense linear system using Gaussian elimination."""

    size = len(rhs)
    aug = [row[:] + [rhs[idx]] for idx, row in enumerate(matrix)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(aug[row][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise UnsupportedCircuitError("singular linear DC matrix")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_value = aug[col][col]
        for item in range(col, size + 1):
            aug[col][item] /= pivot_value
        for row in range(size):
            if row == col:
                continue
            factor = aug[row][col]
            if factor == 0.0:
                continue
            for item in range(col, size + 1):
                aug[row][item] -= factor * aug[col][item]
    return [aug[row][size] for row in range(size)]


def _required_net(circuit: Circuit, ref: str, pin: str) -> str:
    """Return a connected net or raise an unsupported error."""

    net = circuit.net_for_pin(ref, pin)
    if net is None:
        raise UnsupportedCircuitError(f"unconnected pin: {ref}.{pin}")
    return net


def _positive_float(part: PartInstance, parameter: str) -> float:
    """Return a positive float parameter."""

    value = _float_parameter(part, parameter)
    if value <= 0.0:
        raise UnsupportedCircuitError(f"{part.ref}.{parameter} must be positive")
    return value


def _resistive_pins(part: PartInstance) -> tuple[str, str]:
    """Return conductive pin names for resistor-like components."""

    if part.kind in {"pullup_resistor", "pulldown_resistor"}:
        return ("net", "rail")
    return ("1", "2")


def _float_parameter(part: PartInstance, parameter: str) -> float:
    """Return a numeric part parameter."""

    value = part.parameters.get(parameter)
    if not isinstance(value, int | float):
        raise UnsupportedCircuitError(f"{part.ref} lacks numeric {parameter}")
    return float(value)
