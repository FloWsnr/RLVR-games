"""Dense modified nodal analysis solver for circuit diagnosis."""

from collections.abc import Mapping
from dataclasses import dataclass

from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.errors import (
    CircuitSimulationError,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.schema import (
    CircuitDefinition,
)


@dataclass(frozen=True)
class LinearElement:
    """One stamped linear circuit element."""

    element_type: str
    component_id: str
    node_a: str
    node_b: str
    value: float
    measurement_sign: float


@dataclass(frozen=True)
class SolvedLinearCircuit:
    """Raw MNA solution with branch currents."""

    node_voltages_V: dict[str, float]
    branch_currents_A: dict[str, float]


def solve_linear_circuit(
    definition: CircuitDefinition,
    internal_nodes: tuple[str, ...],
    elements: tuple[LinearElement, ...],
) -> SolvedLinearCircuit:
    """Solve a linear circuit using dense modified nodal analysis."""

    node_names = tuple(
        node
        for node in (*definition.nodes, *internal_nodes)
        if node != definition.ground_node
    )
    node_index = {node: index for index, node in enumerate(node_names)}
    voltage_elements = tuple(
        element for element in elements if element.element_type == "voltage_source"
    )
    size = len(node_names) + len(voltage_elements)
    if size == 0:
        return SolvedLinearCircuit(
            node_voltages_V={definition.ground_node: 0.0},
            branch_currents_A={},
        )
    matrix = [[0.0 for _ in range(size)] for _ in range(size)]
    rhs = [0.0 for _ in range(size)]

    for element in elements:
        if element.element_type == "resistor":
            _stamp_resistor(matrix, node_index, element)
        elif element.element_type == "current_source":
            _stamp_current_source(rhs, node_index, element)

    for offset, element in enumerate(voltage_elements):
        branch_index = len(node_names) + offset
        _stamp_voltage_source(matrix, rhs, node_index, branch_index, element)

    solution = _solve_dense_system(matrix, rhs)
    node_voltages = {definition.ground_node: 0.0}
    for node, index in node_index.items():
        node_voltages[node] = solution[index]
    branch_currents = {
        element.component_id: solution[len(node_names) + offset]
        for offset, element in enumerate(voltage_elements)
    }
    return SolvedLinearCircuit(
        node_voltages_V=node_voltages,
        branch_currents_A=branch_currents,
    )


def _stamp_resistor(
    matrix: list[list[float]],
    node_index: Mapping[str, int],
    element: LinearElement,
) -> None:
    """Stamp one resistor into an MNA matrix."""

    conductance = 1.0 / element.value
    a_index = node_index.get(element.node_a)
    b_index = node_index.get(element.node_b)
    if a_index is not None:
        matrix[a_index][a_index] += conductance
    if b_index is not None:
        matrix[b_index][b_index] += conductance
    if a_index is not None and b_index is not None:
        matrix[a_index][b_index] -= conductance
        matrix[b_index][a_index] -= conductance


def _stamp_current_source(
    rhs: list[float],
    node_index: Mapping[str, int],
    element: LinearElement,
) -> None:
    """Stamp one current source into an MNA right-hand side."""

    current = element.value
    a_index = node_index.get(element.node_a)
    b_index = node_index.get(element.node_b)
    if a_index is not None:
        rhs[a_index] -= current
    if b_index is not None:
        rhs[b_index] += current


def _stamp_voltage_source(
    matrix: list[list[float]],
    rhs: list[float],
    node_index: Mapping[str, int],
    branch_index: int,
    element: LinearElement,
) -> None:
    """Stamp one ideal voltage source into an MNA matrix."""

    a_index = node_index.get(element.node_a)
    b_index = node_index.get(element.node_b)
    if a_index is not None:
        matrix[a_index][branch_index] += 1.0
        matrix[branch_index][a_index] += 1.0
    if b_index is not None:
        matrix[b_index][branch_index] -= 1.0
        matrix[branch_index][b_index] -= 1.0
    rhs[branch_index] = element.value


def _solve_dense_system(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    """Solve a dense linear system with partial-pivot Gaussian elimination."""

    size = len(rhs)
    augmented = [row[:] + [rhs[index]] for index, row in enumerate(matrix)]
    for pivot_col in range(size):
        pivot_row = max(
            range(pivot_col, size), key=lambda row: abs(augmented[row][pivot_col])
        )
        pivot_value = augmented[pivot_row][pivot_col]
        if abs(pivot_value) < 1.0e-12:
            raise CircuitSimulationError("singular circuit matrix")
        if pivot_row != pivot_col:
            augmented[pivot_col], augmented[pivot_row] = (
                augmented[pivot_row],
                augmented[pivot_col],
            )
        pivot_value = augmented[pivot_col][pivot_col]
        for column in range(pivot_col, size + 1):
            augmented[pivot_col][column] /= pivot_value
        for row in range(size):
            if row == pivot_col:
                continue
            factor = augmented[row][pivot_col]
            if factor == 0.0:
                continue
            for column in range(pivot_col, size + 1):
                augmented[row][column] -= factor * augmented[pivot_col][column]
    return [augmented[row][size] for row in range(size)]
