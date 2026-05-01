"""Verification helpers for circuit diagnosis."""

from dataclasses import dataclass

from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.constants import (
    CHECK_TOLERANCE,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.errors import (
    CircuitSimulationError,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.schema import (
    CircuitDefinition,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.simulation import (
    SimulationResult,
    component_current,
    component_power,
    node_voltage,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.values import (
    float_parameter,
    str_parameter,
)


@dataclass(frozen=True)
class TargetCheckResult:
    """Result of one post-repair behavior check."""

    check_id: str
    kind: str
    passed: bool
    value: float
    lower_bound: float | None
    upper_bound: float | None


@dataclass(frozen=True)
class FinalCircuitEvaluation:
    """Verifier evaluation for the final repair state."""

    target_restored: bool
    diagnosis_correct: bool
    submitted_faults: tuple[str, ...]
    submitted_repairs: tuple[str, ...]
    expected_faults: tuple[str, ...]
    expected_repairs: tuple[str, ...]
    check_results: tuple[TargetCheckResult, ...]
    simulation_error: str | None


def evaluate_target_checks(
    definition: CircuitDefinition, simulation: SimulationResult
) -> tuple[TargetCheckResult, ...]:
    """Evaluate public post-repair target behavior checks."""

    results: list[TargetCheckResult] = []
    for check in definition.target_checks:
        if check.kind == "voltage_between":
            node_a = str_parameter(check.parameters, "node_a")
            node_b = str_parameter(check.parameters, "node_b")
            value = node_voltage(simulation, node_a) - node_voltage(simulation, node_b)
            lower_bound = float_parameter(check.parameters, "min_V")
            upper_bound = float_parameter(check.parameters, "max_V")
        elif check.kind == "current_range":
            component_id = str_parameter(check.parameters, "component")
            value = component_current(simulation, component_id)
            lower_bound = float_parameter(check.parameters, "min_A")
            upper_bound = float_parameter(check.parameters, "max_A")
        elif check.kind == "power_max":
            component_id = str_parameter(check.parameters, "component")
            value = component_power(simulation, component_id)
            lower_bound = None
            upper_bound = float_parameter(check.parameters, "max_W")
        else:
            raise CircuitSimulationError(f"unsupported target check: {check.kind}")
        passed = _value_passes_bounds(value, lower_bound, upper_bound)
        results.append(
            TargetCheckResult(
                check_id=check.check_id,
                kind=check.kind,
                passed=passed,
                value=value,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
        )
    return tuple(results)


def target_check_public_payloads(
    check_results: tuple[TargetCheckResult, ...],
) -> tuple[dict[str, object], ...]:
    """Return trainer-safe final check result payloads."""

    return tuple(
        {
            "check_id": result.check_id,
            "kind": result.kind,
            "passed": result.passed,
        }
        for result in check_results
    )


def target_check_debug_payloads(
    check_results: tuple[TargetCheckResult, ...],
) -> tuple[dict[str, object], ...]:
    """Return privileged final check result payloads."""

    return tuple(
        {
            "check_id": result.check_id,
            "kind": result.kind,
            "passed": result.passed,
            "value": result.value,
            "lower_bound": result.lower_bound,
            "upper_bound": result.upper_bound,
        }
        for result in check_results
    )


def _value_passes_bounds(
    value: float, lower_bound: float | None, upper_bound: float | None
) -> bool:
    """Return whether a value passes optional inclusive bounds."""

    if lower_bound is not None and value < lower_bound - CHECK_TOLERANCE:
        return False
    if upper_bound is not None and value > upper_bound + CHECK_TOLERANCE:
        return False
    return True
