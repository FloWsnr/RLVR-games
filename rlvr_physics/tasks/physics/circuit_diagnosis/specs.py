"""Public specification helpers for the circuit diagnosis task."""

from dataclasses import dataclass
from math import isfinite

from rlvr_physics.core.specs import (
    RendererSpec,
    RewardSpec,
    SourceSpec,
    TaskSpec,
    VerifierSpec,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.budgets import (
    circuit_budget_limits,
    validate_circuit_budget_limits,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.rewards import (
    DEFAULT_REWARD_CONFIG,
    CircuitRewardConfig,
    reward_config_parameters,
)

CIRCUIT_DIAGNOSIS_KIND = "physics.circuit_diagnosis.v1"
CIRCUIT_DIAGNOSIS_DOMAIN = "physics"
CIRCUIT_TEXT_RENDERER = "circuit_diagnosis.text"
MAX_COMPONENT_COUNT = 10


@dataclass(frozen=True)
class CircuitDiagnosisConfig:
    """Configuration for generated circuit diagnosis task instances.

    Parameters
    ----------
    min_fault_count:
        Minimum number of hidden faults sampled per instance.
    max_fault_count:
        Maximum number of hidden faults sampled per instance.
    component_count:
        Exact number of generated passive resistor components.
    min_diagnosis_measurements:
        Minimum optimal measurement depth accepted during task validation.
    max_diagnosis_measurements:
        Maximum optimal measurement depth accepted during task validation.
    generator_attempt_limit:
        Maximum candidates to sample before failing instance construction.
    max_mna_condition_number:
        Maximum accepted infinity-norm condition estimate for nominal MNA.
    min_observable_delta:
        Minimum observable signature delta required for generated faults.
    target_tolerance_fraction:
        Relative tolerance used when template nominal checks are built.
    target_tolerance_abs:
        Absolute tolerance floor used when template nominal checks are built.
    turn_budget:
        Maximum number of model submissions before truncation.
    probe_budget:
        Maximum number of accepted probing actions.
    repair_budget:
        Maximum number of accepted repair actions.
    final_answer_budget:
        Maximum number of final-answer attempts.
    timeout_seconds:
        Optional wall-clock budget hint for trainers.
    token_budget:
        Optional token budget hint for prompt/completion trainers.
    reward:
        Reward configuration for task events.
    """

    min_fault_count: int
    max_fault_count: int
    component_count: int
    min_diagnosis_measurements: int
    max_diagnosis_measurements: int
    generator_attempt_limit: int
    max_mna_condition_number: float
    min_observable_delta: float
    target_tolerance_fraction: float
    target_tolerance_abs: float
    turn_budget: int
    probe_budget: int
    repair_budget: int
    final_answer_budget: int
    timeout_seconds: float | None
    token_budget: int | None
    reward: CircuitRewardConfig


DEFAULT_CONFIG = CircuitDiagnosisConfig(
    min_fault_count=1,
    max_fault_count=1,
    component_count=8,
    min_diagnosis_measurements=1,
    max_diagnosis_measurements=4,
    generator_attempt_limit=400,
    max_mna_condition_number=1.0e10,
    min_observable_delta=0.02,
    target_tolerance_fraction=0.08,
    target_tolerance_abs=0.05,
    turn_budget=14,
    probe_budget=8,
    repair_budget=2,
    final_answer_budget=1,
    timeout_seconds=None,
    token_budget=1200,
    reward=DEFAULT_REWARD_CONFIG,
)


def validate_circuit_diagnosis_config(config: CircuitDiagnosisConfig) -> None:
    """Validate a circuit diagnosis task configuration."""

    if config.min_fault_count != 1 or config.max_fault_count != 1:
        raise ValueError(
            "procedural circuit diagnosis v1 samples exactly one hidden fault"
        )
    if config.component_count < 2:
        raise ValueError("component_count must be at least 2")
    if config.component_count > MAX_COMPONENT_COUNT:
        raise ValueError(f"component_count must be at most {MAX_COMPONENT_COUNT}")
    if config.min_diagnosis_measurements <= 0:
        raise ValueError("min_diagnosis_measurements must be positive")
    if config.max_diagnosis_measurements < config.min_diagnosis_measurements:
        raise ValueError(
            "max_diagnosis_measurements must be greater than or equal to minimum"
        )
    if config.generator_attempt_limit <= 0:
        raise ValueError("generator_attempt_limit must be positive")
    _validate_finite_float(config.max_mna_condition_number, "max_mna_condition_number")
    if config.max_mna_condition_number <= 0.0:
        raise ValueError("max_mna_condition_number must be positive")
    _validate_finite_float(config.min_observable_delta, "min_observable_delta")
    if config.min_observable_delta <= 0.0:
        raise ValueError("min_observable_delta must be positive")
    if config.max_fault_count > config.repair_budget:
        raise ValueError("repair_budget must cover max_fault_count")
    if config.max_diagnosis_measurements + 1 > config.probe_budget:
        raise ValueError(
            "probe_budget must cover source setup plus max_diagnosis_measurements"
        )
    _validate_finite_float(
        config.target_tolerance_fraction, "target_tolerance_fraction"
    )
    _validate_finite_float(config.target_tolerance_abs, "target_tolerance_abs")
    if config.target_tolerance_fraction <= 0.0:
        raise ValueError("target_tolerance_fraction must be positive")
    if config.target_tolerance_abs <= 0.0:
        raise ValueError("target_tolerance_abs must be positive")
    if config.timeout_seconds is not None:
        _validate_finite_float(config.timeout_seconds, "timeout_seconds")
        if config.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive when provided")
    if config.token_budget is not None and config.token_budget <= 0:
        raise ValueError("token_budget must be positive when provided")
    validate_circuit_budget_limits(
        circuit_budget_limits(
            turn_budget=config.turn_budget,
            probe_budget=config.probe_budget,
            repair_budget=config.repair_budget,
            final_answer_budget=config.final_answer_budget,
        )
    )


def config_parameters(config: CircuitDiagnosisConfig) -> dict[str, object]:
    """Return local construction parameters as plain data."""

    return {
        "min_fault_count": config.min_fault_count,
        "max_fault_count": config.max_fault_count,
        "component_count": config.component_count,
        "min_diagnosis_measurements": config.min_diagnosis_measurements,
        "max_diagnosis_measurements": config.max_diagnosis_measurements,
        "generator_attempt_limit": config.generator_attempt_limit,
        "max_mna_condition_number": config.max_mna_condition_number,
        "min_observable_delta": config.min_observable_delta,
        "target_tolerance_fraction": config.target_tolerance_fraction,
        "target_tolerance_abs": config.target_tolerance_abs,
        "turn_budget": config.turn_budget,
        "probe_budget": config.probe_budget,
        "repair_budget": config.repair_budget,
        "final_answer_budget": config.final_answer_budget,
        "timeout_seconds": config.timeout_seconds,
        "token_budget": config.token_budget,
        "reward": reward_config_parameters(config.reward),
    }


def public_source_parameters(config: CircuitDiagnosisConfig) -> dict[str, object]:
    """Return public source parameters that are safe for task specs."""

    return {
        "fault_count_range": {
            "min": config.min_fault_count,
            "max": config.max_fault_count,
        },
        "generator": "procedural_passive_resistor_network_v1",
        "component_count": config.component_count,
        "diagnosis_measurement_depth": {
            "min": config.min_diagnosis_measurements,
            "max": config.max_diagnosis_measurements,
        },
        "min_observable_delta": config.min_observable_delta,
        "target_tolerance_fraction": config.target_tolerance_fraction,
        "target_tolerance_abs": config.target_tolerance_abs,
    }


def circuit_diagnosis_spec(config: CircuitDiagnosisConfig) -> TaskSpec:
    """Build the public task specification for circuit diagnosis."""

    validate_circuit_diagnosis_config(config)
    return TaskSpec(
        kind=CIRCUIT_DIAGNOSIS_KIND,
        domain=CIRCUIT_DIAGNOSIS_DOMAIN,
        source=SourceSpec(
            source_type="circuit_diagnosis_procedural_graphs",
            seed=0,
            parameters=public_source_parameters(config),
        ),
        renderers=(RendererSpec(renderer_type=CIRCUIT_TEXT_RENDERER, parameters={}),),
        verifier=VerifierSpec(
            verifier_type="procedural_dc_mna_fault_diagnosis",
            parameters={
                "repair_scoring": "target_behavior_restored",
                "diagnosis_metadata": "privileged_fault_labels",
                "topology_validation": "passive_two_terminal_graph",
                "numerical_validation": "dense_mna_rank_conditioning",
            },
        ),
        reward=RewardSpec(
            reward_type="circuit_diagnosis_event_rewards",
            parameters=reward_config_parameters(config.reward),
        ),
        timeout_seconds=config.timeout_seconds,
        token_budget=config.token_budget,
        budget_limits=circuit_budget_limits(
            turn_budget=config.turn_budget,
            probe_budget=config.probe_budget,
            repair_budget=config.repair_budget,
            final_answer_budget=config.final_answer_budget,
        ),
        metadata={
            "task_family": "circuit_diagnosis",
            "interaction_shape": "tool_use_probe_repair_final",
            "physics_model": "piecewise_dc_circuit",
        },
    )


def validate_circuit_renderer_type(renderer_type: str) -> None:
    """Validate a circuit diagnosis renderer identifier."""

    if renderer_type != CIRCUIT_TEXT_RENDERER:
        raise ValueError(f"unsupported circuit diagnosis renderer: {renderer_type}")


def _validate_finite_float(value: float, name: str) -> None:
    """Validate that a numeric config value is finite."""

    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
