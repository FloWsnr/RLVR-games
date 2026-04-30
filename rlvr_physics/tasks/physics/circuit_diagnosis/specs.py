"""Public specification helpers for the circuit diagnosis task."""

from dataclasses import dataclass
from math import isfinite

from rlvr_physics.core.rendering import PNG_MIME_TYPE
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
CIRCUIT_IMAGE_RENDERER = "circuit_diagnosis.image"


@dataclass(frozen=True)
class CircuitDiagnosisConfig:
    """Configuration for generated circuit diagnosis task instances.

    Parameters
    ----------
    min_fault_count:
        Minimum number of hidden faults sampled per instance.
    max_fault_count:
        Maximum number of hidden faults sampled per instance.
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
    max_fault_count=2,
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

    if config.min_fault_count <= 0:
        raise ValueError("min_fault_count must be positive")
    if config.max_fault_count < config.min_fault_count:
        raise ValueError("max_fault_count must be greater than or equal to minimum")
    if config.max_fault_count > 2:
        raise ValueError("max_fault_count must be at most 2 for circuit diagnosis v1")
    if config.max_fault_count > config.repair_budget:
        raise ValueError("repair_budget must cover max_fault_count")
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
        "target_tolerance_fraction": config.target_tolerance_fraction,
        "target_tolerance_abs": config.target_tolerance_abs,
        "templates": (
            "resistor_divider",
            "led_limiter",
            "rc_dc_node",
            "switched_load",
            "internal_source",
            "bridge_balance",
        ),
    }


def circuit_diagnosis_spec(config: CircuitDiagnosisConfig) -> TaskSpec:
    """Build the public task specification for circuit diagnosis."""

    validate_circuit_diagnosis_config(config)
    return TaskSpec(
        kind=CIRCUIT_DIAGNOSIS_KIND,
        domain=CIRCUIT_DIAGNOSIS_DOMAIN,
        source=SourceSpec(
            source_type="circuit_diagnosis_curated_templates",
            seed=0,
            parameters=public_source_parameters(config),
        ),
        renderers=(
            RendererSpec(renderer_type=CIRCUIT_TEXT_RENDERER, parameters={}),
            RendererSpec(
                renderer_type=CIRCUIT_IMAGE_RENDERER,
                parameters={"mime_type": PNG_MIME_TYPE},
            ),
        ),
        verifier=VerifierSpec(
            verifier_type="piecewise_dc_mna_target_behavior",
            parameters={
                "repair_scoring": "target_behavior_restored",
                "diagnosis_metadata": "privileged_fault_labels",
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

    if renderer_type not in {CIRCUIT_TEXT_RENDERER, CIRCUIT_IMAGE_RENDERER}:
        raise ValueError(f"unsupported circuit diagnosis renderer: {renderer_type}")


def _validate_finite_float(value: float, name: str) -> None:
    """Validate that a numeric config value is finite."""

    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
