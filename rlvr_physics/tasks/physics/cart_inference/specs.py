"""Public specification helpers for the cart inference task."""

from dataclasses import dataclass
from math import isfinite

from rlvr_physics.core.specs import (
    RendererSpec,
    RewardSpec,
    SourceSpec,
    TaskSpec,
    VerifierSpec,
)
from rlvr_physics.tasks.physics.cart_inference.budgets import (
    cart_budget_limits,
    validate_cart_budget_limits,
)
from rlvr_physics.tasks.physics.cart_inference.renderers import (
    CART_TEXT_RENDERER,
)
from rlvr_physics.tasks.physics.cart_inference.rewards import (
    DEFAULT_REWARD_CONFIG,
    CartRewardConfig,
    reward_config_parameters,
)

CART_INFERENCE_KIND = "physics.cart_inference.v1"
CART_INFERENCE_DOMAIN = "physics"


@dataclass(frozen=True)
class CartInferenceConfig:
    """Configuration for generated cart inference task instances.

    Parameters
    ----------
    min_measurement_time_s:
        Earliest model-selectable sensor query time in seconds.
    max_measurement_time_s:
        Latest model-selectable sensor query time in seconds.
    target_time_s:
        Future time whose position must be predicted.
    measurement_noise_abs_m:
        Public absolute bound on deterministic synthetic measurement noise.
    answer_tolerance_abs_m:
        Absolute verifier tolerance for the final target position.
    turn_budget:
        Maximum number of model submissions accepted before truncation.
    timeout_seconds:
        Optional wall-clock budget hint for trainers that enforce timeouts.
    token_budget:
        Optional token budget hint for prompt/completion trainers.
    action_budget:
        Maximum number of measurement tool calls allowed.
    final_answer_budget:
        Maximum number of final-answer attempts allowed.
    reward:
        Reward configuration for final-answer, accepted-action, and rejected
        submission events.

    Attributes
    ----------
    min_measurement_time_s:
        Earliest model-selectable sensor query time in seconds.
    max_measurement_time_s:
        Latest model-selectable sensor query time in seconds.
    target_time_s:
        Future time whose position must be predicted.
    measurement_noise_abs_m:
        Public absolute bound on deterministic synthetic measurement noise.
    answer_tolerance_abs_m:
        Absolute verifier tolerance for the final target position.
    turn_budget:
        Maximum number of model submissions accepted before truncation.
    timeout_seconds:
        Optional wall-clock budget hint for trainers that enforce timeouts.
    token_budget:
        Optional token budget hint for prompt/completion trainers.
    action_budget:
        Maximum number of measurement tool calls allowed.
    final_answer_budget:
        Maximum number of final-answer attempts allowed.
    reward:
        Reward configuration for final-answer, accepted-action, and rejected
        submission events.
    """

    min_measurement_time_s: float
    max_measurement_time_s: float
    target_time_s: float
    measurement_noise_abs_m: float
    answer_tolerance_abs_m: float
    turn_budget: int
    timeout_seconds: float | None
    token_budget: int | None
    action_budget: int
    final_answer_budget: int
    reward: CartRewardConfig


DEFAULT_CONFIG = CartInferenceConfig(
    min_measurement_time_s=0.0,
    max_measurement_time_s=10.0,
    target_time_s=12.0,
    measurement_noise_abs_m=0.02,
    answer_tolerance_abs_m=0.05,
    turn_budget=4,
    timeout_seconds=None,
    token_budget=512,
    action_budget=3,
    final_answer_budget=1,
    reward=DEFAULT_REWARD_CONFIG,
)


def validate_cart_inference_config(config: CartInferenceConfig) -> None:
    """Validate a cart inference task configuration.

    Parameters
    ----------
    config:
        Configuration to validate.

    Raises
    ------
    ValueError
        Raised when a field would make generation or verification ambiguous.
    """

    if config.min_measurement_time_s < 0.0:
        raise ValueError("min_measurement_time_s must be non-negative")
    _validate_finite_float(config.min_measurement_time_s, "min_measurement_time_s")
    _validate_finite_float(config.max_measurement_time_s, "max_measurement_time_s")
    _validate_finite_float(config.target_time_s, "target_time_s")
    _validate_finite_float(config.measurement_noise_abs_m, "measurement_noise_abs_m")
    _validate_finite_float(config.answer_tolerance_abs_m, "answer_tolerance_abs_m")
    if config.timeout_seconds is not None:
        _validate_finite_float(config.timeout_seconds, "timeout_seconds")
    if config.max_measurement_time_s <= config.min_measurement_time_s:
        raise ValueError("max_measurement_time_s must exceed min_measurement_time_s")
    if config.target_time_s <= config.max_measurement_time_s:
        raise ValueError("target_time_s must exceed max_measurement_time_s")
    if config.measurement_noise_abs_m < 0.0:
        raise ValueError("measurement_noise_abs_m must be non-negative")
    if config.answer_tolerance_abs_m <= 0.0:
        raise ValueError("answer_tolerance_abs_m must be positive")
    if config.turn_budget <= 0:
        raise ValueError("turn_budget must be positive")
    if config.action_budget <= 0:
        raise ValueError("action_budget must be positive")
    if config.final_answer_budget <= 0:
        raise ValueError("final_answer_budget must be positive")
    if config.final_answer_budget != 1:
        raise ValueError("final_answer_budget must be 1 for cart inference")
    if config.action_budget + config.final_answer_budget > config.turn_budget:
        raise ValueError(
            "action_budget and final_answer_budget must fit within turn_budget"
        )
    validate_cart_budget_limits(
        cart_budget_limits(
            turn_budget=config.turn_budget,
            action_budget=config.action_budget,
            final_answer_budget=config.final_answer_budget,
        )
    )
    if config.timeout_seconds is not None and config.timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be positive when provided")
    if config.token_budget is not None and config.token_budget <= 0:
        raise ValueError("token_budget must be positive when provided")


def _validate_finite_float(value: float, name: str) -> None:
    """Validate that a numeric config value is finite."""

    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def config_parameters(config: CartInferenceConfig) -> dict[str, object]:
    """Return local construction parameters as plain data.

    Parameters
    ----------
    config:
        Configuration to serialize.

    Returns
    -------
    dict[str, object]
        Configuration fields suitable for local play-test overrides and
        deterministic construction.
    """

    return {
        "min_measurement_time_s": config.min_measurement_time_s,
        "max_measurement_time_s": config.max_measurement_time_s,
        "target_time_s": config.target_time_s,
        "measurement_noise_abs_m": config.measurement_noise_abs_m,
        "answer_tolerance_abs_m": config.answer_tolerance_abs_m,
        "turn_budget": config.turn_budget,
        "timeout_seconds": config.timeout_seconds,
        "token_budget": config.token_budget,
        "action_budget": config.action_budget,
        "final_answer_budget": config.final_answer_budget,
        "reward": reward_config_parameters(config.reward),
    }


def public_source_parameters(config: CartInferenceConfig) -> dict[str, object]:
    """Return public source parameters that are safe for task specs.

    Parameters
    ----------
    config:
        Configuration to serialize.

    Returns
    -------
    dict[str, object]
        Public generation fields that exclude privileged verifier settings and
        reward-policy settings advertised elsewhere in the task spec.
    """

    return {
        "min_measurement_time_s": config.min_measurement_time_s,
        "max_measurement_time_s": config.max_measurement_time_s,
        "target_time_s": config.target_time_s,
        "measurement_noise_abs_m": config.measurement_noise_abs_m,
    }


def cart_inference_spec(config: CartInferenceConfig) -> TaskSpec:
    """Build the public task specification for cart inference.

    Parameters
    ----------
    config:
        Public generation, rollout, and verifier configuration.

    Returns
    -------
    TaskSpec
        Trainer-facing task family specification.
    """

    validate_cart_inference_config(config)
    return TaskSpec(
        kind=CART_INFERENCE_KIND,
        domain=CART_INFERENCE_DOMAIN,
        source=SourceSpec(
            source_type="cart_inference_generator",
            seed=0,
            parameters=public_source_parameters(config),
        ),
        renderers=(RendererSpec(renderer_type=CART_TEXT_RENDERER, parameters={}),),
        verifier=VerifierSpec(
            verifier_type="constant_acceleration_numeric",
            parameters={
                "answer_field": "x",
                "answer_units": "m",
                "absolute_tolerance_source": "privileged_instance_payload",
            },
        ),
        reward=RewardSpec(
            reward_type="cart_inference_event_rewards",
            parameters=reward_config_parameters(config.reward),
        ),
        timeout_seconds=config.timeout_seconds,
        token_budget=config.token_budget,
        budget_limits=cart_budget_limits(
            turn_budget=config.turn_budget,
            action_budget=config.action_budget,
            final_answer_budget=config.final_answer_budget,
        ),
        metadata={
            "task_family": "cart_inference",
            "interaction_shape": "tool_use_numeric_final",
            "physics_model": "one_dimensional_constant_acceleration",
        },
    )
