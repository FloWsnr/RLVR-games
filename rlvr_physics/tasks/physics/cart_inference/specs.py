"""Public specification helpers for the cart inference task."""

from dataclasses import dataclass

from rlvr_physics.core.rendering import PNG_MIME_TYPE
from rlvr_physics.core.specs import (
    RendererSpec,
    RewardSpec,
    SourceSpec,
    TaskSpec,
    VerifierSpec,
)
from rlvr_physics.tasks.physics.cart_inference.renderers import (
    CART_IMAGE_RENDERER,
    CART_TEXT_RENDERER,
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
    max_turns:
        Maximum number of submissions accepted before truncation.
    timeout_seconds:
        Optional wall-clock budget hint for trainers that enforce timeouts.
    token_budget:
        Optional token budget hint for prompt/completion trainers.
    action_budget:
        Maximum number of measurement tool calls allowed.

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
    max_turns:
        Maximum number of submissions accepted before truncation.
    timeout_seconds:
        Optional wall-clock budget hint for trainers that enforce timeouts.
    token_budget:
        Optional token budget hint for prompt/completion trainers.
    action_budget:
        Maximum number of measurement tool calls allowed.
    """

    min_measurement_time_s: float
    max_measurement_time_s: float
    target_time_s: float
    measurement_noise_abs_m: float
    answer_tolerance_abs_m: float
    max_turns: int
    timeout_seconds: float | None
    token_budget: int | None
    action_budget: int


DEFAULT_CONFIG = CartInferenceConfig(
    min_measurement_time_s=0.0,
    max_measurement_time_s=10.0,
    target_time_s=12.0,
    measurement_noise_abs_m=0.02,
    answer_tolerance_abs_m=0.05,
    max_turns=5,
    timeout_seconds=None,
    token_budget=512,
    action_budget=3,
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
    if config.max_measurement_time_s <= config.min_measurement_time_s:
        raise ValueError("max_measurement_time_s must exceed min_measurement_time_s")
    if config.target_time_s <= config.max_measurement_time_s:
        raise ValueError("target_time_s must exceed max_measurement_time_s")
    if config.measurement_noise_abs_m < 0.0:
        raise ValueError("measurement_noise_abs_m must be non-negative")
    if config.answer_tolerance_abs_m <= 0.0:
        raise ValueError("answer_tolerance_abs_m must be positive")
    if config.max_turns <= 0:
        raise ValueError("max_turns must be positive")
    if config.action_budget <= 0:
        raise ValueError("action_budget must be positive")
    if config.action_budget >= config.max_turns:
        raise ValueError("action_budget must leave at least one turn for final answer")
    if config.timeout_seconds is not None and config.timeout_seconds <= 0.0:
        raise ValueError("timeout_seconds must be positive when provided")
    if config.token_budget is not None and config.token_budget <= 0:
        raise ValueError("token_budget must be positive when provided")


def config_parameters(config: CartInferenceConfig) -> dict[str, object]:
    """Return public configuration parameters as plain data.

    Parameters
    ----------
    config:
        Configuration to serialize.

    Returns
    -------
    dict[str, object]
        Public configuration fields suitable for task specs and metadata.
    """

    return {
        "min_measurement_time_s": config.min_measurement_time_s,
        "max_measurement_time_s": config.max_measurement_time_s,
        "target_time_s": config.target_time_s,
        "measurement_noise_abs_m": config.measurement_noise_abs_m,
        "answer_tolerance_abs_m": config.answer_tolerance_abs_m,
        "max_turns": config.max_turns,
        "timeout_seconds": config.timeout_seconds,
        "token_budget": config.token_budget,
        "action_budget": config.action_budget,
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
    parameters = config_parameters(config)
    return TaskSpec(
        kind=CART_INFERENCE_KIND,
        domain=CART_INFERENCE_DOMAIN,
        source=SourceSpec(
            source_type="cart_inference_generator",
            seed=0,
            parameters=parameters,
        ),
        renderers=(
            RendererSpec(renderer_type=CART_TEXT_RENDERER, parameters={}),
            RendererSpec(
                renderer_type=CART_IMAGE_RENDERER,
                parameters={"mime_type": PNG_MIME_TYPE},
            ),
        ),
        verifier=VerifierSpec(
            verifier_type="constant_acceleration_numeric",
            parameters={
                "answer_field": "x",
                "answer_units": "m",
                "absolute_tolerance_m": config.answer_tolerance_abs_m,
            },
        ),
        reward=RewardSpec(
            reward_type="threshold_with_linear_partial_credit",
            parameters={
                "perfect_score": 1.0,
                "incorrect_score": 0.0,
                "partial_credit_window_tolerances": 10.0,
            },
        ),
        max_turns=config.max_turns,
        timeout_seconds=config.timeout_seconds,
        token_budget=config.token_budget,
        action_budget=config.action_budget,
        metadata={
            "task_family": "cart_inference",
            "interaction_shape": "tool_use_numeric_final",
            "physics_model": "one_dimensional_constant_acceleration",
        },
    )
