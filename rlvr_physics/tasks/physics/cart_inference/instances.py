"""Immutable instance construction for the cart inference task."""

from random import Random

from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.core.payloads import stable_hash
from rlvr_physics.tasks.physics.cart_inference.backbone import position_from_values
from rlvr_physics.tasks.physics.cart_inference.specs import (
    CART_INFERENCE_DOMAIN,
    CART_INFERENCE_KIND,
    CartInferenceConfig,
    config_parameters,
    validate_cart_inference_config,
)


def build_cart_inference_instance(
    seed: int, config: CartInferenceConfig
) -> TaskInstance:
    """Build one deterministic cart inference task instance.

    Parameters
    ----------
    seed:
        Generator seed for reproducible public and privileged payloads.
    config:
        Public generation, rollout, and verifier configuration.

    Returns
    -------
    TaskInstance
        Immutable scalar task instance ready for session creation.
    """

    validate_cart_inference_config(config)
    rng = Random(seed)
    initial_position_m = round(rng.uniform(-2.0, 2.0), 2)
    initial_velocity_mps = round(_sample_nonzero(rng, -1.5, 1.5, 0.2), 2)
    acceleration_mps2 = round(_sample_nonzero(rng, -0.8, 0.8, 0.15), 2)
    measurement_noise_seed = rng.randrange(0, 2**31)
    exact_target_position_m = position_from_values(
        initial_position_m=initial_position_m,
        initial_velocity_mps=initial_velocity_mps,
        acceleration_mps2=acceleration_mps2,
        time_s=config.target_time_s,
    )

    public_payload: dict[str, object] = {
        "initial_position_m": initial_position_m,
        "initial_velocity_mps": initial_velocity_mps,
        "target_time_s": config.target_time_s,
        "measurement_time_range_s": {
            "min": config.min_measurement_time_s,
            "max": config.max_measurement_time_s,
        },
        "measurement_noise_abs_m": config.measurement_noise_abs_m,
        "required_answer": {"field": "x", "units": "m"},
    }
    privileged_payload: dict[str, object] = {
        "acceleration_mps2": acceleration_mps2,
        "answer_tolerance_abs_m": config.answer_tolerance_abs_m,
        "exact_target_position_m": exact_target_position_m,
        "measurement_noise_seed": measurement_noise_seed,
    }
    task_hash = stable_hash(
        {
            "kind": CART_INFERENCE_KIND,
            "seed": seed,
            "config": config_parameters(config),
            "public_payload": public_payload,
            "privileged_payload": privileged_payload,
        }
    )[:16]

    return TaskInstance(
        task_id=f"cart-inference-v1-{task_hash}",
        kind=CART_INFERENCE_KIND,
        domain=CART_INFERENCE_DOMAIN,
        seed=seed,
        public_payload=public_payload,
        privileged_payload=privileged_payload,
        max_turns=config.max_turns,
        timeout_seconds=config.timeout_seconds,
        token_budget=config.token_budget,
        action_budget=config.action_budget,
        metadata={
            "task_family": "cart_inference",
            "difficulty": "example",
            "config": config_parameters(config),
        },
    )


def _sample_nonzero(rng: Random, low: float, high: float, minimum_abs: float) -> float:
    """Sample a float whose absolute value is not too close to zero.

    Parameters
    ----------
    rng:
        Deterministic random source.
    low:
        Inclusive lower sampling bound.
    high:
        Inclusive upper sampling bound.
    minimum_abs:
        Minimum absolute value accepted.

    Returns
    -------
    float
        Sampled value with ``abs(value) >= minimum_abs``.
    """

    value = rng.uniform(low, high)
    while abs(value) < minimum_abs:
        value = rng.uniform(low, high)
    return value
