"""Playable task descriptor for cart inference."""

from collections.abc import Mapping
from math import isfinite

from rlvr_physics.core.factory import ConfiguredTask
from rlvr_physics.play.interaction import DEFAULT_PUBLIC_INFO_EXCLUDED_KEYS
from rlvr_physics.play.task import PlayableTask
from rlvr_physics.tasks.physics.cart_inference.renderers import (
    CART_IMAGE_RENDERER,
    CART_TEXT_RENDERER,
)
from rlvr_physics.tasks.physics.cart_inference.rewards import (
    CartRewardConfig,
    reward_config_from_mapping,
)
from rlvr_physics.tasks.physics.cart_inference.specs import (
    DEFAULT_CONFIG,
    CartInferenceConfig,
    config_parameters,
)
from rlvr_physics.tasks.physics.cart_inference.task import cart_inference_task

CART_CONFIG_PARAMETER_KEYS = frozenset(config_parameters(DEFAULT_CONFIG))


def build_cart_inference_play_task(
    parameters: Mapping[str, object], renderer_type: str
) -> ConfiguredTask:
    """Build a configured cart task for public play-testing.

    Parameters
    ----------
    parameters:
        Public cart task construction parameters.
    renderer_type:
        Renderer identifier selected for emitted observations.

    Returns
    -------
    ConfiguredTask
        Configured cart inference task.
    """

    return cart_inference_task(
        config=cart_inference_config_from_parameters(parameters),
        renderer_type=renderer_type,
    )


def cart_inference_config_from_parameters(
    parameters: Mapping[str, object],
) -> CartInferenceConfig:
    """Build a cart inference config from public parameters.

    Parameters
    ----------
    parameters:
        Public cart task construction parameters.

    Returns
    -------
    CartInferenceConfig
        Cart inference configuration.
    """

    _reject_unknown_parameters(parameters)
    return CartInferenceConfig(
        min_measurement_time_s=_required_float_parameter(
            parameters, "min_measurement_time_s"
        ),
        max_measurement_time_s=_required_float_parameter(
            parameters, "max_measurement_time_s"
        ),
        target_time_s=_required_float_parameter(parameters, "target_time_s"),
        measurement_noise_abs_m=_required_float_parameter(
            parameters, "measurement_noise_abs_m"
        ),
        answer_tolerance_abs_m=_required_float_parameter(
            parameters, "answer_tolerance_abs_m"
        ),
        turn_budget=_required_int_parameter(parameters, "turn_budget"),
        timeout_seconds=_optional_float_parameter(parameters, "timeout_seconds"),
        token_budget=_optional_int_parameter(parameters, "token_budget"),
        action_budget=_required_int_parameter(parameters, "action_budget"),
        final_answer_budget=_required_int_parameter(parameters, "final_answer_budget"),
        reward=_required_reward_config(parameters, "reward"),
    )


CART_PLAYABLE = PlayableTask(
    name="physics.cart_inference",
    default_renderer=CART_TEXT_RENDERER,
    renderers=(CART_TEXT_RENDERER, CART_IMAGE_RENDERER),
    default_parameters=config_parameters(DEFAULT_CONFIG),
    build_task=build_cart_inference_play_task,
    public_info_excluded_keys=DEFAULT_PUBLIC_INFO_EXCLUDED_KEYS,
)


def _required_float_parameter(parameters: Mapping[str, object], name: str) -> float:
    """Read a required numeric parameter as a float."""

    value = _required_parameter(parameters, name)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    if isinstance(value, int | float):
        numeric_value = float(value)
        if isfinite(numeric_value):
            return numeric_value
        raise ValueError(f"{name} must be finite")
    raise ValueError(f"{name} must be numeric")


def _required_int_parameter(parameters: Mapping[str, object], name: str) -> int:
    """Read a required integer parameter."""

    value = _required_parameter(parameters, name)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if isinstance(value, int):
        return value
    raise ValueError(f"{name} must be an integer")


def _optional_float_parameter(
    parameters: Mapping[str, object], name: str
) -> float | None:
    """Read an optional numeric parameter as a float."""

    value = _required_parameter(parameters, name)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric when provided")
    if isinstance(value, int | float):
        numeric_value = float(value)
        if isfinite(numeric_value):
            return numeric_value
        raise ValueError(f"{name} must be finite when provided")
    raise ValueError(f"{name} must be numeric when provided")


def _optional_int_parameter(parameters: Mapping[str, object], name: str) -> int | None:
    """Read an optional integer parameter."""

    value = _required_parameter(parameters, name)
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer when provided")
    if isinstance(value, int):
        return value
    raise ValueError(f"{name} must be an integer when provided")


def _required_reward_config(
    parameters: Mapping[str, object], name: str
) -> CartRewardConfig:
    """Read a required cart reward configuration parameter."""

    value = _required_parameter(parameters, name)
    if isinstance(value, Mapping):
        return reward_config_from_mapping(value)
    raise ValueError(f"{name} must be an object")


def _required_parameter(parameters: Mapping[str, object], name: str) -> object:
    """Return one required parameter value."""

    try:
        return parameters[name]
    except KeyError as error:
        raise ValueError(f"missing cart inference parameter: {name}") from error


def _reject_unknown_parameters(parameters: Mapping[str, object]) -> None:
    """Reject public cart parameters that are not part of the config."""

    unknown_keys = sorted(set(parameters) - CART_CONFIG_PARAMETER_KEYS)
    if len(unknown_keys) > 0:
        joined_keys = ", ".join(unknown_keys)
        raise ValueError(f"unknown cart inference parameter(s): {joined_keys}")
