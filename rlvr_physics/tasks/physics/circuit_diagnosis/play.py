"""Playable task descriptor for circuit diagnosis."""

from collections.abc import Mapping
from math import isfinite

from rlvr_physics.core.factory import ConfiguredTask
from rlvr_physics.play.interaction import DEFAULT_PUBLIC_INFO_EXCLUDED_KEYS
from rlvr_physics.play.task import PlayableTask
from rlvr_physics.tasks.physics.circuit_diagnosis.rewards import (
    CircuitRewardConfig,
    reward_config_from_mapping,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.specs import (
    CIRCUIT_TEXT_RENDERER,
    DEFAULT_CONFIG,
    CircuitDiagnosisConfig,
    config_parameters,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.task import circuit_diagnosis_task

CIRCUIT_CONFIG_PARAMETER_KEYS = frozenset(config_parameters(DEFAULT_CONFIG))


def build_circuit_diagnosis_play_task(
    parameters: Mapping[str, object], renderer_type: str
) -> ConfiguredTask:
    """Build a configured circuit task for public play-testing."""

    return circuit_diagnosis_task(
        config=circuit_diagnosis_config_from_parameters(parameters),
        renderer_type=renderer_type,
    )


def circuit_diagnosis_config_from_parameters(
    parameters: Mapping[str, object],
) -> CircuitDiagnosisConfig:
    """Build a circuit diagnosis config from public parameters."""

    _reject_unknown_parameters(parameters)
    return CircuitDiagnosisConfig(
        min_fault_count=_required_int_parameter(parameters, "min_fault_count"),
        max_fault_count=_required_int_parameter(parameters, "max_fault_count"),
        component_count=_required_int_parameter(parameters, "component_count"),
        min_diagnosis_measurements=_required_int_parameter(
            parameters, "min_diagnosis_measurements"
        ),
        max_diagnosis_measurements=_required_int_parameter(
            parameters, "max_diagnosis_measurements"
        ),
        generator_attempt_limit=_required_int_parameter(
            parameters, "generator_attempt_limit"
        ),
        max_mna_condition_number=_required_float_parameter(
            parameters, "max_mna_condition_number"
        ),
        min_observable_delta=_required_float_parameter(
            parameters, "min_observable_delta"
        ),
        target_tolerance_fraction=_required_float_parameter(
            parameters, "target_tolerance_fraction"
        ),
        target_tolerance_abs=_required_float_parameter(
            parameters, "target_tolerance_abs"
        ),
        turn_budget=_required_int_parameter(parameters, "turn_budget"),
        probe_budget=_required_int_parameter(parameters, "probe_budget"),
        repair_budget=_required_int_parameter(parameters, "repair_budget"),
        final_answer_budget=_required_int_parameter(parameters, "final_answer_budget"),
        timeout_seconds=_optional_float_parameter(parameters, "timeout_seconds"),
        token_budget=_optional_int_parameter(parameters, "token_budget"),
        reward=_required_reward_config(parameters, "reward"),
    )


CIRCUIT_PLAYABLE = PlayableTask(
    name="physics.circuit_diagnosis",
    default_renderer=CIRCUIT_TEXT_RENDERER,
    renderers=(CIRCUIT_TEXT_RENDERER,),
    default_parameters=config_parameters(DEFAULT_CONFIG),
    build_task=build_circuit_diagnosis_play_task,
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
) -> CircuitRewardConfig:
    """Read a required circuit reward configuration parameter."""

    value = _required_parameter(parameters, name)
    if isinstance(value, Mapping):
        return reward_config_from_mapping(value)
    raise ValueError(f"{name} must be an object")


def _required_parameter(parameters: Mapping[str, object], name: str) -> object:
    """Return one required parameter value."""

    try:
        return parameters[name]
    except KeyError as error:
        raise ValueError(f"missing circuit diagnosis parameter: {name}") from error


def _reject_unknown_parameters(parameters: Mapping[str, object]) -> None:
    """Reject public circuit parameters that are not part of the config."""

    unknown_keys = sorted(set(parameters) - CIRCUIT_CONFIG_PARAMETER_KEYS)
    if len(unknown_keys) > 0:
        joined_keys = ", ".join(unknown_keys)
        raise ValueError(f"unknown circuit diagnosis parameter(s): {joined_keys}")
