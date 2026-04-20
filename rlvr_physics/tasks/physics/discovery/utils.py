"""Utility validation helpers for physics discovery."""

import math
from typing import Mapping

from rlvr_physics.tasks.physics.discovery.constants import PHYSICS_DISCOVERY_PRIOR_MODES


def validate_prior_mode(prior_mode: str) -> None:
    """Validate a physics discovery prior mode.

    Parameters
    ----------
    prior_mode:
        Candidate prior exposure mode.
    """

    if prior_mode not in PHYSICS_DISCOVERY_PRIOR_MODES:
        raise ValueError(f"unknown physics discovery prior mode: {prior_mode}")


def validate_positive_quota(value: int, name: str) -> None:
    """Validate that a quota is positive.

    Parameters
    ----------
    value:
        Quota value.
    name:
        Human-readable quota name for errors.
    """

    if value <= 0:
        raise ValueError(f"{name} must be positive")


def coerce_float(value: object, name: str) -> float:
    """Return a finite float from numeric payload data."""

    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def float_mapping(values: Mapping[str, object]) -> Mapping[str, float]:
    """Return a string-keyed float mapping from payload data."""

    floats: dict[str, float] = {}
    for key, value in values.items():
        floats[str(key)] = coerce_float(value, str(key))
    return floats


def range_pair(value: object, name: str) -> tuple[float, float]:
    """Return a validated numeric range pair."""

    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} must be a two-item tuple")
    low = coerce_float(value[0], f"{name} lower bound")
    high = coerce_float(value[1], f"{name} upper bound")
    if low >= high:
        raise ValueError(f"{name} lower bound must be less than upper bound")
    return low, high


def single_mapping_key(values: Mapping[str, object], name: str) -> str:
    """Return the only key in a mapping."""

    if len(values) != 1:
        raise ValueError(f"{name} must contain exactly one item")
    return str(next(iter(values.keys())))
