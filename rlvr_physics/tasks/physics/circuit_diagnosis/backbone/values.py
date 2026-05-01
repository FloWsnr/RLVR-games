"""Typed value readers shared by circuit backbone modules."""

from collections.abc import Mapping
from math import isfinite


def str_sequence(values: Mapping[str, object], name: str) -> tuple[str, ...]:
    """Return a required string sequence field from a mapping."""

    raw_values = sequence_field(values, name)
    strings: list[str] = []
    for value in raw_values:
        if not isinstance(value, str) or value == "":
            raise TypeError(f"{name} entries must be non-empty strings")
        strings.append(value)
    return tuple(strings)


def sequence_field(values: Mapping[str, object], name: str) -> tuple[object, ...]:
    """Return a required sequence field from a mapping."""

    value = values[name]
    if isinstance(value, list | tuple):
        return tuple(value)
    raise TypeError(f"{name} must be a sequence")


def mapping_field(values: Mapping[str, object], name: str) -> Mapping[str, object]:
    """Return a required mapping field from a mapping."""

    value = values[name]
    if isinstance(value, Mapping):
        return value
    raise TypeError(f"{name} must be a mapping")


def str_field(values: Mapping[str, object], name: str) -> str:
    """Return a required string field from a mapping."""

    value = values[name]
    if isinstance(value, str) and value != "":
        return value
    raise TypeError(f"{name} must be a non-empty string")


def int_field(values: Mapping[str, object], name: str) -> int:
    """Return a required integer field from a mapping."""

    value = values[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def float_field(values: Mapping[str, object], name: str) -> float:
    """Return a required numeric field from a mapping."""

    return numeric_value(values[name], name)


def str_parameter(values: Mapping[str, object], name: str) -> str:
    """Return a required string parameter."""

    return str_field(values, name)


def float_parameter(values: Mapping[str, object], name: str) -> float:
    """Return a required numeric parameter."""

    return float_field(values, name)


def optional_float_parameter(
    values: Mapping[str, object], name: str, fallback: float
) -> float:
    """Return an optional numeric parameter."""

    value = values.get(name)
    if value is None:
        return fallback
    return numeric_value(value, name)


def bool_parameter(values: Mapping[str, object], name: str) -> bool:
    """Return a required boolean parameter."""

    value = values[name]
    if isinstance(value, bool):
        return value
    raise TypeError(f"{name} must be a boolean")


def numeric_value(value: object, name: str) -> float:
    """Return a finite numeric value."""

    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    if isinstance(value, int | float):
        numeric_value = float(value)
        if isfinite(numeric_value):
            return numeric_value
        raise TypeError(f"{name} must be finite")
    raise TypeError(f"{name} must be numeric")


def format_code_number(value: float) -> str:
    """Return a stable compact number for repair labels."""

    return f"{value:g}"
