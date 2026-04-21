"""Packaged record loading for physics discovery."""

from functools import lru_cache
from importlib import resources
import json
from typing import Mapping

from rlvr_physics.tasks.physics.discovery.constants import (
    DEFAULT_RANGE,
    PHYSICS_DISCOVERY_RECORDS_FILE,
)
from rlvr_physics.tasks.physics.discovery.types import PhysicsDiscoveryRecord


def physics_discovery_records() -> tuple[PhysicsDiscoveryRecord, ...]:
    """Return the packaged PhysGym-derived discovery records."""

    return _load_physics_discovery_records()


def record_by_source_id(source_id: int) -> PhysicsDiscoveryRecord:
    """Return a packaged record by source id."""

    for record in physics_discovery_records():
        if record.source_id == source_id:
            return record
    raise ValueError(f"unknown physics discovery source id: {source_id}")


@lru_cache(maxsize=1)
def _load_physics_discovery_records() -> tuple[PhysicsDiscoveryRecord, ...]:
    raw_text = (
        resources.files("rlvr_physics.tasks.physics.discovery.data")
        .joinpath(PHYSICS_DISCOVERY_RECORDS_FILE)
        .read_text(encoding="utf-8")
    )
    raw_records = json.loads(raw_text)
    if not isinstance(raw_records, list):
        raise ValueError("physics discovery records JSON must contain a list")
    records: list[PhysicsDiscoveryRecord] = []
    source_ids: set[int] = set()
    for index, raw_record in enumerate(raw_records):
        record = _parse_physics_discovery_record(raw_record, index)
        if record.source_id in source_ids:
            raise ValueError(
                f"duplicate physics discovery source_id: {record.source_id}"
            )
        source_ids.add(record.source_id)
        records.append(record)
    return tuple(records)


def _parse_physics_discovery_record(
    raw_record: object, index: int
) -> PhysicsDiscoveryRecord:
    if not isinstance(raw_record, dict):
        raise ValueError(f"physics discovery record {index} must be an object")
    source_id = raw_record.get("source_id")
    if not isinstance(source_id, int) or isinstance(source_id, bool):
        raise ValueError(f"physics discovery record {index} has invalid source_id")
    tag = raw_record.get("tag")
    context = raw_record.get("context")
    equation = raw_record.get("equation")
    input_variables = _require_string_mapping(
        raw_record.get("input_variables"),
        f"physics discovery record {source_id} has invalid input_variables",
    )
    output_variable = _require_string_mapping(
        raw_record.get("output_variable"),
        f"physics discovery record {source_id} has invalid output_variable",
    )
    parameter_ranges = _parse_parameter_ranges(
        raw_record.get("parameter_ranges"),
        input_variables,
        f"physics discovery record {source_id} has invalid parameter_ranges",
    )
    if not isinstance(tag, str):
        raise ValueError(f"physics discovery record {source_id} has invalid tag")
    if not isinstance(context, str):
        raise ValueError(f"physics discovery record {source_id} has invalid context")
    if not isinstance(equation, str):
        raise ValueError(f"physics discovery record {source_id} has invalid equation")
    if len(output_variable) != 1:
        raise ValueError(
            f"physics discovery record {source_id} must have one output_variable"
        )
    return PhysicsDiscoveryRecord(
        source_id=source_id,
        tag=tag,
        context=context,
        equation=equation,
        input_variables=input_variables,
        output_variable=output_variable,
        parameter_ranges=parameter_ranges,
    )


def _require_string_mapping(value: object, error_message: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not value:
        raise ValueError(error_message)
    values: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ValueError(error_message)
        values[key] = item
    return values


def _parse_parameter_ranges(
    value: object, input_variables: Mapping[str, object], error_message: str
) -> Mapping[str, object]:
    ranges: dict[str, object] = {
        str(name): DEFAULT_RANGE for name in input_variables.keys()
    }
    if value is None:
        return ranges
    if not isinstance(value, dict):
        raise ValueError(error_message)
    for key, item in value.items():
        if not isinstance(key, str) or key not in ranges:
            raise ValueError(error_message)
        ranges[key] = _require_range_pair(item, error_message)
    return ranges


def _require_range_pair(value: object, error_message: str) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(error_message)
    low = _require_number(value[0], error_message)
    high = _require_number(value[1], error_message)
    if low >= high:
        raise ValueError(error_message)
    return (low, high)


def _require_number(value: object, error_message: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(error_message)
    return float(value)
