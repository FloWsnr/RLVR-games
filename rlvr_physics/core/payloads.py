"""Core payload freezing, plain-data conversion, and hashing helpers."""

from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any, Mapping


def freeze_mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    """Recursively freeze a mapping for dataclass payload storage."""

    frozen: dict[str, object] = {}
    for key, value in values.items():
        frozen[key] = _freeze_value(value)
    return MappingProxyType(frozen)


def _freeze_value(value: object) -> object:
    """Recursively freeze containers while preserving scalar values."""

    if isinstance(value, Mapping):
        string_keyed: dict[str, object] = {}
        for key, item in value.items():
            string_keyed[str(key)] = _freeze_value(item)
        return MappingProxyType(string_keyed)
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return tuple(_freeze_value(item) for item in sorted(value, key=repr))
    return value


def to_plain_data(value: object) -> object:
    """Convert frozen payload data into JSON-serializable containers."""

    if isinstance(value, Mapping):
        return {str(key): to_plain_data(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [to_plain_data(item) for item in value]
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    if isinstance(value, bytes):
        return value.hex()
    return value


def stable_hash(value: object) -> str:
    """Return a SHA-256 hash for JSON-compatible task data."""

    encoded = json.dumps(
        to_plain_data(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def mapping_to_dict(values: Mapping[str, object]) -> dict[str, Any]:
    """Return a mutable plain dictionary from frozen payload data."""

    plain = to_plain_data(values)
    if not isinstance(plain, dict):
        raise TypeError("mapping conversion did not produce a dictionary")
    return plain
