"""Core payload freezing, plain-data conversion, and hashing helpers."""

from hashlib import sha256
import json
from types import MappingProxyType
from typing import Any, Mapping


def freeze_mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    """Recursively freeze a mapping for dataclass payload storage.

    Parameters
    ----------
    values:
        Mapping whose values may contain nested mappings, lists, tuples, sets,
        or scalar values.

    Returns
    -------
    Mapping[str, object]
        Read-only mapping proxy. Nested mappings are also read-only, lists and
        tuples become tuples, and sets become tuples sorted by ``repr``.
    """

    frozen: dict[str, object] = {}
    for key, value in values.items():
        frozen[key] = _freeze_value(value)
    return MappingProxyType(frozen)


def _freeze_value(value: object) -> object:
    """Recursively freeze containers while preserving scalar values.

    Parameters
    ----------
    value:
        Value to freeze.

    Returns
    -------
    object
        Frozen value. Nested mapping keys are converted to strings, sequence
        containers become tuples, sets become deterministically ordered tuples,
        and scalar values are returned unchanged.
    """

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
    """Convert frozen payload data into plain JSON-oriented containers.

    Parameters
    ----------
    value:
        Frozen or mutable payload value to convert.

    Returns
    -------
    object
        Plain value where mappings become dictionaries with string keys, tuple
        and list values become lists, bytes become hexadecimal strings, and
        scalar values are returned unchanged.

    Raises
    ------
    TypeError
        Raised when mapping keys cannot be sorted during deterministic
        conversion.
    """

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
    """Return a stable SHA-256 hash for task data.

    Parameters
    ----------
    value:
        Payload value that can be converted by :func:`to_plain_data` and encoded
        as JSON.

    Returns
    -------
    str
        SHA-256 hex digest of the canonical JSON representation.

    Raises
    ------
    TypeError
        Raised when conversion or JSON encoding encounters unsupported data.
    """

    encoded = json.dumps(
        to_plain_data(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def mapping_to_dict(values: Mapping[str, object]) -> dict[str, Any]:
    """Return a mutable plain dictionary from frozen payload data.

    Parameters
    ----------
    values:
        Mapping to convert into plain containers.

    Returns
    -------
    dict[str, Any]
        Mutable dictionary produced by :func:`to_plain_data`.

    Raises
    ------
    TypeError
        Raised if conversion does not produce a dictionary.
    """

    plain = to_plain_data(values)
    if not isinstance(plain, dict):
        raise TypeError("mapping conversion did not produce a dictionary")
    return plain
