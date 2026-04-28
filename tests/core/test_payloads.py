"""Tests for core payload helpers."""

from typing import MutableMapping, cast

import pytest

from rlvr_physics.core.payloads import (
    freeze_mapping,
    mapping_to_dict,
    stable_hash,
    to_plain_data,
)


def test_freeze_mapping_recursively_freezes_plain_payloads() -> None:
    payload = freeze_mapping(
        {
            "numbers": [1, 2],
            "nested": {"enabled": True},
            "labels": {"beta", "alpha"},
        }
    )

    assert payload["numbers"] == (1, 2)
    assert payload["nested"] == {"enabled": True}
    assert payload["labels"] == ("alpha", "beta")
    with pytest.raises(TypeError):
        payload["extra"] = "blocked"  # type: ignore[index]
    nested = cast(MutableMapping[str, object], payload["nested"])
    with pytest.raises(TypeError):
        nested["enabled"] = False


def test_plain_data_conversion_and_hashing_are_deterministic() -> None:
    frozen = freeze_mapping({"payload": {"b": (2, 3), "a": b"\x0f"}})

    assert to_plain_data(frozen) == {"payload": {"a": "0f", "b": [2, 3]}}
    assert mapping_to_dict(frozen) == {"payload": {"a": "0f", "b": [2, 3]}}
    assert stable_hash({"b": [2, 3], "a": "0f"}) == stable_hash(
        {"a": "0f", "b": (2, 3)}
    )
