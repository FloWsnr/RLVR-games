"""Tests for immutable task instances."""

from typing import Mapping, cast

import pytest

from rlvr_physics.core.instances import (
    TaskInstance,
)
from rlvr_physics.core.payloads import (
    stable_hash,
    to_plain_data,
)


def test_task_instance_freezes_payloads_and_separates_public_view(
    example_task_instance: TaskInstance,
) -> None:
    instance = example_task_instance

    assert instance.public_payload["prompt"] == "configured task prompt"
    with pytest.raises(TypeError):
        instance.public_payload["new"] = "blocked"  # type: ignore[index]

    public_view = instance.public_view()
    assert public_view["payload"] == {"prompt": "configured task prompt"}
    assert public_view["limits"] == {"budget_limits": {"turns": 1}}
    assert "seed" not in public_view
    assert "privileged_payload" not in public_view
    assert "answer" not in str(to_plain_data(public_view))
    assert "42" not in str(to_plain_data(public_view))


def test_content_hash_is_stable_for_equivalent_instances() -> None:
    first = TaskInstance(
        task_id="task-1",
        kind="example.v1",
        domain="tests",
        seed=11,
        public_payload={"values": [3, 2, 1]},
        privileged_payload={"answer": {"x": 1}},
        budget_limits={"turns": 2, "actions": 2},
        metadata={"difficulty": "tiny"},
    )
    second = TaskInstance(
        task_id="task-1",
        kind="example.v1",
        domain="tests",
        seed=11,
        public_payload={"values": (3, 2, 1)},
        privileged_payload={"answer": {"x": 1}},
        budget_limits={"turns": 2, "actions": 2},
        metadata={"difficulty": "tiny"},
    )

    assert first.content_hash() == second.content_hash()
    assert stable_hash(first.public_view()) == stable_hash(second.public_view())
    assert first.public_limits() == {"budget_limits": {"turns": 2, "actions": 2}}


def test_public_limits_include_named_budget_limits() -> None:
    instance = TaskInstance(
        task_id="budget-test",
        kind="tests.instance.v1",
        domain="tests",
        seed=3,
        public_payload={},
        privileged_payload={},
        budget_limits={"turns": 4, "final_answers": 1},
    )

    assert instance.public_limits() == {
        "budget_limits": {"turns": 4, "final_answers": 1},
    }


def test_task_instance_rejects_invalid_budget_names() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        TaskInstance(
            task_id="budget-test",
            kind="tests.instance.v1",
            domain="tests",
            seed=3,
            public_payload={},
            privileged_payload={},
            budget_limits=cast(Mapping[str, int], {1: 1}),
        )
