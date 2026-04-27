"""Tests for immutable task instances."""

import pytest

from rlvr_physics.core.instances import (
    TaskInstance,
    TaskLimits,
)
from rlvr_physics.core.payloads import (
    stable_hash,
    to_plain_data,
)


def test_task_instance_freezes_payloads_and_separates_public_view() -> None:
    instance = TaskInstance(
        task_id="task-1",
        kind="example.v1",
        domain="tests",
        seed=11,
        public_payload={"numbers": [1, 2, 3]},
        privileged_payload={"answer": 6},
        limits=TaskLimits(max_turns=1),
        metadata={"split": "unit"},
    )

    assert instance.public_payload["numbers"] == (1, 2, 3)
    with pytest.raises(TypeError):
        instance.public_payload["new"] = "blocked"  # type: ignore[index]

    public_view = instance.public_view()
    assert public_view["payload"] == {"numbers": (1, 2, 3)}
    assert "privileged_payload" not in public_view
    assert "answer" not in str(to_plain_data(public_view))


def test_content_hash_is_stable_for_equivalent_instances() -> None:
    first = TaskInstance(
        task_id="task-1",
        kind="example.v1",
        domain="tests",
        seed=11,
        public_payload={"values": [3, 2, 1]},
        privileged_payload={"answer": {"x": 1}},
        limits=TaskLimits(max_turns=2, action_budget=2),
        metadata={"difficulty": "tiny"},
    )
    second = TaskInstance(
        task_id="task-1",
        kind="example.v1",
        domain="tests",
        seed=11,
        public_payload={"values": (3, 2, 1)},
        privileged_payload={"answer": {"x": 1}},
        limits=TaskLimits(max_turns=2, action_budget=2),
        metadata={"difficulty": "tiny"},
    )

    assert first.content_hash() == second.content_hash()
    assert stable_hash(first.public_view()) == stable_hash(second.public_view())
