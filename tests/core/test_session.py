"""Tests for shared session result types."""

import pytest

from rlvr_physics.core.rewards import RewardResult
from rlvr_physics.core.session import (
    TaskResetResult,
    TaskStepResult,
    TaskSubmission,
    TaskTurn,
    new_session_id,
)


def test_submission_parsed_payload_is_frozen() -> None:
    submission = TaskSubmission.action("left")

    assert submission.kind == "action"
    assert submission.parsed["action"] == "left"


def test_step_result_done_property_and_turn_payload(example_turn: TaskTurn) -> None:
    result = TaskStepResult(
        accepted=True,
        reward_result=RewardResult(reward=1.0, score=1.0),
        terminal=True,
        truncated=False,
        observation=example_turn,
        public_info={"reason": "correct"},
        debug_info={"answer": 42},
    )

    assert result.done
    assert result.reward == 1.0
    assert result.score == 1.0
    assert result.observation is example_turn


def test_reset_result_payloads_are_frozen(example_turn: TaskTurn) -> None:
    result = TaskResetResult(
        session_id="session-1",
        turn=example_turn,
        public_info={"task_id": "task-1"},
        debug_info={"answer": 42},
    )

    assert result.public_info["task_id"] == "task-1"
    assert result.debug_info["answer"] == 42
    with pytest.raises(TypeError):
        result.public_info["extra"] = "blocked"  # type: ignore[index]


def test_session_ids_do_not_collide_for_same_task_and_seed() -> None:
    first = new_session_id("task-1", 7)
    second = new_session_id("task-1", 7)

    assert first != second
    assert first.startswith("session-")
