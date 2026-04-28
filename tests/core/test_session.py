"""Tests for shared session result types."""

from rlvr_physics.core.rendering import text_observation
from rlvr_physics.core.rewards import RewardResult
from rlvr_physics.core.session import (
    TaskStepResult,
    TaskSubmission,
    TaskTurn,
    new_session_id,
)
from rlvr_physics.core.trajectory import TaskTrajectory


def test_submission_parsed_payload_is_frozen() -> None:
    submission = TaskSubmission.action("left")

    assert submission.kind == "action"
    assert submission.parsed["action"] == "left"


def test_step_result_done_property_and_turn_payload() -> None:
    turn = TaskTurn(
        turn_index=0,
        observation=text_observation("text", "prompt"),
        submission_modes=("final_text",),
        action_schema={},
        public_limits={"max_turns": 1},
        public_info={"task_id": "task-1"},
    )
    trajectory = TaskTrajectory(task_id="task-1", session_id="session-1")
    event = trajectory.append("reward", 0, {"reward": 1.0}, {"answer": 42})
    result = TaskStepResult(
        accepted=True,
        reward_result=RewardResult(reward=1.0, score=1.0),
        terminal=True,
        truncated=False,
        observation=turn,
        public_info={"reason": "correct"},
        debug_info={"answer": 42},
        events=(event,),
    )

    assert result.done
    assert result.reward == 1.0
    assert result.score == 1.0
    assert result.observation is turn
    assert result.events == (event,)


def test_session_ids_do_not_collide_for_same_task_and_seed() -> None:
    first = new_session_id("task-1", 7)
    second = new_session_id("task-1", 7)

    assert first != second
    assert first.startswith("session-")
