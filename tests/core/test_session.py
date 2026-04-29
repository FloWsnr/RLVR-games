"""Tests for shared session result types."""

from typing import Mapping, cast

import pytest

from rlvr_physics.core.rendering import text_observation
from rlvr_physics.core.rewards import RewardResult
from rlvr_physics.core.session import (
    TaskResetResult,
    TaskStepResult,
    TaskTurn,
    new_session_id,
)
from rlvr_physics.core.submissions import (
    InvalidSubmissionPolicy,
    TaskSubmission,
    invalid_submission_policies_payload,
    validate_invalid_submission_policies,
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


def test_invalid_submission_policy_payload_and_budget_validation() -> None:
    policy = InvalidSubmissionPolicy(
        category="invalid_action",
        consumes_budget={"turns": 1},
        reward=0.0,
        terminal=False,
        truncated=False,
    )
    policies = {"invalid_action": policy}

    validate_invalid_submission_policies(policies, {"budget_limits": {"turns": 3}})
    payload = invalid_submission_policies_payload(policies)

    assert payload["invalid_action"] == {
        "category": "invalid_action",
        "consumes_budget": {"turns": 1},
        "reward": 0.0,
        "terminal": False,
        "truncated": False,
    }


def test_invalid_submission_policy_rejects_unknown_budget() -> None:
    policy = InvalidSubmissionPolicy(
        category="invalid_action",
        consumes_budget={"unknown_budget": 1},
        reward=0.0,
        terminal=False,
        truncated=False,
    )

    with pytest.raises(ValueError, match="unknown budget"):
        validate_invalid_submission_policies(
            {"invalid_action": policy}, {"budget_limits": {"turns": 3}}
        )


def test_invalid_submission_policy_rejects_invalid_budget_names() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        InvalidSubmissionPolicy(
            category="invalid_action",
            consumes_budget=cast(Mapping[str, int], {1: 1}),
            reward=0.0,
            terminal=False,
            truncated=False,
        )


def test_policy_validation_rejects_invalid_public_budget_names() -> None:
    policy = InvalidSubmissionPolicy(
        category="invalid_action",
        consumes_budget={"turns": 1},
        reward=0.0,
        terminal=False,
        truncated=False,
    )

    with pytest.raises(ValueError, match="public budget name"):
        validate_invalid_submission_policies(
            {"invalid_action": policy},
            {"budget_limits": cast(Mapping[str, int], {1: 3})},
        )


def test_task_turn_rejects_action_schema_unknown_budget() -> None:
    with pytest.raises(ValueError, match="unknown budget"):
        TaskTurn(
            turn_index=0,
            observation=text_observation("text", "prompt"),
            submission_modes=("action",),
            submission_format={},
            action_schema={
                "actions": {
                    "measure": {
                        "consumes_budget": {"unknown_budget": 1},
                        "arguments": {},
                    }
                }
            },
            invalid_submission_policies={},
            public_limits={"budget_limits": {"turns": 1}},
            public_info={},
        )


def test_task_turn_rejects_invalid_policy_budget_reference() -> None:
    policy = InvalidSubmissionPolicy(
        category="invalid_action",
        consumes_budget={"unknown_budget": 1},
        reward=0.0,
        terminal=False,
        truncated=False,
    )

    with pytest.raises(ValueError, match="unknown budget"):
        TaskTurn(
            turn_index=0,
            observation=text_observation("text", "prompt"),
            submission_modes=("action",),
            submission_format={},
            action_schema={},
            invalid_submission_policies={"invalid_action": policy},
            public_limits={"budget_limits": {"turns": 1}},
            public_info={},
        )
