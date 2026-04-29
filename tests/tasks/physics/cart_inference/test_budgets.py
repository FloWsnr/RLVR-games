"""Tests for cart inference rollout budget accounting."""

from rlvr_physics.core.submissions import InvalidSubmissionPolicy
from rlvr_physics.tasks.physics.cart_inference.budgets import (
    CartRolloutBudgetState,
)


def test_rollout_budget_state_applies_retryable_invalid_policy() -> None:
    """Retryable invalid submissions consume turns but not task actions."""

    state = CartRolloutBudgetState({"turns": 4, "actions": 3, "final_answers": 1})
    policy = InvalidSubmissionPolicy(
        category="retryable_invalid_submission",
        consumes_budget={"turns": 1},
        terminal=False,
        truncated=False,
    )

    state.record_invalid_submission(policy)
    public_status = state.public_status(
        actions_used=0,
        actions_remaining=3,
        extra_info={"reason": "unparseable_action"},
    )

    assert public_status["budget_usage"] == {
        "turns": 1,
        "actions": 0,
        "final_answers": 0,
    }
    assert public_status["budget_remaining"] == {
        "turns": 3,
        "actions": 3,
        "final_answers": 1,
    }
    assert public_status["submissions_used"] == 1
    assert public_status["invalid_submissions"] == 1
    assert public_status["reason"] == "unparseable_action"


def test_rollout_budget_state_counts_invalid_final_answer_once() -> None:
    """Invalid final answers do not double-consume already-counted budgets."""

    state = CartRolloutBudgetState({"turns": 4, "actions": 3, "final_answers": 1})

    state.record_final_answer_submission()
    state.record_invalid_after_counted_submission()
    public_status = state.public_status(
        actions_used=0,
        actions_remaining=3,
        extra_info={},
    )

    assert public_status["budget_usage"] == {
        "turns": 1,
        "actions": 0,
        "final_answers": 1,
    }
    assert public_status["budget_remaining"] == {
        "turns": 3,
        "actions": 3,
        "final_answers": 0,
    }
    assert public_status["submissions_used"] == 1
    assert public_status["invalid_submissions"] == 1
    assert public_status["final_answers_used"] == 1


def test_rollout_budget_state_rejects_action_consuming_invalid_policy() -> None:
    """The backbone owns accepted action budget accounting."""

    state = CartRolloutBudgetState({"turns": 4, "actions": 3, "final_answers": 1})
    policy = InvalidSubmissionPolicy(
        category="invalid_action",
        consumes_budget={"turns": 1, "actions": 1},
        terminal=False,
        truncated=False,
    )

    try:
        state.record_invalid_submission(policy)
    except ValueError as error:
        assert "accepted action budget" in str(error)
    else:
        raise AssertionError("action-consuming invalid policy was accepted")
