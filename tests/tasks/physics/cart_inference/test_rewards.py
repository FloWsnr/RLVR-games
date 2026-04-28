"""Tests for cart inference reward policy."""

from rlvr_physics.tasks.physics.cart_inference import (
    FinalAnswerEvaluation,
    reward_final_answer,
)


def test_reward_final_answer_gives_full_credit_inside_tolerance() -> None:
    evaluation = FinalAnswerEvaluation(
        submitted_position_m=1.02,
        exact_position_m=1.0,
        absolute_error_m=0.02,
        tolerance_abs_m=0.05,
        correct=True,
    )

    reward = reward_final_answer(evaluation)

    assert reward.reward == 1.0
    assert reward.score == 1.0


def test_reward_final_answer_gives_linear_partial_credit_outside_tolerance() -> None:
    evaluation = FinalAnswerEvaluation(
        submitted_position_m=1.3,
        exact_position_m=1.0,
        absolute_error_m=0.3,
        tolerance_abs_m=0.05,
        correct=False,
    )

    reward = reward_final_answer(evaluation)

    assert reward.reward == 0.5
    assert reward.score == 0.5
