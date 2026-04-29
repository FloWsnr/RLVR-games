"""Tests for cart inference reward policy."""

from rlvr_physics.tasks.physics.cart_inference.backbone import (
    FinalAnswerEvaluation,
    MeasurementResult,
)
from rlvr_physics.tasks.physics.cart_inference.rewards import (
    DEFAULT_REWARD_CONFIG,
    CartRewardConfig,
    reward_accepted_measurement,
    reward_budget_exceeded,
    reward_final_answer,
    reward_invalid_submission,
)


def test_reward_final_answer_gives_full_credit_inside_tolerance() -> None:
    evaluation = FinalAnswerEvaluation(
        submitted_position_m=1.02,
        exact_position_m=1.0,
        absolute_error_m=0.02,
        tolerance_abs_m=0.05,
        correct=True,
    )

    reward = reward_final_answer(evaluation, DEFAULT_REWARD_CONFIG)

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

    reward = reward_final_answer(evaluation, DEFAULT_REWARD_CONFIG)

    assert reward.reward == 0.5
    assert reward.score == 0.5


def test_reward_accepted_measurement_uses_intermediate_reward_config() -> None:
    config = CartRewardConfig(
        correct_final_answer_reward=1.0,
        incorrect_final_answer_reward=0.0,
        partial_credit_window_tolerances=10.0,
        accepted_measurement_reward=0.125,
        invalid_submission_reward=-0.25,
        budget_exceeded_reward=-0.5,
        session_already_done_reward=0.0,
    )
    measurement = MeasurementResult(
        measurement_index=2,
        time_s=5.0,
        true_position_m=1.0,
        measured_position_m=1.01,
        noise_m=0.01,
    )

    reward = reward_accepted_measurement(measurement, config)

    assert reward.reward == 0.125
    assert reward.score is None
    assert reward.public_info["reward_event"] == "accepted_action"
    assert reward.public_info["measurement_index"] == 2


def test_reward_invalid_submission_uses_rejection_reward_config() -> None:
    config = CartRewardConfig(
        correct_final_answer_reward=1.0,
        incorrect_final_answer_reward=0.0,
        partial_credit_window_tolerances=10.0,
        accepted_measurement_reward=0.125,
        invalid_submission_reward=-0.25,
        budget_exceeded_reward=-0.5,
        session_already_done_reward=0.0,
    )

    reward = reward_invalid_submission(
        policy_category="retryable_invalid_submission",
        reason_category="unparseable_action",
        config=config,
    )

    assert reward.reward == -0.25
    assert reward.public_info["invalid_submission_policy"] == (
        "retryable_invalid_submission"
    )


def test_reward_budget_exceeded_uses_budget_reward_config() -> None:
    config = CartRewardConfig(
        correct_final_answer_reward=1.0,
        incorrect_final_answer_reward=0.0,
        partial_credit_window_tolerances=10.0,
        accepted_measurement_reward=0.125,
        invalid_submission_reward=-0.25,
        budget_exceeded_reward=-0.5,
        session_already_done_reward=0.0,
    )

    reward = reward_budget_exceeded(
        policy_category="budget_exceeded",
        reason_category="budget_exceeded",
        config=config,
    )

    assert reward.reward == -0.5
    assert reward.public_info["reward_event"] == "budget_exceeded"
