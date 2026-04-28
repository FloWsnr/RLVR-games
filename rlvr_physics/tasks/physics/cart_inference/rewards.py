"""Reward policy for the cart inference task."""

from rlvr_physics.core.rewards import RewardResult
from rlvr_physics.tasks.physics.cart_inference.backbone import (
    FinalAnswerEvaluation,
)


def reward_final_answer(evaluation: FinalAnswerEvaluation) -> RewardResult:
    """Reward a final-answer evaluation.

    Parameters
    ----------
    evaluation:
        Reward-free verifier evaluation from the cart inference backbone.

    Returns
    -------
    RewardResult
        Trainer-facing scalar reward and domain score.
    """

    if evaluation.correct:
        score = 1.0
    else:
        partial_window_m = evaluation.tolerance_abs_m * 10.0
        score = max(
            0.0,
            1.0
            - (
                (evaluation.absolute_error_m - evaluation.tolerance_abs_m)
                / partial_window_m
            ),
        )
    return RewardResult(
        reward=score,
        score=score,
        public_info={"correct": evaluation.correct},
    )
