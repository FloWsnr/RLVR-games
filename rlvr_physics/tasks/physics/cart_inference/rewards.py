"""Reward policy for the cart inference task."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

from rlvr_physics.core.rewards import RewardResult
from rlvr_physics.tasks.physics.cart_inference.backbone import (
    FinalAnswerEvaluation,
    MeasurementResult,
)
from rlvr_physics.tasks.physics.cart_inference.backbone import MEASURE_POSITION_ACTION

_REWARD_PARAMETER_NAMES = frozenset(
    {
        "correct_final_answer_reward",
        "incorrect_final_answer_reward",
        "partial_credit_window_tolerances",
        "accepted_measurement_reward",
        "invalid_submission_reward",
        "budget_exceeded_reward",
        "session_already_done_reward",
    }
)


@dataclass(frozen=True)
class CartRewardConfig:
    """Reward configuration for cart inference events.

    Parameters
    ----------
    correct_final_answer_reward:
        Reward and score assigned to a final answer inside tolerance.
    incorrect_final_answer_reward:
        Minimum reward and score assigned to a final answer outside the partial
        credit window.
    partial_credit_window_tolerances:
        Number of answer tolerances over which outside-tolerance final answers
        receive linear partial credit.
    accepted_measurement_reward:
        Intermediate reward assigned to an accepted measurement action.
    invalid_submission_reward:
        Reward assigned to rejected submissions governed by invalid-submission
        policies.
    budget_exceeded_reward:
        Reward assigned when an action attempt exceeds a task budget.
    session_already_done_reward:
        Reward assigned to submissions made after the session has ended.

    Attributes
    ----------
    correct_final_answer_reward:
        Reward and score assigned to a final answer inside tolerance.
    incorrect_final_answer_reward:
        Minimum reward and score assigned to a final answer outside the partial
        credit window.
    partial_credit_window_tolerances:
        Number of answer tolerances over which outside-tolerance final answers
        receive linear partial credit.
    accepted_measurement_reward:
        Intermediate reward assigned to an accepted measurement action.
    invalid_submission_reward:
        Reward assigned to rejected submissions governed by invalid-submission
        policies.
    budget_exceeded_reward:
        Reward assigned when an action attempt exceeds a task budget.
    session_already_done_reward:
        Reward assigned to submissions made after the session has ended.
    """

    correct_final_answer_reward: float
    incorrect_final_answer_reward: float
    partial_credit_window_tolerances: float
    accepted_measurement_reward: float
    invalid_submission_reward: float
    budget_exceeded_reward: float
    session_already_done_reward: float

    def __post_init__(self) -> None:
        """Validate reward configuration values."""

        _validate_finite_reward(
            self.correct_final_answer_reward, "correct_final_answer_reward"
        )
        _validate_finite_reward(
            self.incorrect_final_answer_reward, "incorrect_final_answer_reward"
        )
        _validate_finite_reward(
            self.accepted_measurement_reward, "accepted_measurement_reward"
        )
        _validate_finite_reward(
            self.invalid_submission_reward, "invalid_submission_reward"
        )
        _validate_finite_reward(self.budget_exceeded_reward, "budget_exceeded_reward")
        _validate_finite_reward(
            self.session_already_done_reward, "session_already_done_reward"
        )
        _validate_finite_reward(
            self.partial_credit_window_tolerances,
            "partial_credit_window_tolerances",
        )
        if self.correct_final_answer_reward < self.incorrect_final_answer_reward:
            raise ValueError(
                "correct_final_answer_reward must be greater than or equal to "
                "incorrect_final_answer_reward"
            )
        if self.partial_credit_window_tolerances <= 0.0:
            raise ValueError("partial_credit_window_tolerances must be positive")


def _validate_finite_reward(value: float, name: str) -> None:
    """Validate one finite reward configuration value."""

    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


DEFAULT_REWARD_CONFIG = CartRewardConfig(
    correct_final_answer_reward=1.0,
    incorrect_final_answer_reward=0.0,
    partial_credit_window_tolerances=10.0,
    accepted_measurement_reward=0.0,
    invalid_submission_reward=0.0,
    budget_exceeded_reward=0.0,
    session_already_done_reward=0.0,
)


def reward_final_answer(
    evaluation: FinalAnswerEvaluation, config: CartRewardConfig
) -> RewardResult:
    """Reward a final-answer evaluation.

    Parameters
    ----------
    evaluation:
        Reward-free verifier evaluation from the cart inference backbone.
    config:
        Reward policy configuration.

    Returns
    -------
    RewardResult
        Trainer-facing scalar reward and domain score.
    """

    if evaluation.correct:
        score = config.correct_final_answer_reward
    else:
        partial_window_m = (
            evaluation.tolerance_abs_m * config.partial_credit_window_tolerances
        )
        reward_range = (
            config.correct_final_answer_reward - config.incorrect_final_answer_reward
        )
        score = max(
            config.incorrect_final_answer_reward,
            config.correct_final_answer_reward
            - reward_range
            * (
                (evaluation.absolute_error_m - evaluation.tolerance_abs_m)
                / partial_window_m
            ),
        )
    return RewardResult(
        reward=score,
        score=score,
        public_info={
            "reward_event": "final_answer",
            "correct": evaluation.correct,
        },
    )


def reward_accepted_measurement(
    measurement: MeasurementResult, config: CartRewardConfig
) -> RewardResult:
    """Reward an accepted measurement action.

    Parameters
    ----------
    measurement:
        Authoritative measurement result produced by the backbone.
    config:
        Reward policy configuration.

    Returns
    -------
    RewardResult
        Trainer-facing scalar reward and reward metadata.
    """

    return RewardResult(
        reward=config.accepted_measurement_reward,
        score=None,
        public_info={
            "reward_event": "accepted_action",
            "accepted_action": MEASURE_POSITION_ACTION,
            "measurement_index": measurement.measurement_index,
        },
    )


def reward_invalid_submission(
    policy_category: str, reason_category: str, config: CartRewardConfig
) -> RewardResult:
    """Reward a rejected submission.

    Parameters
    ----------
    policy_category:
        Public invalid-submission policy category that controlled the result.
    reason_category:
        Public reason category for this particular rejection.
    config:
        Reward policy configuration.

    Returns
    -------
    RewardResult
        Trainer-facing scalar reward and reward metadata.
    """

    return RewardResult(
        reward=config.invalid_submission_reward,
        score=None,
        public_info={
            "reward_event": "invalid_submission",
            "invalid_submission_policy": policy_category,
            "invalid_submission_category": reason_category,
        },
    )


def reward_budget_exceeded(
    policy_category: str, reason_category: str, config: CartRewardConfig
) -> RewardResult:
    """Reward a submission rejected because a public budget was exceeded.

    Parameters
    ----------
    policy_category:
        Public invalid-submission policy category that controlled the result.
    reason_category:
        Public reason category for this particular rejection.
    config:
        Reward policy configuration.

    Returns
    -------
    RewardResult
        Trainer-facing scalar reward and reward metadata.
    """

    return RewardResult(
        reward=config.budget_exceeded_reward,
        score=None,
        public_info={
            "reward_event": "budget_exceeded",
            "invalid_submission_policy": policy_category,
            "invalid_submission_category": reason_category,
        },
    )


def reward_session_already_done(config: CartRewardConfig) -> RewardResult:
    """Reward a submission received after the session ended.

    Parameters
    ----------
    config:
        Reward policy configuration.

    Returns
    -------
    RewardResult
        Trainer-facing scalar reward and reward metadata.
    """

    return RewardResult(
        reward=config.session_already_done_reward,
        score=None,
        public_info={"reward_event": "session_already_done"},
    )


def reward_config_parameters(config: CartRewardConfig) -> dict[str, object]:
    """Return a public reward configuration payload.

    Parameters
    ----------
    config:
        Reward policy configuration.

    Returns
    -------
    dict[str, object]
        Plain reward configuration suitable for specs and metadata.
    """

    return {
        "correct_final_answer_reward": config.correct_final_answer_reward,
        "incorrect_final_answer_reward": config.incorrect_final_answer_reward,
        "partial_credit_window_tolerances": (config.partial_credit_window_tolerances),
        "accepted_measurement_reward": config.accepted_measurement_reward,
        "invalid_submission_reward": config.invalid_submission_reward,
        "budget_exceeded_reward": config.budget_exceeded_reward,
        "session_already_done_reward": config.session_already_done_reward,
    }


def reward_config_from_mapping(
    parameters: Mapping[str, object],
) -> CartRewardConfig:
    """Build a cart reward config from public parameters.

    Parameters
    ----------
    parameters:
        Public reward configuration mapping.

    Returns
    -------
    CartRewardConfig
        Validated reward policy configuration.
    """

    _reject_unknown_reward_parameters(parameters)
    return CartRewardConfig(
        correct_final_answer_reward=_required_float_parameter(
            parameters, "correct_final_answer_reward"
        ),
        incorrect_final_answer_reward=_required_float_parameter(
            parameters, "incorrect_final_answer_reward"
        ),
        partial_credit_window_tolerances=_required_float_parameter(
            parameters, "partial_credit_window_tolerances"
        ),
        accepted_measurement_reward=_required_float_parameter(
            parameters, "accepted_measurement_reward"
        ),
        invalid_submission_reward=_required_float_parameter(
            parameters, "invalid_submission_reward"
        ),
        budget_exceeded_reward=_required_float_parameter(
            parameters, "budget_exceeded_reward"
        ),
        session_already_done_reward=_required_float_parameter(
            parameters, "session_already_done_reward"
        ),
    )


def _reject_unknown_reward_parameters(parameters: Mapping[str, object]) -> None:
    """Reject reward parameters that are not part of the cart reward config."""

    unknown_keys = sorted(
        (key for key in parameters if key not in _REWARD_PARAMETER_NAMES),
        key=str,
    )
    if len(unknown_keys) > 0:
        joined_keys = ", ".join(str(key) for key in unknown_keys)
        raise ValueError(f"unknown cart reward parameter(s): {joined_keys}")


def _required_float_parameter(parameters: Mapping[str, object], name: str) -> float:
    """Read a required numeric reward parameter as a float."""

    try:
        value = parameters[name]
    except KeyError as error:
        raise ValueError(f"missing cart reward parameter: {name}") from error
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    if isinstance(value, int | float):
        numeric_value = float(value)
        if isfinite(numeric_value):
            return numeric_value
        raise ValueError(f"{name} must be finite")
    raise ValueError(f"{name} must be numeric")
