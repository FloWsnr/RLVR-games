"""Reward policy for the circuit diagnosis task."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

from rlvr_physics.core.rewards import RewardResult

_REWARD_PARAMETER_NAMES = frozenset(
    {
        "restored_reward",
        "failed_repair_reward",
        "accepted_probe_reward",
        "accepted_repair_reward",
        "invalid_submission_reward",
        "budget_exceeded_reward",
        "session_already_done_reward",
    }
)


def _validate_finite_reward(value: float, name: str) -> None:
    """Validate one finite reward value."""

    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class CircuitRewardConfig:
    """Reward configuration for circuit diagnosis events.

    Parameters
    ----------
    restored_reward:
        Reward and score assigned when final verification passes.
    failed_repair_reward:
        Reward and score assigned when final verification fails.
    accepted_probe_reward:
        Intermediate reward assigned to accepted probe actions.
    accepted_repair_reward:
        Intermediate reward assigned to accepted repair actions.
    invalid_submission_reward:
        Reward assigned to rejected submissions.
    budget_exceeded_reward:
        Reward assigned when a public budget is exceeded.
    session_already_done_reward:
        Reward assigned to submissions after session completion.
    """

    restored_reward: float
    failed_repair_reward: float
    accepted_probe_reward: float
    accepted_repair_reward: float
    invalid_submission_reward: float
    budget_exceeded_reward: float
    session_already_done_reward: float

    def __post_init__(self) -> None:
        """Validate reward configuration values."""

        _validate_finite_reward(self.restored_reward, "restored_reward")
        _validate_finite_reward(self.failed_repair_reward, "failed_repair_reward")
        _validate_finite_reward(self.accepted_probe_reward, "accepted_probe_reward")
        _validate_finite_reward(self.accepted_repair_reward, "accepted_repair_reward")
        _validate_finite_reward(
            self.invalid_submission_reward, "invalid_submission_reward"
        )
        _validate_finite_reward(self.budget_exceeded_reward, "budget_exceeded_reward")
        _validate_finite_reward(
            self.session_already_done_reward, "session_already_done_reward"
        )
        if self.restored_reward < self.failed_repair_reward:
            raise ValueError(
                "restored_reward must be greater than or equal to failed_repair_reward"
            )


DEFAULT_REWARD_CONFIG = CircuitRewardConfig(
    restored_reward=1.0,
    failed_repair_reward=0.0,
    accepted_probe_reward=0.0,
    accepted_repair_reward=0.0,
    invalid_submission_reward=0.0,
    budget_exceeded_reward=0.0,
    session_already_done_reward=0.0,
)


def reward_final_verification(
    target_restored: bool,
    diagnosis_correct: bool,
    config: CircuitRewardConfig,
) -> RewardResult:
    """Reward a final repair verification event.

    Parameters
    ----------
    target_restored:
        Whether the repaired circuit passes target behavior checks.
    diagnosis_correct:
        Whether submitted diagnosis strings matched privileged fault labels.
    config:
        Reward policy configuration.

    Returns
    -------
    RewardResult
        Trainer-facing scalar reward and domain score.
    """

    score = config.restored_reward if target_restored else config.failed_repair_reward
    return RewardResult(
        reward=score,
        score=score,
        public_info={
            "reward_event": "final_verification",
            "target_restored": target_restored,
            "diagnosis_correct": diagnosis_correct,
        },
    )


def reward_accepted_probe(
    action_name: str, config: CircuitRewardConfig
) -> RewardResult:
    """Reward an accepted probing action."""

    return RewardResult(
        reward=config.accepted_probe_reward,
        score=None,
        public_info={
            "reward_event": "accepted_probe",
            "accepted_action": action_name,
        },
    )


def reward_accepted_repair(
    component_id: str, config: CircuitRewardConfig
) -> RewardResult:
    """Reward an accepted repair action."""

    return RewardResult(
        reward=config.accepted_repair_reward,
        score=None,
        public_info={
            "reward_event": "accepted_repair",
            "component": component_id,
        },
    )


def reward_invalid_submission(
    policy_category: str, reason_category: str, config: CircuitRewardConfig
) -> RewardResult:
    """Reward a rejected submission."""

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
    policy_category: str, reason_category: str, config: CircuitRewardConfig
) -> RewardResult:
    """Reward a budget-exceeded submission."""

    return RewardResult(
        reward=config.budget_exceeded_reward,
        score=None,
        public_info={
            "reward_event": "budget_exceeded",
            "invalid_submission_policy": policy_category,
            "invalid_submission_category": reason_category,
        },
    )


def reward_session_already_done(config: CircuitRewardConfig) -> RewardResult:
    """Reward a submission received after the session ended."""

    return RewardResult(
        reward=config.session_already_done_reward,
        score=None,
        public_info={"reward_event": "session_already_done"},
    )


def reward_config_parameters(config: CircuitRewardConfig) -> dict[str, object]:
    """Return a public reward configuration payload."""

    return {
        "restored_reward": config.restored_reward,
        "failed_repair_reward": config.failed_repair_reward,
        "accepted_probe_reward": config.accepted_probe_reward,
        "accepted_repair_reward": config.accepted_repair_reward,
        "invalid_submission_reward": config.invalid_submission_reward,
        "budget_exceeded_reward": config.budget_exceeded_reward,
        "session_already_done_reward": config.session_already_done_reward,
    }


def reward_config_from_mapping(
    parameters: Mapping[str, object],
) -> CircuitRewardConfig:
    """Build a circuit reward config from public parameters."""

    _reject_unknown_reward_parameters(parameters)
    return CircuitRewardConfig(
        restored_reward=_required_float_parameter(parameters, "restored_reward"),
        failed_repair_reward=_required_float_parameter(
            parameters, "failed_repair_reward"
        ),
        accepted_probe_reward=_required_float_parameter(
            parameters, "accepted_probe_reward"
        ),
        accepted_repair_reward=_required_float_parameter(
            parameters, "accepted_repair_reward"
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
    """Reject reward parameters that are not part of this reward config."""

    unknown_keys = sorted(
        (key for key in parameters if key not in _REWARD_PARAMETER_NAMES),
        key=str,
    )
    if len(unknown_keys) > 0:
        joined_keys = ", ".join(str(key) for key in unknown_keys)
        raise ValueError(f"unknown circuit reward parameter(s): {joined_keys}")


def _required_float_parameter(parameters: Mapping[str, object], name: str) -> float:
    """Read a required numeric reward parameter as a float."""

    try:
        value = parameters[name]
    except KeyError as error:
        raise ValueError(f"missing circuit reward parameter: {name}") from error
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    if isinstance(value, int | float):
        numeric_value = float(value)
        if isfinite(numeric_value):
            return numeric_value
        raise ValueError(f"{name} must be finite")
    raise ValueError(f"{name} must be numeric")
