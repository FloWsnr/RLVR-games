"""Scalar task session protocols and result types."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Protocol
from uuid import uuid4

from rlvr_physics.core.payloads import freeze_mapping
from rlvr_physics.core.rendering import RenderedObservation
from rlvr_physics.core.rewards import RewardResult
from rlvr_physics.core.submissions import (
    InvalidSubmissionPolicy,
    TaskSubmission,
    validate_action_schema_budget_references,
    validate_invalid_submission_policies,
)


def new_session_id(task_id: str, seed: int) -> str:
    """Return an opaque process-local session identifier for one rollout.

    Parameters
    ----------
    task_id:
        Stable identifier for the task instance being rolled out. This value is
        accepted for call-site convenience but is not encoded into the returned
        public identifier.
    seed:
        Session seed used for the rollout. This value is accepted for call-site
        convenience but is not encoded into the returned public identifier.

    Returns
    -------
    str
        Opaque process-local session identifier. Repeated calls with the same
        arguments return different identifiers.
    """

    _ = task_id, seed
    return f"session-{uuid4().hex[:16]}"


@dataclass(frozen=True)
class TaskTurn:
    """One model-facing task turn.

    Attributes
    ----------
    turn_index:
        Zero-based turn number.
    observation:
        Renderer output for this turn.
    submission_modes:
        Accepted submission modes for this turn.
    submission_format:
        Public description of how submissions should be encoded.
    action_schema:
        Public structured action schema, when relevant.
    invalid_submission_policies:
        Public policies for rejected model submissions.
    public_limits:
        Public rollout limits.
    public_info:
        Additional trainer-safe metadata.
    """

    turn_index: int
    observation: RenderedObservation
    submission_modes: tuple[str, ...]
    submission_format: Mapping[str, object]
    action_schema: Mapping[str, object]
    invalid_submission_policies: Mapping[str, InvalidSubmissionPolicy]
    public_limits: Mapping[str, object]
    public_info: Mapping[str, object]

    def __post_init__(self) -> None:
        """Freeze mapping payloads after construction."""

        validate_action_schema_budget_references(self.action_schema, self.public_limits)
        validate_invalid_submission_policies(
            self.invalid_submission_policies, self.public_limits
        )
        object.__setattr__(
            self, "submission_format", freeze_mapping(self.submission_format)
        )
        object.__setattr__(self, "action_schema", freeze_mapping(self.action_schema))
        object.__setattr__(self, "public_limits", freeze_mapping(self.public_limits))
        object.__setattr__(
            self,
            "invalid_submission_policies",
            MappingProxyType(dict(self.invalid_submission_policies)),
        )
        object.__setattr__(self, "public_info", freeze_mapping(self.public_info))


@dataclass(frozen=True)
class TaskResetResult:
    """Result returned when a scalar task session resets.

    Attributes
    ----------
    session_id:
        Identifier for the newly started session.
    turn:
        First model-facing turn of the rollout.
    public_info:
        Trainer-safe reset metadata.
    debug_info:
        Privileged local reset metadata.
    """

    session_id: str
    turn: TaskTurn
    public_info: Mapping[str, object]
    debug_info: Mapping[str, object]

    def __post_init__(self) -> None:
        """Freeze mapping payloads after construction."""

        object.__setattr__(self, "public_info", freeze_mapping(self.public_info))
        object.__setattr__(self, "debug_info", freeze_mapping(self.debug_info))


@dataclass(frozen=True)
class TaskStepResult:
    """Result returned after a model submission.

    Attributes
    ----------
    accepted:
        Whether the submission was well-formed enough to evaluate or apply.
    reward_result:
        Structured trainer-facing reward result.
    terminal:
        Whether the task naturally ended.
    truncated:
        Whether limits ended the task.
    observation:
        Next turn when the task continues.
    public_info:
        Trainer-safe result metadata.
    debug_info:
        Privileged local result metadata.
    """

    accepted: bool
    reward_result: RewardResult
    terminal: bool
    truncated: bool
    observation: TaskTurn | None
    public_info: Mapping[str, object]
    debug_info: Mapping[str, object]

    def __post_init__(self) -> None:
        """Freeze mapping payloads after construction."""

        object.__setattr__(self, "public_info", freeze_mapping(self.public_info))
        object.__setattr__(self, "debug_info", freeze_mapping(self.debug_info))

    @property
    def reward(self) -> float:
        """Return the trainer-facing scalar reward.

        Returns
        -------
        float
            Scalar reward from ``reward_result``.
        """

        return self.reward_result.reward

    @property
    def score(self) -> float | None:
        """Return the optional domain score.

        Returns
        -------
        float or None
            Domain score from ``reward_result``.
        """

        return self.reward_result.score

    @property
    def done(self) -> bool:
        """Return whether this result ended the rollout.

        Returns
        -------
        bool
            ``True`` when the result is terminal or truncated.
        """

        return self.terminal or self.truncated


class TaskSession(Protocol):
    """Minimal scalar task session protocol.

    Implementations manage one scalar rollout at a time. Trainer integrations
    own batching and any rollout recording they need.
    """

    def reset(self, seed: int) -> TaskResetResult:
        """Start a fresh rollout and return the first turn.

        Parameters
        ----------
        seed:
            Deterministic seed for the rollout.

        Returns
        -------
        TaskResetResult
            New session identifier, first turn, and reset metadata.
        """
        ...

    @property
    def turn(self) -> TaskTurn | None:
        """Return the current turn.

        Returns
        -------
        TaskTurn or None
            Current model-facing turn, or ``None`` after termination.
        """
        ...

    def submit(self, submission: TaskSubmission) -> TaskStepResult:
        """Apply or verify a model submission.

        Parameters
        ----------
        submission:
            Raw and optionally parsed model output to evaluate or apply.

        Returns
        -------
        TaskStepResult
            Step outcome, structured reward result, optional next turn,
            and metadata.
        """
        ...
