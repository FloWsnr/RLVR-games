"""Scalar task session protocols and result types."""

from dataclasses import dataclass, field
from itertools import count
from typing import Mapping, Protocol

from rlvr_physics.core.payloads import freeze_mapping, stable_hash
from rlvr_physics.core.rendering import RenderedObservation
from rlvr_physics.core.rewards import RewardResult
from rlvr_physics.core.trajectory import TaskTrajectory, TrajectoryEvent

_SESSION_COUNTER = count()


def new_session_id(task_id: str, seed: int) -> str:
    """Return a process-local session identifier for one rollout.

    Parameters
    ----------
    task_id:
        Stable identifier for the task instance being rolled out.
    seed:
        Session seed used for the rollout.

    Returns
    -------
    str
        Session identifier derived from the task id, seed, and a process-local
        ordinal. Repeated calls with the same arguments include different
        ordinals within the current process.
    """

    ordinal = next(_SESSION_COUNTER)
    return (
        "session-"
        + stable_hash({"task_id": task_id, "seed": seed, "rollout_ordinal": ordinal})[
            :16
        ]
    )


@dataclass(frozen=True)
class TaskSubmission:
    """A raw model submission plus optional interpreted payload.

    Attributes
    ----------
    kind:
        Submission category, such as ``final_text`` or ``action``.
    raw:
        Raw model text or action string.
    parsed:
        Task- or integration-interpreted payload.
    """

    kind: str
    raw: str
    parsed: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze parsed payload after construction."""

        object.__setattr__(self, "parsed", freeze_mapping(self.parsed))

    @classmethod
    def final_text(cls, text: str) -> "TaskSubmission":
        """Create a raw final-text submission.

        Parameters
        ----------
        text:
            Final answer text emitted by the model.

        Returns
        -------
        TaskSubmission
            Submission with kind ``final_text`` and an empty parsed payload.
        """

        return cls(kind="final_text", raw=text, parsed={})

    @classmethod
    def action(cls, action: str) -> "TaskSubmission":
        """Create an action submission.

        Parameters
        ----------
        action:
            Action text emitted by the model.

        Returns
        -------
        TaskSubmission
            Submission with kind ``action`` and the action mirrored into the
            parsed payload.
        """

        return cls(kind="action", raw=action, parsed={"action": action})


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
    action_schema:
        Public structured action schema, when relevant.
    public_limits:
        Public rollout limits.
    public_info:
        Additional trainer-safe metadata.
    """

    turn_index: int
    observation: RenderedObservation
    submission_modes: tuple[str, ...]
    action_schema: Mapping[str, object]
    public_limits: Mapping[str, object]
    public_info: Mapping[str, object]

    def __post_init__(self) -> None:
        """Freeze mapping payloads after construction."""

        object.__setattr__(self, "action_schema", freeze_mapping(self.action_schema))
        object.__setattr__(self, "public_limits", freeze_mapping(self.public_limits))
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
    trajectory:
        Trajectory object associated with the session.
    """

    session_id: str
    turn: TaskTurn
    trajectory: TaskTrajectory


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
    events:
        Trajectory events emitted by the step.
    """

    accepted: bool
    reward_result: RewardResult
    terminal: bool
    truncated: bool
    observation: TaskTurn | None
    public_info: Mapping[str, object]
    debug_info: Mapping[str, object]
    events: tuple[TrajectoryEvent, ...]

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

    Implementations manage one scalar rollout at a time and expose a verified
    trajectory for trainer integration.
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
            New session identifier, first turn, and trajectory state.
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
            metadata, and events.
        """
        ...

    @property
    def trajectory(self) -> TaskTrajectory:
        """Return the verified session trajectory.

        Returns
        -------
        TaskTrajectory
            Append-only trajectory for the active or completed session.
        """
        ...
