"""Scalar task session protocols and result types."""

from dataclasses import dataclass, field
from itertools import count
from typing import Mapping, Protocol

from rlvr_physics.core.instances import freeze_mapping, stable_hash
from rlvr_physics.core.rendering import RenderedObservation
from rlvr_physics.core.trajectory import TaskTrajectory, TrajectoryEvent

_SESSION_COUNTER = count()


def new_session_id(task_id: str, seed: int) -> str:
    """Return a process-unique session id for one rollout."""

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

    Parameters
    ----------
    kind:
        Submission category, such as ``final_text`` or ``action``.
    raw:
        Raw model text or action string.
    parsed:
        Adapter- or task-interpreted payload.
    """

    kind: str
    raw: str
    parsed: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze parsed payload after construction."""

        object.__setattr__(self, "parsed", freeze_mapping(self.parsed))

    @classmethod
    def final_text(cls, text: str) -> "TaskSubmission":
        """Create a raw final-text submission."""

        return cls(kind="final_text", raw=text, parsed={})

    @classmethod
    def action(cls, action: str) -> "TaskSubmission":
        """Create an action submission."""

        return cls(kind="action", raw=action, parsed={"action": action})


@dataclass(frozen=True)
class TaskTurn:
    """One model-facing task turn.

    Parameters
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
    """Result returned when a scalar task session resets."""

    session_id: str
    turn: TaskTurn
    trajectory: TaskTrajectory


@dataclass(frozen=True)
class TaskStepResult:
    """Result returned after a model submission.

    Parameters
    ----------
    accepted:
        Whether the submission was well-formed enough to evaluate or apply.
    reward:
        Trainer-facing scalar reward.
    score:
        Domain score for filtering or reporting.
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
    reward: float
    score: float | None
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
    def done(self) -> bool:
        """Return whether this result ended the rollout."""

        return self.terminal or self.truncated


class TaskSession(Protocol):
    """Minimal scalar task session protocol."""

    def reset(self, seed: int) -> TaskResetResult:
        """Start a fresh rollout and return the first turn."""
        ...

    @property
    def turn(self) -> TaskTurn | None:
        """Return the current turn, or ``None`` after termination."""
        ...

    def submit(self, submission: TaskSubmission) -> TaskStepResult:
        """Apply or verify a model submission."""
        ...

    @property
    def trajectory(self) -> TaskTrajectory:
        """Return the verified session trajectory."""
        ...
