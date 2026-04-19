"""Task-native session contracts for executable verifier rollouts."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar

from rlvr_games.core.action_context import ActionContext
from rlvr_games.core.exceptions import EpisodeFinishedError, EnvironmentNotResetError
from rlvr_games.core.messages import ChatMessage
from rlvr_games.core.protocol import Environment
from rlvr_games.core.rollout import PreparedTurn, prepare_turn
from rlvr_games.core.types import Observation

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


@dataclass(slots=True, frozen=True)
class TaskInstance:
    """Immutable identity and metadata for one verifier-owned task instance.

    Attributes
    ----------
    task_instance_id : str
        Stable identifier shared by one or more scalar sessions over the same
        task payload.
    task_kind : str
        Domain-defined task kind, such as ``"connect4"`` or
        ``"arithmetic"``.
    seed : int
        Deterministic seed used to construct or sample the task instance.
    prompt_key : str | None
        Optional stable prompt/data key used for grouping rollouts.
    metadata : dict[str, object]
        Public-safe task-instance metadata.
    """

    task_instance_id: str
    task_kind: str
    seed: int
    prompt_key: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and detach task-instance metadata."""
        if not self.task_instance_id:
            raise ValueError("TaskInstance task_instance_id must be non-empty.")
        if not self.task_kind:
            raise ValueError("TaskInstance task_kind must be non-empty.")
        object.__setattr__(
            self,
            "metadata",
            _snapshot_mapping(self.metadata, context="TaskInstance.metadata"),
        )


@dataclass(slots=True)
class TaskTurn:
    """Trainer-facing package for one model-action opportunity.

    Attributes
    ----------
    observation : Observation
        Current model-facing observation.
    action_context : ActionContext
        Structured public-safe context for the next submission.
    messages : tuple[ChatMessage, ...]
        Chat-formatted messages derived from the observation and action
        context.
    """

    observation: Observation
    action_context: ActionContext
    messages: tuple[ChatMessage, ...]


@dataclass(slots=True)
class TaskResetResult:
    """Result returned when one scalar task session is reset.

    Attributes
    ----------
    task_instance_id : str
        Stable task-instance identifier for the reset session.
    observation : Observation | None
        Initial or terminal observation emitted during reset, if any.
    turn : TaskTurn | None
        Next actionable model turn, or ``None`` when the session finished
        during reset.
    info : dict[str, object]
        Public-safe reset metadata.
    """

    task_instance_id: str
    observation: Observation | None
    turn: TaskTurn | None
    info: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and detach reset metadata."""
        if not self.task_instance_id:
            raise ValueError("TaskResetResult task_instance_id must be non-empty.")
        self.info = _snapshot_mapping(self.info, context="TaskResetResult.info")


@dataclass(slots=True)
class TaskSubmissionResult:
    """Result returned after one assistant output is submitted.

    Attributes
    ----------
    task_instance_id : str
        Stable task-instance identifier for the active session.
    assistant_output : str
        Full assistant output received by the session.
    raw_submission : str
        Extracted submission passed to the executable verifier.
    parsed_output : object | None
        Parsed verifier-facing output, if parsing succeeded and the task
        exposes one.
    valid_submission : bool
        Whether the submission was parseable/verifiable for the task. This is
        not the same as answer correctness.
    reward : float
        Reward assigned by executable verification.
    terminated : bool
        Whether the task reached natural completion.
    truncated : bool
        Whether the task ended because of an external cutoff or failure mode.
    observation : Observation | None
        Observation emitted by the task after the submission, if any.
    turn : TaskTurn | None
        Next actionable model turn, or ``None`` when the session is done.
    info : dict[str, object]
        Public-safe verifier metadata.
    debug_info : dict[str, object]
        Privileged verifier metadata for offline debugging.
    """

    task_instance_id: str
    assistant_output: str
    raw_submission: str
    parsed_output: object | None
    valid_submission: bool
    reward: float
    terminated: bool
    truncated: bool
    observation: Observation | None
    turn: TaskTurn | None
    info: dict[str, object] = field(default_factory=dict)
    debug_info: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate terminal flags and detach metadata."""
        if not self.task_instance_id:
            raise ValueError("TaskSubmissionResult task_instance_id must be non-empty.")
        if self.terminated and self.truncated:
            raise ValueError(
                "TaskSubmissionResult cannot be both terminated and truncated."
            )
        if self.done and self.turn is not None:
            raise ValueError("Finished task submissions must not expose a next turn.")
        self.info = _snapshot_mapping(self.info, context="TaskSubmissionResult.info")
        self.debug_info = _snapshot_mapping(
            self.debug_info,
            context="TaskSubmissionResult.debug_info",
        )

    @property
    def done(self) -> bool:
        """Return whether this submission finished the scalar session."""
        return self.terminated or self.truncated


@dataclass(slots=True)
class TaskSubmissionRecord:
    """Recorded task-session submission.

    Attributes
    ----------
    assistant_output : str
        Full assistant output received by the session.
    raw_submission : str
        Extracted submission passed to the verifier.
    parsed_output : object | None
        Parsed verifier-facing output, if available.
    valid_submission : bool
        Whether the submission was parseable/verifiable for the task.
    reward : float
        Reward assigned by executable verification.
    terminated : bool
        Whether the submission naturally finished the session.
    truncated : bool
        Whether the submission truncated the session.
    observation : Observation | None
        Observation emitted after the submission, if any.
    info : dict[str, object]
        Public-safe verifier metadata.
    debug_info : dict[str, object]
        Privileged verifier metadata.
    details : dict[str, object]
        Optional adapter-specific details that should remain public-safe.
    """

    assistant_output: str
    raw_submission: str
    parsed_output: object | None
    valid_submission: bool
    reward: float
    terminated: bool
    truncated: bool
    observation: Observation | None = None
    info: dict[str, object] = field(default_factory=dict)
    debug_info: dict[str, object] = field(default_factory=dict)
    details: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate terminal flags and detach metadata."""
        if self.terminated and self.truncated:
            raise ValueError(
                "TaskSubmissionRecord cannot be both terminated and truncated."
            )
        self.info = _snapshot_mapping(self.info, context="TaskSubmissionRecord.info")
        self.debug_info = _snapshot_mapping(
            self.debug_info,
            context="TaskSubmissionRecord.debug_info",
        )
        self.details = _snapshot_mapping(
            self.details,
            context="TaskSubmissionRecord.details",
        )

    @property
    def done(self) -> bool:
        """Return whether this recorded submission finished the session."""
        return self.terminated or self.truncated


@dataclass(slots=True)
class TaskTrajectory:
    """Task-level trajectory shared by environment and verifier sessions.

    Attributes
    ----------
    task_instance_id : str
        Stable identifier for the task instance solved by this session.
    initial_turn : TaskTurn | None
        Initial actionable turn, if reset produced one.
    reset_info : dict[str, object]
        Public-safe reset metadata.
    debug_reset_info : dict[str, object]
        Privileged reset metadata.
    submissions : list[TaskSubmissionRecord]
        Ordered submission records.
    """

    task_instance_id: str
    initial_turn: TaskTurn | None
    reset_info: dict[str, object] = field(default_factory=dict)
    debug_reset_info: dict[str, object] = field(default_factory=dict)
    submissions: list[TaskSubmissionRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate trajectory identity and detach reset metadata."""
        if not self.task_instance_id:
            raise ValueError("TaskTrajectory task_instance_id must be non-empty.")
        self.reset_info = _snapshot_mapping(
            self.reset_info,
            context="TaskTrajectory.reset_info",
        )
        self.debug_reset_info = _snapshot_mapping(
            self.debug_reset_info,
            context="TaskTrajectory.debug_reset_info",
        )

    @property
    def total_reward(self) -> float:
        """Return the cumulative reward for the task session."""
        return sum(submission.reward for submission in self.submissions)

    @property
    def done(self) -> bool:
        """Return whether the recorded trajectory has finished."""
        if not self.submissions:
            return self.initial_turn is None
        return self.submissions[-1].done

    def append(
        self, result: TaskSubmissionResult, *, details: dict[str, object] | None = None
    ) -> None:
        """Append a submission result to the trajectory.

        Parameters
        ----------
        result : TaskSubmissionResult
            Result to record.
        details : dict[str, object] | None
            Optional public-safe adapter-specific details.
        """
        if result.task_instance_id != self.task_instance_id:
            raise ValueError(
                "TaskSubmissionResult task_instance_id does not match trajectory."
            )
        self.submissions.append(
            TaskSubmissionRecord(
                assistant_output=result.assistant_output,
                raw_submission=result.raw_submission,
                parsed_output=deepcopy(result.parsed_output),
                valid_submission=result.valid_submission,
                reward=result.reward,
                terminated=result.terminated,
                truncated=result.truncated,
                observation=result.observation,
                info=result.info,
                debug_info=result.debug_info,
                details={} if details is None else details,
            )
        )


class TaskSessionProtocol(Protocol):
    """Shared scalar task-session contract used by trainer-facing code."""

    @property
    def done(self) -> bool:
        """Return whether the current scalar session has finished."""
        ...

    @property
    def task_instance_id(self) -> str:
        """Return the active task-instance id."""
        ...

    @property
    def turn(self) -> TaskTurn | None:
        """Return the current actionable turn, if one exists."""
        ...

    @property
    def trajectory(self) -> TaskTrajectory:
        """Return the task-level trajectory for the active session."""
        ...

    @property
    def episode_return(self) -> float:
        """Return cumulative reward for the active scalar session."""
        ...

    def reset(self, *, seed: int) -> TaskResetResult:
        """Start a fresh scalar task session."""
        ...

    def submit(self, assistant_output: str) -> TaskSubmissionResult:
        """Submit one assistant output to the executable task."""
        ...

    def close(self) -> None:
        """Close resources owned by the scalar session."""
        ...


class EnvironmentTaskSession(Generic[StateT, ActionT]):
    """Task-session adapter around one stateful environment."""

    def __init__(
        self,
        *,
        env: Environment[StateT, ActionT],
        task_kind: str = "environment",
        action_extractor: Callable[[str], str] | None = None,
    ) -> None:
        """Initialize an environment-backed task session.

        Parameters
        ----------
        env : Environment[StateT, ActionT]
            Stateful environment to wrap.
        task_kind : str
            Public task kind used when building task-instance ids.
        action_extractor : Callable[[str], str] | None
            Optional extractor that converts full assistant output into the raw
            environment action.
        """
        if not task_kind:
            raise ValueError("EnvironmentTaskSession task_kind must be non-empty.")
        self._env = env
        self._task_kind = task_kind
        self._action_extractor = (
            action_extractor if action_extractor is not None else _identity_submission
        )
        self._task_instance_id: str | None = None
        self._turn: TaskTurn | None = None
        self._trajectory: TaskTrajectory | None = None
        self._episode_return = 0.0
        self._reset_count = 0
        self._closed = False

    @property
    def env(self) -> Environment[StateT, ActionT]:
        """Return the wrapped environment."""
        return self._env

    @property
    def done(self) -> bool:
        """Return whether the current scalar session has finished."""
        if self._trajectory is None:
            return False
        return self._turn is None

    @property
    def task_instance_id(self) -> str:
        """Return the active task-instance id.

        Raises
        ------
        EnvironmentNotResetError
            If ``reset()`` has not been called yet.
        """
        if self._task_instance_id is None:
            raise EnvironmentNotResetError(
                "Call EnvironmentTaskSession.reset() before accessing task_instance_id."
            )
        return self._task_instance_id

    @property
    def turn(self) -> TaskTurn | None:
        """Return the current actionable turn, if one exists."""
        self._require_reset()
        return self._turn

    @property
    def trajectory(self) -> TaskTrajectory:
        """Return the task-level trajectory for the active session."""
        if self._trajectory is None:
            raise EnvironmentNotResetError(
                "Call EnvironmentTaskSession.reset() before accessing trajectory."
            )
        return self._trajectory

    @property
    def episode_return(self) -> float:
        """Return cumulative reward for the active scalar session."""
        return self._episode_return

    def reset(self, *, seed: int) -> TaskResetResult:
        """Start a fresh environment-backed scalar session."""
        self._reset_count += 1
        observation, reset_info = self._env.reset(seed=seed)
        task_instance = TaskInstance(
            task_instance_id=_environment_task_instance_id(
                task_kind=self._task_kind,
                seed=seed,
                reset_index=self._reset_count - 1,
            ),
            task_kind=self._task_kind,
            seed=seed,
            prompt_key=str(seed),
            metadata={"adapter": "environment"},
        )
        turn = None
        if not self._env.episode_finished:
            turn = _task_turn_from_prepared_turn(
                observation=observation,
                prepared_turn=prepare_turn(env=self._env, observation=observation),
            )

        self._task_instance_id = task_instance.task_instance_id
        self._turn = turn
        self._episode_return = 0.0
        self._trajectory = TaskTrajectory(
            task_instance_id=task_instance.task_instance_id,
            initial_turn=turn,
            reset_info=reset_info,
            debug_reset_info=self._env.trajectory.debug_reset_info,
        )
        return TaskResetResult(
            task_instance_id=task_instance.task_instance_id,
            observation=observation,
            turn=turn,
            info=reset_info,
        )

    def submit(self, assistant_output: str) -> TaskSubmissionResult:
        """Submit one assistant output to the wrapped environment."""
        self._require_reset()
        if self._turn is None:
            raise EpisodeFinishedError(
                "The current task session has finished. Call reset() first."
            )
        raw_submission = self._action_extractor(assistant_output)
        if not isinstance(raw_submission, str):
            raise TypeError(
                "EnvironmentTaskSession action_extractor must return a string."
            )

        step_result = self._env.step(raw_submission)
        observation = step_result.observation
        turn = None
        if not self._env.episode_finished:
            turn = _task_turn_from_prepared_turn(
                observation=observation,
                prepared_turn=prepare_turn(env=self._env, observation=observation),
            )
        trajectory_step = self._env.trajectory.steps[-1]
        result = TaskSubmissionResult(
            task_instance_id=self.task_instance_id,
            assistant_output=assistant_output,
            raw_submission=raw_submission,
            parsed_output=deepcopy(trajectory_step.action),
            valid_submission=step_result.accepted,
            reward=step_result.reward,
            terminated=step_result.terminated,
            truncated=step_result.truncated,
            observation=observation,
            turn=turn,
            info=step_result.info,
            debug_info=trajectory_step.debug_info,
        )

        self._turn = turn
        self._episode_return += step_result.reward
        self.trajectory.append(
            result,
            details={
                "adapter": "environment",
                "accepted": step_result.accepted,
                "transition_count": len(trajectory_step.transitions),
            },
        )
        return result

    def close(self) -> None:
        """Close the wrapped environment."""
        if self._closed:
            return
        self._env.close()
        self._closed = True

    def _require_reset(self) -> None:
        """Raise if the session has not been reset yet."""
        if self._trajectory is None:
            raise EnvironmentNotResetError(
                "Call EnvironmentTaskSession.reset() before using the session."
            )


def _snapshot_mapping(
    value: dict[str, object],
    *,
    context: str,
) -> dict[str, object]:
    """Return a detached copy of a metadata mapping."""
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a dict.")
    return deepcopy(value)


def _identity_submission(assistant_output: str) -> str:
    """Return the assistant output unchanged."""
    return assistant_output


def _environment_task_instance_id(
    *,
    task_kind: str,
    seed: int,
    reset_index: int,
) -> str:
    """Build a stable public id for one environment-backed session."""
    return f"{task_kind}:seed={seed}:episode={reset_index}"


def _task_turn_from_prepared_turn(
    *,
    observation: Observation,
    prepared_turn: PreparedTurn,
) -> TaskTurn:
    """Convert one environment prepared turn into a task turn."""
    return TaskTurn(
        observation=observation,
        action_context=prepared_turn.action_context,
        messages=prepared_turn.messages,
    )


__all__ = [
    "EnvironmentTaskSession",
    "TaskInstance",
    "TaskResetResult",
    "TaskSessionProtocol",
    "TaskSubmissionRecord",
    "TaskSubmissionResult",
    "TaskTrajectory",
    "TaskTurn",
]
