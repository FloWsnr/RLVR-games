"""Single-step verifier sessions for prompt/completion RLVR tasks."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar

from rlvr_games.core.action_context import ActionContext
from rlvr_games.core.exceptions import EpisodeFinishedError, EnvironmentNotResetError
from rlvr_games.core.messages import (
    DefaultObservationMessageAdapter,
    DefaultObservationMessagePolicy,
    ObservationMessageAdapter,
)
from rlvr_games.core.session import (
    TaskInstance,
    TaskResetResult,
    TaskSubmissionResult,
    TaskTrajectory,
    TaskTurn,
)
from rlvr_games.core.types import Observation

TaskPayloadT = TypeVar("TaskPayloadT")


@dataclass(slots=True, frozen=True)
class SingleStepTask(Generic[TaskPayloadT]):
    """Sampled task payload plus stable task identity.

    Attributes
    ----------
    instance : TaskInstance
        Immutable task identity and public metadata.
    payload : TaskPayloadT
        Verifier-owned task payload.
    """

    instance: TaskInstance
    payload: TaskPayloadT


@dataclass(slots=True)
class VerificationResult:
    """Executable verification result for one completion.

    Attributes
    ----------
    parsed_output : object | None
        Parsed verifier-facing output, if available.
    valid_submission : bool
        Whether the submitted completion was parseable/verifiable. This is not
        equivalent to correctness.
    reward : float
        Reward assigned by executable verification.
    terminated : bool
        Whether verification naturally completed the task.
    truncated : bool
        Whether verification ended due to an external cutoff or failure mode.
    info : dict[str, object]
        Public-safe verification metadata.
    debug_info : dict[str, object]
        Privileged verification metadata for offline debugging.
    """

    parsed_output: object | None
    valid_submission: bool
    reward: float
    terminated: bool = True
    truncated: bool = False
    info: dict[str, object] = field(default_factory=dict)
    debug_info: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate verification boundary flags and detach metadata."""
        if self.terminated == self.truncated:
            raise ValueError(
                "VerificationResult requires exactly one of terminated or truncated."
            )
        self.info = _snapshot_mapping(self.info, context="VerificationResult.info")
        self.debug_info = _snapshot_mapping(
            self.debug_info,
            context="VerificationResult.debug_info",
        )


class TaskSource(Protocol[TaskPayloadT]):
    """Protocol for deterministic single-step task sampling."""

    def sample(self, *, seed: int) -> SingleStepTask[TaskPayloadT]:
        """Return one task instance for the supplied seed."""
        ...


class PromptRenderer(Protocol[TaskPayloadT]):
    """Protocol for rendering a task payload into a model observation."""

    def render(self, task: SingleStepTask[TaskPayloadT]) -> Observation:
        """Render a model-facing prompt observation."""
        ...


class SingleStepVerifier(Protocol[TaskPayloadT]):
    """Protocol for verifying one completion against a task payload."""

    def verify(
        self,
        *,
        task: SingleStepTask[TaskPayloadT],
        completion: str,
    ) -> VerificationResult:
        """Verify one completion and return reward metadata."""
        ...


class SingleStepVerifierSession(Generic[TaskPayloadT]):
    """Scalar task session for prompt/completion/verifier workloads."""

    def __init__(
        self,
        *,
        task_source: TaskSource[TaskPayloadT],
        prompt_renderer: PromptRenderer[TaskPayloadT],
        verifier: SingleStepVerifier[TaskPayloadT],
        observation_message_adapter: ObservationMessageAdapter | None = None,
        submission_extractor: Callable[[str], str] | None = None,
    ) -> None:
        """Initialize the single-step verifier session."""
        self._task_source = task_source
        self._prompt_renderer = prompt_renderer
        self._verifier = verifier
        self._observation_message_adapter = (
            observation_message_adapter
            if observation_message_adapter is not None
            else DefaultObservationMessageAdapter(
                policy=DefaultObservationMessagePolicy()
            )
        )
        self._submission_extractor = (
            submission_extractor
            if submission_extractor is not None
            else _identity_completion
        )
        self._task: SingleStepTask[TaskPayloadT] | None = None
        self._turn: TaskTurn | None = None
        self._trajectory: TaskTrajectory | None = None
        self._episode_return = 0.0
        self._closed = False

    @property
    def done(self) -> bool:
        """Return whether the current scalar session has finished."""
        if self._trajectory is None:
            return False
        return self._turn is None

    @property
    def task_instance_id(self) -> str:
        """Return the active task-instance id."""
        task = self._require_task()
        return task.instance.task_instance_id

    @property
    def turn(self) -> TaskTurn | None:
        """Return the current actionable turn, if one exists."""
        self._require_task()
        return self._turn

    @property
    def trajectory(self) -> TaskTrajectory:
        """Return the task-level trajectory for the active session."""
        if self._trajectory is None:
            raise EnvironmentNotResetError(
                "Call SingleStepVerifierSession.reset() before accessing trajectory."
            )
        return self._trajectory

    @property
    def episode_return(self) -> float:
        """Return cumulative reward for the active scalar session."""
        return self._episode_return

    def reset(self, *, seed: int) -> TaskResetResult:
        """Sample and render a fresh single-step task instance."""
        task = self._task_source.sample(seed=seed)
        observation = self._prompt_renderer.render(task)
        action_context = ActionContext(turn_index=0)
        turn = TaskTurn(
            observation=observation,
            action_context=action_context,
            messages=self._observation_message_adapter.to_messages(
                observation=observation,
                action_context=action_context,
            ),
        )
        reset_info = {
            "task_kind": task.instance.task_kind,
            "seed": task.instance.seed,
            "prompt_key": task.instance.prompt_key,
            "metadata": deepcopy(task.instance.metadata),
        }

        self._task = task
        self._turn = turn
        self._episode_return = 0.0
        self._trajectory = TaskTrajectory(
            task_instance_id=task.instance.task_instance_id,
            initial_turn=turn,
            reset_info=reset_info,
            debug_reset_info={},
        )
        return TaskResetResult(
            task_instance_id=task.instance.task_instance_id,
            observation=observation,
            turn=turn,
            info=reset_info,
        )

    def submit(self, assistant_output: str) -> TaskSubmissionResult:
        """Verify one assistant output and finish the scalar session."""
        task = self._require_task()
        if self._turn is None:
            raise EpisodeFinishedError(
                "The current task session has finished. Call reset() first."
            )
        raw_submission = self._submission_extractor(assistant_output)
        if not isinstance(raw_submission, str):
            raise TypeError(
                "SingleStepVerifierSession submission_extractor must return a string."
            )

        verification = self._verifier.verify(
            task=task,
            completion=raw_submission,
        )
        result = TaskSubmissionResult(
            task_instance_id=task.instance.task_instance_id,
            assistant_output=assistant_output,
            raw_submission=raw_submission,
            parsed_output=verification.parsed_output,
            valid_submission=verification.valid_submission,
            reward=verification.reward,
            terminated=verification.terminated,
            truncated=verification.truncated,
            observation=None,
            turn=None,
            info=verification.info,
            debug_info=verification.debug_info,
        )
        self._turn = None
        self._episode_return += result.reward
        self.trajectory.append(
            result,
            details={"adapter": "single_step"},
        )
        return result

    def close(self) -> None:
        """Close closeable collaborators owned by the session."""
        if self._closed:
            return
        for component in (
            self._task_source,
            self._prompt_renderer,
            self._verifier,
            self._observation_message_adapter,
        ):
            close_method = getattr(component, "close", None)
            if callable(close_method):
                close_method()
        self._closed = True

    def _require_task(self) -> SingleStepTask[TaskPayloadT]:
        """Return the active task or fail if reset has not happened."""
        if self._task is None:
            raise EnvironmentNotResetError(
                "Call SingleStepVerifierSession.reset() before using the session."
            )
        return self._task


def _identity_completion(assistant_output: str) -> str:
    """Return the assistant output unchanged."""
    return assistant_output


def _snapshot_mapping(
    value: dict[str, object],
    *,
    context: str,
) -> dict[str, object]:
    """Return a detached copy of a metadata mapping."""
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a dict.")
    return deepcopy(value)


__all__ = [
    "PromptRenderer",
    "SingleStepTask",
    "SingleStepVerifier",
    "SingleStepVerifierSession",
    "TaskSource",
    "VerificationResult",
]
