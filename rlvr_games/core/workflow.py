"""Trainer-agnostic workflow sessions built on top of environments."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

from rlvr_games.core.action_context import ActionContext
from rlvr_games.core.exceptions import EnvironmentNotResetError
from rlvr_games.core.messages import ChatMessage
from rlvr_games.core.protocol import Environment
from rlvr_games.core.rollout import PreparedTurn, prepare_turn
from rlvr_games.core.trajectory import EpisodeTrajectory
from rlvr_games.core.types import Observation, StepResult

if TYPE_CHECKING:
    from rlvr_games.core.async_env import AsyncEnvPool

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


@dataclass(slots=True)
class WorkflowTurn:
    """Trainer-facing package for one live agent turn.

    Attributes
    ----------
    observation : Observation
        Current observation shown to the model for this turn.
    action_context : ActionContext
        Structured public-safe context for the next action.
    messages : tuple[ChatMessage, ...]
        Chat-formatted messages derived from ``observation`` and
        ``action_context``.
    """

    observation: Observation
    action_context: ActionContext
    messages: tuple[ChatMessage, ...]


@dataclass(slots=True)
class WorkflowResetResult:
    """Result returned when a workflow session starts a new episode."""

    observation: Observation
    reset_info: dict[str, object]
    turn: WorkflowTurn | None


@dataclass(slots=True)
class WorkflowSubmission:
    """Result returned after one assistant output is submitted."""

    assistant_output: str
    raw_action: str
    step_result: StepResult
    turn: WorkflowTurn | None

    @property
    def done(self) -> bool:
        """Return whether this submission finished the episode."""
        return self.step_result.terminated or self.step_result.truncated


class WorkflowSessionProtocol(Protocol):
    """Shared public contract implemented by workflow session wrappers."""

    @property
    def done(self) -> bool:
        """Return whether the current episode has finished."""
        ...

    @property
    def current_observation(self) -> Observation:
        """Return the current observation for the active episode."""
        ...

    @property
    def reset_info(self) -> dict[str, object]:
        """Return the reset metadata for the active episode."""
        ...

    @property
    def turn(self) -> WorkflowTurn | None:
        """Return the currently actionable turn, if one exists."""
        ...

    @property
    def episode_return(self) -> float:
        """Return the cumulative reward for the active episode."""
        ...

    def reset(self, *, seed: int) -> WorkflowResetResult:
        """Start a fresh workflow episode."""
        ...

    def submit(self, assistant_output: str) -> WorkflowSubmission:
        """Submit one assistant output to the wrapped environment."""
        ...

    def close(self) -> None:
        """Close the resources owned by the workflow session."""
        ...


@dataclass(slots=True)
class _DriverResetResult:
    """Normalized reset payload returned by one workflow backend."""

    observation: Observation
    reset_info: dict[str, object]
    turn: PreparedTurn | None


@dataclass(slots=True)
class _DriverStepResult:
    """Normalized step payload returned by one workflow backend."""

    step_result: StepResult
    turn: PreparedTurn | None


class _WorkflowSessionBase(Generic[StateT, ActionT]):
    """Shared implementation for local and async workflow sessions."""

    def __init__(self, *, action_extractor: Callable[[str], str] | None = None) -> None:
        self._action_extractor = (
            action_extractor if action_extractor is not None else _identity_action
        )
        self._current_observation: Observation | None = None
        self._current_turn: WorkflowTurn | None = None
        self._reset_info: dict[str, object] | None = None
        self._episode_finished = False
        self._episode_return = 0.0

    @property
    def done(self) -> bool:
        """Return whether the current episode has finished."""
        if self._current_observation is None:
            return False
        return self._episode_finished

    @property
    def current_observation(self) -> Observation:
        """Return the current observation for the active episode.

        Raises
        ------
        EnvironmentNotResetError
            If ``reset()`` has not been called yet.
        """
        if self._current_observation is None:
            raise EnvironmentNotResetError(
                f"Call {type(self).__name__}.reset() before accessing the observation."
            )
        return self._current_observation

    @property
    def reset_info(self) -> dict[str, object]:
        """Return the reset metadata for the active episode.

        Raises
        ------
        EnvironmentNotResetError
            If ``reset()`` has not been called yet.
        """
        if self._reset_info is None:
            raise EnvironmentNotResetError(
                f"Call {type(self).__name__}.reset() before accessing reset_info."
            )
        return deepcopy(self._reset_info)

    @property
    def turn(self) -> WorkflowTurn | None:
        """Return the currently actionable turn, if one exists.

        Raises
        ------
        EnvironmentNotResetError
            If ``reset()`` has not been called yet.
        """
        self.current_observation
        return self._current_turn

    @property
    def episode_return(self) -> float:
        """Return the cumulative reward for the active episode."""
        return self._episode_return

    def reset(self, *, seed: int) -> WorkflowResetResult:
        """Start a fresh workflow episode."""
        driver_result = self._reset_backend(seed=seed)
        observation = driver_result.observation
        turn = _workflow_turn_from_prepared_turn(
            observation=observation,
            prepared_turn=driver_result.turn,
        )
        reset_info = deepcopy(driver_result.reset_info)

        self._current_observation = observation
        self._current_turn = turn
        self._reset_info = reset_info
        self._episode_finished = turn is None
        self._episode_return = 0.0

        return WorkflowResetResult(
            observation=observation,
            reset_info=deepcopy(reset_info),
            turn=turn,
        )

    def submit(self, assistant_output: str) -> WorkflowSubmission:
        """Submit one assistant output to the wrapped environment.

        Raises
        ------
        EnvironmentNotResetError
            If ``reset()`` has not been called yet.
        TypeError
            If the configured action extractor does not return a string.
        """
        self.current_observation
        raw_action = self._action_extractor(assistant_output)
        if not isinstance(raw_action, str):
            raise TypeError(
                f"{type(self).__name__} action_extractor must return a string action."
            )

        driver_result = self._step_backend(raw_action=raw_action)
        step_result = driver_result.step_result
        observation = step_result.observation
        turn = _workflow_turn_from_prepared_turn(
            observation=observation,
            prepared_turn=driver_result.turn,
        )

        self._current_observation = observation
        self._current_turn = turn
        self._episode_finished = step_result.terminated or step_result.truncated
        self._episode_return += step_result.reward

        return WorkflowSubmission(
            assistant_output=assistant_output,
            raw_action=raw_action,
            step_result=step_result,
            turn=turn,
        )

    def close(self) -> None:
        """Close the resources owned by the workflow session."""
        self._close_backend()

    def _reset_backend(self, *, seed: int) -> _DriverResetResult:
        """Start a new episode and return the normalized reset payload."""
        raise NotImplementedError

    def _step_backend(self, *, raw_action: str) -> _DriverStepResult:
        """Submit one action and return the normalized step payload."""
        raise NotImplementedError

    def _close_backend(self) -> None:
        """Release any resources owned by the workflow backend."""


class LocalWorkflowSession(_WorkflowSessionBase[StateT, ActionT]):
    """Workflow session backed by one in-process environment instance."""

    def __init__(
        self,
        *,
        env: Environment[StateT, ActionT],
        action_extractor: Callable[[str], str] | None = None,
    ) -> None:
        super().__init__(action_extractor=action_extractor)
        self._env = env

    @property
    def env(self) -> Environment[StateT, ActionT]:
        """Return the wrapped environment instance."""
        return self._env

    @property
    def trajectory(self) -> EpisodeTrajectory[ActionT]:
        """Return the recorded trajectory for the active episode."""
        self.current_observation
        return self._env.trajectory

    def _reset_backend(self, *, seed: int) -> _DriverResetResult:
        observation, reset_info = self._env.reset(seed=seed)
        turn = None
        if not self._env.episode_finished:
            turn = prepare_turn(env=self._env, observation=observation)
        return _DriverResetResult(
            observation=observation,
            reset_info=reset_info,
            turn=turn,
        )

    def _step_backend(self, *, raw_action: str) -> _DriverStepResult:
        step_result = self._env.step(raw_action)
        turn = None
        if not self._env.episode_finished:
            turn = prepare_turn(env=self._env, observation=step_result.observation)
        return _DriverStepResult(
            step_result=step_result,
            turn=turn,
        )

    def _close_backend(self) -> None:
        self._env.close()


class AsyncWorkflowSession(_WorkflowSessionBase[Any, Any]):
    """Workflow session backed by one slot in an async environment pool."""

    def __init__(
        self,
        *,
        pool: "AsyncEnvPool",
        slot_id: int,
        lease_token: int,
        action_extractor: Callable[[str], str] | None = None,
        close_pool: bool = False,
    ) -> None:
        super().__init__(action_extractor=action_extractor)
        self._pool = pool
        self._slot_id = slot_id
        self._lease_token: int | None = lease_token
        self._close_pool = close_pool

    @property
    def pool(self) -> "AsyncEnvPool":
        """Return the owning async environment pool."""
        return self._pool

    @property
    def slot_id(self) -> int:
        """Return the pool slot owned by this session."""
        return self._slot_id

    def _reset_backend(self, *, seed: int) -> _DriverResetResult:
        from rlvr_games.core.async_env import AsyncResetResult

        command_id = self._pool._enqueue_reset(
            slot_id=self._slot_id,
            seed=seed,
            allow_leased=True,
            lease_token=self._lease_token,
        )
        result = self._pool._recv_slot(
            slot_id=self._slot_id,
            command_id=command_id,
            timeout_seconds=None,
            allow_leased=True,
            lease_token=self._lease_token,
        )
        if not isinstance(result, AsyncResetResult):
            raise RuntimeError(
                "Expected AsyncResetResult when resetting async workflow session."
            )
        return _DriverResetResult(
            observation=result.observation,
            reset_info=result.reset_info,
            turn=result.turn,
        )

    def _step_backend(self, *, raw_action: str) -> _DriverStepResult:
        from rlvr_games.core.async_env import AsyncStepResult

        command_id = self._pool._enqueue_step(
            slot_id=self._slot_id,
            raw_action=raw_action,
            allow_leased=True,
            lease_token=self._lease_token,
        )
        result = self._pool._recv_slot(
            slot_id=self._slot_id,
            command_id=command_id,
            timeout_seconds=None,
            allow_leased=True,
            lease_token=self._lease_token,
        )
        if not isinstance(result, AsyncStepResult):
            raise RuntimeError(
                "Expected AsyncStepResult when stepping async workflow session."
            )
        return _DriverStepResult(
            step_result=result.step_result,
            turn=result.turn,
        )

    def _close_backend(self) -> None:
        if self._close_pool:
            self._pool.close()
            self._lease_token = None
            return
        if self._lease_token is None:
            return
        self._pool._release_slot(
            slot_id=self._slot_id,
            lease_token=self._lease_token,
        )
        self._lease_token = None


WorkflowSession = LocalWorkflowSession


def _identity_action(assistant_output: str) -> str:
    """Return the assistant output unchanged."""
    return assistant_output


def _workflow_turn_from_prepared_turn(
    *,
    observation: Observation,
    prepared_turn: PreparedTurn | None,
) -> WorkflowTurn | None:
    """Convert one prepared turn into the public workflow turn type."""
    if prepared_turn is None:
        return None
    return WorkflowTurn(
        observation=observation,
        action_context=prepared_turn.action_context,
        messages=prepared_turn.messages,
    )


__all__ = [
    "AsyncWorkflowSession",
    "LocalWorkflowSession",
    "WorkflowResetResult",
    "WorkflowSession",
    "WorkflowSessionProtocol",
    "WorkflowSubmission",
    "WorkflowTurn",
]
