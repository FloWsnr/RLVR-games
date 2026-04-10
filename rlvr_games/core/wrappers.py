"""Environment wrappers for policy-controlled runtime behavior."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar

from rlvr_games.core.exceptions import (
    EpisodeFinishedError,
    EnvironmentNotResetError,
    InvalidActionError,
)
from rlvr_games.core.protocol import Environment
from rlvr_games.core.trajectory import EpisodeTrajectory, TrajectoryStep
from rlvr_games.core.types import Observation, StepResult

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


class InvalidActionMode(StrEnum):
    """Policy modes for how invalid actions should be handled."""

    RAISE = "raise"
    PENALIZE_CONTINUE = "penalize-continue"
    PENALIZE_TRUNCATE = "penalize-truncate"


@dataclass(slots=True)
class InvalidActionPolicy:
    """Configuration describing how invalid actions should be handled.

    Attributes
    ----------
    mode : InvalidActionMode
        High-level invalid-action handling mode.
    penalty : float | None
        Reward assigned to rejected actions when the mode penalizes them. This
        must be `None` for `raise` mode and a concrete float otherwise.
    """

    mode: InvalidActionMode
    penalty: float | None

    def __post_init__(self) -> None:
        """Validate that the configured mode and penalty are coherent.

        Raises
        ------
        ValueError
            If the policy mode and penalty combination is inconsistent.
        """
        if self.mode == InvalidActionMode.RAISE and self.penalty is not None:
            raise ValueError("Raise mode does not accept an invalid-action penalty.")
        if self.mode != InvalidActionMode.RAISE and self.penalty is None:
            raise ValueError("Penalized invalid-action modes require a penalty.")


class InvalidActionPolicyEnv(Generic[StateT, ActionT]):
    """Environment wrapper that makes invalid-action handling explicit.

    The wrapped environment remains the verifier-backed source of truth for
    legal transitions. This wrapper only intercepts `InvalidActionError` to
    either re-raise it or convert the rejected attempt into a recorded
    trajectory step with an explicit penalty policy.
    """

    def __init__(
        self,
        *,
        env: Environment[StateT, ActionT],
        policy: InvalidActionPolicy,
    ) -> None:
        """Initialize the invalid-action policy wrapper.

        Parameters
        ----------
        env : Environment[StateT, ActionT]
            Underlying environment that performs legal action verification and
            state transitions.
        policy : InvalidActionPolicy
            Policy describing how rejected raw actions should be handled.
        """
        self.env = env
        self.policy = policy
        self.backend = env.backend

        self._trajectory: EpisodeTrajectory[ActionT] | None = None
        self._last_observation: Observation | None = None
        self._episode_finished = False

    @property
    def state(self) -> StateT:
        """Return the current canonical state from the wrapped environment."""
        return self.env.state

    @property
    def trajectory(self) -> EpisodeTrajectory[ActionT]:
        """Return the wrapper-owned trajectory for the active episode.

        Returns
        -------
        EpisodeTrajectory[ActionT]
            Recorded trajectory including both accepted and rejected attempts.

        Raises
        ------
        EnvironmentNotResetError
            If `reset()` has not been called yet.
        """
        if self._trajectory is None:
            raise EnvironmentNotResetError(
                "Call reset() before accessing env.trajectory."
            )
        return self._trajectory

    def reset(self, *, seed: int) -> tuple[Observation, dict[str, object]]:
        """Reset the wrapped environment and initialize wrapper state.

        Parameters
        ----------
        seed : int
            Explicit seed forwarded to the wrapped environment reset.

        Returns
        -------
        tuple[Observation, dict[str, object]]
            Initial observation and reset metadata.
        """
        observation, info = self.env.reset(seed=seed)
        initial_observation = _copy_observation(observation)
        self._trajectory = EpisodeTrajectory(
            initial_observation=initial_observation,
            reset_info=dict(info),
        )
        self._last_observation = initial_observation
        self._episode_finished = False
        return initial_observation, dict(info)

    def step(self, raw_action: str) -> StepResult:
        """Advance the episode according to the configured invalid-action mode.

        Parameters
        ----------
        raw_action : str
            Raw agent action to validate against the wrapped environment.

        Returns
        -------
        StepResult
            Accepted transition result or a policy-generated rejected attempt
            result.

        Raises
        ------
        EnvironmentNotResetError
            If `reset()` has not been called yet.
        EpisodeFinishedError
            If the wrapper episode has already terminated or been truncated.
        InvalidActionError
            If the wrapped environment rejects the action and the policy is
            configured to re-raise verifier failures.
        """
        if self._episode_finished:
            raise EpisodeFinishedError(
                "The current episode has finished. Call reset() first."
            )

        self.trajectory
        try:
            step_result = self.env.step(raw_action)
        except InvalidActionError as exc:
            return self._handle_invalid_action(raw_action=raw_action, error=exc)

        recorded_step = self.env.trajectory.steps[-1]
        observation = _copy_observation(step_result.observation)
        info = dict(step_result.info)
        self._last_observation = observation
        self._episode_finished = step_result.terminated or step_result.truncated
        self.trajectory.steps.append(
            TrajectoryStep(
                raw_action=recorded_step.raw_action,
                action=recorded_step.action,
                accepted=recorded_step.accepted,
                observation=observation,
                reward=step_result.reward,
                terminated=step_result.terminated,
                truncated=step_result.truncated,
                info=info,
            )
        )
        return StepResult(
            observation=observation,
            reward=step_result.reward,
            accepted=step_result.accepted,
            terminated=step_result.terminated,
            truncated=step_result.truncated,
            info=info,
        )

    def _handle_invalid_action(
        self, *, raw_action: str, error: InvalidActionError
    ) -> StepResult:
        """Convert a verifier rejection into an explicit policy outcome."""
        if self.policy.mode == InvalidActionMode.RAISE:
            raise error

        if self._last_observation is None:
            raise EnvironmentNotResetError("Call reset() before calling step().")

        truncated = self.policy.mode == InvalidActionMode.PENALIZE_TRUNCATE
        info: dict[str, object] = {
            "accepted": False,
            "error": str(error),
            "invalid_action": True,
            "invalid_action_policy": self.policy.mode.value,
            "raw_action": raw_action,
        }
        if truncated:
            info["truncated_reason"] = "invalid_action"

        penalty = self.policy.penalty
        if penalty is None:
            raise ValueError("Penalized invalid-action handling requires a penalty.")

        observation = _copy_observation(self._last_observation)
        self._last_observation = observation
        self._episode_finished = truncated
        self.trajectory.steps.append(
            TrajectoryStep(
                raw_action=raw_action,
                action=None,
                accepted=False,
                observation=observation,
                reward=penalty,
                terminated=False,
                truncated=truncated,
                info=info,
            )
        )
        return StepResult(
            observation=observation,
            reward=penalty,
            accepted=False,
            terminated=False,
            truncated=truncated,
            info=info,
        )


def _copy_observation(observation: Observation) -> Observation:
    """Create a detached copy of an observation object."""
    return Observation(
        text=observation.text,
        image_paths=observation.image_paths,
        metadata=dict(observation.metadata),
    )
