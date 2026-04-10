"""Canonical environment implementation for turn-based RLVR tasks."""

from dataclasses import dataclass
from typing import Generic, TypeVar

from rlvr_games.core.exceptions import (
    EpisodeFinishedError,
    EnvironmentNotResetError,
    InvalidActionError,
)
from rlvr_games.core.protocol import GameBackend, Renderer, RewardFn, Scenario
from rlvr_games.core.trajectory import EpisodeTrajectory, TrajectoryStep
from rlvr_games.core.types import (
    EpisodeConfig,
    InvalidActionMode,
    Observation,
    ParseResult,
    StepResult,
)

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


@dataclass(slots=True)
class _AttemptOutcome(Generic[StateT, ActionT]):
    """Internal normalized outcome for one attempted environment step."""

    action: ActionT | None
    next_state: StateT
    observation: Observation
    reward: float
    accepted: bool
    terminated: bool
    truncated: bool
    info: dict[str, object]


class TurnBasedEnv(Generic[StateT, ActionT]):
    """Minimal stateful environment with reset/step semantics.

    The environment coordinates four reusable components: a scenario that
    creates the initial canonical state, a backend that verifies actions and
    applies transitions, a renderer that turns state into observations, and a
    reward function that scores verified transitions.
    """

    def __init__(
        self,
        *,
        backend: GameBackend[StateT, ActionT],
        scenario: Scenario[StateT],
        renderer: Renderer[StateT],
        reward_fn: RewardFn[StateT, ActionT],
        config: EpisodeConfig,
    ) -> None:
        """Initialize a turn-based environment.

        Parameters
        ----------
        backend : GameBackend[StateT, ActionT]
            Rules engine responsible for action parsing, legality checks,
            transitions, and terminal detection.
        scenario : Scenario[StateT]
            Component that creates the starting canonical state for each new
            episode.
        renderer : Renderer[StateT]
            Adapter that converts canonical state into the observation exposed
            to the model.
        reward_fn : RewardFn[StateT, ActionT]
            Reward function used to score verified transitions.
        config : EpisodeConfig
            Episode-wide configuration such as optional attempt/transition
            limits, invalid-action handling policy, and metadata.
        """
        self.backend = backend
        self.scenario = scenario
        self.renderer = renderer
        self.reward_fn = reward_fn
        self.config = config

        self._state: StateT | None = None
        self._trajectory: EpisodeTrajectory[ActionT] | None = None
        self._attempt_count = 0
        self._transition_count = 0
        self._episode_finished = False

    @property
    def state(self) -> StateT:
        """Return the current canonical state.

        Returns
        -------
        StateT
            The current verifier-backed state for the active episode.

        Raises
        ------
        EnvironmentNotResetError
            If `reset()` has not been called yet.
        """
        if self._state is None:
            raise EnvironmentNotResetError("Call reset() before accessing env.state.")
        return self._state

    @property
    def trajectory(self) -> EpisodeTrajectory[ActionT]:
        """Return the recorded trajectory for the active episode.

        Returns
        -------
        EpisodeTrajectory[ActionT]
            The trajectory object containing the initial observation and all
            subsequent verified transitions.

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
        """Start a fresh episode from the configured scenario.

        Parameters
        ----------
        seed : int
            Explicit seed forwarded to the scenario reset.

        Returns
        -------
        tuple[Observation, dict[str, object]]
            A pair containing the initial observation shown to the model and
            the reset metadata returned by the scenario.
        """
        self._attempt_count = 0
        self._transition_count = 0
        self._episode_finished = False
        self._state, info = self.scenario.reset(seed=seed)
        observation = self.renderer.render(self._state)
        self._trajectory = EpisodeTrajectory(
            initial_observation=observation,
            reset_info=info,
        )
        return observation, info

    def step(self, raw_action: str) -> StepResult:
        """Advance the episode by one verified model action.

        Parameters
        ----------
        raw_action : str
            Raw model output to be parsed and validated by the backend.

        Returns
        -------
        StepResult
            The rendered next observation, reward, terminal flags, and
            transition metadata for the applied action.

        Raises
        ------
        EnvironmentNotResetError
            If `reset()` has not been called yet.
        EpisodeFinishedError
            If the current episode has already terminated or been truncated.
        InvalidActionError
            If the backend rejects the action as malformed or illegal for the
            current state and the configured invalid-action policy is
            `raise`.
        """
        if self._episode_finished:
            raise EpisodeFinishedError(
                "The current episode has finished. Call reset() first."
            )

        previous_state = self.state
        parse_result = self.backend.parse_action(previous_state, raw_action)
        if parse_result.error is not None:
            return self._handle_invalid_action(
                previous_state=previous_state,
                raw_action=raw_action,
                parse_result=parse_result,
            )
        action = parse_result.require_action()

        self._attempt_count += 1
        next_state, transition_info = self.backend.apply_action(previous_state, action)
        reward = self.reward_fn.evaluate(
            previous_state=previous_state,
            action=action,
            next_state=next_state,
            transition_info=transition_info,
        )

        self._transition_count += 1
        terminated = self.backend.is_terminal(next_state)
        truncated = False
        info = dict(transition_info)
        info["accepted"] = True
        info["attempt_count"] = self._attempt_count
        info["transition_count"] = self._transition_count

        truncated_reason = self._limit_truncated_reason(terminated=terminated)
        if truncated_reason is not None:
            truncated = True
            info.setdefault("truncated_reason", truncated_reason)

        observation = self.renderer.render(next_state)
        outcome = _AttemptOutcome(
            action=action,
            next_state=next_state,
            observation=observation,
            reward=reward,
            accepted=True,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        return self._commit_attempt(raw_action=raw_action, outcome=outcome)

    def _handle_invalid_action(
        self,
        *,
        previous_state: StateT,
        raw_action: str,
        parse_result: ParseResult[ActionT],
    ) -> StepResult:
        """Handle a verifier-rejected action according to env policy."""
        error = parse_result.error
        if error is None:
            raise ValueError(
                "_handle_invalid_action() requires a rejected parse result."
            )

        policy = self.config.invalid_action_policy
        if policy.mode == InvalidActionMode.RAISE:
            raise InvalidActionError(error)

        penalty = policy.penalty
        if penalty is None:
            raise ValueError("Penalized invalid-action handling requires a penalty.")

        self._attempt_count += 1
        observation = self.renderer.render(previous_state)
        terminated = False
        truncated = policy.mode == InvalidActionMode.PENALIZE_TRUNCATE
        info: dict[str, object] = {
            "accepted": False,
            "attempt_count": self._attempt_count,
            "error": error,
            "invalid_action": True,
            "invalid_action_policy": policy.mode.value,
            "raw_action": raw_action,
            "transition_count": self._transition_count,
        }
        if truncated:
            info["truncated_reason"] = "invalid_action"

        limit_truncated_reason = self._limit_truncated_reason(terminated=False)
        if limit_truncated_reason is not None and not truncated:
            truncated = True
            info["truncated_reason"] = limit_truncated_reason

        outcome = _AttemptOutcome(
            action=None,
            next_state=previous_state,
            observation=observation,
            reward=penalty,
            accepted=False,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        return self._commit_attempt(raw_action=raw_action, outcome=outcome)

    def _commit_attempt(
        self,
        *,
        raw_action: str,
        outcome: _AttemptOutcome[StateT, ActionT],
    ) -> StepResult:
        """Persist one normalized attempt outcome and return the step result."""
        self._episode_finished = outcome.terminated or outcome.truncated
        self._state = outcome.next_state

        self.trajectory.steps.append(
            TrajectoryStep(
                raw_action=raw_action,
                action=outcome.action,
                accepted=outcome.accepted,
                observation=outcome.observation,
                reward=outcome.reward,
                terminated=outcome.terminated,
                truncated=outcome.truncated,
                info=outcome.info,
            )
        )
        return StepResult(
            observation=outcome.observation,
            reward=outcome.reward,
            accepted=outcome.accepted,
            terminated=outcome.terminated,
            truncated=outcome.truncated,
            info=outcome.info,
        )

    def _limit_truncated_reason(self, *, terminated: bool) -> str | None:
        """Return the truncation reason implied by configured episode limits."""
        if terminated:
            return None
        if (
            self.config.max_attempts is not None
            and self._attempt_count >= self.config.max_attempts
        ):
            return "max_attempts"
        if (
            self.config.max_transitions is not None
            and self._transition_count >= self.config.max_transitions
        ):
            return "max_transitions"
        return None
