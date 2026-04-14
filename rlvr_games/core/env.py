"""Canonical environment implementation for turn-based RLVR tasks."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from rlvr_games.core.exceptions import (
    EpisodeFinishedError,
    EnvironmentNotResetError,
    InvalidActionError,
)
from rlvr_games.core.protocol import (
    AutoAdvancePolicy,
    GameBackend,
    Renderer,
    RewardFn,
    Scenario,
)
from rlvr_games.core.trajectory import (
    EpisodeTrajectory,
    RecordedTransition,
    TrajectoryStep,
)
from rlvr_games.core.types import (
    EpisodeConfig,
    EpisodeBoundary,
    InvalidActionMode,
    Observation,
    RenderedImage,
    StepResult,
)

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


@dataclass(slots=True)
class _AcceptedTransition(Generic[StateT, ActionT]):
    """Internal accepted backend transition applied during one env step."""

    source: str
    raw_action: str
    action: ActionT
    next_state: StateT
    info: dict[str, object]


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
    transitions: tuple[_AcceptedTransition[StateT, ActionT], ...] = ()


@dataclass(slots=True)
class _ResolutionOutcome(Generic[StateT, ActionT]):
    """Internal normalized state after any required internal transitions."""

    next_state: StateT
    terminated: bool
    truncated: bool
    info: dict[str, object]
    transitions: tuple[_AcceptedTransition[StateT, ActionT], ...] = ()


class TurnBasedEnv(Generic[StateT, ActionT]):
    """Minimal stateful environment with reset/step semantics.

    The environment coordinates reusable components for scenario reset,
    verifier-backed transitions, rendering, state inspection, reward
    evaluation, and optional internal auto-advancement between agent turns.
    """

    def __init__(
        self,
        *,
        backend: GameBackend[StateT, ActionT],
        scenario: Scenario[StateT],
        renderer: Renderer[StateT],
        inspect_state_fn: Callable[[StateT], dict[str, object]],
        reward_fn: RewardFn[StateT, ActionT],
        config: EpisodeConfig,
        auto_advance_policy: AutoAdvancePolicy[StateT, ActionT] | None = None,
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
        inspect_state_fn : Callable[[StateT], dict[str, object]]
            Function that converts canonical state into a structured debug view
            used by CLI and rollout tooling.
        reward_fn : RewardFn[StateT, ActionT]
            Reward function used to score accepted environment steps.
        config : EpisodeConfig
            Episode-wide configuration such as optional attempt/transition
            limits, invalid-action handling policy, and metadata.
        auto_advance_policy : AutoAdvancePolicy[StateT, ActionT] | None
            Optional policy that can apply internal verifier-backed actions
            such as opponent replies after the agent action until control
            returns to the agent or the episode finishes.
        """
        self.backend = backend
        self.scenario = scenario
        self.renderer = renderer
        self.inspect_state_fn = inspect_state_fn
        self.reward_fn = reward_fn
        self.config = config
        self.auto_advance_policy = auto_advance_policy

        self._state: StateT | None = None
        self._trajectory: EpisodeTrajectory[ActionT] | None = None
        self._attempt_count = 0
        self._transition_count = 0
        self._episode_finished = False
        self._closed = False

    @property
    def episode_finished(self) -> bool:
        """Return whether the active episode can accept more steps.

        Returns
        -------
        bool
            `True` when the active episode has already terminated or been
            truncated.
        """
        return self._episode_finished

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

    def legal_actions(self) -> tuple[str, ...]:
        """Return the legal serialized actions for the current state.

        Returns
        -------
        tuple[str, ...]
            Legal model-facing actions accepted by the environment.

        Raises
        ------
        EnvironmentNotResetError
            If `reset()` has not been called yet.
        """
        return tuple(self.backend.legal_actions(self.state))

    def inspect_state(self) -> dict[str, object]:
        """Return a debug-oriented snapshot of the current canonical state.

        Returns
        -------
        dict[str, object]
            Deep-copied state summary safe for inspection tooling.

        Raises
        ------
        EnvironmentNotResetError
            If `reset()` has not been called yet.
        """
        return deepcopy(self.inspect_state_fn(self.state))

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
            the reset metadata returned by the scenario after any required
            internal auto-advanced transitions have been resolved.
        """
        self._attempt_count = 0
        self._transition_count = 0
        initial_state, info = self.scenario.reset(seed=seed)
        if self.auto_advance_policy is not None:
            self.auto_advance_policy.reset(initial_state=initial_state)
        resolution = self._resolve_to_agent_turn(state=initial_state)
        self._state = resolution.next_state
        observation = self.renderer.render(self._state)
        self._episode_finished = resolution.terminated or resolution.truncated
        info = self._build_reset_info(base_info=info, resolution=resolution)
        self._trajectory = EpisodeTrajectory(
            initial_observation=self._snapshot_observation(observation),
            reset_info=self._snapshot_info(info),
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
        if self.auto_advance_policy is not None and not self.auto_advance_policy.is_agent_turn(
            state=previous_state
        ):
            raise RuntimeError(
                "Environment cannot accept an agent action before control "
                "returns to the agent."
            )
        parse_result = self.backend.parse_action(previous_state, raw_action)
        if parse_result.error is not None:
            return self._handle_invalid_action(
                previous_state=previous_state,
                raw_action=raw_action,
                error=parse_result.error,
            )
        action = parse_result.require_action()
        try:
            outcome = self._accepted_attempt_outcome(
                previous_state=previous_state,
                raw_action=raw_action,
                action=action,
            )
        except InvalidActionError as exc:
            return self._handle_invalid_action(
                previous_state=previous_state,
                raw_action=raw_action,
                error=str(exc),
            )
        return self._commit_attempt(raw_action=raw_action, outcome=outcome)

    def close(self) -> None:
        """Release any closeable collaborators owned by the environment.

        Returns
        -------
        None
            This method is idempotent.
        """
        if self._closed:
            return
        for component in (
            self.backend,
            self.scenario,
            self.renderer,
            self.inspect_state_fn,
            self.reward_fn,
            self.auto_advance_policy,
        ):
            close_method = getattr(component, "close", None)
            if callable(close_method):
                close_method()
        self._closed = True

    def _accepted_attempt_outcome(
        self,
        *,
        previous_state: StateT,
        raw_action: str,
        action: ActionT,
    ) -> _AttemptOutcome[StateT, ActionT]:
        """Apply one accepted agent action and any internal follow-up actions.

        Parameters
        ----------
        previous_state : StateT
            Canonical state before the accepted action.
        raw_action : str
            Raw agent action string recorded for the first accepted
            transition.
        action : ActionT
            Parsed action accepted by the backend parser.

        Returns
        -------
        _AttemptOutcome[StateT, ActionT]
            Normalized accepted-attempt payload ready to record in the
            trajectory.
        """
        self._attempt_count += 1
        agent_transition = self._apply_accepted_transition(
            state=previous_state,
            source="agent",
            raw_action=raw_action,
            action=action,
        )
        resolution = self._resolve_to_agent_turn(state=agent_transition.next_state)
        transitions = [agent_transition, *resolution.transitions]

        info = self._build_accepted_step_info(
            transitions=transitions,
            next_state=resolution.next_state,
            boundary_info=resolution.info,
        )
        reward = self.reward_fn.evaluate(
            previous_state=previous_state,
            action=action,
            next_state=resolution.next_state,
            transition_info=info,
        )
        observation = self.renderer.render(resolution.next_state)
        return _AttemptOutcome(
            action=action,
            next_state=resolution.next_state,
            observation=observation,
            reward=reward,
            accepted=True,
            terminated=resolution.terminated,
            truncated=resolution.truncated,
            info=info,
            transitions=tuple(transitions),
        )

    def _handle_invalid_action(
        self,
        *,
        previous_state: StateT,
        raw_action: str,
        error: str | None,
    ) -> StepResult:
        """Handle a verifier-rejected action according to env policy."""
        if error is None:
            raise ValueError("_handle_invalid_action() requires an error message.")

        policy = self.config.invalid_action_policy
        if policy.mode == InvalidActionMode.RAISE:
            raise InvalidActionError(error)

        outcome = self._rejected_attempt_outcome(
            previous_state=previous_state,
            raw_action=raw_action,
            error=error,
        )
        return self._commit_attempt(raw_action=raw_action, outcome=outcome)

    def _rejected_attempt_outcome(
        self,
        *,
        previous_state: StateT,
        raw_action: str,
        error: str,
    ) -> _AttemptOutcome[StateT, ActionT]:
        """Build a normalized outcome for a penalized rejected action.

        Parameters
        ----------
        previous_state : StateT
            Canonical state that remains current after the rejection.
        raw_action : str
            Raw model output rejected by the backend parser.
        error : str
            Backend rejection message.

        Returns
        -------
        _AttemptOutcome[StateT, ActionT]
            Normalized rejected-attempt payload ready to record in the
            trajectory.

        Raises
        ------
        ValueError
            If the active invalid-action policy is missing the required
            penalty value.
        """
        policy = self.config.invalid_action_policy
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
        return outcome

    def _commit_attempt(
        self,
        *,
        raw_action: str,
        outcome: _AttemptOutcome[StateT, ActionT],
    ) -> StepResult:
        """Persist one normalized attempt outcome and return the step result."""
        self._episode_finished = outcome.terminated or outcome.truncated
        self._state = outcome.next_state
        trajectory_observation = self._snapshot_observation(outcome.observation)
        trajectory_info = self._snapshot_info(outcome.info)

        self.trajectory.steps.append(
            TrajectoryStep(
                raw_action=raw_action,
                action=outcome.action,
                accepted=outcome.accepted,
                observation=trajectory_observation,
                reward=outcome.reward,
                terminated=outcome.terminated,
                truncated=outcome.truncated,
                info=trajectory_info,
                transitions=self._snapshot_transitions(outcome.transitions),
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

    def _snapshot_info(self, info: dict[str, object]) -> dict[str, object]:
        """Return a trajectory-safe snapshot of transition metadata.

        Parameters
        ----------
        info : dict[str, object]
            Metadata dictionary to snapshot.

        Returns
        -------
        dict[str, object]
            Deep-copied metadata suitable for trajectory storage.
        """
        return deepcopy(info)

    def _snapshot_observation(self, observation: Observation) -> Observation:
        """Return a trajectory-safe snapshot of an observation.

        Parameters
        ----------
        observation : Observation
            Observation to snapshot.

        Returns
        -------
        Observation
            Observation copy whose metadata does not alias caller-owned state.
        """
        return Observation(
            text=observation.text,
            images=self._snapshot_images(observation.images),
            metadata=deepcopy(observation.metadata),
        )

    def _snapshot_images(
        self,
        images: tuple[RenderedImage, ...],
    ) -> tuple[RenderedImage, ...]:
        """Return trajectory-safe copies of observation images.

        Parameters
        ----------
        images : tuple[RenderedImage, ...]
            Image payloads to snapshot.

        Returns
        -------
        tuple[RenderedImage, ...]
            Copied images whose raster data does not alias the caller-owned
            payloads.
        """
        return tuple(image.copy() for image in images)

    def _snapshot_transitions(
        self,
        transitions: tuple[_AcceptedTransition[StateT, ActionT], ...],
    ) -> tuple[RecordedTransition[ActionT], ...]:
        """Return trajectory-safe copies of accepted backend transitions."""
        return tuple(
            RecordedTransition(
                source=transition.source,
                raw_action=transition.raw_action,
                action=deepcopy(transition.action),
                info=self._snapshot_info(transition.info),
            )
            for transition in transitions
        )

    def _apply_accepted_transition(
        self,
        *,
        state: StateT,
        source: str,
        raw_action: str,
        action: ActionT,
    ) -> _AcceptedTransition[StateT, ActionT]:
        """Apply one accepted backend transition and record its metadata."""
        next_state, transition_info = self.backend.apply_action(state, action)
        self._transition_count += 1
        return _AcceptedTransition(
            source=source,
            raw_action=raw_action,
            action=action,
            next_state=next_state,
            info=dict(transition_info),
        )

    def _build_accepted_step_info(
        self,
        *,
        transitions: list[_AcceptedTransition[StateT, ActionT]],
        next_state: StateT,
        boundary_info: dict[str, object],
    ) -> dict[str, object]:
        """Build step metadata for an accepted agent action."""
        agent_transition = transitions[0]
        info = dict(agent_transition.info)
        info.update(self.inspect_state_fn(next_state))
        info["accepted"] = True
        info["attempt_count"] = self._attempt_count
        info["transition_count"] = self._transition_count
        info["transition_count_delta"] = len(transitions)
        info["auto_advanced"] = len(transitions) > 1
        info["agent_transition"] = self._serialize_transition(agent_transition)
        info["internal_transitions"] = tuple(
            self._serialize_transition(transition) for transition in transitions[1:]
        )
        info["transitions"] = tuple(
            self._serialize_transition(transition) for transition in transitions
        )
        info.update(boundary_info)
        return info

    def _build_reset_info(
        self,
        *,
        base_info: dict[str, object],
        resolution: _ResolutionOutcome[StateT, ActionT],
    ) -> dict[str, object]:
        """Build reset metadata after any required internal transitions."""
        info = dict(base_info)
        if resolution.terminated or resolution.truncated:
            info["terminated"] = resolution.terminated
            info["truncated"] = resolution.truncated
        if resolution.transitions:
            info["auto_advanced"] = True
            info["transition_count"] = self._transition_count
            info["transition_count_delta"] = len(resolution.transitions)
            info["initial_transitions"] = tuple(
                self._serialize_transition(transition)
                for transition in resolution.transitions
            )
        info.update(resolution.info)
        return info

    def _serialize_transition(
        self,
        transition: _AcceptedTransition[StateT, ActionT],
    ) -> dict[str, object]:
        """Serialize one accepted backend transition into step metadata."""
        return {
            "source": transition.source,
            "raw_action": transition.raw_action,
            "info": self._snapshot_info(transition.info),
        }

    def _episode_boundary(self, *, state: StateT) -> EpisodeBoundary | None:
        """Return any explicit episode boundary from the auto-advance policy."""
        if self.auto_advance_policy is None:
            return None
        return self.auto_advance_policy.episode_boundary(state=state)

    def _resolve_to_agent_turn(
        self,
        *,
        state: StateT,
    ) -> _ResolutionOutcome[StateT, ActionT]:
        """Apply internal transitions until the agent can act or the episode ends."""
        transitions: list[_AcceptedTransition[StateT, ActionT]] = []
        current_state = state
        terminated = False
        truncated = False
        info: dict[str, object] = {}

        while True:
            terminated = self.backend.is_terminal(current_state)
            if terminated:
                break

            boundary = self._episode_boundary(state=current_state)
            if boundary is not None:
                terminated = boundary.terminated
                truncated = boundary.truncated
                info.update(boundary.info)
                break

            if self.auto_advance_policy is None:
                break
            if self.auto_advance_policy.is_agent_turn(state=current_state):
                break

            auto_action = self.auto_advance_policy.select_internal_action(
                state=current_state,
                backend=self.backend,
            )
            if auto_action is None:
                raise RuntimeError(
                    "Auto-advance policy returned no internal action before "
                    "control returned to the agent."
                )
            try:
                transition = self._apply_accepted_transition(
                    state=current_state,
                    source=auto_action.source,
                    raw_action=auto_action.raw_action,
                    action=auto_action.action,
                )
            except InvalidActionError:
                raise RuntimeError(
                    "Auto-advance policy produced an invalid internal action: "
                    f"{auto_action.raw_action!r}."
                ) from None
            transitions.append(transition)
            current_state = transition.next_state

        if not terminated and not truncated:
            truncated_reason = self._limit_truncated_reason(terminated=False)
            if truncated_reason is not None:
                truncated = True
                info["truncated_reason"] = truncated_reason

        return _ResolutionOutcome(
            next_state=current_state,
            terminated=terminated,
            truncated=truncated,
            info=info,
            transitions=tuple(transitions),
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
