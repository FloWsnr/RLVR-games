"""Canonical environment implementation for turn-based RLVR tasks."""

from typing import Generic, TypeVar

from rlvr_games.core.exceptions import EpisodeFinishedError, EnvironmentNotResetError
from rlvr_games.core.protocol import GameBackend, Renderer, RewardFn, Scenario
from rlvr_games.core.trajectory import EpisodeTrajectory, TrajectoryStep
from rlvr_games.core.types import EpisodeConfig, Observation, StepResult

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


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
            Episode-wide configuration such as optional turn limits and
            metadata.
        """
        self.backend = backend
        self.scenario = scenario
        self.renderer = renderer
        self.reward_fn = reward_fn
        self.config = config

        self._state: StateT | None = None
        self._trajectory: EpisodeTrajectory[ActionT] | None = None
        self._turn_count = 0
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
        self._turn_count = 0
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
            current state.
        """
        if self._episode_finished:
            raise EpisodeFinishedError(
                "The current episode has finished. Call reset() first."
            )

        previous_state = self.state
        action = self.backend.parse_action(previous_state, raw_action)
        next_state, transition_info = self.backend.apply_action(previous_state, action)
        reward = self.reward_fn.evaluate(
            previous_state=previous_state,
            action=action,
            next_state=next_state,
            transition_info=transition_info,
        )

        self._turn_count += 1
        terminated = self.backend.is_terminal(next_state)
        truncated = False
        info = dict(transition_info)

        if (
            self.config.max_turns is not None
            and self._turn_count >= self.config.max_turns
            and not terminated
        ):
            truncated = True
            info.setdefault("truncated_reason", "max_turns")

        self._episode_finished = terminated or truncated
        self._state = next_state
        observation = self.renderer.render(next_state)

        result = StepResult(
            observation=observation,
            reward=reward,
            accepted=True,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        self.trajectory.steps.append(
            TrajectoryStep(
                raw_action=raw_action,
                action=action,
                accepted=True,
                observation=observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
        )
        return result
