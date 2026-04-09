"""Canonical environment implementation for turn-based RLVR tasks."""

from typing import Generic, TypeVar

from rlvr_games.core.exceptions import EpisodeFinishedError, EnvironmentNotResetError
from rlvr_games.core.protocol import GameBackend, Renderer, RewardFn, Scenario
from rlvr_games.core.rewards import ZeroReward
from rlvr_games.core.trajectory import EpisodeTrajectory, TrajectoryStep
from rlvr_games.core.types import EpisodeConfig, Observation, StepResult

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


class TurnBasedEnv(Generic[StateT, ActionT]):
    """Minimal stateful environment with reset/step semantics."""

    def __init__(
        self,
        *,
        backend: GameBackend[StateT, ActionT],
        scenario: Scenario[StateT],
        renderer: Renderer[StateT],
        reward_fn: RewardFn[StateT, ActionT] | None = None,
        config: EpisodeConfig | None = None,
    ) -> None:
        self.backend = backend
        self.scenario = scenario
        self.renderer = renderer
        self.reward_fn = reward_fn or ZeroReward()
        self.config = config or EpisodeConfig()

        self._state: StateT | None = None
        self._trajectory: EpisodeTrajectory[ActionT] | None = None
        self._turn_count = 0
        self._episode_finished = False

    @property
    def state(self) -> StateT:
        """Return the current canonical state."""
        if self._state is None:
            raise EnvironmentNotResetError("Call reset() before accessing env.state.")
        return self._state

    @property
    def trajectory(self) -> EpisodeTrajectory[ActionT]:
        """Return the current episode trajectory."""
        if self._trajectory is None:
            raise EnvironmentNotResetError(
                "Call reset() before accessing env.trajectory."
            )
        return self._trajectory

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[Observation, dict[str, object]]:
        """Start a fresh episode."""
        effective_seed = self.config.seed if seed is None else seed
        self._turn_count = 0
        self._episode_finished = False
        self._state, info = self.scenario.reset(seed=effective_seed)
        observation = self.renderer.render(self._state)
        self._trajectory = EpisodeTrajectory(
            initial_observation=observation,
            reset_info=info,
        )
        return observation, info

    def step(self, raw_action: str) -> StepResult:
        """Advance the episode by one model action."""
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
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        self.trajectory.steps.append(
            TrajectoryStep(
                raw_action=raw_action,
                action=action,
                observation=observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
        )
        return result
