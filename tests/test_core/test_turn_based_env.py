"""Generic turn-based env tests."""

from dataclasses import dataclass

import pytest

from rlvr_games.core import AutoAction, EpisodeBoundary
from rlvr_games.core.exceptions import EnvironmentNotResetError
from rlvr_games.core.protocol import GameBackend
from rlvr_games.core.types import EpisodeConfig

from tests.test_core.support import (
    CounterAction,
    CounterBackend,
    CounterState,
    make_counter_env,
)


@dataclass(slots=True, frozen=True)
class MacroCounterReward:
    """Reward the total value change across one env step."""

    def evaluate(
        self,
        *,
        previous_state: CounterState,
        action: CounterAction,
        next_state: CounterState,
        transition_info: dict[str, object],
    ) -> float:
        """Return the net state delta across the full step."""
        del action
        del transition_info
        return float(next_state.value - previous_state.value)


class CounterAutoAdvancePolicy:
    """Auto-advance counter states until control returns on even values."""

    def reset(self, *, initial_state: CounterState) -> None:
        """Start a new episode with no extra policy state."""
        del initial_state

    def is_agent_turn(self, *, state: CounterState) -> bool:
        """Return whether the counter is at an even-valued agent turn."""
        return state.value % 2 == 0

    def select_internal_action(
        self,
        *,
        state: CounterState,
        backend: GameBackend[CounterState, CounterAction],
    ) -> AutoAction[CounterAction] | None:
        """Return one internal increment on odd-valued states."""
        del state
        del backend
        return AutoAction(
            source="opponent",
            raw_action="1",
            action=CounterAction(delta=1),
        )

    def episode_boundary(self, *, state: CounterState) -> EpisodeBoundary | None:
        """Return no additional task boundary for the counter tests."""
        del state
        return None


class CounterSecondPlayerAutoAdvancePolicy(CounterAutoAdvancePolicy):
    """Auto-advance one opening move before the agent can act."""

    def is_agent_turn(self, *, state: CounterState) -> bool:
        """Return whether the counter is at an odd-valued agent turn."""
        return state.value % 2 == 1


def test_step_requires_reset() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(),
    )

    with pytest.raises(EnvironmentNotResetError):
        env.step("1")


def test_records_trajectory_until_terminal() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(max_transitions=10),
    )

    observation, info = env.reset(seed=7)

    assert observation.text == "value=0"
    assert info["seed"] == 7

    first = env.step("1")
    second = env.step("1")
    third = env.step("1")

    assert first.terminated is False
    assert second.terminated is False
    assert third.terminated is True
    assert first.accepted is True
    assert env.legal_actions() == ("1",)
    assert env.inspect_state()["value"] == 3
    assert env.trajectory.steps[-1].accepted is True
    assert env.trajectory.total_reward == 3.0
    assert len(env.trajectory.steps) == 3
    assert env.trajectory.steps[-1].observation.metadata["value"] == 3


def test_reset_can_auto_advance_before_the_first_agent_turn() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(),
        auto_advance_policy=CounterSecondPlayerAutoAdvancePolicy(),
    )

    observation, info = env.reset(seed=8)
    initial_transitions = info["initial_transitions"]

    assert observation.metadata["value"] == 1
    assert env.state.value == 1
    assert info["auto_advanced"] is True
    assert info["transition_count"] == 1
    assert info["transition_count_delta"] == 1
    assert isinstance(initial_transitions, tuple)
    assert len(initial_transitions) == 1
    assert initial_transitions[0]["source"] == "opponent"
    assert env.trajectory.initial_observation.metadata["value"] == 1


def test_step_can_auto_advance_internal_transitions_before_returning() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(),
        reward_fn=MacroCounterReward(),
        auto_advance_policy=CounterAutoAdvancePolicy(),
    )
    env.reset(seed=4)

    result = env.step("1")

    assert result.accepted is True
    assert result.reward == 2.0
    assert result.observation.metadata["value"] == 2
    assert env.state.value == 2
    assert result.info["transition_count"] == 2
    assert result.info["transition_count_delta"] == 2
    assert result.info["auto_advanced"] is True
    assert len(env.trajectory.steps) == 1
    assert len(env.trajectory.steps[0].transitions) == 2
    assert env.trajectory.steps[0].transitions[0].source == "agent"
    assert env.trajectory.steps[0].transitions[1].source == "opponent"


def test_step_applies_auto_advance_before_max_attempts_truncation() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(max_attempts=1),
        reward_fn=MacroCounterReward(),
        auto_advance_policy=CounterAutoAdvancePolicy(),
    )
    env.reset(seed=9)

    result = env.step("1")

    assert result.accepted is True
    assert result.reward == 2.0
    assert result.truncated is True
    assert result.info["truncated_reason"] == "max_attempts"
    assert result.observation.metadata["value"] == 2
    assert env.state.value == 2
    assert result.info["transition_count_delta"] == 2
    assert env.trajectory.steps[0].transitions[1].source == "opponent"


def test_step_applies_auto_advance_before_max_transitions_truncation() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(max_transitions=1),
        reward_fn=MacroCounterReward(),
        auto_advance_policy=CounterAutoAdvancePolicy(),
    )
    env.reset(seed=10)

    result = env.step("1")

    assert result.accepted is True
    assert result.reward == 2.0
    assert result.truncated is True
    assert result.info["truncated_reason"] == "max_transitions"
    assert result.observation.metadata["value"] == 2
    assert env.state.value == 2
    assert result.info["transition_count"] == 2
    assert len(env.trajectory.steps[0].transitions) == 2


def test_trajectory_snapshots_do_not_alias_returned_results() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(),
    )

    observation, _ = env.reset(seed=5)
    step_result = env.step("1")

    observation.metadata["value"] = 99
    step_result.info["value"] = 42
    step_result.observation.metadata["value"] = 77

    assert env.trajectory.initial_observation.metadata["value"] == 0
    assert env.trajectory.steps[0].info["value"] == 1
    assert env.trajectory.steps[0].observation.metadata["value"] == 1
