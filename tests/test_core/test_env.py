"""Core environment tests."""

from dataclasses import dataclass
from typing import Any

import pytest

from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.exceptions import EnvironmentNotResetError
from rlvr_games.core.types import (
    EpisodeConfig,
    InvalidActionMode,
    InvalidActionPolicy,
    Observation,
    ParseResult,
)


@dataclass(slots=True)
class CounterState:
    value: int


@dataclass(slots=True)
class CounterAction:
    delta: int


class CounterScenario:
    def reset(self, *, seed: int) -> tuple[CounterState, dict[str, Any]]:
        return CounterState(value=0), {"scenario": "counter", "seed": seed}


class CounterRenderer:
    def render(self, state: CounterState) -> Observation:
        return Observation(text=f"value={state.value}", metadata={"value": state.value})


class CounterBackend:
    def parse_action(
        self, state: CounterState, raw_action: str
    ) -> ParseResult[CounterAction]:
        del state
        if raw_action == "bad":
            return ParseResult(action=None, error="bad action")
        return ParseResult(action=CounterAction(delta=int(raw_action)), error=None)

    def legal_actions(self, state: CounterState) -> list[str]:
        return ["1"]

    def apply_action(
        self, state: CounterState, action: CounterAction
    ) -> tuple[CounterState, dict[str, Any]]:
        next_state = CounterState(value=state.value + action.delta)
        return next_state, {"value": next_state.value}

    def is_terminal(self, state: CounterState) -> bool:
        return state.value >= 3


class CounterReward:
    def evaluate(
        self,
        *,
        previous_state: CounterState,
        action: CounterAction,
        next_state: CounterState,
        transition_info: dict[str, Any],
    ) -> float:
        return float(action.delta)


def test_step_requires_reset() -> None:
    env = TurnBasedEnv(
        backend=CounterBackend(),
        scenario=CounterScenario(),
        renderer=CounterRenderer(),
        reward_fn=CounterReward(),
        config=EpisodeConfig(),
    )

    with pytest.raises(EnvironmentNotResetError):
        env.step("1")


def test_records_trajectory_until_terminal() -> None:
    env = TurnBasedEnv(
        backend=CounterBackend(),
        scenario=CounterScenario(),
        renderer=CounterRenderer(),
        reward_fn=CounterReward(),
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
    assert env.trajectory.steps[-1].accepted is True
    assert env.trajectory.total_reward == 3.0
    assert len(env.trajectory.steps) == 3
    assert env.trajectory.steps[-1].observation.metadata["value"] == 3


def test_penalize_continue_records_rejected_attempt_without_state_transition() -> None:
    env = TurnBasedEnv(
        backend=CounterBackend(),
        scenario=CounterScenario(),
        renderer=CounterRenderer(),
        reward_fn=CounterReward(),
        config=EpisodeConfig(
            invalid_action_policy=InvalidActionPolicy(
                mode=InvalidActionMode.PENALIZE_CONTINUE,
                penalty=-2.0,
            )
        ),
    )
    env.reset(seed=1)

    rejected = env.step("bad")
    accepted = env.step("1")

    assert rejected.accepted is False
    assert rejected.reward == -2.0
    assert rejected.info["attempt_count"] == 1
    assert rejected.info["transition_count"] == 0
    assert rejected.observation.metadata["value"] == 0
    assert accepted.accepted is True
    assert accepted.info["attempt_count"] == 2
    assert accepted.info["transition_count"] == 1
    assert len(env.trajectory.steps) == 2
    assert env.trajectory.steps[0].action is None
    assert env.trajectory.steps[1].action is not None


def test_max_attempts_counts_penalized_invalid_attempts() -> None:
    env = TurnBasedEnv(
        backend=CounterBackend(),
        scenario=CounterScenario(),
        renderer=CounterRenderer(),
        reward_fn=CounterReward(),
        config=EpisodeConfig(
            max_attempts=2,
            invalid_action_policy=InvalidActionPolicy(
                mode=InvalidActionMode.PENALIZE_CONTINUE,
                penalty=-1.0,
            ),
        ),
    )
    env.reset(seed=2)

    first = env.step("bad")
    second = env.step("1")

    assert first.truncated is False
    assert second.accepted is True
    assert second.truncated is True
    assert second.info["truncated_reason"] == "max_attempts"


def test_max_transitions_only_counts_accepted_state_changes() -> None:
    env = TurnBasedEnv(
        backend=CounterBackend(),
        scenario=CounterScenario(),
        renderer=CounterRenderer(),
        reward_fn=CounterReward(),
        config=EpisodeConfig(
            max_transitions=1,
            invalid_action_policy=InvalidActionPolicy(
                mode=InvalidActionMode.PENALIZE_CONTINUE,
                penalty=-1.0,
            ),
        ),
    )
    env.reset(seed=3)

    rejected = env.step("bad")
    accepted = env.step("1")

    assert rejected.truncated is False
    assert accepted.accepted is True
    assert accepted.truncated is True
    assert accepted.info["truncated_reason"] == "max_transitions"
