"""Generic turn-based env tests."""

import pytest

from rlvr_games.core.exceptions import EnvironmentNotResetError
from rlvr_games.core.types import EpisodeConfig

from tests.test_core.support import CounterBackend, make_counter_env


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
