"""Tests for rollout turn preparation helpers."""

import pytest

from rlvr_games.core import (
    EpisodeConfig,
    TextMessagePart,
    prepare_turn,
)

from tests.test_core.support import CounterBackend, make_counter_env


def test_prepare_turn_packages_action_context_and_messages() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(),
    )

    observation, _ = env.reset(seed=7)
    prepared = prepare_turn(env=env, observation=observation)

    assert prepared.action_context.turn_index == 0
    assert len(prepared.messages) == 1
    text_part = prepared.messages[0].content[0]
    assert isinstance(text_part, TextMessagePart)
    assert text_part.text == "Observation:\nvalue=0"

    step_result = env.step("1")
    next_prepared = prepare_turn(env=env, observation=step_result.observation)

    assert next_prepared.action_context.turn_index == 1


def test_prepare_turn_rejects_finished_episodes() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(max_attempts=1),
    )

    env.reset(seed=9)
    step_result = env.step("1")

    assert step_result.truncated is True
    with pytest.raises(ValueError, match="finished episode"):
        prepare_turn(env=env, observation=step_result.observation)
