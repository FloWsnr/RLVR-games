"""Core invalid-action policy integration tests."""

import pytest

from rlvr_games.core import (
    EpisodeFinishedError,
    EpisodeConfig,
    InvalidActionError,
    InvalidActionMode,
    InvalidActionPolicy,
)
from rlvr_games.core.rollout import run_episode

from tests.test_core.support import (
    STANDARD_START_FEN,
    ScriptedAgent,
    make_chess_env_for_core_tests,
)


def test_raise_mode_preserves_existing_invalid_action_behavior() -> None:
    env = make_chess_env_for_core_tests(
        config=EpisodeConfig(
            invalid_action_policy=InvalidActionPolicy(
                mode=InvalidActionMode.RAISE,
                penalty=None,
            )
        ),
        initial_fen=STANDARD_START_FEN,
    )
    env.reset(seed=17)

    with pytest.raises(InvalidActionError):
        env.step("e2e5")

    assert env.state.fen == STANDARD_START_FEN
    assert len(env.trajectory.steps) == 0


def test_penalize_continue_records_rejected_attempt_and_keeps_episode_open() -> None:
    env = make_chess_env_for_core_tests(
        config=EpisodeConfig(
            invalid_action_policy=InvalidActionPolicy(
                mode=InvalidActionMode.PENALIZE_CONTINUE,
                penalty=-1.5,
            )
        ),
        initial_fen=STANDARD_START_FEN,
    )
    env.reset(seed=3)

    rejected = env.step("e2e5")
    accepted = env.step("e2e4")

    assert rejected.accepted is False
    assert rejected.reward == -1.5
    assert rejected.terminated is False
    assert rejected.truncated is False
    assert rejected.info["invalid_action"] is True
    assert env.trajectory.steps[0].action is None
    assert env.trajectory.steps[0].accepted is False
    assert env.trajectory.steps[1].accepted is True
    assert env.trajectory.steps[1].action is not None
    assert env.state.fen != STANDARD_START_FEN
    assert accepted.accepted is True


def test_penalize_truncate_records_rejected_attempt_and_finishes_episode() -> None:
    env = make_chess_env_for_core_tests(
        config=EpisodeConfig(
            invalid_action_policy=InvalidActionPolicy(
                mode=InvalidActionMode.PENALIZE_TRUNCATE,
                penalty=-2.0,
            )
        ),
        initial_fen=STANDARD_START_FEN,
    )
    env.reset(seed=5)

    result = env.step("e2e5")

    assert result.accepted is False
    assert result.reward == -2.0
    assert result.terminated is False
    assert result.truncated is True
    assert result.info["truncated_reason"] == "invalid_action"
    assert env.trajectory.steps[0].accepted is False

    with pytest.raises(EpisodeFinishedError):
        env.step("e2e4")


def test_run_episode_works_with_penalize_truncate_policy() -> None:
    env = make_chess_env_for_core_tests(
        config=EpisodeConfig(
            invalid_action_policy=InvalidActionPolicy(
                mode=InvalidActionMode.PENALIZE_TRUNCATE,
                penalty=-3.0,
            )
        ),
        initial_fen=STANDARD_START_FEN,
    )
    agent = ScriptedAgent(actions=["e2e5"])

    result = run_episode(env=env, agent=agent, seed=9)

    assert result.terminated is False
    assert result.truncated is True
    assert result.turn_count == 0
    assert result.trajectory.steps[0].accepted is False
    assert result.trajectory.steps[0].action is None
    assert result.trajectory.total_reward == -3.0


def test_run_episode_turn_count_only_counts_accepted_transitions() -> None:
    env = make_chess_env_for_core_tests(
        config=EpisodeConfig(
            max_attempts=2,
            invalid_action_policy=InvalidActionPolicy(
                mode=InvalidActionMode.PENALIZE_CONTINUE,
                penalty=-1.0,
            ),
        ),
        initial_fen=STANDARD_START_FEN,
    )
    agent = ScriptedAgent(actions=["e2e5", "e2e4"])

    result = run_episode(env=env, agent=agent, seed=4)

    assert result.terminated is False
    assert result.truncated is True
    assert len(result.trajectory.steps) == 2
    assert result.turn_count == 1
    assert result.trajectory.accepted_step_count == 1
