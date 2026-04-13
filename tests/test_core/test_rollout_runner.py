"""Rollout runner tests."""

from rlvr_games.core import EpisodeConfig
from rlvr_games.core.rollout import run_episode

from tests.test_core.support import (
    STANDARD_START_FEN,
    ScriptedAgent,
    make_chess_env_for_core_tests,
)

TERMINAL_FEN = "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"


def test_run_episode_records_scripted_checkmate_trajectory() -> None:
    env = make_chess_env_for_core_tests(
        config=EpisodeConfig(),
        initial_fen=STANDARD_START_FEN,
    )
    agent = ScriptedAgent(actions=["f2f3", "e7e5", "g2g4", "d8h4"])

    result = run_episode(env=env, agent=agent, seed=11)

    assert result.terminated is True
    assert result.truncated is False
    assert result.turn_count == 4
    assert result.trajectory.reset_info["seed"] == 11
    assert len(result.trajectory.steps) == 4
    assert result.trajectory.steps[-1].info["termination"] == "checkmate"
    assert result.trajectory.steps[-1].info["winner"] == "black"
    assert agent.contexts[0].turn_index == 0
    assert "f2f3" in agent.contexts[0].legal_actions


def test_run_episode_passes_sorted_legal_actions_in_context() -> None:
    env = make_chess_env_for_core_tests(
        config=EpisodeConfig(max_transitions=1),
        initial_fen=STANDARD_START_FEN,
    )
    agent = ScriptedAgent(actions=["e2e4"])

    result = run_episode(env=env, agent=agent, seed=3)

    assert result.terminated is False
    assert result.truncated is True
    assert result.turn_count == 1
    assert agent.contexts[0].legal_actions == tuple(
        sorted(agent.contexts[0].legal_actions)
    )
    assert len(agent.contexts[0].legal_actions) == 20
    assert result.trajectory.steps[0].accepted is True
    assert result.trajectory.steps[0].action is not None
    assert result.trajectory.steps[0].action.uci == "e2e4"


def test_run_episode_finishes_immediately_for_terminal_reset_positions() -> None:
    env = make_chess_env_for_core_tests(
        config=EpisodeConfig(),
        initial_fen=TERMINAL_FEN,
    )
    agent = ScriptedAgent(actions=[])

    result = run_episode(env=env, agent=agent, seed=19)

    assert result.terminated is True
    assert result.truncated is False
    assert result.turn_count == 0
    assert len(result.trajectory.steps) == 0
