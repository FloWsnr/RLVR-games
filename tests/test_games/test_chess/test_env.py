"""Chess environment integration tests."""

import chess
import pytest

from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.exceptions import EpisodeFinishedError
from rlvr_games.core.types import EpisodeConfig
from rlvr_games.games.chess import (
    AsciiBoardFormatter,
    ChessBackend,
    ChessObservationRenderer,
    ChessStateInspector,
    StartingPositionScenario,
    TerminalOutcomeReward,
)
from rlvr_games.games.chess.scenarios import STANDARD_START_FEN

PROMOTION_FEN = "k7/4P3/8/8/8/8/8/7K w - - 0 1"
TERMINAL_FEN = "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"
STATE_INSPECTOR = ChessStateInspector()


def make_renderer() -> ChessObservationRenderer:
    """Return the standard text-only chess renderer."""
    return ChessObservationRenderer(
        board_formatter=AsciiBoardFormatter(orientation=chess.WHITE),
        image_renderer=None,
    )


def make_reward() -> TerminalOutcomeReward:
    """Return a sparse terminal reward used by env tests."""
    return TerminalOutcomeReward(
        perspective="white",
        win_reward=1.0,
        draw_reward=0.0,
        loss_reward=-1.0,
    )


def test_checkmate_sequence_terminates_with_winner_metadata() -> None:
    env = TurnBasedEnv(
        backend=ChessBackend(),
        scenario=StartingPositionScenario(),
        renderer=make_renderer(),
        state_inspector=STATE_INSPECTOR,
        reward_fn=make_reward(),
        config=EpisodeConfig(),
    )
    env.reset(seed=11)

    for raw_action in ["f2f3", "e7e5", "g2g4"]:
        result = env.step(raw_action)
        assert result.terminated is False

    final_result = env.step("d8h4")

    assert final_result.terminated is True
    assert final_result.truncated is False
    assert final_result.accepted is True
    assert final_result.info["move_san"] == "Qh4#"
    assert final_result.info["winner"] == "black"
    assert final_result.info["result"] == "0-1"
    assert final_result.info["termination"] == "checkmate"
    assert final_result.observation.metadata["is_terminal"] is True
    assert final_result.observation.metadata["winner"] == "black"


def test_threefold_repetition_terminates_on_the_third_occurrence() -> None:
    env = TurnBasedEnv(
        backend=ChessBackend(),
        scenario=StartingPositionScenario(),
        renderer=make_renderer(),
        state_inspector=STATE_INSPECTOR,
        reward_fn=make_reward(),
        config=EpisodeConfig(),
    )
    env.reset(seed=5)

    repeated_knight_moves = [
        "g1f3",
        "g8f6",
        "f3g1",
        "f6g8",
        "g1f3",
        "g8f6",
        "f3g1",
    ]
    for raw_action in repeated_knight_moves:
        result = env.step(raw_action)
        assert result.terminated is False

    final_result = env.step("f6g8")

    assert final_result.terminated is True
    assert final_result.info["winner"] is None
    assert final_result.info["result"] == "1/2-1/2"
    assert final_result.info["termination"] == "threefold_repetition"
    assert final_result.info["repetition_count"] == 3


def test_custom_valid_fen_reset_is_normalized() -> None:
    env = TurnBasedEnv(
        backend=ChessBackend(),
        scenario=StartingPositionScenario(initial_fen=PROMOTION_FEN),
        renderer=make_renderer(),
        state_inspector=STATE_INSPECTOR,
        reward_fn=make_reward(),
        config=EpisodeConfig(),
    )

    observation, info = env.reset(seed=17)

    assert info["scenario"] == "fen_position"
    assert info["initial_fen"] == PROMOTION_FEN
    assert observation.metadata["fen"] == PROMOTION_FEN
    assert observation.metadata["side_to_move"] == "white"


def test_terminal_reset_marks_episode_finished_and_rejects_steps() -> None:
    env = TurnBasedEnv(
        backend=ChessBackend(),
        scenario=StartingPositionScenario(initial_fen=TERMINAL_FEN),
        renderer=make_renderer(),
        state_inspector=STATE_INSPECTOR,
        reward_fn=make_reward(),
        config=EpisodeConfig(),
    )

    observation, _ = env.reset(seed=23)

    assert env.episode_finished is True
    assert observation.metadata["is_terminal"] is True

    with pytest.raises(EpisodeFinishedError):
        env.step("h8g8")


def test_invalid_fen_reset_fails_fast() -> None:
    env = TurnBasedEnv(
        backend=ChessBackend(),
        scenario=StartingPositionScenario(initial_fen="not-a-fen"),
        renderer=make_renderer(),
        state_inspector=STATE_INSPECTOR,
        reward_fn=make_reward(),
        config=EpisodeConfig(),
    )

    with pytest.raises(ValueError):
        env.reset(seed=3)


def test_env_records_trajectory_with_real_backend() -> None:
    env = TurnBasedEnv(
        backend=ChessBackend(),
        scenario=StartingPositionScenario(),
        renderer=make_renderer(),
        state_inspector=STATE_INSPECTOR,
        reward_fn=make_reward(),
        config=EpisodeConfig(),
    )
    observation, info = env.reset(seed=123)

    result = env.step("e2e4")

    assert info["seed"] == 123
    assert STANDARD_START_FEN in (observation.text or "")
    assert result.reward == 0.0
    assert result.accepted is True
    assert len(env.trajectory.steps) == 1
    assert env.trajectory.steps[0].accepted is True
    assert env.trajectory.steps[0].action is not None
    assert env.trajectory.steps[0].action.uci == "e2e4"
    assert env.trajectory.steps[0].info["move_san"] == "e4"
