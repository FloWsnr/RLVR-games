"""Connect 4 environment integration tests."""

import pytest

from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.exceptions import EpisodeFinishedError
from rlvr_games.core.types import EpisodeConfig
from rlvr_games.games.connect4 import (
    Connect4Action,
    Connect4State,
    FixedBoardScenario,
    RandomPositionScenario,
    TerminalOutcomeReward,
    make_connect4_env,
)
from tests.test_games.test_connect4.support import PRE_WIN_BOARD, X_WIN_BOARD


def make_reward() -> TerminalOutcomeReward:
    """Return a sparse terminal reward used by env tests."""
    return TerminalOutcomeReward(
        perspective="mover",
        win_reward=1.0,
        draw_reward=0.0,
        loss_reward=-1.0,
    )


def make_empty_start_env() -> TurnBasedEnv[Connect4State, Connect4Action]:
    """Return a standard Connect 4 environment from the empty board."""
    return make_connect4_env(
        scenario=RandomPositionScenario(
            rows=6,
            columns=7,
            connect_length=4,
            min_start_moves=0,
            max_start_moves=0,
        ),
        reward_fn=make_reward(),
        config=EpisodeConfig(),
        include_images=False,
        image_size=360,
    )


def test_horizontal_win_sequence_terminates_with_reward_and_metadata() -> None:
    env = make_connect4_env(
        scenario=FixedBoardScenario(
            initial_board=PRE_WIN_BOARD,
            connect_length=4,
        ),
        reward_fn=make_reward(),
        config=EpisodeConfig(),
        include_images=False,
        image_size=360,
    )
    env.reset(seed=1)

    result = env.step("4")

    assert result.accepted is True
    assert result.reward == 1.0
    assert result.terminated is True
    assert result.truncated is False
    assert result.info["player"] == "x"
    assert result.info["column"] == 4
    assert result.info["winner"] == "x"
    assert result.info["termination"] == "connect_length"
    assert env.state.board[5][:4] == ("x", "x", "x", "x")
    assert result.observation.metadata["is_terminal"] is True
    assert result.observation.metadata["winner"] == "x"


def test_terminal_reset_marks_episode_finished_and_rejects_steps() -> None:
    env = make_connect4_env(
        scenario=FixedBoardScenario(
            initial_board=X_WIN_BOARD,
            connect_length=4,
        ),
        reward_fn=make_reward(),
        config=EpisodeConfig(),
        include_images=False,
        image_size=360,
    )

    observation, _ = env.reset(seed=23)

    assert env.episode_finished is True
    assert observation.metadata["is_terminal"] is True
    assert observation.metadata["winner"] == "x"

    with pytest.raises(EpisodeFinishedError):
        env.step("1")


def test_env_records_trajectory_with_real_backend() -> None:
    env = make_empty_start_env()
    observation, info = env.reset(seed=123)

    result = env.step("1")

    assert info["scenario"] == "random_position"
    assert info["applied_start_moves"] == 0
    assert "Connect 4 board:" in (observation.text or "")
    assert result.reward == 0.0
    assert result.accepted is True
    assert len(env.trajectory.steps) == 1
    assert env.trajectory.steps[0].accepted is True
    assert env.trajectory.steps[0].action is not None
    assert env.trajectory.steps[0].action.label == "1"
    assert env.trajectory.steps[0].info["player"] == "x"
