"""Minesweeper environment integration tests."""

import pytest

from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.exceptions import EpisodeFinishedError
from rlvr_games.core.types import EpisodeConfig
from rlvr_games.games.minesweeper import (
    MinesweeperAction,
    MinesweeperState,
    OutcomeReward,
    make_minesweeper_env,
    normalize_initial_board,
)

FIXED_BOARD = ("*..", "...", "..*")


def make_env() -> TurnBasedEnv[MinesweeperState, MinesweeperAction]:
    """Return a standard fixed-board Minesweeper environment."""
    return make_minesweeper_env(
        rows=3,
        columns=3,
        mine_count=2,
        initial_board=normalize_initial_board(board=FIXED_BOARD),
        reward_fn=OutcomeReward(win_reward=1.0, loss_reward=-1.0),
        config=EpisodeConfig(),
        include_images=False,
        image_size=240,
    )


def test_clearing_board_terminates_with_reward_and_public_metadata() -> None:
    env = make_env()
    observation, info = env.reset(seed=1)

    first_result = env.step("reveal 1 3")
    second_result = env.step("reveal 3 1")

    assert info["scenario"] == "fixed_board"
    assert "Minesweeper board:" in (observation.text or "")
    assert first_result.reward == 0.0
    assert first_result.terminated is False
    assert second_result.reward == 1.0
    assert second_result.terminated is True
    assert second_result.info["termination"] == "cleared"
    assert second_result.observation.metadata["won"] is True
    assert "hidden_board" not in second_result.observation.metadata


def test_hitting_a_mine_finishes_episode_and_rejects_further_steps() -> None:
    env = make_env()
    env.reset(seed=2)

    result = env.step("reveal 1 1")

    assert result.reward == -1.0
    assert result.terminated is True
    assert result.info["termination"] == "mine"

    with pytest.raises(EpisodeFinishedError):
        env.step("reveal 1 2")


def test_env_records_trajectory_and_keeps_hidden_board_debug_only() -> None:
    env = make_minesweeper_env(
        rows=3,
        columns=3,
        mine_count=2,
        initial_board=None,
        reward_fn=OutcomeReward(win_reward=1.0, loss_reward=-1.0),
        config=EpisodeConfig(),
        include_images=False,
        image_size=240,
    )
    env.reset(seed=7)

    result = env.step("reveal 2 2")

    assert result.accepted is True
    assert len(env.trajectory.steps) == 1
    assert env.trajectory.steps[0].action is not None
    assert env.trajectory.steps[0].action.label == "reveal 2 2"
    assert env.inspect_state()["hidden_board"] is not None
    assert "hidden_board" not in result.observation.metadata
