"""Minesweeper CLI interaction tests."""

from io import StringIO
import sys

from _pytest.monkeypatch import MonkeyPatch

from rlvr_games.cli.main import run_cli, run_play_session
from rlvr_games.core import EpisodeConfig
from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.games.minesweeper import (
    MinesweeperAction,
    MinesweeperState,
    OutcomeReward,
    make_minesweeper_env,
    normalize_initial_board,
)
from rlvr_games.games.minesweeper.cli import MINESWEEPER_CLI_SPEC

FIXED_BOARD = ("*..", "...", "..*")


def make_env() -> TurnBasedEnv[MinesweeperState, MinesweeperAction]:
    """Construct a fixed-board Minesweeper environment for CLI tests."""
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


def test_run_play_session_handles_minesweeper_commands_and_moves() -> None:
    env = make_env()
    input_stream = StringIO(
        "help\ndebug-legal\nshow remaining_safe_cells\nreveal 1 3\ntrajectory\nquit\n"
    )
    output_stream = StringIO()

    exit_code = run_play_session(
        env=env,
        game_spec=MINESWEEPER_CLI_SPEC,
        seed=7,
        image_output_dir=None,
        input_stream=input_stream,
        output_stream=output_stream,
    )

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "Reset info:" in output
    assert (
        "Commands: help state show <key> debug-state debug-show <key> "
        "debug-legal trajectory quit exit"
    ) in output
    assert "Legal actions (18):" in output
    assert "remaining_safe_cells: 7" in output
    assert "Action: reveal row=1 col=3" in output
    assert "Newly revealed: 4" in output
    assert "Trajectory steps: 1" in output
    assert "Session ended." in output


def test_run_play_session_reports_invalid_moves_without_state_change() -> None:
    env = make_env()
    input_stream = StringIO("unflag 1 1\nquit\n")
    output_stream = StringIO()

    exit_code = run_play_session(
        env=env,
        game_spec=MINESWEEPER_CLI_SPEC,
        seed=13,
        image_output_dir=None,
        input_stream=input_stream,
        output_stream=output_stream,
    )

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "Invalid action:" in output
    assert len(env.trajectory.steps) == 0


def test_run_play_session_separates_public_and_debug_state_views() -> None:
    env = make_env()
    input_stream = StringIO(
        "show hidden_board\n"
        "show legal_action_count\n"
        "debug-show hidden_board\n"
        "debug-show legal_action_count\n"
        "quit\n"
    )
    output_stream = StringIO()

    exit_code = run_play_session(
        env=env,
        game_spec=MINESWEEPER_CLI_SPEC,
        seed=19,
        image_output_dir=None,
        input_stream=input_stream,
        output_stream=output_stream,
    )

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "Metadata key unavailable: hidden_board" in output
    assert "Metadata key unavailable: legal_action_count" in output
    assert 'hidden_board: ["*..", "...", "..*"]' in output
    assert "legal_action_count: 18" in output


def test_run_cli_can_start_and_exit_a_random_minesweeper_play_session(
    monkeypatch: MonkeyPatch,
) -> None:
    input_stream = StringIO("quit\n")
    output_stream = StringIO()
    monkeypatch.setattr(sys, "stdin", input_stream)
    monkeypatch.setattr(sys, "stdout", output_stream)

    exit_code = run_cli(["play", "minesweeper", "--seed", "5"])

    output = output_stream.getvalue()
    assert exit_code == 0
    assert '"seed":' not in output
    assert "Minesweeper board:" in output
    assert "Session ended." in output


def test_run_cli_can_use_a_custom_minesweeper_board(monkeypatch: MonkeyPatch) -> None:
    input_stream = StringIO("quit\n")
    output_stream = StringIO()
    monkeypatch.setattr(sys, "stdin", input_stream)
    monkeypatch.setattr(sys, "stdout", output_stream)

    exit_code = run_cli(
        [
            "play",
            "minesweeper",
            "--seed",
            "5",
            "--board",
            "*../.../..*",
        ]
    )

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "Mines: 2" in output
    assert "Session ended." in output
