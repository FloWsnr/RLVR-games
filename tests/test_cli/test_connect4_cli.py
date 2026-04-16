"""Connect 4 CLI interaction tests."""

from io import StringIO
import sys

from _pytest.monkeypatch import MonkeyPatch
import pytest

from rlvr_games.cli.main import build_parser, run_cli, run_play_session
from rlvr_games.core import EpisodeConfig
from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.games.connect4 import (
    Connect4Action,
    Connect4SolverAutoAdvancePolicy,
    Connect4State,
    FixedBoardScenario,
    RandomPositionScenario,
    SolverMoveScoreReward,
    TerminalOutcomeReward,
    make_connect4_env,
)
from rlvr_games.games.connect4.cli import CONNECT4_CLI_SPEC, build_connect4_environment
from tests.test_games.test_connect4.support import (
    FULL_FIRST_COLUMN_BOARD,
    X_WIN_BOARD,
)


def make_reward() -> TerminalOutcomeReward:
    """Return a sparse terminal reward for Connect 4 CLI tests."""
    return TerminalOutcomeReward(
        perspective="mover",
        win_reward=1.0,
        draw_reward=0.0,
        loss_reward=-1.0,
    )


def make_env() -> TurnBasedEnv[Connect4State, Connect4Action]:
    """Return a standard Connect 4 environment for interactive CLI tests."""
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


def test_run_play_session_handles_connect4_commands_and_moves() -> None:
    env = make_env()
    input_stream = StringIO(
        "help\ndebug-legal\nshow current_player\n1\ntrajectory\nquit\n"
    )
    output_stream = StringIO()

    exit_code = run_play_session(
        env=env,
        game_spec=CONNECT4_CLI_SPEC,
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
    assert "Legal actions (7): 1 2 3 4 5 6 7" in output
    assert "current_player: x" in output
    assert "Player: x" in output
    assert "Column: 1" in output
    assert "Trajectory steps: 1" in output
    assert "Session ended." in output


def test_run_play_session_reports_invalid_moves_without_state_change() -> None:
    env = make_connect4_env(
        scenario=FixedBoardScenario(
            initial_board=FULL_FIRST_COLUMN_BOARD,
            connect_length=4,
        ),
        reward_fn=make_reward(),
        config=EpisodeConfig(),
        include_images=False,
        image_size=360,
    )
    input_stream = StringIO("1\nquit\n")
    output_stream = StringIO()

    exit_code = run_play_session(
        env=env,
        game_spec=CONNECT4_CLI_SPEC,
        seed=13,
        image_output_dir=None,
        input_stream=input_stream,
        output_stream=output_stream,
    )

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "Invalid action:" in output
    assert env.state.board == FULL_FIRST_COLUMN_BOARD
    assert len(env.trajectory.steps) == 0


def test_run_play_session_finishes_immediately_for_terminal_connect4_positions() -> (
    None
):
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
    input_stream = StringIO("quit\n")
    output_stream = StringIO()

    exit_code = run_play_session(
        env=env,
        game_spec=CONNECT4_CLI_SPEC,
        seed=2,
        image_output_dir=None,
        input_stream=input_stream,
        output_stream=output_stream,
    )

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "Terminal: yes" in output
    assert "Winner: x" in output
    assert "Episode finished." in output
    assert "turn[0]>" not in output


def test_run_cli_can_start_and_exit_a_connect4_play_session(
    monkeypatch: MonkeyPatch,
) -> None:
    input_stream = StringIO("quit\n")
    output_stream = StringIO()
    monkeypatch.setattr(sys, "stdin", input_stream)
    monkeypatch.setattr(sys, "stdout", output_stream)

    exit_code = run_cli(["play", "connect4", "--seed", "5", "--max-start-moves", "0"])

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "Connect 4 board:" in output
    assert "Session ended." in output


def test_run_cli_can_use_a_custom_connect4_board(monkeypatch: MonkeyPatch) -> None:
    input_stream = StringIO("quit\n")
    output_stream = StringIO()
    monkeypatch.setattr(sys, "stdin", input_stream)
    monkeypatch.setattr(sys, "stdout", output_stream)

    exit_code = run_cli(
        [
            "play",
            "connect4",
            "--seed",
            "5",
            "--board",
            "......./......./......./......./ooo..../xxx....",
        ]
    )

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "| x x x . . . . |" in output
    assert "Session ended." in output


def test_run_cli_solver_opponent_auto_replies(monkeypatch: MonkeyPatch) -> None:
    input_stream = StringIO("4\nquit\n")
    output_stream = StringIO()
    monkeypatch.setattr(sys, "stdin", input_stream)
    monkeypatch.setattr(sys, "stdout", output_stream)

    exit_code = run_cli(
        [
            "play",
            "connect4",
            "--seed",
            "5",
            "--max-start-moves",
            "0",
            "--opponent",
            "solver",
        ]
    )

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "Agent Column: 4" in output
    assert "Opponent Column: 4" in output
    assert '"transition_count_delta": 2' in output


def test_build_connect4_environment_can_use_solver_move_dense_reward() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "play",
            "connect4",
            "--reward",
            "solver-move-dense",
        ]
    )

    env = build_connect4_environment(args, parser)

    assert isinstance(env, TurnBasedEnv)
    assert isinstance(env.reward_fn, SolverMoveScoreReward)
    assert env.reward_fn.perspective == "mover"


def test_build_connect4_environment_can_use_solver_opponent() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "play",
            "connect4",
            "--opponent",
            "solver",
        ]
    )

    env = build_connect4_environment(args, parser)

    assert isinstance(env, TurnBasedEnv)
    assert isinstance(env.auto_advance_policy, Connect4SolverAutoAdvancePolicy)


def test_connect4_cli_rejects_removed_rows_option() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "play",
                "connect4",
                "--rows",
                "5",
            ]
        )


def test_build_connect4_environment_rejects_non_standard_fixed_board() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "play",
            "connect4",
            "--board",
            "......./......./......./......./.......",
        ]
    )

    with pytest.raises(SystemExit):
        build_connect4_environment(args, parser)
