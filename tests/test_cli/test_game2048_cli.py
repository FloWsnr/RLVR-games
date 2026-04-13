"""2048 CLI interaction tests."""

from io import StringIO
import sys

from _pytest.monkeypatch import MonkeyPatch

from rlvr_games.cli.main import run_cli, run_play_session
from rlvr_games.core import EpisodeConfig
from rlvr_games.games.game2048 import Game2048Env, TargetTileReward, make_game2048_env
from rlvr_games.games.game2048.cli import GAME2048_CLI_SPEC

NOOP_UP_BOARD = (
    (2, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
)
NO_MOVES_BOARD = (
    (2, 4, 2, 4),
    (4, 2, 4, 2),
    (2, 4, 2, 4),
    (4, 2, 4, 8),
)


def make_env() -> Game2048Env:
    """Construct a 2048 environment for interactive CLI tests.

    Returns
    -------
    object
        Fully wired 2048 environment instance.
    """
    return make_game2048_env(
        size=4,
        target_value=2048,
        initial_board=None,
        initial_score=0,
        initial_move_count=0,
        reward_fn=TargetTileReward(),
        config=EpisodeConfig(),
        include_images=False,
        image_size=360,
    )


def test_run_play_session_handles_2048_commands_and_moves() -> None:
    env = make_env()
    input_stream = StringIO(
        "help\nlegal\nstate\nshow max_tile\nleft\ntrajectory\nquit\n"
    )
    output_stream = StringIO()

    exit_code = run_play_session(
        env=env,
        game_spec=GAME2048_CLI_SPEC,
        seed=0,
        image_output_dir=None,
        input_stream=input_stream,
        output_stream=output_stream,
    )

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "Reset info:" in output
    assert "Commands: help legal state show <key> trajectory quit exit" in output
    assert "Legal actions (4): up right down left" in output
    assert "State:" in output
    assert "max_tile: 2" in output
    assert "Direction: left" in output
    assert "Score gain: 0" in output
    assert "Trajectory steps: 1" in output
    assert "Session ended." in output


def test_run_play_session_reports_invalid_2048_moves_without_state_change() -> None:
    env = make_game2048_env(
        size=4,
        target_value=2048,
        initial_board=NOOP_UP_BOARD,
        initial_score=0,
        initial_move_count=0,
        reward_fn=TargetTileReward(),
        config=EpisodeConfig(),
        include_images=False,
        image_size=360,
    )
    input_stream = StringIO("up\nquit\n")
    output_stream = StringIO()

    exit_code = run_play_session(
        env=env,
        game_spec=GAME2048_CLI_SPEC,
        seed=13,
        image_output_dir=None,
        input_stream=input_stream,
        output_stream=output_stream,
    )

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "Invalid action:" in output
    assert env.state.board == NOOP_UP_BOARD
    assert len(env.trajectory.steps) == 0


def test_run_play_session_finishes_immediately_for_terminal_2048_positions() -> None:
    env = make_game2048_env(
        size=4,
        target_value=2048,
        initial_board=NO_MOVES_BOARD,
        initial_score=0,
        initial_move_count=0,
        reward_fn=TargetTileReward(),
        config=EpisodeConfig(),
        include_images=False,
        image_size=360,
    )
    input_stream = StringIO("quit\n")
    output_stream = StringIO()

    exit_code = run_play_session(
        env=env,
        game_spec=GAME2048_CLI_SPEC,
        seed=2,
        image_output_dir=None,
        input_stream=input_stream,
        output_stream=output_stream,
    )

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "Terminal: yes" in output
    assert "Termination: no_moves" in output
    assert "Episode finished." in output
    assert "turn[0]>" not in output


def test_run_cli_can_start_and_exit_a_2048_play_session(
    monkeypatch: MonkeyPatch,
) -> None:
    input_stream = StringIO("quit\n")
    output_stream = StringIO()
    monkeypatch.setattr(sys, "stdin", input_stream)
    monkeypatch.setattr(sys, "stdout", output_stream)

    exit_code = run_cli(["play", "2048", "--seed", "5"])

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "2048 board:" in output
    assert "Session ended." in output


def test_run_cli_can_use_a_custom_2048_board(monkeypatch: MonkeyPatch) -> None:
    input_stream = StringIO("quit\n")
    output_stream = StringIO()
    monkeypatch.setattr(sys, "stdin", input_stream)
    monkeypatch.setattr(sys, "stdout", output_stream)

    exit_code = run_cli(
        [
            "play",
            "2048",
            "--seed",
            "5",
            "--board",
            "2,0,0,0/0,2,0,0/0,0,0,0/0,0,0,0",
        ]
    )

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "|     2|" in output
    assert "Session ended." in output
