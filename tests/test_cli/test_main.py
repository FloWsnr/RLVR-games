"""CLI interaction tests."""

from io import StringIO
import sys

from _pytest.monkeypatch import MonkeyPatch
from rlvr_games.cli.main import run_cli, run_play_session
from rlvr_games.core import (
    EpisodeConfig,
    InvalidActionMode,
    InvalidActionPolicy,
)
from rlvr_games.games.chess import (
    ChessBoardOrientation,
    ChessEnv,
    ChessTextRendererKind,
    make_chess_env,
)
from rlvr_games.games.chess.scenarios import STANDARD_START_FEN


def make_env() -> ChessEnv:
    """Construct a chess environment for interactive CLI tests.

    Returns
    -------
    ChessEnv
        Fully wired chess environment instance.
    """
    return make_chess_env(
        initial_fen=STANDARD_START_FEN,
        config=EpisodeConfig(),
        text_renderer_kind=ChessTextRendererKind.ASCII,
        image_output_dir=None,
        image_size=360,
        image_coordinates=True,
        orientation=ChessBoardOrientation.WHITE,
    )


def test_run_play_session_handles_commands_and_moves() -> None:
    env = make_env()
    input_stream = StringIO("help\nlegal\ne2e4\ntrajectory\nquit\n")
    output_stream = StringIO()

    exit_code = run_play_session(
        env=env,
        seed=7,
        input_stream=input_stream,
        output_stream=output_stream,
    )

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "Reset info:" in output
    assert "Commands: help legal fen trajectory quit exit" in output
    assert "Legal actions (20):" in output
    assert "Move SAN: e4" in output
    assert "Trajectory steps: 1" in output
    assert "Session ended." in output


def test_run_play_session_reports_invalid_moves_without_state_change() -> None:
    env = make_env()
    input_stream = StringIO("e2e5\nquit\n")
    output_stream = StringIO()

    exit_code = run_play_session(
        env=env,
        seed=13,
        input_stream=input_stream,
        output_stream=output_stream,
    )

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "Invalid action:" in output
    assert env.state.fen == STANDARD_START_FEN
    assert len(env.trajectory.steps) == 0


def test_run_play_session_reports_penalized_invalid_moves_from_env_policy() -> None:
    env = make_chess_env(
        initial_fen=STANDARD_START_FEN,
        config=EpisodeConfig(
            invalid_action_policy=InvalidActionPolicy(
                mode=InvalidActionMode.PENALIZE_CONTINUE,
                penalty=-1.0,
            ),
        ),
        text_renderer_kind=ChessTextRendererKind.ASCII,
        image_output_dir=None,
        image_size=360,
        image_coordinates=True,
        orientation=ChessBoardOrientation.WHITE,
    )
    input_stream = StringIO("e2e5\ntrajectory\nquit\n")
    output_stream = StringIO()

    exit_code = run_play_session(
        env=env,
        seed=21,
        input_stream=input_stream,
        output_stream=output_stream,
    )

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "Accepted: False" in output
    assert "Reward: -1.0" in output
    assert "Trajectory steps: 1" in output
    assert "accepted=False" in output


def test_run_cli_can_start_and_exit_a_chess_play_session(
    monkeypatch: MonkeyPatch,
) -> None:
    input_stream = StringIO("quit\n")
    output_stream = StringIO()
    monkeypatch.setattr(sys, "stdin", input_stream)
    monkeypatch.setattr(sys, "stdout", output_stream)

    exit_code = run_cli(["play", "chess", "--seed", "5"])

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "Chess board:" in output
    assert "Session ended." in output


def test_run_cli_can_use_black_board_orientation(
    monkeypatch: MonkeyPatch,
) -> None:
    """CLI orientation controls the rendered text board perspective."""
    input_stream = StringIO("quit\n")
    output_stream = StringIO()
    monkeypatch.setattr(sys, "stdin", input_stream)
    monkeypatch.setattr(sys, "stdout", output_stream)

    exit_code = run_cli(["play", "chess", "--seed", "5", "--orientation", "black"])

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "1 R N B K Q B N R" in output
    assert "  h g f e d c b a" in output


def test_run_cli_can_use_penalize_truncate_invalid_action_policy(
    monkeypatch: MonkeyPatch,
) -> None:
    input_stream = StringIO("e2e5\n")
    output_stream = StringIO()
    monkeypatch.setattr(sys, "stdin", input_stream)
    monkeypatch.setattr(sys, "stdout", output_stream)

    exit_code = run_cli(
        [
            "play",
            "chess",
            "--seed",
            "8",
            "--invalid-action-policy",
            "penalize-truncate",
            "--invalid-action-penalty",
            "-2",
        ]
    )

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "Accepted: False" in output
    assert "Episode finished." in output


def test_run_cli_penalize_continue_respects_max_attempts(
    monkeypatch: MonkeyPatch,
) -> None:
    input_stream = StringIO("e2e5\ne2e4\n")
    output_stream = StringIO()
    monkeypatch.setattr(sys, "stdin", input_stream)
    monkeypatch.setattr(sys, "stdout", output_stream)

    exit_code = run_cli(
        [
            "play",
            "chess",
            "--seed",
            "8",
            "--invalid-action-policy",
            "penalize-continue",
            "--invalid-action-penalty",
            "-2",
            "--max-attempts",
            "2",
        ]
    )

    output = output_stream.getvalue()
    assert exit_code == 0
    assert '"truncated_reason": "max_attempts"' in output
    assert "Episode finished." in output
