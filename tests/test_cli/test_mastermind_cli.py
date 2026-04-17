"""Mastermind CLI interaction tests."""

from io import StringIO
from pathlib import Path
import sys

from _pytest.monkeypatch import MonkeyPatch

from rlvr_games.cli.main import run_cli, run_play_session
from rlvr_games.core import EpisodeConfig
from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.games.mastermind import (
    FixedCodeScenario,
    MastermindAction,
    MastermindState,
    TerminalOutcomeReward,
    make_mastermind_env,
)
from rlvr_games.games.mastermind.cli import MASTERMIND_CLI_SPEC


def make_env() -> TurnBasedEnv[MastermindState, MastermindAction]:
    """Construct a fixed-code Mastermind environment for CLI tests."""
    return make_mastermind_env(
        scenario=FixedCodeScenario(secret_code=(1, 1, 2, 2)),
        reward_fn=TerminalOutcomeReward(win_reward=1.0, loss_reward=-1.0),
        config=EpisodeConfig(),
        include_images=False,
        image_size=320,
    )


def test_run_play_session_handles_mastermind_commands_and_moves() -> None:
    env = make_env()
    input_stream = StringIO("show candidate_count\nguess 1 2 1 2\ntrajectory\nquit\n")
    output_stream = StringIO()

    exit_code = run_play_session(
        env=env,
        game_spec=MASTERMIND_CLI_SPEC,
        seed=7,
        image_output_dir=None,
        input_stream=input_stream,
        output_stream=output_stream,
    )

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "candidate_count: 1296" in output
    assert "Guess: 1 2 1 2" in output
    assert "Feedback: 2 black, 2 white" in output
    assert "Trajectory steps: 1" in output
    assert "Session ended." in output


def test_run_play_session_reports_invalid_moves_without_state_change() -> None:
    env = make_env()
    input_stream = StringIO("guess 1 2 3 7\nquit\n")
    output_stream = StringIO()

    exit_code = run_play_session(
        env=env,
        game_spec=MASTERMIND_CLI_SPEC,
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
        "show secret_code\nshow candidate_count\ndebug-show secret_code\nquit\n"
    )
    output_stream = StringIO()

    exit_code = run_play_session(
        env=env,
        game_spec=MASTERMIND_CLI_SPEC,
        seed=19,
        image_output_dir=None,
        input_stream=input_stream,
        output_stream=output_stream,
    )

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "Metadata key unavailable: secret_code" in output
    assert "candidate_count: 1296" in output
    assert "secret_code: [1, 1, 2, 2]" in output


def test_run_cli_can_start_and_exit_a_random_mastermind_play_session(
    monkeypatch: MonkeyPatch,
) -> None:
    input_stream = StringIO("quit\n")
    output_stream = StringIO()
    monkeypatch.setattr(sys, "stdin", input_stream)
    monkeypatch.setattr(sys, "stdout", output_stream)

    exit_code = run_cli(["play", "mastermind", "--seed", "5"])

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "Mastermind board:" in output
    assert "Session ended." in output


def test_run_cli_can_use_a_custom_fixed_code(monkeypatch: MonkeyPatch) -> None:
    input_stream = StringIO("quit\n")
    output_stream = StringIO()
    monkeypatch.setattr(sys, "stdin", input_stream)
    monkeypatch.setattr(sys, "stdout", output_stream)

    exit_code = run_cli(
        [
            "play",
            "mastermind",
            "--seed",
            "5",
            "--code",
            "1122",
        ]
    )

    output = output_stream.getvalue()
    assert exit_code == 0
    assert '"scenario": "fixed_code"' in output
    assert "Consistent candidates: 1296" in output
    assert "Session ended." in output


def test_run_play_session_persists_small_mastermind_images_when_requested(
    tmp_path: Path,
) -> None:
    env = make_mastermind_env(
        scenario=FixedCodeScenario(secret_code=(1, 1, 2, 2)),
        reward_fn=TerminalOutcomeReward(win_reward=1.0, loss_reward=-1.0),
        config=EpisodeConfig(),
        include_images=True,
        image_size=128,
    )
    input_stream = StringIO("quit\n")
    output_stream = StringIO()

    exit_code = run_play_session(
        env=env,
        game_spec=MASTERMIND_CLI_SPEC,
        seed=6,
        image_output_dir=tmp_path,
        input_stream=input_stream,
        output_stream=output_stream,
    )

    output = output_stream.getvalue()
    saved_paths = tuple(tmp_path.glob("*.png"))
    assert exit_code == 0
    assert "Image paths:" in output
    assert len(saved_paths) == 1
    assert saved_paths[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
