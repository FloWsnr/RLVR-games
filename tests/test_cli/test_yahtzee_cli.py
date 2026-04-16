"""Yahtzee CLI interaction tests."""

from io import StringIO
import sys

from _pytest.monkeypatch import MonkeyPatch

from rlvr_games.cli.main import run_cli, run_play_session
from rlvr_games.core import EpisodeConfig
from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.games.yahtzee import (
    ScoreDeltaReward,
    YahtzeeAction,
    YahtzeeState,
    make_yahtzee_env,
)
from rlvr_games.games.yahtzee.chance import YahtzeeChanceModel
from rlvr_games.games.yahtzee.cli import YAHTZEE_CLI_SPEC


def make_fixed_env() -> TurnBasedEnv[YahtzeeState, YahtzeeAction]:
    """Construct a fixed-state Yahtzee environment for CLI tests."""
    initial_state = YahtzeeState(
        dice=(6, 6, 6, 2, 2),
        rolls_used_in_turn=2,
        turns_completed=0,
        awaiting_roll=False,
        rng_state=YahtzeeChanceModel().initial_rng_state(seed=0),
    )
    return make_yahtzee_env(
        initial_state=initial_state,
        reward_fn=ScoreDeltaReward(),
        config=EpisodeConfig(),
        include_images=False,
        image_size=320,
    )


def test_run_play_session_handles_yahtzee_commands_and_scoring() -> None:
    env = make_fixed_env()
    input_stream = StringIO(
        "help\ndebug-legal\nstate\nshow total_score\nscore full-house\ntrajectory\nquit\n"
    )
    output_stream = StringIO()

    exit_code = run_play_session(
        env=env,
        game_spec=YAHTZEE_CLI_SPEC,
        seed=0,
        image_output_dir=None,
        input_stream=input_stream,
        output_stream=output_stream,
    )

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "Reset info:" in output
    assert "Yahtzee state:" in output
    assert "Legal actions (44):" in output
    assert "total_score: 0" in output
    assert "Scored: full-house = 25" in output
    assert "Next roll: 4 4 1 3 5" in output
    assert "Total score: 25" in output
    assert "Reset events: 0" in output
    assert "Trajectory steps: 1" in output
    assert "Session ended." in output


def test_run_cli_can_start_and_exit_a_yahtzee_play_session(
    monkeypatch: MonkeyPatch,
) -> None:
    input_stream = StringIO("quit\n")
    output_stream = StringIO()
    monkeypatch.setattr(sys, "stdin", input_stream)
    monkeypatch.setattr(sys, "stdout", output_stream)

    exit_code = run_cli(["play", "yahtzee", "--seed", "0"])

    output = output_stream.getvalue()
    assert exit_code == 0
    assert '"seed":' not in output
    assert "Yahtzee state:" in output
    assert "Session ended." in output
