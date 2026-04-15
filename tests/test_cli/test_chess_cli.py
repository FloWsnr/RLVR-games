"""Chess CLI tests."""

from io import StringIO
from pathlib import Path
import sys

from _pytest.monkeypatch import MonkeyPatch
import pytest

from rlvr_games.cli.main import build_parser, run_cli, run_play_session
from rlvr_games.core import (
    AutoAction,
    EpisodeConfig,
    InvalidActionMode,
    InvalidActionPolicy,
)
from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.games.chess import (
    ChessAction,
    ChessBackend,
    ChessBoardOrientation,
    ChessState,
    ChessTextRendererKind,
    EngineEvalDenseReward,
    TerminalOutcomeReward,
    make_chess_env,
)
from rlvr_games.games.chess.cli import CHESS_CLI_SPEC, build_chess_environment
from rlvr_games.games.chess.scenarios import (
    STANDARD_START_FEN,
    StartingPositionScenario,
)

TERMINAL_FEN = "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"
ENGINE_REWARD_ARGS = (
    "--reward",
    "engine-eval-dense",
    "--engine-depth",
    "12",
    "--engine-mate-score",
    "100000",
)


class StubEvaluator:
    """Minimal evaluator stub for CLI reward construction tests."""

    def evaluate(self, *, state: object, perspective: str) -> float:
        """Return a constant dummy evaluation."""
        del state
        del perspective
        return 0.0

    def close(self) -> None:
        """Provide a no-op close hook for reward cleanup tests."""


class StubMoveSelector:
    """Minimal move selector stub for CLI auto-reply tests."""

    def select_action(
        self,
        *,
        state: ChessState,
        backend: ChessBackend,
    ) -> AutoAction[ChessAction]:
        """Return a deterministic legal reply from the initial position."""
        parse_result = backend.parse_action(state, "e7e5")
        return AutoAction(
            source="opponent",
            raw_action="e7e5",
            action=parse_result.require_action(),
        )

    def close(self) -> None:
        """Provide a no-op close hook for move-selector cleanup tests."""


def patch_stockfish_runtime(monkeypatch: MonkeyPatch) -> None:
    """Replace Stockfish construction with pure in-memory stubs."""
    monkeypatch.setattr(
        "rlvr_games.games.chess.cli._build_stockfish_evaluator",
        lambda *, args: StubEvaluator(),
    )
    monkeypatch.setattr(
        "rlvr_games.games.chess.cli._build_stockfish_move_selector",
        lambda *, args: StubMoveSelector(),
    )


def make_reward() -> TerminalOutcomeReward:
    """Return a sparse terminal reward for chess CLI tests."""
    return TerminalOutcomeReward(
        perspective="white",
        win_reward=1.0,
        draw_reward=0.0,
        loss_reward=-1.0,
    )


def make_env() -> TurnBasedEnv[ChessState, ChessAction]:
    """Return a standard chess environment for interactive CLI tests."""
    return make_chess_env(
        scenario=StartingPositionScenario(initial_fen=STANDARD_START_FEN),
        reward_fn=make_reward(),
        config=EpisodeConfig(),
        text_renderer_kind=ChessTextRendererKind.ASCII,
        include_images=False,
        image_size=360,
        image_coordinates=True,
        orientation=ChessBoardOrientation.WHITE,
    )


def fixture_puzzle_manifest_path() -> Path:
    """Return the checked-in processed Lichess puzzle subset manifest."""
    return (
        Path(__file__).resolve().parents[1]
        / "test_games"
        / "test_chess"
        / "fixtures"
        / "lichess_puzzles_subset"
        / "manifest.json"
    )


def test_run_play_session_handles_commands_and_moves() -> None:
    env = make_env()
    input_stream = StringIO("help\ndebug-legal\nshow fen\ne2e4\ntrajectory\nquit\n")
    output_stream = StringIO()

    exit_code = run_play_session(
        env=env,
        game_spec=CHESS_CLI_SPEC,
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
    assert "Legal actions (20):" in output
    assert f"fen: {STANDARD_START_FEN}" in output
    assert "Move SAN: e4" in output
    assert "Trajectory steps: 1" in output
    assert "Session ended." in output


def test_run_play_session_reports_invalid_moves_without_state_change() -> None:
    env = make_env()
    input_stream = StringIO("e2e5\nquit\n")
    output_stream = StringIO()

    exit_code = run_play_session(
        env=env,
        game_spec=CHESS_CLI_SPEC,
        seed=13,
        image_output_dir=None,
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
        scenario=StartingPositionScenario(initial_fen=STANDARD_START_FEN),
        reward_fn=make_reward(),
        config=EpisodeConfig(
            invalid_action_policy=InvalidActionPolicy(
                mode=InvalidActionMode.PENALIZE_CONTINUE,
                penalty=-1.0,
            ),
        ),
        text_renderer_kind=ChessTextRendererKind.ASCII,
        include_images=False,
        image_size=360,
        image_coordinates=True,
        orientation=ChessBoardOrientation.WHITE,
    )
    input_stream = StringIO("e2e5\ntrajectory\nquit\n")
    output_stream = StringIO()

    exit_code = run_play_session(
        env=env,
        game_spec=CHESS_CLI_SPEC,
        seed=21,
        image_output_dir=None,
        input_stream=input_stream,
        output_stream=output_stream,
    )

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "Accepted: False" in output
    assert "Reward: -1.0" in output
    assert "Trajectory steps: 1" in output
    assert "accepted=False" in output


def test_run_play_session_finishes_immediately_for_terminal_reset_positions() -> None:
    env = make_chess_env(
        scenario=StartingPositionScenario(initial_fen=TERMINAL_FEN),
        reward_fn=make_reward(),
        config=EpisodeConfig(),
        text_renderer_kind=ChessTextRendererKind.ASCII,
        include_images=False,
        image_size=360,
        image_coordinates=True,
        orientation=ChessBoardOrientation.WHITE,
    )
    input_stream = StringIO("quit\n")
    output_stream = StringIO()

    exit_code = run_play_session(
        env=env,
        game_spec=CHESS_CLI_SPEC,
        seed=2,
        image_output_dir=None,
        input_stream=input_stream,
        output_stream=output_stream,
    )

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "Terminal: yes" in output
    assert "Episode finished." in output
    assert "turn[0]>" not in output


def test_run_play_session_persists_rendered_images_when_requested(
    tmp_path: Path,
) -> None:
    env = make_chess_env(
        scenario=StartingPositionScenario(initial_fen=STANDARD_START_FEN),
        reward_fn=make_reward(),
        config=EpisodeConfig(max_transitions=1),
        text_renderer_kind=ChessTextRendererKind.ASCII,
        include_images=True,
        image_size=360,
        image_coordinates=True,
        orientation=ChessBoardOrientation.WHITE,
    )
    input_stream = StringIO("e2e4\n")
    output_stream = StringIO()

    exit_code = run_play_session(
        env=env,
        game_spec=CHESS_CLI_SPEC,
        seed=6,
        image_output_dir=tmp_path,
        input_stream=input_stream,
        output_stream=output_stream,
    )

    output = output_stream.getvalue()
    saved_paths = tuple(tmp_path.glob("*.png"))
    assert exit_code == 0
    assert "Image paths:" in output
    assert len(saved_paths) == 2
    assert all(
        path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in saved_paths
    )


def test_run_cli_can_start_and_exit_a_chess_play_session(
    monkeypatch: MonkeyPatch,
) -> None:
    input_stream = StringIO("quit\n")
    output_stream = StringIO()
    monkeypatch.setattr(sys, "stdin", input_stream)
    monkeypatch.setattr(sys, "stdout", output_stream)
    patch_stockfish_runtime(monkeypatch)

    exit_code = run_cli(["play", "chess", "--seed", "5", *ENGINE_REWARD_ARGS])

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "Chess board:" in output
    assert "Session ended." in output


def test_run_cli_auto_replies_with_the_engine_move(
    monkeypatch: MonkeyPatch,
) -> None:
    input_stream = StringIO("e2e4\nquit\n")
    output_stream = StringIO()
    monkeypatch.setattr(sys, "stdin", input_stream)
    monkeypatch.setattr(sys, "stdout", output_stream)
    patch_stockfish_runtime(monkeypatch)

    exit_code = run_cli(["play", "chess", "--seed", "5", *ENGINE_REWARD_ARGS])

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "Agent Move SAN: e4" in output
    assert "Opponent Move SAN: e5" in output
    assert '"transition_count_delta": 2' in output


def test_run_cli_can_use_black_board_orientation(
    monkeypatch: MonkeyPatch,
) -> None:
    input_stream = StringIO("quit\n")
    output_stream = StringIO()
    monkeypatch.setattr(sys, "stdin", input_stream)
    monkeypatch.setattr(sys, "stdout", output_stream)
    patch_stockfish_runtime(monkeypatch)

    exit_code = run_cli(
        [
            "play",
            "chess",
            "--seed",
            "5",
            "--orientation",
            "black",
            *ENGINE_REWARD_ARGS,
        ]
    )

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
    patch_stockfish_runtime(monkeypatch)

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
            *ENGINE_REWARD_ARGS,
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
    patch_stockfish_runtime(monkeypatch)

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
            *ENGINE_REWARD_ARGS,
        ]
    )

    output = output_stream.getvalue()
    assert exit_code == 0
    assert '"truncated_reason": "max_attempts"' in output
    assert "Episode finished." in output


def test_run_cli_can_start_a_chess_puzzle_session(
    monkeypatch: MonkeyPatch,
) -> None:
    input_stream = StringIO("quit\n")
    output_stream = StringIO()
    monkeypatch.setattr(sys, "stdin", input_stream)
    monkeypatch.setattr(sys, "stdout", output_stream)

    exit_code = run_cli(
        [
            "play",
            "chess",
            "--scenario",
            "lichess-puzzles",
            "--dataset-manifest",
            str(fixture_puzzle_manifest_path()),
            "--reward",
            "puzzle-sparse",
            "--seed",
            "0",
        ]
    )

    output = output_stream.getvalue()
    assert exit_code == 0
    assert '"scenario": "dataset_puzzle"' in output
    assert "Chess board:" in output
    assert "Session ended." in output


def test_build_chess_environment_can_use_engine_eval_dense_reward(
    monkeypatch: MonkeyPatch,
) -> None:
    patch_stockfish_runtime(monkeypatch)
    parser = build_parser()
    args = parser.parse_args(
        [
            "play",
            "chess",
            "--reward",
            "engine-eval-dense",
            "--engine-depth",
            "12",
            "--engine-mate-score",
            "100000",
        ]
    )

    env = build_chess_environment(args, parser)

    assert isinstance(env, TurnBasedEnv)
    assert isinstance(env.reward_fn, EngineEvalDenseReward)
    assert env.reward_fn.perspective == "mover"


def test_build_chess_environment_rejects_puzzle_rewards_for_real_games() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "play",
            "chess",
            "--reward",
            "puzzle-sparse",
        ]
    )

    with pytest.raises(SystemExit):
        build_chess_environment(args, parser)


def test_build_chess_environment_rejects_engine_rewards_for_puzzles() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "play",
            "chess",
            "--scenario",
            "lichess-puzzles",
            "--dataset-manifest",
            str(fixture_puzzle_manifest_path()),
            "--reward",
            "engine-eval-dense",
            "--engine-depth",
            "12",
            "--engine-mate-score",
            "100000",
        ]
    )

    with pytest.raises(SystemExit):
        build_chess_environment(args, parser)
