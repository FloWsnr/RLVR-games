"""Chess environment integration tests."""

import chess
from pathlib import Path
import pytest

from rlvr_games.datasets import DatasetSplit
from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.protocol import GameBackend
from rlvr_games.core.types import AutoAction
from rlvr_games.core.exceptions import EpisodeFinishedError
from rlvr_games.core.types import EpisodeConfig
from rlvr_games.games.chess import (
    AsciiBoardFormatter,
    ChessAction,
    ChessBackend,
    ChessBoardOrientation,
    ChessEngineAutoAdvancePolicy,
    ChessPuzzleAutoAdvancePolicy,
    ChessObservationRenderer,
    ChessPuzzleDatasetScenario,
    ChessState,
    StockfishMoveSelector,
    ChessTextRendererKind,
    PuzzleOnlyMoveDenseReward,
    PuzzleOnlyMoveSparseReward,
    StartingPositionScenario,
    TerminalOutcomeReward,
    make_chess_env,
)
from rlvr_games.games.chess.scenarios import STANDARD_START_FEN
from rlvr_games.games.chess.stockfish_runtime import resolve_stockfish_binary_path
from rlvr_games.games.chess.state import inspect_chess_state

PROMOTION_FEN = "k7/4P3/8/8/8/8/8/7K w - - 0 1"
TERMINAL_FEN = "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"


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


def fixture_puzzle_manifest_path() -> Path:
    """Return the checked-in processed Lichess puzzle subset manifest."""
    return (
        Path(__file__).resolve().parent
        / "fixtures"
        / "lichess_puzzles_subset"
        / "manifest.json"
    )


def fixture_stockfish_binary_path() -> Path:
    """Return a usable local Stockfish binary path or skip the test."""
    try:
        return resolve_stockfish_binary_path()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))


class ScriptedMoveSelector:
    """Return one scripted opponent move for chess auto-advance tests."""

    def __init__(self, *, raw_action: str) -> None:
        """Store the scripted reply move."""
        self.raw_action = raw_action

    def select_action(
        self,
        *,
        state: ChessState,
        backend: GameBackend[ChessState, ChessAction],
    ) -> AutoAction[ChessAction]:
        """Return the scripted reply parsed through the backend."""
        parse_result = backend.parse_action(state, self.raw_action)
        return AutoAction(
            source="opponent",
            raw_action=self.raw_action,
            action=parse_result.require_action(),
        )


def test_checkmate_sequence_terminates_with_winner_metadata() -> None:
    env = TurnBasedEnv(
        backend=ChessBackend(),
        scenario=StartingPositionScenario(),
        renderer=make_renderer(),
        inspect_state_fn=inspect_chess_state,
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
        inspect_state_fn=inspect_chess_state,
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
        inspect_state_fn=inspect_chess_state,
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
        inspect_state_fn=inspect_chess_state,
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
        inspect_state_fn=inspect_chess_state,
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
        inspect_state_fn=inspect_chess_state,
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


def test_engine_auto_advance_returns_to_agent_turn_and_records_reply() -> None:
    env = make_chess_env(
        scenario=StartingPositionScenario(),
        reward_fn=make_reward(),
        config=EpisodeConfig(),
        text_renderer_kind=ChessTextRendererKind.ASCII,
        include_images=False,
        image_size=360,
        image_coordinates=True,
        orientation=ChessBoardOrientation.WHITE,
        auto_advance_policy=ChessEngineAutoAdvancePolicy(
            move_selector=ScriptedMoveSelector(raw_action="e7e5"),
        ),
    )
    env.reset(seed=31)

    result = env.step("e2e4")

    board = chess.Board(STANDARD_START_FEN)
    board.push(chess.Move.from_uci("e2e4"))
    board.push(chess.Move.from_uci("e7e5"))
    assert result.accepted is True
    assert result.terminated is False
    assert result.info["transition_count_delta"] == 2
    assert result.info["auto_advanced"] is True
    assert env.state.fen == board.fen()
    assert result.observation.metadata["side_to_move"] == "white"
    assert len(env.trajectory.steps[0].transitions) == 2
    assert env.trajectory.steps[0].transitions[0].source == "agent"
    assert env.trajectory.steps[0].transitions[1].source == "opponent"
    assert env.trajectory.steps[0].transitions[1].raw_action == "e7e5"


def test_engine_auto_advance_can_play_multiple_real_stockfish_replies() -> None:
    env = make_chess_env(
        scenario=StartingPositionScenario(),
        reward_fn=make_reward(),
        config=EpisodeConfig(),
        text_renderer_kind=ChessTextRendererKind.ASCII,
        include_images=False,
        image_size=360,
        image_coordinates=True,
        orientation=ChessBoardOrientation.WHITE,
        auto_advance_policy=ChessEngineAutoAdvancePolicy(
            move_selector=StockfishMoveSelector.from_engine_path(
                engine_path=fixture_stockfish_binary_path(),
                depth=1,
            ),
        ),
    )

    try:
        observation, info = env.reset(seed=41)
        assert info["scenario"] == "starting_position"
        assert observation.metadata["side_to_move"] == "white"

        candidate_moves_by_turn = (
            ("e2e4",),
            ("g1f3", "d2d4", "b1c3", "f1c4"),
            ("d2d4", "b1c3", "f1c4", "c2c3"),
        )

        for turn_index, candidate_moves in enumerate(candidate_moves_by_turn):
            legal_actions = env.legal_actions()
            assert any(
                candidate_move in legal_actions for candidate_move in candidate_moves
            )
            raw_action = next(
                candidate_move
                for candidate_move in candidate_moves
                if candidate_move in legal_actions
            )
            result = env.step(raw_action)

            assert result.accepted is True
            assert result.terminated is False
            assert result.truncated is False
            assert result.info["auto_advanced"] is True
            assert result.info["transition_count_delta"] == 2
            assert len(env.trajectory.steps) == turn_index + 1
            assert len(env.trajectory.steps[-1].transitions) == 2
            assert env.trajectory.steps[-1].transitions[0].source == "agent"
            assert env.trajectory.steps[-1].transitions[1].source == "opponent"
            assert result.observation.metadata["side_to_move"] == "white"
            assert env.state.side_to_move == "white"
            assert env.trajectory.steps[-1].transitions[0].raw_action == raw_action
            assert env.trajectory.steps[-1].transitions[1].raw_action != raw_action

        assert env.trajectory.accepted_step_count == 3
        assert env.trajectory.steps[-1].info["transition_count"] == 6
    finally:
        env.close()


def test_puzzle_auto_advance_replays_canonical_reply_and_finishes_on_solution() -> None:
    env = make_chess_env(
        scenario=ChessPuzzleDatasetScenario(
            manifest_path=fixture_puzzle_manifest_path(),
            split=DatasetSplit.TRAIN,
        ),
        reward_fn=PuzzleOnlyMoveSparseReward(
            success_reward=1.0,
            incorrect_move_reward=-1.0,
        ),
        config=EpisodeConfig(),
        text_renderer_kind=ChessTextRendererKind.ASCII,
        include_images=False,
        image_size=360,
        image_coordinates=True,
        orientation=ChessBoardOrientation.WHITE,
        auto_advance_policy=ChessPuzzleAutoAdvancePolicy(),
    )
    env.reset(seed=0)

    first_result = env.step("f8d8")

    assert first_result.accepted is True
    assert first_result.reward == 0.0
    assert first_result.terminated is False
    assert first_result.info["auto_advanced"] is True
    assert first_result.observation.metadata["side_to_move"] == "black"
    assert len(first_result.info["internal_transitions"]) == 1
    assert first_result.info["internal_transitions"][0]["raw_action"] == "d6d8"
    assert len(env.trajectory.steps[0].transitions) == 2

    final_result = env.step("f6d8")

    assert final_result.accepted is True
    assert final_result.reward == 1.0
    assert final_result.terminated is True
    assert final_result.truncated is False
    assert final_result.info["episode_completion_reason"] == (
        "puzzle_solution_complete"
    )
    assert len(env.trajectory.steps[1].transitions) == 1


def test_puzzle_auto_advance_terminates_when_agent_leaves_solution_path() -> None:
    env = make_chess_env(
        scenario=ChessPuzzleDatasetScenario(
            manifest_path=fixture_puzzle_manifest_path(),
            split=DatasetSplit.TRAIN,
        ),
        reward_fn=PuzzleOnlyMoveDenseReward(
            correct_move_reward=1.0,
            incorrect_move_reward=-1.0,
        ),
        config=EpisodeConfig(),
        text_renderer_kind=ChessTextRendererKind.ASCII,
        include_images=False,
        image_size=360,
        image_coordinates=True,
        orientation=ChessBoardOrientation.WHITE,
        auto_advance_policy=ChessPuzzleAutoAdvancePolicy(),
    )
    env.reset(seed=0)
    wrong_move = next(
        legal_action for legal_action in env.legal_actions() if legal_action != "f8d8"
    )

    result = env.step(wrong_move)

    assert result.accepted is True
    assert result.reward == -1.0
    assert result.terminated is True
    assert result.info["episode_completion_reason"] == "puzzle_off_path"
    assert result.info["auto_advanced"] is False
    assert len(env.trajectory.steps[0].transitions) == 1


def test_puzzle_scenarios_do_not_auto_advance_without_an_explicit_policy() -> None:
    env = make_chess_env(
        scenario=ChessPuzzleDatasetScenario(
            manifest_path=fixture_puzzle_manifest_path(),
            split=DatasetSplit.TRAIN,
        ),
        reward_fn=PuzzleOnlyMoveSparseReward(
            success_reward=1.0,
            incorrect_move_reward=-1.0,
        ),
        config=EpisodeConfig(),
        text_renderer_kind=ChessTextRendererKind.ASCII,
        include_images=False,
        image_size=360,
        image_coordinates=True,
        orientation=ChessBoardOrientation.WHITE,
        auto_advance_policy=None,
    )
    env.reset(seed=0)

    result = env.step("f8d8")

    assert env.auto_advance_policy is None
    assert result.accepted is True
    assert result.terminated is False
    assert result.info["auto_advanced"] is False
    assert result.observation.metadata["side_to_move"] == "white"
    assert len(env.trajectory.steps[0].transitions) == 1
