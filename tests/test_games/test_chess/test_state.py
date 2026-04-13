"""Chess state tests."""

from dataclasses import FrozenInstanceError

import chess
import pytest

from rlvr_games.games.chess import ChessState
from rlvr_games.games.chess.scenarios import STANDARD_START_FEN
from rlvr_games.games.chess.state import repetition_key_from_board


def test_chess_state_exposes_copies_of_mutable_payloads() -> None:
    board = chess.Board(STANDARD_START_FEN)
    repetition_key = repetition_key_from_board(board)
    state = ChessState.from_board(
        board=board,
        repetition_counts={repetition_key: 1},
        metadata={"scenario": "copy-test"},
    )

    board_copy = state.board
    board_copy.push(chess.Move.from_uci("e2e4"))
    repetition_counts = state.repetition_counts
    repetition_counts.clear()
    metadata = state.metadata
    metadata["scenario"] = "mutated"

    assert state.fen == STANDARD_START_FEN
    assert state.board.fen() == STANDARD_START_FEN
    assert state.legal_action_count == 20
    assert state.repetition_counts
    assert state.metadata["scenario"] == "copy-test"
    with pytest.raises(FrozenInstanceError):
        setattr(state, "fen", "mutated")
