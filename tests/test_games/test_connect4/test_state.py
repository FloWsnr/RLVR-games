"""Connect 4 state tests."""

import pytest

from rlvr_games.games.connect4.state import Connect4State
from tests.test_games.test_connect4.support import (
    DRAW_BOARD,
    PRE_WIN_BOARD,
    X_WIN_BOARD,
    board_from_rows,
)


def test_state_derives_player_and_legal_actions_from_valid_board() -> None:
    state = Connect4State(
        board=PRE_WIN_BOARD,
        connect_length=4,
    )

    assert state.current_player == "x"
    assert state.legal_actions == ("1", "2", "3", "4", "5", "6", "7")
    assert state.column_heights == (2, 2, 2, 0, 0, 0, 0)
    assert state.move_count == 6
    assert state.is_terminal is False


def test_state_rejects_column_gaps() -> None:
    invalid_board = board_from_rows(
        ".......",
        ".......",
        ".......",
        "x......",
        ".......",
        ".......",
    )

    with pytest.raises(ValueError, match="gaps"):
        Connect4State(board=invalid_board, connect_length=4)


def test_state_rejects_piece_count_imbalance() -> None:
    invalid_board = board_from_rows(
        ".......",
        ".......",
        ".......",
        ".......",
        ".......",
        "oo.....",
    )

    with pytest.raises(ValueError, match="equal pieces"):
        Connect4State(board=invalid_board, connect_length=4)


def test_state_detects_wins_and_draws() -> None:
    winning_state = Connect4State(
        board=X_WIN_BOARD,
        connect_length=4,
    )
    draw_state = Connect4State(
        board=DRAW_BOARD,
        connect_length=4,
    )

    assert winning_state.is_terminal is True
    assert winning_state.current_player == "terminal"
    assert winning_state.outcome.winner == "x"
    assert winning_state.outcome.termination == "connect_length"
    assert winning_state.outcome.winning_cells == (
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
    )

    assert draw_state.is_terminal is True
    assert draw_state.outcome.winner is None
    assert draw_state.outcome.termination == "draw"
    assert draw_state.legal_actions == ()


def test_state_rejects_non_standard_board_size() -> None:
    invalid_board = board_from_rows(
        ".......",
        ".......",
        ".......",
        ".......",
        ".......",
    )

    with pytest.raises(ValueError, match="standard 6x7 board"):
        Connect4State(board=invalid_board, connect_length=4)


def test_state_rejects_non_standard_connect_length() -> None:
    with pytest.raises(ValueError, match="standard 6x7 board"):
        Connect4State(board=PRE_WIN_BOARD, connect_length=5)
