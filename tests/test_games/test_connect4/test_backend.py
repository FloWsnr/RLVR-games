"""Connect 4 backend tests."""

import pytest

from rlvr_games.games.connect4 import Connect4Backend, Connect4State
from tests.test_games.test_connect4.support import (
    FULL_FIRST_COLUMN_BOARD,
    VERTICAL_THREAT_BOARD,
    X_WIN_BOARD,
)


def test_legal_actions_exclude_full_columns() -> None:
    backend = Connect4Backend()
    state = Connect4State(
        board=FULL_FIRST_COLUMN_BOARD,
        connect_length=4,
    )

    legal_actions = backend.legal_actions(state)

    assert legal_actions == ["2", "3", "4", "5", "6", "7"]


@pytest.mark.parametrize("raw_action", ["", "8", "foo", "1"])
def test_parse_action_rejects_invalid_or_full_columns(raw_action: str) -> None:
    backend = Connect4Backend()
    state = Connect4State(
        board=FULL_FIRST_COLUMN_BOARD,
        connect_length=4,
    )

    parse_result = backend.parse_action(state, raw_action)

    assert parse_result.action is None
    assert parse_result.error is not None


def test_apply_action_records_vertical_win_metadata() -> None:
    backend = Connect4Backend()
    state = Connect4State(
        board=VERTICAL_THREAT_BOARD,
        connect_length=4,
    )

    action = backend.parse_action(state, "1").require_action()
    next_state, info = backend.apply_action(state, action)

    assert next_state.board == X_WIN_BOARD
    assert next_state.current_player == "terminal"
    assert info["player"] == "x"
    assert info["column"] == 1
    assert info["column_index"] == 0
    assert info["row_index"] == 2
    assert info["row_from_bottom"] == 4
    assert info["winner"] == "x"
    assert info["termination"] == "connect_length"
    assert info["winning_cells"] == (
        (2, 0),
        (3, 0),
        (4, 0),
        (5, 0),
    )
