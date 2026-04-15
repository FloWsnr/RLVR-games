"""Chess backend tests."""

import pytest

from rlvr_games.games.chess import ChessBackend, ChessState
from rlvr_games.games.chess.scenarios import STANDARD_START_FEN

PROMOTION_FEN = "k7/4P3/8/8/8/8/8/7K w - - 0 1"


def test_legal_actions_from_start_position_are_sorted_uci() -> None:
    backend = ChessBackend()
    state = ChessState(fen=STANDARD_START_FEN)

    legal_actions = backend.legal_actions(state)

    assert len(legal_actions) == 20
    assert legal_actions == sorted(legal_actions)
    assert "e2e4" in legal_actions
    assert "g1f3" in legal_actions


def test_apply_action_updates_state_and_transition_info() -> None:
    backend = ChessBackend()
    state = ChessState(fen=STANDARD_START_FEN)

    action = backend.parse_action(state, "e2e4").require_action()
    next_state, info = backend.apply_action(state, action)

    assert action.uci == "e2e4"
    assert (
        next_state.fen == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    )
    assert info == {
        "move_uci": "e2e4",
        "move_san": "e4",
        "fen": next_state.fen,
        "side_to_move": "black",
        "repetition_count": 1,
        "is_check": False,
        "is_terminal": False,
    }


@pytest.mark.parametrize("raw_action", ["", "bad", "e2e5"])
def test_parse_action_rejects_invalid_or_illegal_uci(raw_action: str) -> None:
    backend = ChessBackend()
    state = ChessState(fen=STANDARD_START_FEN)

    parse_result = backend.parse_action(state, raw_action)

    assert parse_result.action is None
    assert parse_result.error is not None


def test_promotion_requires_suffix_and_applies_correctly() -> None:
    backend = ChessBackend()
    state = ChessState(fen=PROMOTION_FEN)

    rejected = backend.parse_action(state, "e7e8")
    assert rejected.action is None
    assert rejected.error is not None

    action = backend.parse_action(state, "e7e8q").require_action()
    next_state, info = backend.apply_action(state, action)

    assert next_state.fen == "k3Q3/8/8/8/8/8/8/7K b - - 0 1"
    assert info["move_san"] == "e8=Q+"
    assert info["is_terminal"] is False
