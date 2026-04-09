"""Chess backend placeholder."""

from typing import Any

from rlvr_games.core.exceptions import InvalidActionError
from rlvr_games.games.chess.actions import ChessAction
from rlvr_games.games.chess.state import ChessState


class ChessBackend:
    """Placeholder chess backend.

    The next implementation step is to back this with `python-chess` so that
    action parsing, legal moves, transitions, and terminal conditions are all
    verified by executable game logic.
    """

    def parse_action(self, state: ChessState, raw_action: str) -> ChessAction:
        move = raw_action.strip()
        if not move:
            raise InvalidActionError("Chess actions must be a non-empty move string.")
        return ChessAction(move=move)

    def legal_actions(self, state: ChessState) -> list[str]:
        raise NotImplementedError("ChessBackend.legal_actions() is not implemented yet.")

    def apply_action(self, state: ChessState, action: ChessAction) -> tuple[ChessState, dict[str, Any]]:
        raise NotImplementedError(
            "ChessBackend.apply_action() is not implemented yet. "
            "Back this with python-chess or another verified rule engine."
        )

    def is_terminal(self, state: ChessState) -> bool:
        return False
