"""Rule-verified chess backend backed by `python-chess`."""

from typing import Any

from rlvr_games.core.exceptions import InvalidActionError
from rlvr_games.core.types import ParseResult
from rlvr_games.games.chess.actions import ChessAction
from rlvr_games.games.chess.state import ChessState, repetition_key_from_board


class ChessBackend:
    """Chess rules engine backed by `python-chess`.

    The backend is the authoritative verifier for move legality, state
    transitions, terminal outcomes, and repetition-based draw termination.
    """

    def parse_action(
        self, state: ChessState, raw_action: str
    ) -> ParseResult[ChessAction]:
        """Parse and validate a raw model action as a legal chess move.

        Parameters
        ----------
        state : ChessState
            Current canonical chess state.
        raw_action : str
            Model-produced move string expected to be in UCI notation.

        Returns
        -------
        ParseResult[ChessAction]
            Structured parse result containing either a normalized legal move
            or an explicit rejection message for the current position.
        """
        move_text = raw_action.strip().lower()
        if not move_text:
            return ParseResult(
                action=None,
                error="Chess actions must be a non-empty move string.",
            )
        try:
            move = state.board.parse_uci(move_text)
        except ValueError:
            return ParseResult(
                action=None,
                error=(
                    "Chess actions must be legal UCI moves for the current "
                    f"position: {raw_action!r}."
                ),
            )
        return ParseResult(action=ChessAction(uci=move.uci()), error=None)

    def legal_actions(self, state: ChessState) -> list[str]:
        """Enumerate legal model-facing actions for the current position.

        Parameters
        ----------
        state : ChessState
            Canonical chess state to inspect.

        Returns
        -------
        list[str]
            Sorted list of legal moves in UCI notation.
        """
        return list(state.legal_actions)

    def apply_action(
        self, state: ChessState, action: ChessAction
    ) -> tuple[ChessState, dict[str, Any]]:
        """Apply a verified chess move and return the resulting transition.

        Parameters
        ----------
        state : ChessState
            Canonical state before the move.
        action : ChessAction
            Parsed move to apply.

        Returns
        -------
        tuple[ChessState, dict[str, Any]]
            Updated canonical state and metadata describing the move, resulting
            position, legal move count, repetition state, and any terminal
            outcome.

        Raises
        ------
        InvalidActionError
            If `action` is not legal in the supplied state.
        """
        board = state.board.copy(stack=False)
        try:
            move = board.parse_uci(action.uci)
        except ValueError as exc:
            raise InvalidActionError(
                f"Chess actions must be legal UCI moves for the current position: {action.uci!r}."
            ) from exc

        move_san = board.san(move)
        board.push(move)
        next_repetition_counts = dict(state.repetition_counts)
        position_key = repetition_key_from_board(board)
        next_repetition_counts[position_key] = (
            next_repetition_counts.get(position_key, 0) + 1
        )
        next_state = ChessState.from_board(
            board=board,
            repetition_counts=next_repetition_counts,
            metadata=dict(state.metadata),
        )
        transition_info = {
            "move_uci": move.uci(),
            "move_san": move_san,
            "fen": next_state.fen,
            "side_to_move": next_state.side_to_move,
            "legal_action_count": next_state.legal_action_count,
            "repetition_count": next_state.repetition_count,
            "is_check": next_state.is_check,
            "is_terminal": next_state.is_terminal,
        }
        transition_info.update(next_state.outcome.metadata())
        return next_state, transition_info

    def is_terminal(self, state: ChessState) -> bool:
        """Return whether the supplied chess state ends the episode.

        Parameters
        ----------
        state : ChessState
            Canonical chess state to inspect.

        Returns
        -------
        bool
            `True` when the state is checkmate, stalemate, another draw outcome,
            or a completed threefold repetition.
        """
        return state.is_terminal
