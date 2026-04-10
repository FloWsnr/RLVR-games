"""Rule-verified chess backend backed by `python-chess`."""

from typing import Any

import chess

from rlvr_games.core.exceptions import InvalidActionError
from rlvr_games.core.types import ParseResult
from rlvr_games.games.chess.actions import ChessAction
from rlvr_games.games.chess.state import (
    ChessState,
    repetition_key_from_board,
    winner_name,
)


class ChessBackend:
    """Chess rules engine backed by `python-chess`.

    The backend is the authoritative verifier for move legality, state
    transitions, terminal outcomes, and repetition-based draw termination.
    """

    def _board_from_state(self, state: ChessState) -> chess.Board:
        """Materialize a mutable `python-chess` board from canonical state.

        Parameters
        ----------
        state : ChessState
            Canonical chess state storing the current FEN string.

        Returns
        -------
        chess.Board
            Board instance initialized from `state.fen`.
        """
        return chess.Board(state.fen)

    def _repetition_count(self, state: ChessState, board: chess.Board) -> int:
        """Return how often the current repetition key has been observed.

        Parameters
        ----------
        state : ChessState
            Canonical state carrying repetition counters.
        board : chess.Board
            Board representing the same current position.

        Returns
        -------
        int
            Recorded count for the board's repetition-significant position.
        """
        position_key = repetition_key_from_board(board)
        return state.repetition_counts.get(position_key, 1)

    def _outcome_info(self, state: ChessState, board: chess.Board) -> dict[str, Any]:
        """Build terminal metadata for the current board position.

        Parameters
        ----------
        state : ChessState
            Canonical state used for repetition-based draw checks.
        board : chess.Board
            Board representing the current position after a move has been
            applied.

        Returns
        -------
        dict[str, Any]
            Outcome metadata including winner, result string, and termination
            reason. Returns an empty dictionary for non-terminal positions.
        """
        if self._repetition_count(state, board) >= 3:
            return {
                "winner": None,
                "result": "1/2-1/2",
                "termination": "threefold_repetition",
            }

        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            return {}

        return {
            "winner": winner_name(outcome.winner),
            "result": board.result(claim_draw=True),
            "termination": outcome.termination.name.lower(),
        }

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
        board = self._board_from_state(state)
        try:
            move = board.parse_uci(move_text)
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
        board = self._board_from_state(state)
        return sorted(move.uci() for move in board.legal_moves)

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
        board = self._board_from_state(state)
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
        next_state = ChessState(
            fen=board.fen(),
            repetition_counts=next_repetition_counts,
            metadata=dict(state.metadata),
        )
        terminal_info = self._outcome_info(next_state, board)
        side_to_move = "white" if board.turn == chess.WHITE else "black"
        transition_info = {
            "move_uci": move.uci(),
            "move_san": move_san,
            "fen": next_state.fen,
            "side_to_move": side_to_move,
            "legal_action_count": board.legal_moves.count(),
            "repetition_count": next_state.repetition_counts[position_key],
            "is_check": board.is_check(),
            "is_terminal": bool(terminal_info),
        }
        transition_info.update(terminal_info)
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
        board = self._board_from_state(state)
        if self._repetition_count(state, board) >= 3:
            return True
        return board.outcome(claim_draw=True) is not None
