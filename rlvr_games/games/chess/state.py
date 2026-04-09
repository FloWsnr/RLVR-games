"""Canonical chess state types."""

from dataclasses import dataclass, field
from typing import Any

import chess


def repetition_key_from_board(board: chess.Board) -> str:
    """Return the repetition-significant part of a chess position.

    Parameters
    ----------
    board : chess.Board
        Board whose position should be reduced to the fields relevant for
        repetition detection.

    Returns
    -------
    str
        Prefix of the FEN string containing piece placement, side to move,
        castling rights, and en passant status.
    """
    return " ".join(board.fen().split(" ")[:4])


def winner_name(winner: bool | None) -> str | None:
    """Return the structured winner label used in chess metadata.

    Parameters
    ----------
    winner : bool | None
        Winner flag returned by `python-chess`, or `None` for a draw.

    Returns
    -------
    str | None
        `"white"`, `"black"`, or `None` when the game is drawn.
    """
    if winner is None:
        return None
    return "white" if winner == chess.WHITE else "black"


@dataclass(slots=True)
class ChessState:
    """Canonical chess state.

    FEN remains the source-of-truth payload so states stay serializable and
    replayable. Repetition counts preserve the extra draw-claim state that a
    bare FEN cannot encode by itself.

    Attributes
    ----------
    fen : str
        Full FEN string describing the current board position.
    repetition_counts : dict[str, int]
        Counts for repetition-significant positions encountered in the current
        episode. These counts are used to detect threefold repetition claims.
    metadata : dict[str, Any]
        Free-form game-specific metadata that should travel with the state
        without becoming part of the authoritative chess rules payload.
    """

    fen: str
    repetition_counts: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
