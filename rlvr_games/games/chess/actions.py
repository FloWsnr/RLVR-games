"""Chess action types."""

from dataclasses import dataclass

import chess


@dataclass(slots=True, frozen=True)
class ChessAction:
    """Canonical chess action expressed as a parsed chess move.

    Attributes
    ----------
    move : chess.Move
        Parsed move accepted by `python-chess` for the position where the
        action was produced.
    """

    move: chess.Move

    @property
    def uci(self) -> str:
        """Return the normalized UCI representation of the move.

        Returns
        -------
        str
            Move serialized in UCI notation, for example ``"e2e4"`` or
            ``"a7a8q"`` for promotions.
        """
        return self.move.uci()
