"""Canonical action types for Minesweeper."""

from dataclasses import dataclass
from enum import StrEnum


class MinesweeperVerb(StrEnum):
    """Supported Minesweeper action kinds."""

    REVEAL = "reveal"
    FLAG = "flag"
    UNFLAG = "unflag"


VERB_ALIASES = {
    "reveal": MinesweeperVerb.REVEAL,
    "r": MinesweeperVerb.REVEAL,
    "open": MinesweeperVerb.REVEAL,
    "flag": MinesweeperVerb.FLAG,
    "f": MinesweeperVerb.FLAG,
    "mark": MinesweeperVerb.FLAG,
    "unflag": MinesweeperVerb.UNFLAG,
    "u": MinesweeperVerb.UNFLAG,
}


@dataclass(slots=True, frozen=True)
class MinesweeperAction:
    """Canonical Minesweeper action.

    Attributes
    ----------
    verb : MinesweeperVerb
        Action kind to apply to the target cell.
    row : int
        Zero-based row index of the target cell.
    col : int
        Zero-based column index of the target cell.
    """

    verb: MinesweeperVerb
    row: int
    col: int

    @property
    def label(self) -> str:
        """Return the canonical serialized action label.

        Returns
        -------
        str
            Canonical one-based action label such as ``"reveal 3 5"``.
        """
        return serialize_minesweeper_action(
            verb=self.verb,
            row=self.row,
            col=self.col,
        )


def serialize_minesweeper_action(
    *,
    verb: MinesweeperVerb,
    row: int,
    col: int,
) -> str:
    """Return the canonical Minesweeper action string.

    Parameters
    ----------
    verb : MinesweeperVerb
        Verb to serialize.
    row : int
        Zero-based row index.
    col : int
        Zero-based column index.

    Returns
    -------
    str
        Canonical one-based action label.
    """
    return f"{verb.value} {row + 1} {col + 1}"
