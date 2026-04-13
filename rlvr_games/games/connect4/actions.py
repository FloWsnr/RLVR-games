"""Connect 4 action types."""

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Connect4Action:
    """Canonical Connect 4 action expressed as one drop column.

    Attributes
    ----------
    column : int
        Zero-based column index where the current player's token should be
        dropped.
    """

    column: int

    @property
    def label(self) -> str:
        """Return the normalized human-facing column label.

        Returns
        -------
        str
            One-based column label such as ``"1"`` or ``"7"``.
        """
        return str(self.column + 1)
