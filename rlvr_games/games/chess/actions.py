"""Chess action types."""

from dataclasses import dataclass


@dataclass(slots=True)
class ChessAction:
    """Canonical chess action expressed in UCI notation.

    Attributes
    ----------
    uci : str
        Legal move string in UCI format, for example ``"e2e4"`` or
        ``"a7a8q"`` for promotions.
    """

    uci: str
