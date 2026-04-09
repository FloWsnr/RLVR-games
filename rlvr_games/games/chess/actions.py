"""Chess action types."""

from dataclasses import dataclass


@dataclass(slots=True)
class ChessAction:
    """Model-produced chess move string.

    We will likely normalize this to UCI first, then add SAN support if needed.
    """

    move: str
