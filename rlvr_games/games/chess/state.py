"""Canonical chess state types."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ChessState:
    """Minimal chess state placeholder.

    The first implementation can use FEN as the canonical state payload.
    """

    fen: str
    metadata: dict[str, Any] = field(default_factory=dict)
