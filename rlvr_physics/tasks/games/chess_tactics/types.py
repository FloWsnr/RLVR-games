"""Types for chess tactic verification."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ChessVerification:
    """Verification result for one chess tactic submission."""

    accepted: bool
    correct: bool
    reward: float
    reason: str
    move_uci: str | None
    move_san: str | None
