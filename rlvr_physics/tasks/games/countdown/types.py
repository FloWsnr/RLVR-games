"""Types for Countdown verification."""

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class CountdownVerification:
    """Verification result for one Countdown expression."""

    accepted: bool
    correct: bool
    reward: float
    reason: str
    expression: str
    value: Fraction | None
    used_numbers: tuple[int, ...]
