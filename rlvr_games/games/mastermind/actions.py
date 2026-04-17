"""Mastermind action types and serialization helpers."""

from dataclasses import dataclass
from typing import Sequence

from rlvr_games.games.mastermind.engine import (
    MastermindCode,
    format_code,
    normalize_code,
)


def serialize_mastermind_action(*, code: MastermindCode) -> str:
    """Serialize one Mastermind code into the canonical action label."""
    return f"guess {format_code(code=code)}"


@dataclass(init=False, slots=True, frozen=True)
class MastermindAction:
    """Canonical Mastermind action containing one complete guess."""

    code: MastermindCode

    def __init__(self, *, code: MastermindCode | Sequence[int | str] | str) -> None:
        """Normalize and store one guess action."""
        object.__setattr__(self, "code", normalize_code(code=code))

    @property
    def label(self) -> str:
        """Return the canonical serialized action label."""
        return serialize_mastermind_action(code=self.code)
