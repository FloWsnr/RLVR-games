"""Canonical action types for Yahtzee."""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from rlvr_games.games.yahtzee.engine import YahtzeeCategory


class YahtzeeActionKind(StrEnum):
    """Supported Yahtzee action kinds."""

    OPENING_ROLL = "opening-roll"
    REROLL = "reroll"
    SCORE = "score"


@dataclass(slots=True, frozen=True)
class YahtzeeAction:
    """Canonical Yahtzee action.

    Attributes
    ----------
    kind : YahtzeeActionKind
        Action kind to apply.
    reroll_positions : tuple[int, ...]
        Zero-based die positions to reroll for `reroll` actions.
    category : YahtzeeCategory | None
        Category to score for `score` actions.
    """

    kind: YahtzeeActionKind
    reroll_positions: tuple[int, ...] = ()
    category: YahtzeeCategory | None = None

    def __post_init__(self) -> None:
        """Validate that the action payload matches `kind`."""
        if self.kind == YahtzeeActionKind.OPENING_ROLL:
            if self.category is not None:
                raise ValueError(
                    "Opening-roll actions must not carry a score category."
                )
            if self.reroll_positions:
                raise ValueError(
                    "Opening-roll actions must not include reroll positions."
                )
            return

        if self.kind == YahtzeeActionKind.REROLL:
            if self.category is not None:
                raise ValueError("Reroll actions must not carry a score category.")
            if not self.reroll_positions:
                raise ValueError(
                    "Reroll actions must include at least one die position."
                )
            if len(set(self.reroll_positions)) != len(self.reroll_positions):
                raise ValueError("Reroll action positions must be unique.")
            if any(position < 0 or position > 4 for position in self.reroll_positions):
                raise ValueError("Reroll action positions must be in the range 0..4.")
            object.__setattr__(
                self,
                "reroll_positions",
                tuple(sorted(self.reroll_positions)),
            )
            return

        if self.kind == YahtzeeActionKind.SCORE:
            if self.category is None:
                raise ValueError("Score actions must specify a Yahtzee category.")
            if self.reroll_positions:
                raise ValueError("Score actions must not include reroll positions.")
            return

        raise ValueError(f"Unsupported Yahtzee action kind: {self.kind!r}.")

    @property
    def label(self) -> str:
        """Return the canonical serialized action label."""
        if self.kind == YahtzeeActionKind.OPENING_ROLL:
            return serialize_yahtzee_opening_roll_action()
        if self.kind == YahtzeeActionKind.REROLL:
            return serialize_yahtzee_reroll_action(
                positions=self.reroll_positions,
            )
        if self.category is None:
            raise ValueError("Score actions require a category to serialize.")
        return serialize_yahtzee_score_action(category=self.category)


def serialize_yahtzee_reroll_action(*, positions: tuple[int, ...]) -> str:
    """Return one canonical Yahtzee reroll action string."""
    normalized_positions = normalize_yahtzee_reroll_positions(positions=positions)
    serialized_positions = " ".join(
        str(position + 1) for position in normalized_positions
    )
    return f"reroll {serialized_positions}"


def serialize_yahtzee_score_action(*, category: YahtzeeCategory) -> str:
    """Return one canonical Yahtzee score action string."""
    return f"score {category.value}"


def serialize_yahtzee_opening_roll_action() -> str:
    """Return the canonical serialized opening-roll action string."""
    return YahtzeeActionKind.OPENING_ROLL.value


def normalize_yahtzee_reroll_positions(
    *,
    positions: Sequence[int],
) -> tuple[int, ...]:
    """Return one canonical sorted reroll-position tuple.

    Parameters
    ----------
    positions : Sequence[int]
        Zero-based dice positions to reroll.

    Returns
    -------
    tuple[int, ...]
        Canonical sorted reroll positions.

    Raises
    ------
    ValueError
        If the supplied positions are empty, duplicated, or out of range.
    """
    if not positions:
        raise ValueError("Yahtzee reroll actions must include at least one position.")

    normalized_positions = tuple(int(position) for position in positions)
    if len(set(normalized_positions)) != len(normalized_positions):
        raise ValueError("Yahtzee reroll action positions must be unique.")
    if any(position < 0 or position > 4 for position in normalized_positions):
        raise ValueError("Yahtzee reroll action positions must be in the range 0..4.")
    return tuple(sorted(normalized_positions))
