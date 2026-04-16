"""Canonical Yahtzee state types."""

from dataclasses import dataclass, field
from typing import Any

from rlvr_games.games.yahtzee.actions import (
    serialize_yahtzee_reroll_action,
    serialize_yahtzee_score_action,
)
from rlvr_games.games.yahtzee.engine import (
    CATEGORY_ORDER,
    CategoryScores,
    Dice,
    ZERO_DICE,
    available_categories,
    category_scores_dict,
    empty_category_scores,
    filled_category_count,
    normalize_category_scores,
    normalize_dice,
    reroll_position_sets,
    score_options,
    total_score,
)


@dataclass(slots=True, frozen=True)
class YahtzeeOutcome:
    """Terminal outcome summary for a canonical Yahtzee state.

    Attributes
    ----------
    is_terminal : bool
        Whether the scorecard has been completed.
    final_score : int | None
        Final total score for completed episodes.
    termination : str | None
        Structured terminal reason. Supported value is
        ``"scorecard_complete"``.
    """

    is_terminal: bool
    final_score: int | None = None
    termination: str | None = None

    def __post_init__(self) -> None:
        """Validate that terminal metadata is internally coherent."""
        if self.is_terminal:
            if self.final_score is None or self.termination is None:
                raise ValueError(
                    "Terminal Yahtzee outcomes require final-score metadata."
                )
            if self.termination != "scorecard_complete":
                raise ValueError(
                    "Yahtzee terminal outcomes must terminate by scorecard completion."
                )
            return

        if self.final_score is not None or self.termination is not None:
            raise ValueError(
                "Non-terminal Yahtzee outcomes must not include terminal metadata."
            )

    def metadata(self) -> dict[str, object]:
        """Return dictionary metadata for the outcome."""
        if not self.is_terminal:
            return {}
        return {
            "final_score": self.final_score,
            "termination": self.termination,
        }


@dataclass(init=False, slots=True, frozen=True)
class YahtzeeState:
    """Canonical Yahtzee state.

    Attributes
    ----------
    dice : Dice
        Current five dice. Zero placeholders are used only while
        `awaiting_roll` is `True`.
    rolls_used_in_turn : int
        Number of rolls already used in the current turn.
    turns_completed : int
        Number of categories already scored.
    awaiting_roll : bool
        Whether the state is waiting for an authoritative internal opening
        roll before the agent can act.
    category_scores : CategoryScores
        Scorecard values aligned with `CATEGORY_ORDER`.
    legal_actions : tuple[str, ...]
        Canonical legal action labels for the current state.
    available_categories : tuple[str, ...]
        Remaining score categories in canonical order.
    available_score_options : dict[str, int]
        Immediate score values for the remaining categories on the current
        dice. This is empty while `awaiting_roll` is `True`.
    total_score : int
        Sum of all scored categories. No bonus rules are applied in v1.
    rerolls_remaining : int
        Number of remaining rerolls in the current turn.
    turn_number : int
        One-based turn number for the active turn, or the just-completed final
        turn when the scorecard is terminal.
    outcome : YahtzeeOutcome
        Terminal outcome summary for the current state.
    """

    dice: Dice
    rolls_used_in_turn: int
    turns_completed: int
    awaiting_roll: bool
    category_scores: CategoryScores
    _rng_state: tuple[Any, ...] = field(repr=False)
    _available_score_options: tuple[tuple[str, int], ...] = field(
        init=False,
        repr=False,
    )
    legal_actions: tuple[str, ...] = field(init=False)
    available_categories: tuple[str, ...] = field(init=False)
    total_score: int = field(init=False)
    filled_category_count: int = field(init=False)
    rerolls_remaining: int = field(init=False)
    turn_number: int = field(init=False)
    outcome: YahtzeeOutcome = field(init=False)

    def __init__(
        self,
        *,
        dice: Dice,
        rolls_used_in_turn: int,
        turns_completed: int,
        awaiting_roll: bool,
        category_scores: CategoryScores | None = None,
        rng_state: tuple[Any, ...],
    ) -> None:
        """Create a canonical Yahtzee state."""
        if turns_completed < 0 or turns_completed > len(CATEGORY_ORDER):
            raise ValueError("Yahtzee turns_completed must be in the range 0..13.")
        if rolls_used_in_turn < 0 or rolls_used_in_turn > 3:
            raise ValueError("Yahtzee rolls_used_in_turn must be in the range 0..3.")

        current_scores = (
            empty_category_scores()
            if category_scores is None
            else normalize_category_scores(scores=category_scores)
        )
        current_filled_count = filled_category_count(scores=current_scores)
        if current_filled_count != turns_completed:
            raise ValueError(
                "Yahtzee turns_completed must equal the number of scored categories."
            )

        current_total_score = total_score(scores=current_scores)
        is_terminal = current_filled_count == len(CATEGORY_ORDER)

        if awaiting_roll:
            if is_terminal:
                raise ValueError(
                    "Terminal Yahtzee states cannot still await an opening roll."
                )
            if rolls_used_in_turn != 0:
                raise ValueError(
                    "Yahtzee states awaiting an opening roll must use zero rolls."
                )
            current_dice = normalize_dice(dice=dice, allow_zero=True)
            if current_dice != ZERO_DICE:
                raise ValueError(
                    "Yahtzee states awaiting an opening roll must use zero placeholder dice."
                )
        else:
            current_dice = normalize_dice(dice=dice, allow_zero=False)
            if not is_terminal and rolls_used_in_turn < 1:
                raise ValueError(
                    "Active non-terminal Yahtzee turns must have used at least one roll."
                )

        if is_terminal:
            outcome = YahtzeeOutcome(
                is_terminal=True,
                final_score=current_total_score,
                termination="scorecard_complete",
            )
            current_available_categories: tuple[str, ...] = ()
            current_available_score_options: tuple[tuple[str, int], ...] = ()
            legal_actions: tuple[str, ...] = ()
            rerolls_remaining = 0
        else:
            outcome = YahtzeeOutcome(
                is_terminal=False,
                final_score=None,
                termination=None,
            )
            remaining_categories = available_categories(scores=current_scores)
            current_available_categories = tuple(
                category.value for category in remaining_categories
            )
            if awaiting_roll:
                current_available_score_options = ()
                legal_actions = ()
                rerolls_remaining = 0
            else:
                current_available_score_options = tuple(
                    score_options(
                        dice=current_dice,
                        scores=current_scores,
                    ).items()
                )
                rerolls_remaining = 3 - rolls_used_in_turn
                legal_actions_list = [
                    serialize_yahtzee_score_action(category=category)
                    for category in remaining_categories
                ]
                if rerolls_remaining > 0:
                    legal_actions_list.extend(
                        serialize_yahtzee_reroll_action(positions=positions)
                        for positions in reroll_position_sets()
                    )
                legal_actions = tuple(legal_actions_list)

        turn_number = current_filled_count
        if not is_terminal:
            turn_number += 1

        object.__setattr__(self, "dice", current_dice)
        object.__setattr__(self, "rolls_used_in_turn", rolls_used_in_turn)
        object.__setattr__(self, "turns_completed", turns_completed)
        object.__setattr__(self, "awaiting_roll", awaiting_roll)
        object.__setattr__(self, "category_scores", current_scores)
        object.__setattr__(self, "_rng_state", rng_state)
        object.__setattr__(
            self,
            "_available_score_options",
            current_available_score_options,
        )
        object.__setattr__(self, "legal_actions", legal_actions)
        object.__setattr__(self, "available_categories", current_available_categories)
        object.__setattr__(self, "total_score", current_total_score)
        object.__setattr__(self, "filled_category_count", current_filled_count)
        object.__setattr__(self, "rerolls_remaining", rerolls_remaining)
        object.__setattr__(self, "turn_number", turn_number)
        object.__setattr__(self, "outcome", outcome)

    @property
    def rng_state(self) -> tuple[Any, ...]:
        """Return the deterministic RNG state for future dice rolls."""
        return self._rng_state

    @property
    def available_score_options(self) -> dict[str, int]:
        """Return score options without exposing mutable internal state."""
        return dict(self._available_score_options)

    @property
    def legal_action_count(self) -> int:
        """Return the number of legal actions in the current state."""
        return len(self.legal_actions)

    @property
    def is_terminal(self) -> bool:
        """Return whether the scorecard has been completed."""
        return self.outcome.is_terminal

    @property
    def turns_remaining(self) -> int:
        """Return the number of turns left to score."""
        return len(CATEGORY_ORDER) - self.turns_completed


def public_yahtzee_metadata(state: YahtzeeState) -> dict[str, object]:
    """Return a structured public summary of a Yahtzee state."""
    metadata: dict[str, object] = {
        "dice": state.dice,
        "rolls_used_in_turn": state.rolls_used_in_turn,
        "rerolls_remaining": state.rerolls_remaining,
        "turn_number": state.turn_number,
        "turns_completed": state.turns_completed,
        "turns_remaining": state.turns_remaining,
        "awaiting_roll": state.awaiting_roll,
        "scorecard": category_scores_dict(scores=state.category_scores),
        "available_categories": state.available_categories,
        "available_score_options": dict(state.available_score_options),
        "total_score": state.total_score,
        "filled_category_count": state.filled_category_count,
        "legal_action_count": state.legal_action_count,
        "is_terminal": state.is_terminal,
    }
    metadata.update(state.outcome.metadata())
    return metadata


def inspect_yahtzee_state(state: YahtzeeState) -> dict[str, object]:
    """Return a structured canonical summary of a Yahtzee state."""
    metadata = public_yahtzee_metadata(state)
    metadata.update(
        {
            "legal_actions": state.legal_actions,
            "rng_state": state.rng_state,
        }
    )
    return metadata
