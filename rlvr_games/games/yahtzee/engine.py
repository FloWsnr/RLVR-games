"""Pure Yahtzee rules and scoring helpers."""

from collections import Counter
from enum import StrEnum
from itertools import combinations
from typing import Sequence

type Dice = tuple[int, int, int, int, int]
type CategoryScores = tuple[int | None, ...]


class YahtzeeCategory(StrEnum):
    """Supported Yahtzee score categories."""

    ONES = "ones"
    TWOS = "twos"
    THREES = "threes"
    FOURS = "fours"
    FIVES = "fives"
    SIXES = "sixes"
    THREE_KIND = "three-of-a-kind"
    FOUR_KIND = "four-of-a-kind"
    FULL_HOUSE = "full-house"
    SMALL_STRAIGHT = "small-straight"
    LARGE_STRAIGHT = "large-straight"
    CHANCE = "chance"
    YAHTZEE = "yahtzee"


CATEGORY_ORDER: tuple[YahtzeeCategory, ...] = (
    YahtzeeCategory.ONES,
    YahtzeeCategory.TWOS,
    YahtzeeCategory.THREES,
    YahtzeeCategory.FOURS,
    YahtzeeCategory.FIVES,
    YahtzeeCategory.SIXES,
    YahtzeeCategory.THREE_KIND,
    YahtzeeCategory.FOUR_KIND,
    YahtzeeCategory.FULL_HOUSE,
    YahtzeeCategory.SMALL_STRAIGHT,
    YahtzeeCategory.LARGE_STRAIGHT,
    YahtzeeCategory.CHANCE,
    YahtzeeCategory.YAHTZEE,
)

ZERO_DICE: Dice = (0, 0, 0, 0, 0)

_UPPER_SECTION_FACE_VALUES = {
    YahtzeeCategory.ONES: 1,
    YahtzeeCategory.TWOS: 2,
    YahtzeeCategory.THREES: 3,
    YahtzeeCategory.FOURS: 4,
    YahtzeeCategory.FIVES: 5,
    YahtzeeCategory.SIXES: 6,
}

_CATEGORY_ALIASES = {
    "1s": YahtzeeCategory.ONES,
    "ones": YahtzeeCategory.ONES,
    "2s": YahtzeeCategory.TWOS,
    "twos": YahtzeeCategory.TWOS,
    "3s": YahtzeeCategory.THREES,
    "threes": YahtzeeCategory.THREES,
    "4s": YahtzeeCategory.FOURS,
    "fours": YahtzeeCategory.FOURS,
    "5s": YahtzeeCategory.FIVES,
    "fives": YahtzeeCategory.FIVES,
    "6s": YahtzeeCategory.SIXES,
    "sixes": YahtzeeCategory.SIXES,
    "three kind": YahtzeeCategory.THREE_KIND,
    "three-kind": YahtzeeCategory.THREE_KIND,
    "three of a kind": YahtzeeCategory.THREE_KIND,
    "three-of-a-kind": YahtzeeCategory.THREE_KIND,
    "four kind": YahtzeeCategory.FOUR_KIND,
    "four-kind": YahtzeeCategory.FOUR_KIND,
    "four of a kind": YahtzeeCategory.FOUR_KIND,
    "four-of-a-kind": YahtzeeCategory.FOUR_KIND,
    "full house": YahtzeeCategory.FULL_HOUSE,
    "full-house": YahtzeeCategory.FULL_HOUSE,
    "small straight": YahtzeeCategory.SMALL_STRAIGHT,
    "small-straight": YahtzeeCategory.SMALL_STRAIGHT,
    "large straight": YahtzeeCategory.LARGE_STRAIGHT,
    "large-straight": YahtzeeCategory.LARGE_STRAIGHT,
    "chance": YahtzeeCategory.CHANCE,
    "yahtzee": YahtzeeCategory.YAHTZEE,
}


def normalize_dice(
    *,
    dice: Sequence[int],
    allow_zero: bool = False,
) -> Dice:
    """Normalize one Yahtzee dice sequence.

    Parameters
    ----------
    dice : Sequence[int]
        Dice-like sequence to validate.
    allow_zero : bool
        Whether zero-valued placeholders are allowed.

    Returns
    -------
    Dice
        Normalized five-die tuple.
    """
    if len(dice) != 5:
        raise ValueError("Yahtzee states require exactly five dice.")

    normalized = tuple(int(value) for value in dice)
    minimum = 0 if allow_zero else 1
    if any(value < minimum or value > 6 for value in normalized):
        if allow_zero:
            raise ValueError("Yahtzee dice must be in the range 0..6.")
        raise ValueError("Yahtzee dice must be in the range 1..6.")
    return normalized  # type: ignore[return-value]


def normalize_category_scores(*, scores: Sequence[int | None]) -> CategoryScores:
    """Normalize one category-score sequence.

    Parameters
    ----------
    scores : Sequence[int | None]
        Score-like sequence aligned with `CATEGORY_ORDER`.

    Returns
    -------
    CategoryScores
        Normalized tuple aligned with `CATEGORY_ORDER`.
    """
    if len(scores) != len(CATEGORY_ORDER):
        raise ValueError(
            "Yahtzee category score sequences must cover every category exactly once."
        )

    normalized: list[int | None] = []
    for score in scores:
        if score is not None and score < 0:
            raise ValueError("Yahtzee category scores must be non-negative.")
        normalized.append(score)
    return tuple(normalized)


def empty_category_scores() -> CategoryScores:
    """Return an empty Yahtzee scorecard."""
    return tuple(None for _ in CATEGORY_ORDER)


def normalize_category_name(*, raw_name: str) -> YahtzeeCategory | None:
    """Normalize one user-facing category name.

    Parameters
    ----------
    raw_name : str
        Raw category text.

    Returns
    -------
    YahtzeeCategory | None
        Parsed category or `None` when the name is unknown.
    """
    normalized = raw_name.strip().lower().replace("_", " ").replace("-", " ")
    normalized = " ".join(normalized.split())
    return _CATEGORY_ALIASES.get(normalized)


def category_index(*, category: YahtzeeCategory) -> int:
    """Return the fixed scorecard index for one category."""
    return CATEGORY_ORDER.index(category)


def category_scores_dict(*, scores: CategoryScores) -> dict[str, int | None]:
    """Return one scorecard dictionary keyed by canonical category label."""
    return {
        category.value: scores[index] for index, category in enumerate(CATEGORY_ORDER)
    }


def total_score(*, scores: CategoryScores) -> int:
    """Return the total score for one Yahtzee scorecard."""
    return sum(score for score in scores if score is not None)


def filled_category_count(*, scores: CategoryScores) -> int:
    """Return the number of filled scorecard categories."""
    return sum(1 for score in scores if score is not None)


def available_categories(*, scores: CategoryScores) -> tuple[YahtzeeCategory, ...]:
    """Return the remaining unfilled categories in canonical order."""
    return tuple(
        category
        for index, category in enumerate(CATEGORY_ORDER)
        if scores[index] is None
    )


def score_options(
    *,
    dice: Dice,
    scores: CategoryScores,
) -> dict[str, int]:
    """Return the score available for every unfilled category."""
    return {
        category.value: score_category(dice=dice, category=category)
        for category in available_categories(scores=scores)
    }


def reroll_position_sets() -> tuple[tuple[int, ...], ...]:
    """Return all non-empty reroll position subsets in canonical order."""
    subsets: list[tuple[int, ...]] = []
    positions = range(5)
    for subset_size in range(1, 6):
        subsets.extend(combinations(positions, subset_size))
    return tuple(subsets)


def score_category(*, dice: Dice, category: YahtzeeCategory) -> int:
    """Return the score for one category on the supplied dice.

    Parameters
    ----------
    dice : Dice
        Five rolled dice.
    category : YahtzeeCategory
        Category to evaluate.

    Returns
    -------
    int
        Score assigned by the Yahtzee rules for `category`.
    """
    counts = Counter(dice)
    unique_values = set(dice)
    dice_total = sum(dice)

    upper_face = _UPPER_SECTION_FACE_VALUES.get(category)
    if upper_face is not None:
        return upper_face * counts.get(upper_face, 0)

    if category == YahtzeeCategory.THREE_KIND:
        return dice_total if any(count >= 3 for count in counts.values()) else 0
    if category == YahtzeeCategory.FOUR_KIND:
        return dice_total if any(count >= 4 for count in counts.values()) else 0
    if category == YahtzeeCategory.FULL_HOUSE:
        return 25 if sorted(counts.values()) == [2, 3] else 0
    if category == YahtzeeCategory.SMALL_STRAIGHT:
        straights = ({1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6})
        return (
            30 if any(straight.issubset(unique_values) for straight in straights) else 0
        )
    if category == YahtzeeCategory.LARGE_STRAIGHT:
        return 40 if unique_values in ({1, 2, 3, 4, 5}, {2, 3, 4, 5, 6}) else 0
    if category == YahtzeeCategory.CHANCE:
        return dice_total
    if category == YahtzeeCategory.YAHTZEE:
        return 50 if len(unique_values) == 1 else 0

    raise ValueError(f"Unsupported Yahtzee category: {category!r}.")
