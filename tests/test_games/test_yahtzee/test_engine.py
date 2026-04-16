"""Yahtzee engine tests."""

import pytest

from rlvr_games.games.yahtzee import (
    YahtzeeCategory,
    normalize_category_name,
    reroll_position_sets,
    score_category,
)


@pytest.mark.parametrize(
    ("dice", "category", "expected_score"),
    [
        ((1, 1, 1, 4, 6), YahtzeeCategory.ONES, 3),
        ((6, 6, 6, 2, 2), YahtzeeCategory.FULL_HOUSE, 25),
        ((1, 2, 3, 4, 6), YahtzeeCategory.SMALL_STRAIGHT, 30),
        ((2, 3, 4, 5, 6), YahtzeeCategory.LARGE_STRAIGHT, 40),
        ((5, 5, 5, 5, 5), YahtzeeCategory.YAHTZEE, 50),
    ],
)
def test_score_category_applies_standard_rules(
    dice: tuple[int, int, int, int, int],
    category: YahtzeeCategory,
    expected_score: int,
) -> None:
    assert score_category(dice=dice, category=category) == expected_score


def test_reroll_position_sets_cover_all_non_empty_subsets() -> None:
    subsets = reroll_position_sets()

    assert len(subsets) == 31
    assert subsets[0] == (0,)
    assert subsets[-1] == (0, 1, 2, 3, 4)


@pytest.mark.parametrize(
    ("raw_name", "expected_category"),
    [
        ("full house", YahtzeeCategory.FULL_HOUSE),
        ("3s", YahtzeeCategory.THREES),
        ("large_straight", YahtzeeCategory.LARGE_STRAIGHT),
    ],
)
def test_normalize_category_name_supports_common_aliases(
    raw_name: str,
    expected_category: YahtzeeCategory,
) -> None:
    assert normalize_category_name(raw_name=raw_name) == expected_category
