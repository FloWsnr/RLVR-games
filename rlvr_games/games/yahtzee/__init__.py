"""Yahtzee environment scaffolding."""

from rlvr_games.games.yahtzee.actions import (
    YahtzeeAction,
    YahtzeeActionKind,
    normalize_yahtzee_reroll_positions,
    serialize_yahtzee_opening_roll_action,
    serialize_yahtzee_reroll_action,
    serialize_yahtzee_score_action,
)
from rlvr_games.games.yahtzee.backend import YahtzeeBackend
from rlvr_games.games.yahtzee.chance import DiceRollTransition, YahtzeeChanceModel
from rlvr_games.games.yahtzee.engine import (
    CATEGORY_ORDER,
    CategoryScores,
    Dice,
    YahtzeeCategory,
    ZERO_DICE,
    available_categories,
    category_index,
    category_scores_dict,
    empty_category_scores,
    filled_category_count,
    normalize_category_name,
    normalize_category_scores,
    normalize_dice,
    reroll_position_sets,
    score_category,
    score_options,
    total_score,
)
from rlvr_games.games.yahtzee.factory import make_yahtzee_env
from rlvr_games.games.yahtzee.render import (
    YahtzeeDiceFormatter,
    YahtzeeImageRenderer,
    YahtzeeObservationRenderer,
    YahtzeeScorecardFormatter,
)
from rlvr_games.games.yahtzee.reset_events import YahtzeeOpeningRollPolicy
from rlvr_games.games.yahtzee.rewards import FinalScoreReward, ScoreDeltaReward
from rlvr_games.games.yahtzee.scenarios import (
    STANDARD_YAHTZEE_DICE_COUNT,
    STANDARD_YAHTZEE_TURN_COUNT,
    FixedStateScenario,
    StandardGameScenario,
)
from rlvr_games.games.yahtzee.state import (
    YahtzeeOutcome,
    YahtzeeState,
    inspect_yahtzee_state,
    public_yahtzee_metadata,
)
from rlvr_games.games.yahtzee.turns import YahtzeeOpeningRollAutoAdvancePolicy

__all__ = [
    "CATEGORY_ORDER",
    "CategoryScores",
    "Dice",
    "DiceRollTransition",
    "FinalScoreReward",
    "FixedStateScenario",
    "STANDARD_YAHTZEE_DICE_COUNT",
    "STANDARD_YAHTZEE_TURN_COUNT",
    "ScoreDeltaReward",
    "StandardGameScenario",
    "YahtzeeAction",
    "YahtzeeActionKind",
    "YahtzeeBackend",
    "YahtzeeCategory",
    "YahtzeeChanceModel",
    "YahtzeeDiceFormatter",
    "YahtzeeImageRenderer",
    "YahtzeeObservationRenderer",
    "YahtzeeOpeningRollPolicy",
    "YahtzeeOutcome",
    "YahtzeeScorecardFormatter",
    "YahtzeeState",
    "ZERO_DICE",
    "available_categories",
    "category_index",
    "category_scores_dict",
    "empty_category_scores",
    "filled_category_count",
    "inspect_yahtzee_state",
    "make_yahtzee_env",
    "normalize_category_name",
    "normalize_category_scores",
    "normalize_dice",
    "normalize_yahtzee_reroll_positions",
    "public_yahtzee_metadata",
    "reroll_position_sets",
    "score_category",
    "score_options",
    "serialize_yahtzee_opening_roll_action",
    "serialize_yahtzee_reroll_action",
    "serialize_yahtzee_score_action",
    "total_score",
    "YahtzeeOpeningRollAutoAdvancePolicy",
]
