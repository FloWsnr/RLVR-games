"""Yahtzee scenario initializers."""

from dataclasses import dataclass

from rlvr_games.core.trajectory import ScenarioReset
from rlvr_games.games.yahtzee.chance import YahtzeeChanceModel
from rlvr_games.games.yahtzee.engine import ZERO_DICE
from rlvr_games.games.yahtzee.state import YahtzeeState

STANDARD_YAHTZEE_DICE_COUNT = 5
STANDARD_YAHTZEE_TURN_COUNT = 13


@dataclass(slots=True)
class StandardGameScenario:
    """Scenario that initializes a standard single-player Yahtzee game."""

    chance_model: YahtzeeChanceModel

    def reset(self, *, seed: int) -> ScenarioReset[YahtzeeState]:
        """Create a fresh Yahtzee episode awaiting the opening roll."""
        return ScenarioReset(
            initial_state=YahtzeeState(
                dice=ZERO_DICE,
                rolls_used_in_turn=0,
                turns_completed=0,
                awaiting_roll=True,
                rng_state=self.chance_model.initial_rng_state(seed=seed),
            ),
            reset_info={
                "scenario": "standard_game",
                "turn_count": STANDARD_YAHTZEE_TURN_COUNT,
                "upper_bonus_enabled": False,
                "extra_yahtzee_bonus_enabled": False,
                "joker_rule_enabled": False,
            },
        )


@dataclass(slots=True)
class FixedStateScenario:
    """Scenario that initializes Yahtzee from one explicit canonical state."""

    initial_state: YahtzeeState

    def reset(self, *, seed: int) -> ScenarioReset[YahtzeeState]:
        """Create a fresh Yahtzee episode from the configured state."""
        del seed
        return ScenarioReset(
            initial_state=YahtzeeState(
                dice=self.initial_state.dice,
                rolls_used_in_turn=self.initial_state.rolls_used_in_turn,
                turns_completed=self.initial_state.turns_completed,
                awaiting_roll=self.initial_state.awaiting_roll,
                category_scores=self.initial_state.category_scores,
                rng_state=self.initial_state.rng_state,
            ),
            reset_info={
                "scenario": "fixed_state",
                "turns_completed": self.initial_state.turns_completed,
                "awaiting_roll": self.initial_state.awaiting_roll,
                "total_score": self.initial_state.total_score,
            },
        )
