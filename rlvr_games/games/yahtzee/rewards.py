"""Reward functions for Yahtzee."""

from typing import Any

from rlvr_games.games.yahtzee.actions import YahtzeeAction
from rlvr_games.games.yahtzee.state import YahtzeeState


class ScoreDeltaReward:
    """Dense reward equal to the score added by the accepted action."""

    def evaluate(
        self,
        *,
        previous_state: YahtzeeState,
        action: YahtzeeAction,
        next_state: YahtzeeState,
        transition_info: dict[str, Any],
    ) -> float:
        """Return the immediate score delta for one accepted action."""
        del action
        del transition_info
        return float(next_state.total_score - previous_state.total_score)


class FinalScoreReward:
    """Sparse reward that pays out only when the scorecard is complete."""

    def evaluate(
        self,
        *,
        previous_state: YahtzeeState,
        action: YahtzeeAction,
        next_state: YahtzeeState,
        transition_info: dict[str, Any],
    ) -> float:
        """Return the final total score on terminal transitions only."""
        del previous_state
        del action
        del transition_info
        if not next_state.is_terminal or next_state.outcome.final_score is None:
            return 0.0
        return float(next_state.outcome.final_score)
