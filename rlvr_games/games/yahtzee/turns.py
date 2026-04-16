"""Automatic turn-resolution helpers for Yahtzee environments."""

from dataclasses import dataclass

from rlvr_games.core.protocol import AutoAdvancePolicy, GameBackend
from rlvr_games.core.types import AutoAction, EpisodeBoundary
from rlvr_games.games.yahtzee.actions import (
    YahtzeeAction,
    YahtzeeActionKind,
    serialize_yahtzee_opening_roll_action,
)
from rlvr_games.games.yahtzee.state import YahtzeeState


@dataclass(slots=True)
class YahtzeeOpeningRollAutoAdvancePolicy(
    AutoAdvancePolicy[YahtzeeState, YahtzeeAction]
):
    """Auto-advance Yahtzee states through required opening rolls."""

    source: str = "chance"

    def reset(self, *, initial_state: YahtzeeState) -> None:
        """Start a fresh episode with no extra policy state."""
        del initial_state

    def is_agent_turn(self, *, state: YahtzeeState) -> bool:
        """Return whether the agent may act in `state`."""
        return state.awaiting_roll is False

    def select_internal_action(
        self,
        *,
        state: YahtzeeState,
        backend: GameBackend[YahtzeeState, YahtzeeAction],
    ) -> AutoAction[YahtzeeAction] | None:
        """Return the internal opening-roll action when the turn is pending."""
        del backend
        if state.awaiting_roll is False:
            return None
        raw_action = serialize_yahtzee_opening_roll_action()
        return AutoAction(
            source=self.source,
            raw_action=raw_action,
            action=YahtzeeAction(kind=YahtzeeActionKind.OPENING_ROLL),
        )

    def episode_boundary(self, *, state: YahtzeeState) -> EpisodeBoundary | None:
        """Return no additional task boundary for Yahtzee turns."""
        del state
        return None
