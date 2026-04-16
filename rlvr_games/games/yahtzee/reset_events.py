"""Reset-time event policies for Yahtzee."""

from dataclasses import dataclass

from rlvr_games.core.protocol import ResetEventPolicy
from rlvr_games.core.trajectory import AppliedResetEvent
from rlvr_games.games.yahtzee.backend import YahtzeeBackend
from rlvr_games.games.yahtzee.state import YahtzeeState


@dataclass(slots=True)
class YahtzeeOpeningRollPolicy(ResetEventPolicy[YahtzeeState]):
    """Apply the standard opening roll before the first agent move."""

    backend: YahtzeeBackend
    source: str = "chance"
    _applied: bool = False

    def reset(self, *, initial_state: YahtzeeState) -> None:
        """Start a fresh reset-time opening roll sequence."""
        del initial_state
        self._applied = False

    def apply_next_event(
        self,
        *,
        state: YahtzeeState,
    ) -> AppliedResetEvent[YahtzeeState] | None:
        """Apply the opening roll when it has not yet been produced."""
        if self._applied:
            return None

        next_state, info = self.backend.apply_opening_roll(state)
        dice = next_state.dice
        self._applied = True
        return AppliedResetEvent(
            source=self.source,
            label=f"opening-roll {' '.join(str(value) for value in dice)}",
            next_state=next_state,
            info=info,
        )
