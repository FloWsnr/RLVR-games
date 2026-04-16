"""Reset-time event policies for 2048."""

from dataclasses import dataclass, field

from rlvr_games.core.protocol import ResetEventPolicy
from rlvr_games.core.trajectory import AppliedResetEvent
from rlvr_games.games.game2048.backend import Game2048Backend
from rlvr_games.games.game2048.state import Game2048State


@dataclass(slots=True)
class Game2048StartTilePolicy(ResetEventPolicy[Game2048State]):
    """Apply the standard opening tile spawns before the first agent move.

    Attributes
    ----------
    backend : Game2048Backend
        Authoritative backend used to apply each reset-time spawn.
    start_tile_count : int
        Number of opening tiles that should be spawned before the first
        observation is returned.
    source : str
        Structured source label recorded for each reset-time event.
    """

    backend: Game2048Backend
    start_tile_count: int
    source: str = "chance"
    _applied_count: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        """Validate the requested number of reset-time spawns."""
        if self.start_tile_count <= 0:
            raise ValueError("2048 start_tile_count must be positive.")

    def reset(self, *, initial_state: Game2048State) -> None:
        """Start a fresh reset-time spawn sequence."""
        del initial_state
        self._applied_count = 0

    def apply_next_event(
        self,
        *,
        state: Game2048State,
    ) -> AppliedResetEvent[Game2048State] | None:
        """Apply the next opening spawn when one remains."""
        if self._applied_count >= self.start_tile_count:
            return None

        next_state, info = self.backend.apply_reset_spawn(state)
        spawned_tile = info["spawned_tile"]
        if not isinstance(spawned_tile, dict):
            raise ValueError("2048 reset spawn metadata must include spawned_tile.")

        self._applied_count += 1
        return AppliedResetEvent(
            source=self.source,
            label=(
                "spawn "
                f"row={spawned_tile['row']} "
                f"col={spawned_tile['col']} "
                f"value={spawned_tile['value']}"
            ),
            next_state=next_state,
            info=info,
        )
