"""2048 action types."""

from dataclasses import dataclass
from enum import StrEnum


class MoveDirection(StrEnum):
    """Canonical movement directions for 2048 actions."""

    UP = "up"
    RIGHT = "right"
    DOWN = "down"
    LEFT = "left"


ACTION_ORDER = (
    MoveDirection.UP,
    MoveDirection.RIGHT,
    MoveDirection.DOWN,
    MoveDirection.LEFT,
)


@dataclass(slots=True, frozen=True)
class Game2048Action:
    """Canonical 2048 action expressed as one movement direction.

    Attributes
    ----------
    direction : MoveDirection
        Parsed movement direction accepted for the current board.
    """

    direction: MoveDirection

    @property
    def label(self) -> str:
        """Return the normalized string label for the action.

        Returns
        -------
        str
            Canonical direction label such as ``"left"`` or ``"up"``.
        """
        return self.direction.value
