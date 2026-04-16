"""Automatic turn-resolution helpers for Connect 4 environments."""

from dataclasses import dataclass, field
from typing import Protocol, cast

from rlvr_games.core.protocol import AutoAdvancePolicy, GameBackend
from rlvr_games.core.types import AutoAction, EpisodeBoundary
from rlvr_games.games.connect4.actions import Connect4Action
from rlvr_games.games.connect4.state import Connect4State


class Connect4MoveSelector(Protocol):
    """Protocol for selecting one automatic Connect 4 move."""

    def select_action(
        self,
        *,
        state: Connect4State,
    ) -> Connect4Action:
        """Return one selected automatic move for the supplied state."""
        ...


@dataclass(slots=True)
class Connect4SolverAutoAdvancePolicy(AutoAdvancePolicy[Connect4State, Connect4Action]):
    """Auto-advance Connect 4 turns by letting a move selector reply."""

    move_selector: Connect4MoveSelector
    source: str = "opponent"
    _agent_side: str = field(init=False, default="x", repr=False)

    def reset(self, *, initial_state: Connect4State) -> None:
        """Record which side is controlled by the agent for this episode."""
        validate_state = getattr(self.move_selector, "validate_state", None)
        if callable(validate_state):
            validate_state(state=initial_state)
        self._agent_side = cast(str, initial_state.current_player)

    def is_agent_turn(self, *, state: Connect4State) -> bool:
        """Return whether `state` is controlled by the agent."""
        return state.current_player == self._agent_side

    def select_internal_action(
        self,
        *,
        state: Connect4State,
        backend: GameBackend[Connect4State, Connect4Action],
    ) -> AutoAction[Connect4Action] | None:
        """Return one selector-chosen opponent move."""
        action = self.move_selector.select_action(state=state)
        parse_result = backend.parse_action(state, action.label)
        if parse_result.error is not None:
            raise ValueError(
                "Connect 4 move selector produced an action that backend parsing "
                f"rejected: {action.label!r}."
            )
        return AutoAction(
            source=self.source,
            raw_action=action.label,
            action=parse_result.require_action(),
        )

    def episode_boundary(self, *, state: Connect4State) -> EpisodeBoundary | None:
        """Return no additional task boundary for selector-played games."""
        del state
        return None

    def close(self) -> None:
        """Close the underlying selector when it owns external resources."""
        close_method = getattr(self.move_selector, "close", None)
        if callable(close_method):
            close_method()


__all__ = [
    "Connect4MoveSelector",
    "Connect4SolverAutoAdvancePolicy",
]
