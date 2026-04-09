"""Small reusable reward helpers."""

from typing import Any, Generic, TypeVar

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


class ZeroReward(Generic[StateT, ActionT]):
    """Default reward function for scaffolding and debugging."""

    def evaluate(
        self,
        *,
        previous_state: StateT,
        action: ActionT,
        next_state: StateT,
        transition_info: dict[str, Any],
    ) -> float:
        return 0.0
