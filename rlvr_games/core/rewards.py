"""Small reusable reward helpers."""

from typing import Any, Generic, TypeVar

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


class ZeroReward(Generic[StateT, ActionT]):
    """Reward function that always returns zero.

    This is useful for scaffolding environments before task-specific rewards
    are defined, or for debugging transition logic independently from reward
    shaping.
    """

    def evaluate(
        self,
        *,
        previous_state: StateT,
        action: ActionT,
        next_state: StateT,
        transition_info: dict[str, Any],
    ) -> float:
        """Return a constant zero reward for any transition.

        Parameters
        ----------
        previous_state : StateT
            Canonical state before the transition.
        action : ActionT
            Parsed action that was applied.
        next_state : StateT
            Canonical state after the transition.
        transition_info : dict[str, Any]
            Verifier metadata for the transition. It is ignored by this
            implementation.

        Returns
        -------
        float
            Always `0.0`.
        """
        return 0.0
