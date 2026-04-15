"""Helpers for building agent-facing action context."""

from dataclasses import dataclass
from typing import TypeVar

from rlvr_games.core.protocol import Environment

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


@dataclass(slots=True)
class ActionContext:
    """Agent-facing context for choosing the next action.

    Attributes
    ----------
    turn_index : int
        Zero-based turn index for the next action to be taken.
    """

    turn_index: int


def build_action_context(*, env: Environment[StateT, ActionT]) -> ActionContext:
    """Build the agent-facing action context for the current turn.

    Parameters
    ----------
    env : Environment[StateT, ActionT]
        Active environment whose current state should be exposed to the agent.

    Returns
    -------
    ActionContext
        Turn index for the current agent-visible interaction step.
    """
    return ActionContext(turn_index=len(env.trajectory.steps))
