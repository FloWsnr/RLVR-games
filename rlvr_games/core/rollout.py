"""Helpers for building agent-facing action context."""

from copy import deepcopy
from typing import TypeVar

from rlvr_games.core.action_context import (
    ActionContext,
    ProjectedActionContext,
    PublicResetEvent,
)
from rlvr_games.core.protocol import Environment

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


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
    turn_index = len(env.trajectory.steps)
    projector = env.agent_context_projector
    if projector is None:
        return ActionContext(turn_index=turn_index)

    projected_context = projector.project_action_context(
        state=deepcopy(env.state),
        reset_events=tuple(
            PublicResetEvent(
                source=event.source,
                label=event.label,
                info=event.info,
            )
            for event in env.trajectory.reset_events
        ),
    )
    if not isinstance(projected_context, ProjectedActionContext):
        raise TypeError(
            "AgentContextProjector.project_action_context() must return "
            "ProjectedActionContext."
        )
    return ActionContext(
        turn_index=turn_index,
        opening_events=projected_context.opening_events,
    )
