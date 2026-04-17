"""Helpers for building trainer-facing rollout turns."""

from copy import deepcopy
from dataclasses import dataclass
from typing import TypeVar

from rlvr_games.core.action_context import (
    ActionContext,
    ProjectedActionContext,
    PublicResetEvent,
)
from rlvr_games.core.messages import ChatMessage
from rlvr_games.core.protocol import Environment
from rlvr_games.core.types import Observation

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


@dataclass(slots=True)
class PreparedTurn:
    """Trainer-facing package for one actionable environment turn.

    Attributes
    ----------
    action_context : ActionContext
        Structured context for the next agent action.
    messages : tuple[ChatMessage, ...]
        Chat-formatted prompt messages derived from the current observation and
        action context.
    """

    action_context: ActionContext
    messages: tuple[ChatMessage, ...]


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


def prepare_turn(
    *,
    env: Environment[StateT, ActionT],
    observation: Observation,
) -> PreparedTurn:
    """Prepare one actionable trainer-facing turn from an env observation.

    Parameters
    ----------
    env : Environment[StateT, ActionT]
        Active environment whose current turn should be packaged.
    observation : Observation
        Observation currently shown to the agent.

    Returns
    -------
    PreparedTurn
        Packaged action context and trainer-facing chat messages for the next
        action.

    Raises
    ------
    ValueError
        If the environment has already finished and there is no next action to
        prepare.
    """
    if env.episode_finished:
        raise ValueError("Cannot prepare a turn for a finished episode.")

    action_context = build_action_context(env=env)
    messages = env.messages_for_observation(
        observation,
        action_context=action_context,
    )
    return PreparedTurn(
        action_context=action_context,
        messages=messages,
    )
