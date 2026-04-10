"""Reusable rollout helpers for running agents against environments."""

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from rlvr_games.core.protocol import Environment
from rlvr_games.core.trajectory import EpisodeTrajectory
from rlvr_games.core.types import Observation

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


@dataclass(slots=True)
class ActionContext:
    """Agent-facing context for choosing the next action.

    Attributes
    ----------
    turn_index : int
        Zero-based turn index for the next action to be taken.
    legal_actions : tuple[str, ...]
        Legal serialized actions accepted by the environment for the current
        state.
    """

    turn_index: int
    legal_actions: tuple[str, ...]


class RolloutAgent(Protocol):
    """Protocol for agents that emit raw environment actions."""

    def act(self, observation: Observation, context: ActionContext) -> str:
        """Return the next raw action for the supplied observation.

        Parameters
        ----------
        observation : Observation
            Current model-facing observation emitted by the environment.
        context : ActionContext
            Auxiliary turn data such as the legal action list.

        Returns
        -------
        str
            Raw action string to pass to `env.step(...)`.
        """
        ...


@dataclass(slots=True)
class RolloutResult(Generic[ActionT]):
    """Terminal rollout summary returned by `run_episode`.

    Attributes
    ----------
    trajectory : EpisodeTrajectory[ActionT]
        Recorded episode trajectory accumulated by the environment.
    terminated : bool
        Whether the episode ended via a game-defined terminal condition.
    truncated : bool
        Whether the episode ended because of an external cutoff such as a turn
        limit.
    turn_count : int
        Number of accepted environment steps recorded in `trajectory`.
    """

    trajectory: EpisodeTrajectory[ActionT]
    terminated: bool
    truncated: bool
    turn_count: int


def build_action_context(*, env: Environment[StateT, ActionT]) -> ActionContext:
    """Build the agent-facing action context for the current turn.

    Parameters
    ----------
    env : Environment[StateT, ActionT]
        Active environment whose current state should be exposed to the agent.

    Returns
    -------
    ActionContext
        Turn index and serialized legal actions for the current state.
    """
    return ActionContext(
        turn_index=len(env.trajectory.steps),
        legal_actions=tuple(env.backend.legal_actions(env.state)),
    )


def run_episode(
    *,
    env: Environment[StateT, ActionT],
    agent: RolloutAgent,
    seed: int,
) -> RolloutResult[ActionT]:
    """Run a single episode to completion against the supplied environment.

    Parameters
    ----------
    env : Environment[StateT, ActionT]
        Stateful environment to reset and step in-process.
    agent : RolloutAgent
        Agent that emits raw action strings in response to observations.
    seed : int
        Explicit seed passed into `env.reset(...)`.

    Returns
    -------
    RolloutResult[ActionT]
        Final rollout summary containing the recorded trajectory and terminal
        flags.

    Raises
    ------
    InvalidActionError
        Propagated when the agent emits an invalid or illegal action and the
        environment is configured to fail fast.
    """
    observation, _ = env.reset(seed=seed)

    if env.backend.is_terminal(env.state):
        return RolloutResult(
            trajectory=env.trajectory,
            terminated=True,
            truncated=False,
            turn_count=0,
        )

    while True:
        context = build_action_context(env=env)
        raw_action = agent.act(observation, context)
        step_result = env.step(raw_action)
        observation = step_result.observation

        if step_result.terminated or step_result.truncated:
            return RolloutResult(
                trajectory=env.trajectory,
                terminated=step_result.terminated,
                truncated=step_result.truncated,
                turn_count=len(env.trajectory.steps),
            )
