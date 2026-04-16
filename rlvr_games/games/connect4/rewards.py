"""Reward helpers for Connect 4."""

from dataclasses import dataclass
from typing import Literal, cast

from rlvr_games.games.connect4.actions import Connect4Action
from rlvr_games.games.connect4.solver import Connect4MoveScorer
from rlvr_games.games.connect4.state import Connect4State

Connect4Perspective = Literal["x", "o"]
Connect4RewardPerspective = Literal["x", "o", "mover"]


def resolve_reward_perspective(
    *,
    previous_state: Connect4State,
    perspective: Connect4RewardPerspective,
) -> Connect4Perspective:
    """Resolve a reward perspective into a concrete Connect 4 side label.

    Parameters
    ----------
    previous_state : Connect4State
        Canonical state before the accepted move.
    perspective : {"x", "o", "mover"}
        Configured reward perspective.

    Returns
    -------
    {"x", "o"}
        Concrete side label used for reward evaluation.
    """
    if perspective == "mover":
        return cast(Connect4Perspective, previous_state.current_player)
    return perspective


@dataclass(slots=True, frozen=True)
class TerminalOutcomeReward:
    """Sparse reward based on the terminal game outcome for one side.

    Attributes
    ----------
    perspective : {"x", "o", "mover"}
        Side whose outcome should determine the reward sign.
    win_reward : float
        Reward returned when `perspective` wins the game.
    draw_reward : float
        Reward returned for terminal drawn positions.
    loss_reward : float
        Reward returned when `perspective` loses the game.
    """

    perspective: Connect4RewardPerspective
    win_reward: float
    draw_reward: float
    loss_reward: float

    def __post_init__(self) -> None:
        """Validate that the configured perspective is supported.

        Raises
        ------
        ValueError
            If `perspective` is not one of the supported values.
        """
        if self.perspective not in ("x", "o", "mover"):
            raise ValueError(
                "Connect 4 reward perspective must be 'x', 'o', or 'mover'."
            )

    def evaluate(
        self,
        *,
        previous_state: Connect4State,
        action: Connect4Action,
        next_state: Connect4State,
        transition_info: dict[str, object],
    ) -> float:
        """Return the configured terminal reward for the evaluated side.

        Parameters
        ----------
        previous_state : Connect4State
            Canonical state before the move.
        action : Connect4Action
            Parsed column action. It is ignored by this sparse reward.
        next_state : Connect4State
            Canonical state after the move.
        transition_info : dict[str, object]
            Transition metadata emitted by the backend. It is ignored by this
            reward implementation.

        Returns
        -------
        float
            Configured win, draw, or loss reward for terminal transitions,
            otherwise `0.0`.
        """
        del action
        del transition_info
        if not next_state.outcome.is_terminal:
            return 0.0

        perspective = resolve_reward_perspective(
            previous_state=previous_state,
            perspective=self.perspective,
        )
        winner = next_state.outcome.winner
        if winner is None:
            return self.draw_reward
        if winner == perspective:
            return self.win_reward
        return self.loss_reward


@dataclass(slots=True, frozen=True)
class SolverMoveScoreReward:
    """Dense reward equal to the solver score of the chosen move.

    Attributes
    ----------
    scorer : Connect4MoveScorer
        Move scorer whose scalar output defines the reward scale.
    perspective : {"x", "o", "mover"}
        Side whose point of view determines the reward sign. `"mover"`
        scores each action from the side that just acted.
    """

    scorer: Connect4MoveScorer
    perspective: Connect4RewardPerspective

    def __post_init__(self) -> None:
        """Validate that the configured perspective is supported.

        Raises
        ------
        ValueError
            If `perspective` is not one of the supported values.
        """
        if self.perspective not in ("x", "o", "mover"):
            raise ValueError(
                "Connect 4 reward perspective must be 'x', 'o', or 'mover'."
            )

    def evaluate(
        self,
        *,
        previous_state: Connect4State,
        action: Connect4Action,
        next_state: Connect4State,
        transition_info: dict[str, object],
    ) -> float:
        """Return the solver score for the accepted move.

        Parameters
        ----------
        previous_state : Connect4State
            Canonical state before the move.
        action : Connect4Action
            Parsed move that was applied.
        next_state : Connect4State
            Canonical state after the move. It is ignored by this reward.
        transition_info : dict[str, object]
            Transition metadata emitted by the backend. It is ignored by this
            reward implementation.

        Returns
        -------
        float
            Solver score for `action` from the configured perspective.
        """
        del next_state
        del transition_info
        concrete_perspective = resolve_reward_perspective(
            previous_state=previous_state,
            perspective=self.perspective,
        )
        return self.scorer.score_action(
            state=previous_state,
            action=action,
            perspective=concrete_perspective,
        )

    def reset(self, *, initial_state: Connect4State) -> None:
        """Validate reset-time state compatibility for the configured scorer.

        Parameters
        ----------
        initial_state : Connect4State
            Canonical state visible to the agent after reset-time events.
        """
        validate_state = getattr(self.scorer, "validate_state", None)
        if callable(validate_state):
            validate_state(state=initial_state)

    def close(self) -> None:
        """Close the underlying scorer when it owns external resources.

        Returns
        -------
        None
            This method is a no-op for pure in-memory scorers.
        """
        close_method = getattr(self.scorer, "close", None)
        if callable(close_method):
            close_method()
