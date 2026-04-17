"""Reward helpers for Mastermind."""

from rlvr_games.games.mastermind.actions import MastermindAction
from rlvr_games.games.mastermind.state import MastermindState


class TerminalOutcomeReward:
    """Sparse terminal reward for Mastermind."""

    def __init__(self, *, win_reward: float, loss_reward: float) -> None:
        """Initialize the reward with explicit terminal values."""
        self.win_reward = win_reward
        self.loss_reward = loss_reward

    def evaluate(
        self,
        *,
        previous_state: MastermindState,
        action: MastermindAction,
        next_state: MastermindState,
        transition_info: dict[str, object],
    ) -> float:
        """Return terminal reward for cracking or failing the code."""
        del previous_state
        del action
        del transition_info
        if not next_state.is_terminal:
            return 0.0
        if next_state.outcome.won:
            return self.win_reward
        return self.loss_reward


class CandidateReductionDenseReward:
    """Dense reward based on shrinking the consistent secret-code set."""

    def evaluate(
        self,
        *,
        previous_state: MastermindState,
        action: MastermindAction,
        next_state: MastermindState,
        transition_info: dict[str, object],
    ) -> float:
        """Return normalized candidate-set reduction for one accepted guess.

        Solved transitions return ``1.0`` so the dense reward retains an
        unambiguous success signal in addition to intermediate pruning.
        """
        del action
        del transition_info

        if previous_state.candidate_count <= 0:
            raise ValueError(
                "Mastermind previous_state.candidate_count must be positive."
            )
        if next_state.candidate_count > previous_state.candidate_count:
            raise ValueError(
                "Mastermind next_state.candidate_count cannot increase after a "
                "verified guess."
            )
        if next_state.outcome.won:
            return 1.0
        reduction = previous_state.candidate_count - next_state.candidate_count
        return float(reduction) / float(previous_state.candidate_count)
