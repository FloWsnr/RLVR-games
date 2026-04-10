"""Reward helpers for 2048."""

from rlvr_games.games.game2048.actions import Game2048Action
from rlvr_games.games.game2048.state import Game2048State


class ScoreDeltaReward:
    """Reward function that returns the score gained by a legal move."""

    def evaluate(
        self,
        *,
        previous_state: Game2048State,
        action: Game2048Action,
        next_state: Game2048State,
        transition_info: dict[str, object],
    ) -> float:
        """Return the merge-score delta produced by the transition.

        Parameters
        ----------
        previous_state : Game2048State
            Canonical state before the move.
        action : Game2048Action
            Parsed direction applied to the board.
        next_state : Game2048State
            Canonical state after the move and random tile spawn.
        transition_info : dict[str, object]
            Transition metadata emitted by the backend.

        Returns
        -------
        float
            Score gained by merges in the move.
        """
        del previous_state
        del action
        del next_state
        score_gain = transition_info["score_gain"]
        if not isinstance(score_gain, int):
            raise TypeError("2048 transition_info['score_gain'] must be an int.")
        return float(score_gain)
