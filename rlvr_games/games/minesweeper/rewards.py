"""Reward helpers for Minesweeper."""

from rlvr_games.games.minesweeper.actions import MinesweeperAction, MinesweeperVerb
from rlvr_games.games.minesweeper.state import MinesweeperState


class OutcomeReward:
    """Sparse terminal reward for Minesweeper."""

    def __init__(self, *, win_reward: float, loss_reward: float) -> None:
        """Initialize the reward with explicit terminal values.

        Parameters
        ----------
        win_reward : float
            Reward assigned to successful board clears.
        loss_reward : float
            Reward assigned to mine hits.
        """
        self.win_reward = win_reward
        self.loss_reward = loss_reward

    def evaluate(
        self,
        *,
        previous_state: MinesweeperState,
        action: MinesweeperAction,
        next_state: MinesweeperState,
        transition_info: dict[str, object],
    ) -> float:
        """Return terminal reward for winning or losing transitions."""
        del previous_state
        del action
        del transition_info
        if not next_state.is_terminal:
            return 0.0
        if next_state.outcome.won:
            return self.win_reward
        return self.loss_reward


class SafeRevealCountReward:
    """Dense reward equal to newly revealed safe cells."""

    def __init__(self, *, mine_penalty: float) -> None:
        """Initialize the reward with an explicit mine penalty.

        Parameters
        ----------
        mine_penalty : float
            Reward assigned to reveal actions that hit a mine.
        """
        self.mine_penalty = mine_penalty

    def evaluate(
        self,
        *,
        previous_state: MinesweeperState,
        action: MinesweeperAction,
        next_state: MinesweeperState,
        transition_info: dict[str, object],
    ) -> float:
        """Return newly revealed safe cells for successful reveal actions."""
        del previous_state
        del next_state
        if action.verb != MinesweeperVerb.REVEAL:
            return 0.0
        exploded = transition_info.get("exploded")
        if not isinstance(exploded, bool):
            raise TypeError("Minesweeper transition_info['exploded'] must be a bool.")
        if exploded:
            return self.mine_penalty

        newly_revealed_count = transition_info.get("newly_revealed_count")
        if not isinstance(newly_revealed_count, int):
            raise TypeError(
                "Minesweeper transition_info['newly_revealed_count'] must be an int."
            )
        return float(newly_revealed_count)
