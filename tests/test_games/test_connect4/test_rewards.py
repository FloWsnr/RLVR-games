"""Connect 4 reward tests."""

from rlvr_games.games.connect4 import (
    Connect4Action,
    Connect4State,
    TerminalOutcomeReward,
)
from tests.test_games.test_connect4.support import PRE_WIN_BOARD, X_WIN_BOARD


def test_terminal_outcome_reward_uses_mover_perspective_for_wins() -> None:
    reward_fn = TerminalOutcomeReward(
        perspective="mover",
        win_reward=1.0,
        draw_reward=0.0,
        loss_reward=-1.0,
    )
    previous_state = Connect4State(
        board=PRE_WIN_BOARD,
        connect_length=4,
    )
    next_state = Connect4State(
        board=X_WIN_BOARD,
        connect_length=4,
    )

    reward = reward_fn.evaluate(
        previous_state=previous_state,
        action=Connect4Action(column=0),
        next_state=next_state,
        transition_info={},
    )

    assert reward == 1.0


def test_terminal_outcome_reward_returns_zero_for_non_terminal_transitions() -> None:
    reward_fn = TerminalOutcomeReward(
        perspective="x",
        win_reward=1.0,
        draw_reward=0.0,
        loss_reward=-1.0,
    )
    previous_state = Connect4State(
        board=PRE_WIN_BOARD,
        connect_length=4,
    )
    next_state = Connect4State(
        board=PRE_WIN_BOARD,
        connect_length=4,
    )

    reward = reward_fn.evaluate(
        previous_state=previous_state,
        action=Connect4Action(column=3),
        next_state=next_state,
        transition_info={},
    )

    assert reward == 0.0
