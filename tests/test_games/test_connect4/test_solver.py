"""Connect 4 BitBully solver tests."""

from rlvr_games.games.connect4 import (
    BitBullySolver,
    Connect4Action,
    Connect4State,
)
from rlvr_games.games.connect4.state import make_empty_board
from tests.test_games.test_connect4.support import PRE_WIN_BOARD


def test_bitbully_solver_scores_and_selects_the_center_on_the_empty_board() -> None:
    solver = BitBullySolver()
    state = Connect4State(
        board=make_empty_board(rows=6, columns=7),
        connect_length=4,
    )

    assert solver.score_actions(state=state, perspective="x") == {
        "4": 1.0,
        "3": 0.0,
        "5": 0.0,
        "2": -1.0,
        "6": -1.0,
        "1": -2.0,
        "7": -2.0,
    }
    assert (
        solver.score_action(
            state=state,
            action=Connect4Action(column=3),
            perspective="x",
        )
        == 1.0
    )
    assert (
        solver.score_action(
            state=state,
            action=Connect4Action(column=3),
            perspective="o",
        )
        == -1.0
    )
    assert solver.select_action(state=state).label == "4"


def test_bitbully_solver_identifies_the_forced_winning_move() -> None:
    solver = BitBullySolver()
    state = Connect4State(
        board=PRE_WIN_BOARD,
        connect_length=4,
    )

    assert solver.select_action(state=state).label == "4"
    assert (
        solver.score_action(
            state=state,
            action=Connect4Action(column=3),
            perspective="x",
        )
        == 18.0
    )
    assert next(iter(solver.score_actions(state=state, perspective="x"))) == "4"
