"""2048 state tests."""

from random import Random

from rlvr_games.games.game2048 import Game2048State


def test_target_tile_terminal_state_has_no_legal_actions() -> None:
    state = Game2048State(
        board=((2048, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
        score=2048,
        move_count=1,
        target_value=2048,
        rng_state=Random(0).getstate(),
    )

    assert state.is_terminal is True
    assert state.legal_actions == ()
    assert state.legal_action_count == 0
