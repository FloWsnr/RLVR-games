"""2048 backend tests."""

from random import Random

import pytest

from rlvr_games.games.game2048 import (
    Game2048Backend,
    Game2048ChanceModel,
    Game2048State,
)

MERGE_BOARD = (
    (2, 2, 2, 2),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
)
NOOP_UP_BOARD = (
    (2, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
)


def test_legal_actions_exclude_noop_directions() -> None:
    backend = Game2048Backend(chance_model=Game2048ChanceModel())
    state = Game2048State(
        board=NOOP_UP_BOARD,
        score=0,
        move_count=0,
        target_value=2048,
        rng_state=Random(0).getstate(),
    )

    legal_actions = backend.legal_actions(state)

    assert legal_actions == ["right", "down"]


@pytest.mark.parametrize("raw_action", ["", "north", "up"])
def test_parse_action_rejects_invalid_or_illegal_directions(raw_action: str) -> None:
    backend = Game2048Backend(chance_model=Game2048ChanceModel())
    state = Game2048State(
        board=NOOP_UP_BOARD,
        score=0,
        move_count=0,
        target_value=2048,
        rng_state=Random(0).getstate(),
    )

    parse_result = backend.parse_action(state, raw_action)

    assert parse_result.action is None
    assert parse_result.error is not None


def test_apply_action_merges_pairs_once_and_records_spawn_metadata() -> None:
    backend = Game2048Backend(chance_model=Game2048ChanceModel())
    state = Game2048State(
        board=MERGE_BOARD,
        score=0,
        move_count=0,
        target_value=2048,
        rng_state=Random(5).getstate(),
    )

    action = backend.parse_action(state, "left").require_action()
    next_state, info = backend.apply_action(state, action)

    assert next_state.board == (
        (4, 4, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 2, 0, 0),
    )
    assert next_state.score == 8
    assert next_state.move_count == 1
    assert info["score_gain"] == 8
    assert info["merge_count"] == 2
    assert info["spawned_tile"] == {"row": 3, "col": 1, "value": 2}
    assert info["merges"] == (
        {"row": 0, "col": 0, "value": 4, "sources": ((0, 0), (0, 1))},
        {"row": 0, "col": 1, "value": 4, "sources": ((0, 2), (0, 3))},
    )


def test_apply_reset_spawn_records_authoritative_opening_tile_metadata() -> None:
    backend = Game2048Backend(chance_model=Game2048ChanceModel())
    state = Game2048State(
        board=(
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        ),
        score=0,
        move_count=0,
        target_value=2048,
        rng_state=Random(0).getstate(),
    )

    next_state, info = backend.apply_reset_spawn(state)

    assert next_state.board == (
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 0, 0),
        (0, 2, 0, 0),
    )
    assert next_state.score == 0
    assert next_state.move_count == 0
    assert info["spawned_tile"] == {"row": 3, "col": 1, "value": 2}
    assert info["board"] == next_state.board
