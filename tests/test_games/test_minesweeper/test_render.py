"""Minesweeper renderer tests."""

from rlvr_games.games.minesweeper import (
    MinesweeperAsciiBoardFormatter,
    MinesweeperImageRenderer,
    MinesweeperObservationRenderer,
    MinesweeperState,
    normalize_initial_board,
)

FIXED_BOARD = ("*..", "...", "..*")


def test_observation_renderer_hides_mines_in_text_and_metadata() -> None:
    renderer = MinesweeperObservationRenderer(
        board_formatter=MinesweeperAsciiBoardFormatter(),
        image_renderer=MinesweeperImageRenderer(size=240),
    )
    state = MinesweeperState(
        rows=3,
        columns=3,
        mine_count=2,
        hidden_board=normalize_initial_board(board=FIXED_BOARD),
        move_count=0,
        placement_seed=None,
    )

    observation = renderer.render(state)

    assert "Minesweeper board:" in (observation.text or "")
    assert "Action format: reveal <row> <col>" in (observation.text or "")
    assert "hidden_board" not in observation.metadata
    assert observation.metadata["visible_board"] == (
        ("#", "#", "#"),
        ("#", "#", "#"),
        ("#", "#", "#"),
    )
    assert len(observation.images) == 1
    assert observation.images[0].key.startswith("minesweeper-board-")


def test_image_renderer_is_deterministic_for_the_same_state() -> None:
    renderer = MinesweeperImageRenderer(size=240)
    state = MinesweeperState(
        rows=3,
        columns=3,
        mine_count=2,
        hidden_board=normalize_initial_board(board=FIXED_BOARD),
        revealed=(
            (False, True, True),
            (False, True, True),
            (False, False, False),
        ),
        move_count=1,
        placement_seed=None,
    )

    first_render = renderer.render_images(state)[0]
    second_render = renderer.render_images(state)[0]

    assert second_render.key == first_render.key
    assert second_render.image.tobytes() == first_render.image.tobytes()
