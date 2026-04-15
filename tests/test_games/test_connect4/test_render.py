"""Connect 4 rendering tests."""

from rlvr_games.games.connect4 import (
    Connect4AsciiBoardFormatter,
    Connect4ImageRenderer,
    Connect4ObservationRenderer,
    Connect4State,
)
from tests.test_games.test_connect4.support import PRE_WIN_BOARD


def test_observation_renderer_includes_text_metadata_and_images() -> None:
    renderer = Connect4ObservationRenderer(
        board_formatter=Connect4AsciiBoardFormatter(),
        image_renderer=Connect4ImageRenderer(size=280),
    )
    state = Connect4State(
        board=PRE_WIN_BOARD,
        connect_length=4,
    )

    observation = renderer.render(state)

    assert "Connect 4 board:" in (observation.text or "")
    assert "Current player: x" in (observation.text or "")
    assert observation.metadata["current_player"] == "x"
    assert len(observation.images) == 1
    assert observation.images[0].key.startswith("connect4-board-")
