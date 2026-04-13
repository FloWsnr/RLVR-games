"""Chess renderer tests."""

import chess
from PIL import Image

from rlvr_games.core.types import RenderedImage
from rlvr_games.games.chess import (
    AsciiBoardFormatter,
    ChessFastImageRenderer,
    ChessObservationRenderer,
    ChessState,
    UnicodeBoardFormatter,
)
from rlvr_games.games.chess.scenarios import STANDARD_START_FEN


class StubChessImageRenderer:
    """Small stub image renderer for observation tests."""

    def render_images(self, board: chess.Board) -> tuple[RenderedImage, ...]:
        """Return a single fixed image payload."""
        del board
        return (
            RenderedImage(
                key="stub-chess-board",
                image=Image.new("RGBA", (32, 32), (255, 0, 0, 255)),
            ),
        )


def test_observation_renderer_can_emit_text_and_images() -> None:
    renderer = ChessObservationRenderer(
        board_formatter=AsciiBoardFormatter(orientation=chess.WHITE),
        image_renderer=StubChessImageRenderer(),
    )

    observation = renderer.render(ChessState(fen=STANDARD_START_FEN))

    assert observation.text is not None
    assert "Chess board:" in observation.text
    assert len(observation.images) == 1
    assert observation.images[0].key == "stub-chess-board"
    assert observation.images[0].image.size == (32, 32)


def test_fast_image_renderer_emits_single_raster_image() -> None:
    renderer = ChessObservationRenderer(
        board_formatter=AsciiBoardFormatter(orientation=chess.WHITE),
        image_renderer=ChessFastImageRenderer(
            size=360,
            coordinates=True,
            orientation=chess.WHITE,
        ),
    )

    observation = renderer.render(ChessState(fen=STANDARD_START_FEN))

    assert len(observation.images) == 1
    rendered_image = observation.images[0]
    assert rendered_image.key.startswith("chess-board-")
    assert rendered_image.image.size == (360, 360)
    assert rendered_image.image.mode == "RGBA"


def test_fast_image_renderer_is_deterministic_for_the_same_position() -> None:
    image_renderer = ChessFastImageRenderer(
        size=360,
        coordinates=True,
        orientation=chess.WHITE,
    )
    board = chess.Board(STANDARD_START_FEN)

    first_render = image_renderer.render_images(board)[0]
    second_render = image_renderer.render_images(board)[0]

    assert second_render.key == first_render.key
    assert second_render.image.tobytes() == first_render.image.tobytes()


def test_unicode_board_formatter_can_be_used_in_observation_renderer() -> None:
    renderer = ChessObservationRenderer(
        board_formatter=UnicodeBoardFormatter(orientation=chess.WHITE),
        image_renderer=None,
    )

    observation = renderer.render(ChessState(fen=STANDARD_START_FEN))

    assert observation.text is not None
    assert "♜" in observation.text
    assert "a b c d e f g h" in observation.text


def test_ascii_board_formatter_supports_black_orientation() -> None:
    formatter = AsciiBoardFormatter(orientation=chess.BLACK)

    board_text = formatter.render_text(chess.Board(STANDARD_START_FEN))

    lines = board_text.splitlines()
    assert lines[0] == "1 R N B K Q B N R"
    assert lines[-1] == "  h g f e d c b a"


def test_unicode_board_formatter_supports_black_orientation() -> None:
    formatter = UnicodeBoardFormatter(orientation=chess.BLACK)

    board_text = formatter.render_text(chess.Board(STANDARD_START_FEN))

    assert "1 |♖|♘|♗|♔|♕|♗|♘|♖|" in board_text
    assert "   h g f e d c b a" in board_text
