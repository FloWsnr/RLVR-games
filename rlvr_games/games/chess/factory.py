"""Factory helpers for constructing chess environments."""

from enum import StrEnum

import chess

from rlvr_games.core.rewards import ZeroReward
from rlvr_games.core.types import EpisodeConfig
from rlvr_games.games.chess.backend import ChessBackend
from rlvr_games.games.chess.env import ChessEnv
from rlvr_games.games.chess.render import (
    AsciiBoardFormatter,
    ChessFastImageRenderer,
    ChessObservationRenderer,
    UnicodeBoardFormatter,
)
from rlvr_games.games.chess.scenarios import StartingPositionScenario


class ChessTextRendererKind(StrEnum):
    """Text renderer variants supported by the chess environment factory."""

    ASCII = "ascii"
    UNICODE = "unicode"


class ChessBoardOrientation(StrEnum):
    """Board orientations supported by chess observation rendering."""

    WHITE = "white"
    BLACK = "black"


def make_chess_env(
    *,
    initial_fen: str,
    config: EpisodeConfig,
    text_renderer_kind: ChessTextRendererKind,
    include_images: bool,
    image_size: int,
    image_coordinates: bool,
    orientation: ChessBoardOrientation,
) -> ChessEnv:
    """Construct a fully wired chess environment.

    Parameters
    ----------
    initial_fen : str
        Starting position for the episode scenario.
    config : EpisodeConfig
        Episode execution configuration controlling invalid-action handling and
        optional attempt/transition limits.
    text_renderer_kind : ChessTextRendererKind
        Text board formatter to expose in observations.
    include_images : bool
        Whether observations should include rendered board images.
    image_size : int
        Raster image size in pixels. Ignored when `include_images` is `False`.
    image_coordinates : bool
        Whether raster image renders should include rank/file coordinates.
        Ignored when `include_images` is `False`.
    orientation : ChessBoardOrientation
        Side to place at the bottom of text and raster board renders.

    Returns
    -------
    ChessEnv
        Chess environment wired with the standard backend, scenario, renderer,
        and zero reward function.
    """
    chess_orientation: chess.Color = chess.WHITE
    if orientation == ChessBoardOrientation.BLACK:
        chess_orientation = chess.BLACK

    board_formatter: AsciiBoardFormatter | UnicodeBoardFormatter
    board_formatter = AsciiBoardFormatter(orientation=chess_orientation)
    if text_renderer_kind == ChessTextRendererKind.UNICODE:
        board_formatter = UnicodeBoardFormatter(orientation=chess_orientation)

    image_renderer: ChessFastImageRenderer | None
    image_renderer = None
    if include_images:
        image_renderer = ChessFastImageRenderer(
            size=image_size,
            coordinates=image_coordinates,
            orientation=chess_orientation,
        )

    return ChessEnv(
        backend=ChessBackend(),
        scenario=StartingPositionScenario(initial_fen=initial_fen),
        renderer=ChessObservationRenderer(
            board_formatter=board_formatter,
            image_renderer=image_renderer,
        ),
        reward_fn=ZeroReward(),
        config=config,
    )
