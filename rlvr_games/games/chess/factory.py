"""Factory helpers for constructing chess environments."""

from enum import StrEnum
from pathlib import Path

import chess

from rlvr_games.core.rewards import ZeroReward
from rlvr_games.core.types import EpisodeConfig
from rlvr_games.games.chess.backend import ChessBackend
from rlvr_games.games.chess.env import ChessEnv
from rlvr_games.games.chess.render import (
    AsciiBoardFormatter,
    ChessObservationRenderer,
    ChessRasterBoardImageRenderer,
    UnicodeBoardFormatter,
)
from rlvr_games.games.chess.scenarios import StartingPositionScenario


class ChessTextRendererKind(StrEnum):
    """Text renderer variants supported by the chess environment factory."""

    ASCII = "ascii"
    UNICODE = "unicode"


class ChessImageOrientation(StrEnum):
    """Board orientations supported by raster chess image rendering."""

    WHITE = "white"
    BLACK = "black"


def make_chess_env(
    *,
    initial_fen: str,
    max_turns: int | None,
    text_renderer_kind: ChessTextRendererKind,
    image_output_dir: Path | None,
    image_size: int,
    image_coordinates: bool,
    image_orientation: ChessImageOrientation,
) -> ChessEnv:
    """Construct a fully wired chess environment.

    Parameters
    ----------
    initial_fen : str
        Starting position for the episode scenario.
    max_turns : int | None
        Optional episode turn limit. When `None`, no truncation limit is
        applied.
    text_renderer_kind : ChessTextRendererKind
        Text board formatter to expose in observations.
    image_output_dir : Path | None
        Directory used for PNG board renders. When `None`, the environment
        emits no image paths.
    image_size : int
        Raster image size in pixels. Ignored when `image_output_dir` is
        `None`.
    image_coordinates : bool
        Whether raster image renders should include rank/file coordinates.
        Ignored when `image_output_dir` is `None`.
    image_orientation : ChessImageOrientation
        Side to place at the bottom of raster image renders. Ignored when
        `image_output_dir` is `None`.

    Returns
    -------
    ChessEnv
        Chess environment wired with the standard backend, scenario, renderer,
        and zero reward function.
    """
    board_formatter: AsciiBoardFormatter | UnicodeBoardFormatter
    board_formatter = AsciiBoardFormatter()
    if text_renderer_kind == ChessTextRendererKind.UNICODE:
        board_formatter = UnicodeBoardFormatter()

    image_renderer: ChessRasterBoardImageRenderer | None
    image_renderer = None
    if image_output_dir is not None:
        orientation = chess.WHITE
        if image_orientation == ChessImageOrientation.BLACK:
            orientation = chess.BLACK
        image_renderer = ChessRasterBoardImageRenderer(
            output_dir=image_output_dir,
            size=image_size,
            coordinates=image_coordinates,
            orientation=orientation,
        )

    return ChessEnv(
        backend=ChessBackend(),
        scenario=StartingPositionScenario(initial_fen=initial_fen),
        renderer=ChessObservationRenderer(
            board_formatter=board_formatter,
            image_renderer=image_renderer,
        ),
        reward_fn=ZeroReward(),
        config=EpisodeConfig(max_turns=max_turns),
    )
