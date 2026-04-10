"""Factory helpers for constructing 2048 environments."""

from typing import Sequence

from rlvr_games.core.types import EpisodeConfig
from rlvr_games.games.game2048.backend import Game2048Backend
from rlvr_games.games.game2048.env import Game2048Env
from rlvr_games.games.game2048.render import (
    Game2048AsciiBoardFormatter,
    Game2048ImageRenderer,
    Game2048ObservationRenderer,
)
from rlvr_games.games.game2048.rewards import ScoreDeltaReward
from rlvr_games.games.game2048.scenarios import (
    FixedBoardScenario,
    RandomStartScenario,
    normalize_initial_board,
)


def make_game2048_env(
    *,
    size: int,
    target_value: int,
    initial_board: Sequence[Sequence[int]] | None,
    initial_score: int,
    initial_move_count: int,
    config: EpisodeConfig,
    include_images: bool,
    image_size: int,
) -> Game2048Env:
    """Construct a fully wired 2048 environment.

    Parameters
    ----------
    size : int
        Board size used when `initial_board` is not supplied.
    target_value : int
        Tile value that counts as a win.
    initial_board : Sequence[Sequence[int]] | None
        Explicit initial board to use. When `None`, the environment starts from
        a seeded random board with two spawned tiles.
    initial_score : int
        Starting score used only when `initial_board` is supplied.
    initial_move_count : int
        Starting move count used only when `initial_board` is supplied.
    config : EpisodeConfig
        Episode execution configuration controlling invalid-action handling and
        optional attempt or transition limits.
    include_images : bool
        Whether observations should include rendered board images.
    image_size : int
        Raster image size in pixels. Ignored when `include_images` is `False`.

    Returns
    -------
    Game2048Env
        2048 environment wired with the standard backend, scenario, renderer,
        and score-delta reward function.
    """
    if initial_board is None:
        scenario = RandomStartScenario(
            size=size,
            target_value=target_value,
            start_tile_count=2,
        )
    else:
        normalized_board = normalize_initial_board(board=initial_board)
        if len(normalized_board) != size:
            raise ValueError(
                f"Configured 2048 board size {size} does not match initial board size "
                f"{len(normalized_board)}."
            )
        scenario = FixedBoardScenario(
            initial_board=normalized_board,
            initial_score=initial_score,
            initial_move_count=initial_move_count,
            target_value=target_value,
        )

    image_renderer: Game2048ImageRenderer | None = None
    if include_images:
        image_renderer = Game2048ImageRenderer(size=image_size)

    return Game2048Env(
        backend=Game2048Backend(),
        scenario=scenario,
        renderer=Game2048ObservationRenderer(
            board_formatter=Game2048AsciiBoardFormatter(),
            image_renderer=image_renderer,
        ),
        reward_fn=ScoreDeltaReward(),
        config=config,
    )
