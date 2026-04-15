"""Factory helpers for constructing Minesweeper environments."""

from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.protocol import RewardFn
from rlvr_games.core.types import EpisodeConfig
from rlvr_games.games.minesweeper.actions import MinesweeperAction
from rlvr_games.games.minesweeper.backend import MinesweeperBackend
from rlvr_games.games.minesweeper.engine import MineBoard
from rlvr_games.games.minesweeper.render import (
    MinesweeperAsciiBoardFormatter,
    MinesweeperImageRenderer,
    MinesweeperObservationRenderer,
)
from rlvr_games.games.minesweeper.scenarios import (
    FixedBoardScenario,
    RandomBoardScenario,
)
from rlvr_games.games.minesweeper.state import (
    MinesweeperState,
    inspect_minesweeper_state,
)


def make_minesweeper_env(
    *,
    rows: int,
    columns: int,
    mine_count: int,
    initial_board: MineBoard | None,
    reward_fn: RewardFn[MinesweeperState, MinesweeperAction],
    config: EpisodeConfig,
    include_images: bool,
    image_size: int,
) -> TurnBasedEnv[MinesweeperState, MinesweeperAction]:
    """Construct a fully wired Minesweeper environment.

    Parameters
    ----------
    rows : int
        Board row count used when `initial_board` is not supplied.
    columns : int
        Board column count used when `initial_board` is not supplied.
    mine_count : int
        Number of mines to place when `initial_board` is not supplied.
    initial_board : MineBoard | None
        Explicit hidden mine layout to use. When `None`, the environment uses
        deferred random placement on the first reveal action.
    reward_fn : RewardFn[MinesweeperState, MinesweeperAction]
        Reward function used to score verified transitions.
    config : EpisodeConfig
        Episode execution configuration controlling invalid-action handling and
        optional attempt or transition limits.
    include_images : bool
        Whether observations should include rendered board images.
    image_size : int
        Raster image size in pixels. Ignored when `include_images` is `False`.

    Returns
    -------
    TurnBasedEnv[MinesweeperState, MinesweeperAction]
        Minesweeper environment wired with the standard backend, scenario,
        renderer, and supplied reward function.
    """
    if initial_board is None:
        scenario = RandomBoardScenario(
            rows=rows,
            columns=columns,
            mine_count=mine_count,
        )
    else:
        scenario = FixedBoardScenario(hidden_board=initial_board)

    image_renderer: MinesweeperImageRenderer | None = None
    if include_images:
        image_renderer = MinesweeperImageRenderer(size=image_size)

    return TurnBasedEnv(
        backend=MinesweeperBackend(),
        scenario=scenario,
        renderer=MinesweeperObservationRenderer(
            board_formatter=MinesweeperAsciiBoardFormatter(),
            image_renderer=image_renderer,
        ),
        inspect_state_fn=inspect_minesweeper_state,
        reward_fn=reward_fn,
        config=config,
    )
