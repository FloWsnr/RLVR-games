"""Minesweeper environment scaffolding."""

from rlvr_games.games.minesweeper.actions import (
    MinesweeperAction,
    MinesweeperVerb,
    VERB_ALIASES,
    serialize_minesweeper_action,
)
from rlvr_games.games.minesweeper.backend import MinesweeperBackend
from rlvr_games.games.minesweeper.engine import (
    AdjacentMineBoard,
    BoolBoard,
    Coordinate,
    MineBoard,
    adjacent_mine_counts,
    count_mines,
    make_boolean_board,
    mine_board_text_rows,
    normalize_mine_board,
    place_random_mines,
    reveal_cells,
)
from rlvr_games.games.minesweeper.factory import make_minesweeper_env
from rlvr_games.games.minesweeper.render import (
    MinesweeperAsciiBoardFormatter,
    MinesweeperImageRenderer,
    MinesweeperObservationRenderer,
)
from rlvr_games.games.minesweeper.rewards import OutcomeReward, SafeRevealCountReward
from rlvr_games.games.minesweeper.scenarios import (
    FixedBoardScenario,
    RandomBoardScenario,
    STANDARD_MINESWEEPER_COLUMNS,
    STANDARD_MINESWEEPER_MINE_COUNT,
    STANDARD_MINESWEEPER_ROWS,
    normalize_initial_board,
)
from rlvr_games.games.minesweeper.state import (
    MinesweeperOutcome,
    MinesweeperState,
    inspect_minesweeper_state,
    public_minesweeper_board,
    public_minesweeper_metadata,
)

__all__ = [
    "AdjacentMineBoard",
    "BoolBoard",
    "Coordinate",
    "FixedBoardScenario",
    "MineBoard",
    "MinesweeperAction",
    "MinesweeperAsciiBoardFormatter",
    "MinesweeperBackend",
    "MinesweeperImageRenderer",
    "MinesweeperObservationRenderer",
    "MinesweeperOutcome",
    "MinesweeperState",
    "MinesweeperVerb",
    "OutcomeReward",
    "RandomBoardScenario",
    "STANDARD_MINESWEEPER_COLUMNS",
    "STANDARD_MINESWEEPER_MINE_COUNT",
    "STANDARD_MINESWEEPER_ROWS",
    "SafeRevealCountReward",
    "VERB_ALIASES",
    "adjacent_mine_counts",
    "count_mines",
    "inspect_minesweeper_state",
    "make_boolean_board",
    "make_minesweeper_env",
    "mine_board_text_rows",
    "normalize_initial_board",
    "normalize_mine_board",
    "place_random_mines",
    "public_minesweeper_board",
    "public_minesweeper_metadata",
    "reveal_cells",
    "serialize_minesweeper_action",
]
