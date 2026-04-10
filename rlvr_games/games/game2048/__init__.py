"""2048 environment scaffolding."""

from rlvr_games.games.game2048.actions import (
    ACTION_ORDER,
    Game2048Action,
    MoveDirection,
)
from rlvr_games.games.game2048.backend import Game2048Backend
from rlvr_games.games.game2048.engine import (
    Board,
    MergeSummary,
    MoveSummary,
    SpawnOutcome,
    SpawnSummary,
    spawn_outcomes,
)
from rlvr_games.games.game2048.env import Game2048Env
from rlvr_games.games.game2048.factory import make_game2048_env
from rlvr_games.games.game2048.render import (
    Game2048AsciiBoardFormatter,
    Game2048ImageRenderer,
    Game2048ObservationRenderer,
)
from rlvr_games.games.game2048.rewards import ScoreDeltaReward
from rlvr_games.games.game2048.scenarios import (
    FixedBoardScenario,
    RandomStartScenario,
    STANDARD_2048_SIZE,
    STANDARD_2048_TARGET,
    STANDARD_START_TILE_COUNT,
    normalize_initial_board,
)
from rlvr_games.games.game2048.state import Game2048Outcome, Game2048State

__all__ = [
    "ACTION_ORDER",
    "Board",
    "FixedBoardScenario",
    "Game2048Action",
    "Game2048AsciiBoardFormatter",
    "Game2048Backend",
    "Game2048Env",
    "Game2048ImageRenderer",
    "Game2048ObservationRenderer",
    "Game2048Outcome",
    "Game2048State",
    "MergeSummary",
    "MoveDirection",
    "MoveSummary",
    "RandomStartScenario",
    "ScoreDeltaReward",
    "SpawnOutcome",
    "SpawnSummary",
    "STANDARD_2048_SIZE",
    "STANDARD_2048_TARGET",
    "STANDARD_START_TILE_COUNT",
    "make_game2048_env",
    "normalize_initial_board",
    "spawn_outcomes",
]
