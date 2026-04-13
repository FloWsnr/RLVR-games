"""Connect 4 environment scaffolding."""

from rlvr_games.games.connect4.actions import Connect4Action
from rlvr_games.games.connect4.backend import Connect4Backend
from rlvr_games.games.connect4.factory import make_connect4_env
from rlvr_games.games.connect4.render import (
    Connect4AsciiBoardFormatter,
    Connect4ImageRenderer,
    Connect4ObservationRenderer,
)
from rlvr_games.games.connect4.rewards import (
    Connect4Perspective,
    Connect4RewardPerspective,
    TerminalOutcomeReward,
    resolve_reward_perspective,
)
from rlvr_games.games.connect4.scenarios import (
    DEFAULT_RANDOM_START_MAX_MOVES,
    FixedBoardScenario,
    RandomPositionScenario,
    STANDARD_CONNECT4_COLUMNS,
    STANDARD_CONNECT4_CONNECT_LENGTH,
    STANDARD_CONNECT4_ROWS,
    normalize_initial_board,
)
from rlvr_games.games.connect4.state import (
    Board,
    Connect4Outcome,
    Connect4State,
    inspect_connect4_state,
)

__all__ = [
    "Board",
    "Connect4Action",
    "Connect4AsciiBoardFormatter",
    "Connect4Backend",
    "Connect4ImageRenderer",
    "Connect4ObservationRenderer",
    "Connect4Outcome",
    "Connect4Perspective",
    "Connect4RewardPerspective",
    "Connect4State",
    "DEFAULT_RANDOM_START_MAX_MOVES",
    "FixedBoardScenario",
    "inspect_connect4_state",
    "make_connect4_env",
    "normalize_initial_board",
    "RandomPositionScenario",
    "resolve_reward_perspective",
    "STANDARD_CONNECT4_COLUMNS",
    "STANDARD_CONNECT4_CONNECT_LENGTH",
    "STANDARD_CONNECT4_ROWS",
    "TerminalOutcomeReward",
]
