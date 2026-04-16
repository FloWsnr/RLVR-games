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
    SolverMoveScoreReward,
    TerminalOutcomeReward,
    resolve_reward_perspective,
)
from rlvr_games.games.connect4.scenarios import (
    DEFAULT_RANDOM_START_MAX_MOVES,
    FixedBoardScenario,
    RandomPositionScenario,
    normalize_initial_board,
)
from rlvr_games.games.connect4.state import (
    Board,
    Connect4Outcome,
    Connect4State,
    inspect_connect4_state,
)
from rlvr_games.games.connect4.solver import (
    BitBullyOpeningBook,
    BitBullySolver,
    bitbully_board_from_state,
    ensure_bitbully_supported_state,
)
from rlvr_games.games.connect4.turns import Connect4SolverAutoAdvancePolicy
from rlvr_games.games.connect4.variant import (
    STANDARD_CONNECT4_COLUMNS,
    STANDARD_CONNECT4_CONNECT_LENGTH,
    STANDARD_CONNECT4_ROWS,
)

__all__ = [
    "Board",
    "BitBullyOpeningBook",
    "BitBullySolver",
    "bitbully_board_from_state",
    "Connect4Action",
    "Connect4AsciiBoardFormatter",
    "Connect4Backend",
    "Connect4ImageRenderer",
    "Connect4ObservationRenderer",
    "Connect4Outcome",
    "Connect4Perspective",
    "Connect4RewardPerspective",
    "Connect4SolverAutoAdvancePolicy",
    "Connect4State",
    "DEFAULT_RANDOM_START_MAX_MOVES",
    "ensure_bitbully_supported_state",
    "FixedBoardScenario",
    "inspect_connect4_state",
    "make_connect4_env",
    "normalize_initial_board",
    "RandomPositionScenario",
    "resolve_reward_perspective",
    "SolverMoveScoreReward",
    "STANDARD_CONNECT4_COLUMNS",
    "STANDARD_CONNECT4_CONNECT_LENGTH",
    "STANDARD_CONNECT4_ROWS",
    "TerminalOutcomeReward",
]
