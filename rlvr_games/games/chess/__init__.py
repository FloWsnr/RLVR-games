"""Chess environment scaffolding."""

from rlvr_games.games.chess.actions import ChessAction
from rlvr_games.games.chess.backend import ChessBackend
from rlvr_games.games.chess.env import ChessEnv
from rlvr_games.games.chess.factory import (
    ChessBoardOrientation,
    ChessTextRendererKind,
    make_chess_env,
)
from rlvr_games.games.chess.render import (
    AsciiBoardFormatter,
    ChessObservationRenderer,
    ChessRasterBoardImageRenderer,
    UnicodeBoardFormatter,
)
from rlvr_games.games.chess.scenarios import (
    STANDARD_START_FEN,
    StartingPositionScenario,
)
from rlvr_games.games.chess.state import ChessState

__all__ = [
    "ChessAction",
    "AsciiBoardFormatter",
    "ChessBackend",
    "ChessBoardOrientation",
    "ChessEnv",
    "ChessObservationRenderer",
    "ChessRasterBoardImageRenderer",
    "ChessState",
    "ChessTextRendererKind",
    "STANDARD_START_FEN",
    "StartingPositionScenario",
    "UnicodeBoardFormatter",
    "make_chess_env",
]
