"""Chess environment scaffolding."""

from rlvr_games.games.chess.actions import ChessAction
from rlvr_games.games.chess.backend import ChessBackend
from rlvr_games.games.chess.env import ChessEnv
from rlvr_games.games.chess.render import (
    AsciiBoardFormatter,
    ChessBoardImageRenderer,
    ChessBoardTextFormatter,
    ChessObservationRenderer,
    ChessTextRenderer,
    EmptyChessBoardImageRenderer,
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
    "ChessBoardImageRenderer",
    "ChessBoardTextFormatter",
    "ChessEnv",
    "ChessObservationRenderer",
    "ChessState",
    "ChessTextRenderer",
    "EmptyChessBoardImageRenderer",
    "STANDARD_START_FEN",
    "StartingPositionScenario",
    "UnicodeBoardFormatter",
]
