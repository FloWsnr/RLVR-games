"""Chess environment scaffolding."""

from rlvr_games.games.chess.actions import ChessAction
from rlvr_games.games.chess.backend import ChessBackend
from rlvr_games.games.chess.factory import (
    ChessBoardOrientation,
    ChessTextRendererKind,
    make_chess_env,
)
from rlvr_games.games.chess.rewards import (
    ChessPerspective,
    ChessRewardPerspective,
    ChessStateEvaluator,
    EngineEvalDenseReward,
    EngineEvalSparseReward,
    PuzzleOnlyMoveDenseReward,
    PuzzleOnlyMoveSparseReward,
    TerminalOutcomeReward,
    UciEngineEvaluator,
    puzzle_solution_moves_uci,
    puzzle_solution_progress_index,
    resolve_reward_perspective,
)
from rlvr_games.games.chess.render import (
    AsciiBoardFormatter,
    ChessFastImageRenderer,
    ChessObservationRenderer,
    UnicodeBoardFormatter,
)
from rlvr_games.games.chess.scenarios import (
    ChessPuzzleDatasetScenario,
    STANDARD_START_FEN,
    StartingPositionScenario,
)
from rlvr_games.games.chess.state import ChessState, inspect_chess_state
from rlvr_games.games.chess.stockfish_runtime import StockfishEvaluator
from rlvr_games.games.chess.turns import (
    ChessEngineAutoAdvancePolicy,
    ChessMoveSelector,
    ChessPuzzleAutoAdvancePolicy,
    StockfishMoveSelector,
    UciEngineMoveSelector,
)

__all__ = [
    "ChessAction",
    "AsciiBoardFormatter",
    "ChessBackend",
    "ChessBoardOrientation",
    "ChessEngineAutoAdvancePolicy",
    "ChessMoveSelector",
    "ChessPuzzleAutoAdvancePolicy",
    "ChessFastImageRenderer",
    "ChessObservationRenderer",
    "ChessPerspective",
    "ChessPuzzleDatasetScenario",
    "ChessRewardPerspective",
    "ChessStateEvaluator",
    "ChessState",
    "ChessTextRendererKind",
    "EngineEvalDenseReward",
    "EngineEvalSparseReward",
    "inspect_chess_state",
    "PuzzleOnlyMoveDenseReward",
    "PuzzleOnlyMoveSparseReward",
    "STANDARD_START_FEN",
    "StockfishEvaluator",
    "StockfishMoveSelector",
    "StartingPositionScenario",
    "TerminalOutcomeReward",
    "UciEngineEvaluator",
    "UciEngineMoveSelector",
    "UnicodeBoardFormatter",
    "make_chess_env",
    "puzzle_solution_moves_uci",
    "puzzle_solution_progress_index",
    "resolve_reward_perspective",
]
