"""Move verification for chess tactics."""

import chess

from rlvr_physics.core.instances import TaskInstance, require_str
from rlvr_physics.core.session import TaskSubmission
from rlvr_physics.tasks.games.chess_tactics.types import ChessVerification


def verify_chess_tactic_submission(
    instance: TaskInstance, submission: TaskSubmission
) -> ChessVerification:
    """Verify one chess tactic submission."""

    fen = require_str(instance.public_payload["fen"], "fen")
    board = chess.Board(fen)
    raw = submission.raw.strip()
    if not raw:
        return ChessVerification(False, False, 0.0, "empty_submission", None, None)
    move = _parse_move(board, raw)
    if move is None:
        return ChessVerification(False, False, 0.0, "parse_failure", None, None)
    if move not in board.legal_moves:
        return ChessVerification(False, False, 0.0, "illegal_move", move.uci(), None)
    san = board.san(move)
    board.push(move)
    if board.is_checkmate():
        return ChessVerification(True, True, 1.0, "checkmate", move.uci(), san)
    return ChessVerification(True, False, 0.2, "legal_non_solution", move.uci(), san)


def _parse_move(board: chess.Board, raw: str) -> chess.Move | None:
    cleaned = raw.strip()
    try:
        return board.parse_san(cleaned)
    except ValueError:
        pass
    try:
        return chess.Move.from_uci(cleaned.lower())
    except ValueError:
        return None
