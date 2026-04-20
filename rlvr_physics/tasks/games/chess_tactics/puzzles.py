"""Built-in chess tactic puzzle helpers."""

import chess

from rlvr_physics.tasks.games.chess_tactics.constants import FOOLS_MATE_IN_ONE_FEN


def select_puzzle_fen(seed: int, puzzle_id: str) -> str:
    """Return the FEN for a deterministic puzzle id.

    Parameters
    ----------
    seed:
        Source seed.
    puzzle_id:
        Built-in puzzle identifier.
    """

    if puzzle_id == "fools_mate_mate_in_one":
        return FOOLS_MATE_IN_ONE_FEN
    if puzzle_id == "seeded":
        return FOOLS_MATE_IN_ONE_FEN
    raise ValueError(f"unknown chess tactic puzzle id: {puzzle_id} (seed={seed})")


def mate_in_one_moves(board: chess.Board) -> tuple[str, ...]:
    """Return sorted UCI moves that immediately checkmate.

    Parameters
    ----------
    board:
        Board to solve.
    """

    solutions: list[str] = []
    for move in board.legal_moves:
        candidate = board.copy(stack=False)
        candidate.push(move)
        if candidate.is_checkmate():
            solutions.append(move.uci())
    return tuple(sorted(solutions))
