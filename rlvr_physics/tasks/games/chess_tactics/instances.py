"""Instance construction for chess tactics."""

import chess

from rlvr_physics.core.instances import TaskInstance, TaskLimits, stable_hash
from rlvr_physics.tasks.games.chess_tactics.constants import CHESS_DOMAIN, CHESS_KIND
from rlvr_physics.tasks.games.chess_tactics.puzzles import (
    mate_in_one_moves,
    select_puzzle_fen,
)


def make_chess_tactics_instance(seed: int, puzzle_id: str) -> TaskInstance:
    """Create one immutable mate-in-one chess tactic instance."""

    fen = select_puzzle_fen(seed, puzzle_id)
    board = chess.Board(fen)
    solutions = mate_in_one_moves(board)
    if not solutions:
        raise ValueError("selected puzzle has no mate-in-one solution")
    task_id = (
        "chess-"
        + stable_hash(
            {"seed": seed, "puzzle_id": puzzle_id, "fen": fen, "solutions": solutions}
        )[:16]
    )
    side_to_move = "white" if board.turn == chess.WHITE else "black"
    return TaskInstance(
        task_id=task_id,
        kind=CHESS_KIND,
        domain=CHESS_DOMAIN,
        seed=seed,
        public_payload={
            "fen": fen,
            "side_to_move": side_to_move,
            "mate_depth": 1,
            "allowed_notation": ("SAN", "UCI"),
        },
        privileged_payload={
            "puzzle_id": puzzle_id,
            "solution_moves_uci": solutions,
            "source": "builtin",
        },
        limits=TaskLimits(max_turns=1, token_budget=64),
        metadata={"puzzle_id": puzzle_id, "mate_depth": 1},
    )
