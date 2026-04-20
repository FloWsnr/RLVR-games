"""Chess tactics task package."""

from rlvr_physics.tasks.games.chess_tactics.constants import (
    CHESS_DOMAIN,
    CHESS_KIND,
    FOOLS_MATE_IN_ONE_FEN,
)
from rlvr_physics.tasks.games.chess_tactics.instances import make_chess_tactics_instance
from rlvr_physics.tasks.games.chess_tactics.renderers import (
    render_chess_tactics_image,
    render_chess_tactics_text,
)
from rlvr_physics.tasks.games.chess_tactics.session import ChessTacticsSession
from rlvr_physics.tasks.games.chess_tactics.spec import chess_tactics_task_spec
from rlvr_physics.tasks.games.chess_tactics.types import ChessVerification
from rlvr_physics.tasks.games.chess_tactics.verifier import (
    verify_chess_tactic_submission,
)

__all__ = [
    "CHESS_DOMAIN",
    "CHESS_KIND",
    "FOOLS_MATE_IN_ONE_FEN",
    "ChessTacticsSession",
    "ChessVerification",
    "chess_tactics_task_spec",
    "make_chess_tactics_instance",
    "render_chess_tactics_image",
    "render_chess_tactics_text",
    "verify_chess_tactic_submission",
]
