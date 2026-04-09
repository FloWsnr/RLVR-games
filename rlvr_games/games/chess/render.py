"""Text rendering for chess state."""

from rlvr_games.core.types import Observation
from rlvr_games.games.chess.state import ChessState


class ChessTextRenderer:
    """Minimal text renderer based on FEN."""

    def render(self, state: ChessState) -> Observation:
        fields = state.fen.split(" ")
        side_to_move = fields[1] if len(fields) > 1 else "?"
        return Observation(
            text=f"Chess position (FEN): {state.fen}\nSide to move: {side_to_move}",
            metadata={"fen": state.fen, "side_to_move": side_to_move},
        )
