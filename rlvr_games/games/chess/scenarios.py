"""Chess scenario initializers."""

from dataclasses import dataclass
from typing import Any

from rlvr_games.games.chess.state import ChessState

STANDARD_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@dataclass(slots=True)
class StartingPositionScenario:
    """Start from the standard chess opening position."""

    initial_fen: str = STANDARD_START_FEN

    def reset(self, *, seed: int | None = None) -> tuple[ChessState, dict[str, Any]]:
        return (
            ChessState(fen=self.initial_fen),
            {
                "scenario": "starting_position",
                "initial_fen": self.initial_fen,
                "seed": seed,
            },
        )
