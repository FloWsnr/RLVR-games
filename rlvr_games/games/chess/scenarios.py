"""Chess scenario initializers."""

from dataclasses import dataclass
from typing import Any

import chess

from rlvr_games.games.chess.state import ChessState, repetition_key_from_board

STANDARD_START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@dataclass(slots=True)
class StartingPositionScenario:
    """Scenario that initializes chess from a specified FEN.

    Attributes
    ----------
    initial_fen : str
        Starting position expressed as FEN. The default is the standard chess
        opening position, but custom legal FEN strings are also supported.
    """

    initial_fen: str = STANDARD_START_FEN

    def reset(self, *, seed: int) -> tuple[ChessState, dict[str, Any]]:
        """Create a fresh chess episode from the configured starting position.

        Parameters
        ----------
        seed : int
            Scenario seed forwarded into the returned reset metadata. The
            current implementation does not randomize positions.

        Returns
        -------
        tuple[ChessState, dict[str, Any]]
            Canonical initial chess state and metadata describing the scenario
            type, normalized FEN, and supplied seed.

        Raises
        ------
        ValueError
            If `initial_fen` is not a valid chess position.
        """
        try:
            board = chess.Board(self.initial_fen)
        except ValueError as exc:
            raise ValueError(
                f"Invalid chess FEN for scenario reset: {self.initial_fen}"
            ) from exc

        normalized_fen = board.fen()
        scenario_name = "starting_position"
        if normalized_fen != STANDARD_START_FEN:
            scenario_name = "fen_position"

        return (
            ChessState(
                fen=normalized_fen,
                repetition_counts={repetition_key_from_board(board): 1},
            ),
            {
                "scenario": scenario_name,
                "initial_fen": normalized_fen,
                "seed": seed,
            },
        )
