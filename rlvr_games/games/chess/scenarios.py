"""Chess scenario initializers."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess

from rlvr_games.datasets import DatasetSplit, ParquetScenarioDataset
from rlvr_games.games.chess.datasets import (
    ChessPuzzleRecord,
    parse_chess_puzzle_record,
)
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
            ChessState.from_board(
                board=board,
                repetition_counts={repetition_key_from_board(board): 1},
                metadata={},
            ),
            {
                "scenario": scenario_name,
                "initial_fen": normalized_fen,
                "seed": seed,
            },
        )


@dataclass(slots=True)
class ChessPuzzleDatasetScenario:
    """Scenario that samples preprocessed chess puzzles from dataset shards.

    Attributes
    ----------
    manifest_path : Path
        Processed dataset manifest describing available shards.
    split : DatasetSplit
        Dataset split to sample from.
    """

    manifest_path: Path
    split: DatasetSplit
    _dataset: ParquetScenarioDataset[ChessPuzzleRecord] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Load and validate the processed dataset manifest.

        Raises
        ------
        ValueError
            If the manifest does not describe a chess puzzle dataset.
        """
        self._dataset = ParquetScenarioDataset.from_manifest_path(
            manifest_path=self.manifest_path,
            parser=parse_chess_puzzle_record,
            max_cached_shards=2,
        )
        if self._dataset.manifest.game != "chess":
            raise ValueError("ChessPuzzleDatasetScenario requires a chess manifest.")

    def reset(self, *, seed: int) -> tuple[ChessState, dict[str, Any]]:
        """Sample one puzzle record and create a fresh chess episode.

        Parameters
        ----------
        seed : int
            Deterministic sampling seed used to select the dataset record.

        Returns
        -------
        tuple[ChessState, dict[str, Any]]
            Canonical initial puzzle state and reset metadata describing the
            sampled record.
        """
        record = self._dataset.sample_record(split=self.split, seed=seed)
        board = chess.Board(record.presented_fen)
        state_metadata = {
            "task_type": "puzzle",
            "dataset": record.dataset,
            "record_id": record.record_id,
            "presented_fen": record.presented_fen,
            "solution_moves_uci": record.solution_moves_uci,
            "rating": record.rating,
            "rating_deviation": record.rating_deviation,
            "popularity": record.popularity,
            "play_count": record.play_count,
            "themes": record.themes,
            "source_url": record.source_url,
            **record.metadata,
        }
        return (
            ChessState.from_board(
                board=board,
                repetition_counts={repetition_key_from_board(board): 1},
                metadata=state_metadata,
            ),
            {
                "scenario": "dataset_puzzle",
                "dataset": record.dataset,
                "split": self.split.value,
                "record_id": record.record_id,
                "initial_fen": record.presented_fen,
                "rating": record.rating,
                "themes": record.themes,
                "seed": seed,
            },
        )
