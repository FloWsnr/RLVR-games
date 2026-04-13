"""Offline preprocessing helpers for chess datasets."""

from dataclasses import dataclass
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import chess

from rlvr_games.datasets import (
    DATASET_MANIFEST_SCHEMA_VERSION,
    DatasetManifest,
    ShardedParquetWriter,
    SplitPercentages,
    assign_split_from_key,
    download_file,
    open_text_input,
    sha256_file,
    write_dataset_manifest,
)

LICHESS_PUZZLES_DATASET_NAME = "lichess-puzzles"
LICHESS_PUZZLES_DATASET_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
LICHESS_PUZZLES_LICENSE = "CC0"
LICHESS_PUZZLES_RAW_FILENAME = Path(LICHESS_PUZZLES_DATASET_URL).name


@dataclass(frozen=True, slots=True)
class ChessPuzzleRecord:
    """Normalized runtime record for a chess puzzle scenario.

    Attributes
    ----------
    record_id : str
        Stable dataset record identifier.
    dataset : str
        Dataset family name that produced the record.
    presented_fen : str
        FEN shown to the player at reset time.
    solution_moves_uci : tuple[str, ...]
        Canonical UCI solution line beginning from ``presented_fen``.
    rating : int
        Puzzle difficulty rating.
    rating_deviation : int
        Rating deviation attached to the difficulty estimate.
    popularity : int
        Lichess popularity score.
    play_count : int
        Number of recorded puzzle attempts.
    themes : tuple[str, ...]
        Puzzle theme labels.
    source_url : str
        Source game or puzzle URL.
    metadata : dict[str, Any]
        Additional normalized metadata that should travel with the state.
    """

    record_id: str
    dataset: str
    presented_fen: str
    solution_moves_uci: tuple[str, ...]
    rating: int
    rating_deviation: int
    popularity: int
    play_count: int
    themes: tuple[str, ...]
    source_url: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping for the record.

        Returns
        -------
        dict[str, Any]
            Serialized record payload following the shared dataset schema.
        """
        return {
            "record_id": self.record_id,
            "game": "chess",
            "dataset": self.dataset,
            "task_type": "puzzle",
            "state_format": "fen",
            "state_repr": self.presented_fen,
            "solution": {"moves_uci": list(self.solution_moves_uci)},
            "difficulty": {
                "rating": self.rating,
                "rating_deviation": self.rating_deviation,
                "popularity": self.popularity,
                "play_count": self.play_count,
            },
            "themes": list(self.themes),
            "source_url": self.source_url,
            "license": LICHESS_PUZZLES_LICENSE,
            "metadata": self.metadata,
        }


def parse_chess_puzzle_record(record_payload: dict[str, Any]) -> ChessPuzzleRecord:
    """Parse one normalized puzzle record from processed dataset storage.

    Parameters
    ----------
    record_payload : dict[str, Any]
        Raw JSON-decoded processed record payload.

    Returns
    -------
    ChessPuzzleRecord
        Parsed and validated puzzle record.

    Raises
    ------
    ValueError
        If the processed record payload does not match the expected chess
        puzzle schema.
    """
    if record_payload.get("game") != "chess":
        raise ValueError("Processed puzzle record must belong to the chess game.")
    if record_payload.get("task_type") != "puzzle":
        raise ValueError("Processed chess puzzle records must have task_type='puzzle'.")
    if record_payload.get("state_format") != "fen":
        raise ValueError("Processed chess puzzle records must use FEN state format.")

    solution_payload = record_payload.get("solution")
    if not isinstance(solution_payload, dict):
        raise ValueError("Processed chess puzzle records require a solution object.")
    solution_moves_payload = solution_payload.get("moves_uci")
    if not isinstance(solution_moves_payload, list):
        raise ValueError(
            "Processed chess puzzle records require solution moves in list form."
        )

    difficulty_payload = record_payload.get("difficulty")
    if not isinstance(difficulty_payload, dict):
        raise ValueError("Processed chess puzzle records require difficulty metadata.")

    themes_payload = record_payload.get("themes")
    if not isinstance(themes_payload, list):
        raise ValueError("Processed chess puzzle records require a themes list.")

    metadata_payload = record_payload.get("metadata")
    if not isinstance(metadata_payload, dict):
        raise ValueError("Processed chess puzzle records require a metadata object.")

    return ChessPuzzleRecord(
        record_id=str(record_payload["record_id"]),
        dataset=str(record_payload["dataset"]),
        presented_fen=str(record_payload["state_repr"]),
        solution_moves_uci=tuple(str(move) for move in solution_moves_payload),
        rating=int(difficulty_payload["rating"]),
        rating_deviation=int(difficulty_payload["rating_deviation"]),
        popularity=int(difficulty_payload["popularity"]),
        play_count=int(difficulty_payload["play_count"]),
        themes=tuple(str(theme) for theme in themes_payload),
        source_url=str(record_payload["source_url"]),
        metadata=dict(metadata_payload),
    )


def build_lichess_puzzle_dataset(
    *,
    source_path: Path | None,
    raw_root_dir: Path,
    processed_root_dir: Path,
    chunk_size: int,
    split_percentages: SplitPercentages,
    max_records: int | None,
    overwrite: bool,
) -> Path:
    """Read a local Lichess puzzle dump and build processed shards.

    Parameters
    ----------
    source_path : Path | None
        Optional local raw CSV or CSV.ZST file. When ``None``, preprocessing
        expects the previously downloaded file in ``raw_root_dir``.
    raw_root_dir : Path
        Root directory for raw dataset artifacts.
    processed_root_dir : Path
        Root directory receiving processed shard outputs.
    chunk_size : int
        Maximum number of normalized records per shard.
    split_percentages : SplitPercentages
        Deterministic split percentages used during preprocessing.
    max_records : int | None
        Optional limit on the number of raw rows to process.
    overwrite : bool
        Whether to rebuild an existing processed directory for the same raw
        source hash.

    Returns
    -------
    Path
        Path to the written dataset manifest.

    Raises
    ------
    FileNotFoundError
        If no local raw source file is available for preprocessing.
    """
    raw_source_path = _resolve_raw_source_path(
        source_path=source_path,
        raw_root_dir=raw_root_dir,
    )
    source_sha256 = sha256_file(path=raw_source_path)
    version = _builder_config_version(
        source_sha256=source_sha256,
        chunk_size=chunk_size,
        split_percentages=split_percentages,
        max_records=max_records,
    )

    processed_dataset_dir = (
        processed_root_dir / "chess" / LICHESS_PUZZLES_DATASET_NAME / version
    )
    manifest_path = processed_dataset_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        return manifest_path
    if processed_dataset_dir.exists() and overwrite:
        shutil.rmtree(processed_dataset_dir)
    processed_dataset_dir.mkdir(parents=True, exist_ok=True)

    writer = ShardedParquetWriter(
        output_dir=processed_dataset_dir,
        chunk_size=chunk_size,
    )
    processed_records = 0
    skipped_records = 0

    with open_text_input(path=raw_source_path) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if max_records is not None and processed_records >= max_records:
                break
            try:
                record = normalize_lichess_puzzle_row(row=row)
            except ValueError:
                skipped_records += 1
                continue

            split = assign_split_from_key(
                key=record.record_id,
                split_percentages=split_percentages,
            )
            writer.write(split=split, record=record.to_dict())
            processed_records += 1

    shards = writer.close()
    if not shards:
        raise ValueError(
            "No valid chess puzzle records were written during dataset init."
        )
    manifest = DatasetManifest(
        schema_version=DATASET_MANIFEST_SCHEMA_VERSION,
        game="chess",
        dataset=LICHESS_PUZZLES_DATASET_NAME,
        version=version,
        record_format="parquet",
        shards=shards,
        source_url=None if source_path is not None else LICHESS_PUZZLES_DATASET_URL,
        source_filename=raw_source_path.name,
        source_sha256=source_sha256,
        license=LICHESS_PUZZLES_LICENSE,
        metadata={
            "task_type": "puzzle",
            "builder": "lichess",
            "chunk_size": chunk_size,
            "max_records": max_records,
            "processed_record_count": processed_records,
            "skipped_record_count": skipped_records,
            "split_percentages": split_percentages.to_dict(),
        },
    )
    write_dataset_manifest(path=manifest_path, manifest=manifest)
    return manifest_path


def normalize_lichess_puzzle_row(*, row: dict[str, str]) -> ChessPuzzleRecord:
    """Convert one raw Lichess CSV row into the shared puzzle schema.

    Parameters
    ----------
    row : dict[str, str]
        Raw CSV row from the Lichess puzzle export.

    Returns
    -------
    ChessPuzzleRecord
        Normalized puzzle record ready for sharding.

    Raises
    ------
    ValueError
        If the row is malformed or contains illegal chess moves.
    """
    record_id = _require_csv_value(row=row, field_name="PuzzleId")
    initial_fen = _require_csv_value(row=row, field_name="FEN")
    moves_text = _require_csv_value(row=row, field_name="Moves")
    source_url = _require_csv_value(row=row, field_name="GameUrl")

    board = chess.Board(initial_fen)
    moves = tuple(move for move in moves_text.split() if move)
    if len(moves) < 2:
        raise ValueError(
            "Lichess puzzle rows must contain a setup move and at least one "
            "solution move."
        )

    setup_move = _parse_legal_move(board=board, move_uci=moves[0])
    board.push(setup_move)
    presented_fen = board.fen()

    solution_moves_uci = tuple(moves[1:])
    solution_board = board.copy(stack=False)
    for move_uci in solution_moves_uci:
        solution_move = _parse_legal_move(board=solution_board, move_uci=move_uci)
        solution_board.push(solution_move)

    themes = tuple(
        theme
        for theme in _require_csv_value(row=row, field_name="Themes").split()
        if theme
    )
    opening_tags = tuple(tag for tag in row.get("OpeningTags", "").split() if tag)

    return ChessPuzzleRecord(
        record_id=record_id,
        dataset=LICHESS_PUZZLES_DATASET_NAME,
        presented_fen=presented_fen,
        solution_moves_uci=solution_moves_uci,
        rating=int(_require_csv_value(row=row, field_name="Rating")),
        rating_deviation=int(_require_csv_value(row=row, field_name="RatingDeviation")),
        popularity=int(_require_csv_value(row=row, field_name="Popularity")),
        play_count=int(_require_csv_value(row=row, field_name="NbPlays")),
        themes=themes,
        source_url=source_url,
        metadata={
            "initial_fen": initial_fen,
            "setup_move_uci": moves[0],
            "opening_tags": opening_tags,
            "solution_length": len(solution_moves_uci),
        },
    )


def _resolve_raw_source_path(
    *,
    source_path: Path | None,
    raw_root_dir: Path,
) -> Path:
    """Resolve the local raw source file for the Lichess puzzle dump.

    Parameters
    ----------
    source_path : Path | None
        Optional caller-provided raw source file.
    raw_root_dir : Path
        Root directory for downloaded raw artifacts.

    Returns
    -------
    Path
        Local raw source file path.

    Raises
    ------
    FileNotFoundError
        If no local raw source file is available.
    """
    if source_path is not None:
        if not source_path.exists():
            raise FileNotFoundError(
                f"Chess dataset source file does not exist: {source_path}"
            )
        return source_path

    destination = default_lichess_puzzle_source_path(raw_root_dir=raw_root_dir)
    if destination.exists():
        return destination
    raise FileNotFoundError(
        "No local chess puzzle source file found for preprocessing. Run "
        "`rlvr-games datasets download chess lichess-puzzles` first or pass "
        "`--source-file`."
    )


def download_lichess_puzzle_source(
    *,
    raw_root_dir: Path,
    overwrite: bool,
) -> Path:
    """Download the raw Lichess puzzle dump into the canonical raw-data path.

    Parameters
    ----------
    raw_root_dir : Path
        Root directory for downloaded raw artifacts.
    overwrite : bool
        Whether to re-download even when the destination file already exists.

    Returns
    -------
    Path
        Local raw source file path.
    """
    destination = default_lichess_puzzle_source_path(raw_root_dir=raw_root_dir)
    if destination.exists() and not overwrite:
        return destination
    if destination.exists() and overwrite:
        destination.unlink()
    return download_file(url=LICHESS_PUZZLES_DATASET_URL, destination=destination)


def default_lichess_puzzle_source_path(*, raw_root_dir: Path) -> Path:
    """Return the canonical raw source path for the Lichess puzzle dump.

    Parameters
    ----------
    raw_root_dir : Path
        Root directory for raw dataset artifacts.

    Returns
    -------
    Path
        Canonical local raw source file path.
    """
    raw_dataset_dir = raw_root_dir / "chess" / LICHESS_PUZZLES_DATASET_NAME
    raw_dataset_dir.mkdir(parents=True, exist_ok=True)
    return raw_dataset_dir / LICHESS_PUZZLES_RAW_FILENAME


def _builder_config_version(
    *,
    source_sha256: str,
    chunk_size: int,
    split_percentages: SplitPercentages,
    max_records: int | None,
) -> str:
    """Return a version string that captures source and preprocessing config.

    Parameters
    ----------
    source_sha256 : str
        SHA-256 hash of the raw source file.
    chunk_size : int
        Maximum number of normalized records per output shard.
    split_percentages : SplitPercentages
        Deterministic split percentages used during preprocessing.
    max_records : int | None
        Optional cap on processed records.

    Returns
    -------
    str
        Compact deterministic version identifier.
    """
    config_payload = {
        "chunk_size": chunk_size,
        "max_records": max_records,
        "split_percentages": split_percentages.to_dict(),
    }
    config_digest = hashlib.sha256(
        json.dumps(config_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"{source_sha256[:12]}-{config_digest[:8]}"


def _require_csv_value(
    *,
    row: dict[str, str],
    field_name: str,
) -> str:
    """Return a required non-empty CSV field value.

    Parameters
    ----------
    row : dict[str, str]
        Raw CSV row being normalized.
    field_name : str
        Column name that must be present and non-empty.

    Returns
    -------
    str
        Required field value.

    Raises
    ------
    ValueError
        If the field is missing or empty.
    """
    value = row.get(field_name)
    if value is None or value == "":
        raise ValueError(f"Missing required CSV field: {field_name}.")
    return value


def _parse_legal_move(
    *,
    board: chess.Board,
    move_uci: str,
) -> chess.Move:
    """Parse and validate a legal UCI move for a board.

    Parameters
    ----------
    board : chess.Board
        Board whose legal moves should contain the supplied action.
    move_uci : str
        Serialized UCI move to parse and validate.

    Returns
    -------
    chess.Move
        Parsed legal move.

    Raises
    ------
    ValueError
        If the move is malformed or illegal in the supplied position.
    """
    try:
        move = chess.Move.from_uci(move_uci)
    except ValueError as exc:
        raise ValueError(f"Invalid UCI move in dataset row: {move_uci}.") from exc
    if move not in board.legal_moves:
        raise ValueError(f"Illegal UCI move in dataset row: {move_uci}.")
    return move
