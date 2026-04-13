"""Chess dataset preprocessing and scenario tests."""

from pathlib import Path

import chess
import zstandard

from rlvr_games.datasets import (
    DatasetSplit,
    SplitPercentages,
    load_dataset_manifest,
    read_parquet_records,
)
from rlvr_games.games.chess.datasets import (
    build_lichess_puzzle_dataset,
    parse_chess_puzzle_record,
)
from rlvr_games.games.chess.scenarios import ChessPuzzleDatasetScenario


def write_sample_lichess_csv(*, path: Path) -> Path:
    """Write a tiny Lichess-like puzzle CSV fixture.

    Parameters
    ----------
    path : Path
        Destination path receiving the fixture CSV.

    Returns
    -------
    Path
        The written fixture path.
    """
    content = (
        "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,"
        "Themes,GameUrl,OpeningTags\n"
        "00sHx,q3k1nr/1pp1nQpp/3p4/1P2p3/4P3/B1PP1b2/B5PP/5K2 b k - 0 17,"
        "e8d7 a2e6 d7d8 f7f8,1760,80,83,72,"
        "mate mateIn2 middlegame short,"
        "https://lichess.org/yyznGmXs/black#34,"
        "Italian_Game Italian_Game_Classical_Variation\n"
        "00sJb,Q1b2r1k/p2np2p/5bp1/q7/5P2/4B3/PPP3PP/2KR1B1R w - - 1 17,"
        "d1d7 a5e1 d7d1 e1e3 c1b1 e3b6,2235,76,97,64,"
        "advantage fork long,https://lichess.org/kiuvTFoE#33,"
        "Sicilian_Defense Sicilian_Defense_Dragon_Variation\n"
    )
    if path.suffix == ".zst":
        with zstandard.open(path, mode="wt", encoding="utf-8") as handle:
            handle.write(content)
        return path
    path.write_text(content, encoding="utf-8")
    return path


def test_build_lichess_puzzle_dataset_writes_processed_shards(
    tmp_path: Path,
) -> None:
    csv_path = write_sample_lichess_csv(path=tmp_path / "lichess_sample.csv")

    manifest_path = build_lichess_puzzle_dataset(
        source_path=csv_path,
        raw_root_dir=tmp_path / "raw",
        processed_root_dir=tmp_path / "processed",
        chunk_size=1,
        split_percentages=SplitPercentages(train=100, val=0, test=0),
        max_records=None,
        overwrite=False,
    )

    manifest = load_dataset_manifest(path=manifest_path)
    assert manifest.game == "chess"
    assert manifest.dataset == "lichess-puzzles"
    assert manifest.split_record_counts()["train"] == 2
    assert len(manifest.shards) == 2

    shard_records = [
        parse_chess_puzzle_record(record_payload)
        for shard in manifest.shards
        for record_payload in read_parquet_records(
            path=manifest_path.parent / shard.path
        )
    ]
    assert len(shard_records) == 2
    first_record = shard_records[0]
    board = chess.Board(first_record.metadata["initial_fen"])
    board.push(chess.Move.from_uci(first_record.metadata["setup_move_uci"]))
    assert first_record.presented_fen == board.fen()
    assert first_record.solution_moves_uci
    assert first_record.metadata["solution_length"] == len(
        first_record.solution_moves_uci
    )


def test_chess_puzzle_dataset_scenario_samples_deterministically(
    tmp_path: Path,
) -> None:
    csv_path = write_sample_lichess_csv(path=tmp_path / "lichess_sample.csv")
    manifest_path = build_lichess_puzzle_dataset(
        source_path=csv_path,
        raw_root_dir=tmp_path / "raw",
        processed_root_dir=tmp_path / "processed",
        chunk_size=2,
        split_percentages=SplitPercentages(train=100, val=0, test=0),
        max_records=None,
        overwrite=False,
    )
    scenario = ChessPuzzleDatasetScenario(
        manifest_path=manifest_path,
        split=DatasetSplit.TRAIN,
    )

    state_a, info_a = scenario.reset(seed=0)
    state_b, info_b = scenario.reset(seed=0)

    assert state_a.fen == state_b.fen
    assert info_a == info_b
    assert info_a["scenario"] == "dataset_puzzle"
    assert state_a.metadata["task_type"] == "puzzle"
    assert state_a.metadata["presented_fen"] == state_a.fen
    assert state_a.metadata["record_id"] == info_a["record_id"]
    assert isinstance(state_a.metadata["solution_moves_uci"], tuple)
