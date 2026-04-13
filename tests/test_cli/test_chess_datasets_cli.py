"""Chess dataset CLI tests."""

from io import StringIO
from pathlib import Path
import sys

from _pytest.monkeypatch import MonkeyPatch
import pytest
import zstandard

from rlvr_games.cli.main import build_parser as build_play_parser
from rlvr_games.games.chess.datasets import default_lichess_puzzle_source_path
from rlvr_games.games.chess.datasets_cli import run_cli


def write_sample_lichess_csv(*, path: Path) -> Path:
    """Write a tiny Lichess-like puzzle CSV fixture."""
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


def test_play_cli_does_not_register_dataset_commands() -> None:
    parser = build_play_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["datasets", "download"])


def test_run_cli_dataset_download_reuses_existing_local_source(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_stream = StringIO()
    output_stream = StringIO()
    raw_dir = tmp_path / "raw"
    source_path = default_lichess_puzzle_source_path(raw_root_dir=raw_dir)
    write_sample_lichess_csv(path=source_path)
    monkeypatch.setattr(sys, "stdin", input_stream)
    monkeypatch.setattr(sys, "stdout", output_stream)

    exit_code = run_cli(["download", "--raw-dir", str(raw_dir)])

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "Downloaded dataset source:" in output
    downloaded_path = Path(
        output.strip().split("Downloaded dataset source: ", maxsplit=1)[1]
    )
    assert downloaded_path == source_path
    assert downloaded_path.exists()


def test_run_cli_can_preprocess_a_downloaded_lichess_puzzle_dataset(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_stream = StringIO()
    output_stream = StringIO()
    raw_dir = tmp_path / "raw"
    source_path = default_lichess_puzzle_source_path(raw_root_dir=raw_dir)
    write_sample_lichess_csv(path=source_path)
    monkeypatch.setattr(sys, "stdin", input_stream)
    monkeypatch.setattr(sys, "stdout", output_stream)

    exit_code = run_cli(
        [
            "preprocess",
            "--processed-dir",
            str(tmp_path / "processed"),
            "--raw-dir",
            str(raw_dir),
            "--train-percentage",
            "100",
            "--val-percentage",
            "0",
            "--test-percentage",
            "0",
        ]
    )

    output = output_stream.getvalue()
    assert exit_code == 0
    assert "Dataset manifest:" in output
    manifest_path = Path(output.strip().split("Dataset manifest: ", maxsplit=1)[1])
    assert manifest_path.exists()


def test_run_cli_preprocess_requires_a_local_source_file(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_stream = StringIO()
    output_stream = StringIO()
    monkeypatch.setattr(sys, "stdin", input_stream)
    monkeypatch.setattr(sys, "stdout", output_stream)

    with pytest.raises(SystemExit):
        run_cli(
            [
                "preprocess",
                "--processed-dir",
                str(tmp_path / "processed"),
                "--raw-dir",
                str(tmp_path / "raw"),
            ]
        )
