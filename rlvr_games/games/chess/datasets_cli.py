"""Dedicated dataset CLI for chess corpus preparation."""

from argparse import ArgumentParser, Namespace
from pathlib import Path
import sys
from typing import Sequence

from rlvr_games.datasets import SplitPercentages
from rlvr_games.games.chess.datasets import (
    build_lichess_puzzle_dataset,
    download_lichess_puzzle_source,
)


def build_parser() -> ArgumentParser:
    """Build the chess dataset command-line parser.

    Returns
    -------
    ArgumentParser
        Parser supporting raw download and preprocessing workflows for the
        chess Lichess puzzle dataset.
    """
    parser = ArgumentParser(prog="rlvr-games-chess-datasets")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download")
    register_download_arguments(download_parser)

    preprocess_parser = subparsers.add_parser("preprocess")
    register_preprocess_arguments(preprocess_parser)

    return parser


def register_download_arguments(parser: ArgumentParser) -> None:
    """Attach raw-download arguments to the chess dataset parser.

    Parameters
    ----------
    parser : ArgumentParser
        Parser to extend with raw-download arguments.
    """
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--overwrite", action="store_true")


def register_preprocess_arguments(parser: ArgumentParser) -> None:
    """Attach preprocessing arguments to the chess dataset parser.

    Parameters
    ----------
    parser : ArgumentParser
        Parser to extend with dataset preprocessing arguments.
    """
    parser.add_argument("--source-file", type=Path)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--train-percentage", type=int, default=98)
    parser.add_argument("--val-percentage", type=int, default=1)
    parser.add_argument("--test-percentage", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")


def run_cli(argv: Sequence[str]) -> int:
    """Run the chess dataset CLI for the supplied arguments.

    Parameters
    ----------
    argv : Sequence[str]
        Argument vector excluding the executable name.

    Returns
    -------
    int
        Process-style exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "download":
        raw_path = download_dataset(args=args)
        print(f"Downloaded dataset source: {raw_path}")
        return 0

    if args.command == "preprocess":
        manifest_path = preprocess_dataset(args=args, parser=parser)
        print(f"Dataset manifest: {manifest_path}")
        return 0

    raise ValueError(f"Unsupported command arguments: {argv!r}")


def download_dataset(*, args: Namespace) -> Path:
    """Download the raw Lichess puzzle dump or reuse an existing local copy.

    Parameters
    ----------
    args : Namespace
        Parsed dataset-download CLI arguments.

    Returns
    -------
    Path
        Path to the raw dataset artifact.
    """
    return download_lichess_puzzle_source(
        raw_root_dir=args.raw_dir,
        overwrite=args.overwrite,
    )


def preprocess_dataset(
    *,
    args: Namespace,
    parser: ArgumentParser,
) -> Path:
    """Build the processed Lichess puzzle dataset from parsed arguments.

    Parameters
    ----------
    args : Namespace
        Parsed dataset-preprocess CLI arguments.
    parser : ArgumentParser
        Parser used to report invalid argument combinations.

    Returns
    -------
    Path
        Path to the written processed dataset manifest.
    """
    try:
        return build_lichess_puzzle_dataset(
            source_path=args.source_file,
            raw_root_dir=args.raw_dir,
            processed_root_dir=args.processed_dir,
            chunk_size=args.chunk_size,
            split_percentages=build_split_percentages(args=args, parser=parser),
            max_records=args.max_records,
            overwrite=args.overwrite,
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))


def build_split_percentages(
    *,
    args: Namespace,
    parser: ArgumentParser,
) -> SplitPercentages:
    """Build validated split percentages from parsed CLI arguments.

    Parameters
    ----------
    args : Namespace
        Parsed dataset CLI arguments.
    parser : ArgumentParser
        Parser used to report invalid argument combinations.

    Returns
    -------
    SplitPercentages
        Validated split percentages for deterministic dataset sharding.
    """
    try:
        return SplitPercentages(
            train=args.train_percentage,
            val=args.val_percentage,
            test=args.test_percentage,
        )
    except ValueError as exc:
        parser.error(str(exc))


def main() -> int:
    """Run the chess dataset CLI using ``sys.argv``.

    Returns
    -------
    int
        Process-style exit code.
    """
    return run_cli(sys.argv[1:])
