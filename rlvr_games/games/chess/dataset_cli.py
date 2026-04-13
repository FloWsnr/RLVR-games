"""Dataset CLI registration for chess corpora."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from rlvr_games.cli.common import (
    add_common_dataset_download_arguments,
    add_common_dataset_preprocess_arguments,
    build_split_percentages,
)
from rlvr_games.cli.specs import DatasetCliSpec
from rlvr_games.games.chess.datasets import (
    build_lichess_puzzle_dataset,
    download_lichess_puzzle_source,
)


def register_lichess_puzzle_download_arguments(parser: ArgumentParser) -> None:
    """Attach Lichess puzzle dataset arguments to a dataset-download parser.

    Parameters
    ----------
    parser : ArgumentParser
        Parser to extend with dataset download arguments.
    """
    add_common_dataset_download_arguments(parser)


def register_lichess_puzzle_preprocess_arguments(parser: ArgumentParser) -> None:
    """Attach Lichess puzzle dataset arguments to a dataset-preprocess parser.

    Parameters
    ----------
    parser : ArgumentParser
        Parser to extend with dataset preprocessing arguments.
    """
    add_common_dataset_preprocess_arguments(parser)


def download_lichess_puzzles(
    args: Namespace,
    parser: ArgumentParser,
) -> Path:
    """Download the raw Lichess puzzle dump or reuse an existing local copy.

    Parameters
    ----------
    args : Namespace
        Parsed dataset-download CLI arguments.
    parser : ArgumentParser
        Parser provided for command-handler consistency.

    Returns
    -------
    Path
        Path to the raw dataset artifact.
    """
    del parser
    return download_lichess_puzzle_source(
        raw_root_dir=args.raw_dir,
        overwrite=args.overwrite,
    )


def preprocess_lichess_puzzle_dataset(
    args: Namespace,
    parser: ArgumentParser,
) -> Path:
    """Build a processed Lichess puzzle dataset from parsed CLI arguments.

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


CHESS_LICHESS_PUZZLES_DATASET_SPEC = DatasetCliSpec(
    game="chess",
    name="lichess-puzzles",
    register_download_arguments=register_lichess_puzzle_download_arguments,
    download_dataset=download_lichess_puzzles,
    register_preprocess_arguments=register_lichess_puzzle_preprocess_arguments,
    preprocess_dataset=preprocess_lichess_puzzle_dataset,
)
