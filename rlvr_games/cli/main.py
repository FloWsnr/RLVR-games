"""Command-line entrypoints for interactive RLVR sessions and datasets."""

from argparse import ArgumentParser
from collections import defaultdict
import sys
from typing import Any, Sequence

from rlvr_games.cli.common import add_common_play_arguments
from rlvr_games.cli.registry import DATASET_SPECS, PLAY_GAME_SPECS
from rlvr_games.cli.session import run_play_session
from rlvr_games.cli.specs import DatasetCliSpec, GameCliSpec


def build_parser() -> ArgumentParser:
    """Build the top-level argument parser for the RLVR CLI.

    Returns
    -------
    ArgumentParser
        Configured parser supporting interactive play commands.
    """
    parser = ArgumentParser(prog="rlvr-games")
    subparsers = parser.add_subparsers(dest="command", required=True)

    play_parser = subparsers.add_parser("play")
    play_subparsers = play_parser.add_subparsers(dest="game", required=True)
    for game_spec in PLAY_GAME_SPECS:
        game_parser = play_subparsers.add_parser(game_spec.name)
        add_common_play_arguments(game_parser)
        game_spec.register_arguments(game_parser)
        game_parser.set_defaults(game_spec=game_spec)

    datasets_parser = subparsers.add_parser("datasets")
    dataset_command_subparsers = datasets_parser.add_subparsers(
        dest="dataset_command",
        required=True,
    )
    dataset_specs_by_game = _dataset_specs_by_game()
    _register_dataset_command_parsers(
        command_subparsers=dataset_command_subparsers,
        command_name="download",
        dataset_specs_by_game=dataset_specs_by_game,
    )
    _register_dataset_command_parsers(
        command_subparsers=dataset_command_subparsers,
        command_name="preprocess",
        dataset_specs_by_game=dataset_specs_by_game,
    )

    return parser


def run_cli(argv: Sequence[str]) -> int:
    """Run the RLVR command-line interface for the supplied arguments.

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
    if args.command == "play":
        game_spec = getattr(args, "game_spec", None)
        if not isinstance(game_spec, GameCliSpec):
            raise ValueError(f"Unsupported command arguments: {argv!r}")

        env = game_spec.build_environment(args, parser)
        return run_play_session(
            env=env,
            game_spec=game_spec,
            seed=args.seed,
            image_output_dir=args.image_output_dir,
            input_stream=sys.stdin,
            output_stream=sys.stdout,
        )

    if args.command == "datasets":
        dataset_spec = getattr(args, "dataset_spec", None)
        if not isinstance(dataset_spec, DatasetCliSpec):
            raise ValueError(f"Unsupported command arguments: {argv!r}")

        if args.dataset_command == "download":
            raw_path = dataset_spec.download_dataset(args, parser)
            print(f"Downloaded dataset source: {raw_path}")
            return 0

        if args.dataset_command == "preprocess":
            manifest_path = dataset_spec.preprocess_dataset(args, parser)
            print(f"Dataset manifest: {manifest_path}")
            return 0

        raise ValueError(f"Unsupported command arguments: {argv!r}")

    raise ValueError(f"Unsupported command arguments: {argv!r}")


def main() -> int:
    """Run the RLVR CLI using `sys.argv`.

    Returns
    -------
    int
        Process-style exit code.
    """
    return run_cli(sys.argv[1:])


def _dataset_specs_by_game() -> dict[str, tuple[DatasetCliSpec, ...]]:
    """Group registered dataset specs by owning game.

    Returns
    -------
    dict[str, tuple[DatasetCliSpec, ...]]
        Dataset CLI specs grouped by their game name.
    """
    grouped_specs: dict[str, list[DatasetCliSpec]] = defaultdict(list)
    for dataset_spec in DATASET_SPECS:
        grouped_specs[dataset_spec.game].append(dataset_spec)
    return {game_name: tuple(specs) for game_name, specs in grouped_specs.items()}


def _register_dataset_command_parsers(
    *,
    command_subparsers: Any,
    command_name: str,
    dataset_specs_by_game: dict[str, tuple[DatasetCliSpec, ...]],
) -> None:
    """Register nested parsers for one dataset command.

    Parameters
    ----------
    command_subparsers : Any
        Top-level ``datasets`` subparser collection.
    command_name : str
        Dataset subcommand name, for example ``"download"``.
    dataset_specs_by_game : dict[str, tuple[DatasetCliSpec, ...]]
        Dataset specs grouped by owning game.
    """
    command_parser = command_subparsers.add_parser(command_name)
    game_subparsers = command_parser.add_subparsers(dest="dataset_game", required=True)
    for game_name, dataset_specs in dataset_specs_by_game.items():
        game_parser = game_subparsers.add_parser(game_name)
        dataset_subparsers = game_parser.add_subparsers(
            dest="dataset_name",
            required=True,
        )
        for dataset_spec in dataset_specs:
            dataset_parser = dataset_subparsers.add_parser(dataset_spec.name)
            if command_name == "download":
                dataset_spec.register_download_arguments(dataset_parser)
            else:
                dataset_spec.register_preprocess_arguments(dataset_parser)
            dataset_parser.set_defaults(dataset_spec=dataset_spec)
