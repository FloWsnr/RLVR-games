"""Command-line entrypoint for interactive RLVR play sessions."""

from argparse import ArgumentParser
import sys
from typing import Sequence

from rlvr_games.cli.common import add_common_play_arguments
from rlvr_games.cli.registry import PLAY_GAME_SPECS
from rlvr_games.cli.session import run_play_session
from rlvr_games.cli.specs import GameCliSpec


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

    raise ValueError(f"Unsupported command arguments: {argv!r}")


def main() -> int:
    """Run the RLVR CLI using `sys.argv`.

    Returns
    -------
    int
        Process-style exit code.
    """
    return run_cli(sys.argv[1:])
