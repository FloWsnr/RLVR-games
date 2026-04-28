"""Generic command line entrypoint for playable task interactions."""

from argparse import ArgumentParser
import sys
from typing import Sequence, TextIO

from rlvr_physics.play.registry import get_playable_task, playable_task_names
from rlvr_physics.play.task import run_playable_interaction_cli


def main(argv: Sequence[str] | None = None) -> int:
    """Run the generic play command line interface.

    Parameters
    ----------
    argv:
        Command line arguments excluding the program name. ``None`` reads
        arguments from :data:`sys.argv`.

    Returns
    -------
    int
        Process exit code.
    """

    return run_play_cli(
        argv=argv,
        input_stream=sys.stdin,
        output_stream=sys.stdout,
        error_stream=sys.stderr,
    )


def run_play_cli(
    argv: Sequence[str] | None,
    input_stream: TextIO,
    output_stream: TextIO,
    error_stream: TextIO,
) -> int:
    """Run a registered playable task through the JSONL CLI.

    Parameters
    ----------
    argv:
        Command line arguments excluding the program name.
    input_stream:
        Stream containing one model submission per line.
    output_stream:
        Stream receiving either task names for ``--list`` or public JSONL
        interaction events.
    error_stream:
        Stream receiving CLI and process-level errors.

    Returns
    -------
    int
        Process exit code.
    """

    raw_args = tuple(sys.argv[1:] if argv is None else argv)
    parser = play_argument_parser()
    if len(raw_args) == 0:
        parser.error("task is required unless --list is used")
    if raw_args[0] in {"-h", "--help"}:
        parser.parse_args(raw_args)
        return 0
    if raw_args[0] == "--list":
        parser.parse_args(raw_args)
        _write_playable_task_names(output_stream)
        return 0
    if raw_args[0].startswith("-"):
        parser.parse_args(raw_args)
        return 0

    task_name = raw_args[0]
    playable = get_playable_task(task_name)
    if playable is None:
        choices = ", ".join(playable_task_names())
        parser.error(f"unknown task {task_name!r}; choose one of: {choices}")

    return run_playable_interaction_cli(
        playable=playable,
        argv=raw_args[1:],
        input_stream=input_stream,
        output_stream=output_stream,
        error_stream=error_stream,
        program_name=f"play {task_name}",
    )


def play_argument_parser() -> ArgumentParser:
    """Build the generic play command argument parser.

    Returns
    -------
    ArgumentParser
        Parser for selecting playable tasks.
    """

    parser = ArgumentParser(
        prog="play",
        description="Run a registered RLVR physics task through JSONL play mode.",
    )
    parser.add_argument(
        "task",
        nargs="?",
        help="playable task name or alias",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list available playable task names and aliases",
    )
    return parser


def _write_playable_task_names(output_stream: TextIO) -> None:
    """Write available playable task names and aliases."""

    for task_name in playable_task_names():
        output_stream.write(task_name)
        output_stream.write("\n")
    output_stream.flush()


if __name__ == "__main__":
    raise SystemExit(main())
