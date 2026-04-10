"""Command-line entrypoints for interactive RLVR sessions."""

from argparse import ArgumentParser
import json
from pathlib import Path
import sys
from typing import Any, Sequence, TextIO

from rlvr_games.core.exceptions import InvalidActionError
from rlvr_games.core.protocol import Environment
from rlvr_games.core.rollout import build_action_context
from rlvr_games.core.types import Observation, StepResult
from rlvr_games.games.chess import (
    ChessImageOrientation,
    ChessTextRendererKind,
    STANDARD_START_FEN,
    make_chess_env,
)


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
    play_parser.add_argument("game", choices=("chess",))
    play_parser.add_argument("--seed", type=int, default=0)
    play_parser.add_argument("--fen", default=STANDARD_START_FEN)
    play_parser.add_argument("--max-turns", type=int)
    play_parser.add_argument(
        "--renderer",
        choices=tuple(kind.value for kind in ChessTextRendererKind),
        default=ChessTextRendererKind.ASCII.value,
    )
    play_parser.add_argument("--image-output-dir", type=Path)
    play_parser.add_argument("--image-size", type=int, default=360)
    play_parser.add_argument("--image-coordinates", action="store_true")
    play_parser.add_argument(
        "--image-orientation",
        choices=tuple(orientation.value for orientation in ChessImageOrientation),
        default=ChessImageOrientation.WHITE.value,
    )

    return parser


def run_play_session(
    *,
    env: Environment[Any, Any],
    seed: int,
    input_stream: TextIO,
    output_stream: TextIO,
) -> int:
    """Run an interactive play session against an in-process environment.

    Parameters
    ----------
    env : Environment[Any, Any]
        Environment to reset and step during the session.
    seed : int
        Explicit scenario seed passed into `env.reset(...)`.
    input_stream : TextIO
        Input stream used to read commands or raw actions.
    output_stream : TextIO
        Output stream used to print observations and transition summaries.

    Returns
    -------
    int
        Process-style exit code, where `0` denotes a clean session exit.
    """
    observation, reset_info = env.reset(seed=seed)
    _write_line(output_stream, f"Reset info: {_format_json(reset_info)}")
    _write_observation(output_stream, observation)

    while True:
        context = build_action_context(env=env)
        output_stream.write(f"turn[{context.turn_index}]> ")
        output_stream.flush()

        raw_input = input_stream.readline()
        if raw_input == "":
            _write_line(output_stream, "")
            return 0

        command = raw_input.strip()
        if not command:
            continue

        if command in {"quit", "exit"}:
            _write_line(output_stream, "Session ended.")
            return 0

        if command == "help":
            _write_help(output_stream)
            continue

        if command == "legal":
            legal_actions = " ".join(context.legal_actions)
            _write_line(
                output_stream,
                f"Legal actions ({len(context.legal_actions)}): {legal_actions}",
            )
            continue

        if command == "fen":
            fen = observation.metadata.get("fen")
            _write_line(output_stream, f"FEN: {fen}")
            continue

        if command == "trajectory":
            _write_trajectory(output_stream, env)
            continue

        try:
            step_result = env.step(command)
        except InvalidActionError as exc:
            _write_line(output_stream, f"Invalid action: {exc}")
            continue

        observation = step_result.observation
        _write_step_result(output_stream, step_result)
        _write_observation(output_stream, observation)

        if step_result.terminated or step_result.truncated:
            _write_line(output_stream, "Episode finished.")
            return 0


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

    if args.command == "play" and args.game == "chess":
        env = make_chess_env(
            initial_fen=args.fen,
            max_turns=args.max_turns,
            text_renderer_kind=ChessTextRendererKind(args.renderer),
            image_output_dir=args.image_output_dir,
            image_size=args.image_size,
            image_coordinates=args.image_coordinates,
            image_orientation=ChessImageOrientation(args.image_orientation),
        )
        return run_play_session(
            env=env,
            seed=args.seed,
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


def _format_json(payload: dict[str, object]) -> str:
    """Serialize a metadata dictionary for CLI output."""
    return json.dumps(payload, default=str, sort_keys=True)


def _write_help(output_stream: TextIO) -> None:
    """Print the supported interactive commands."""
    _write_line(output_stream, "Commands: help legal fen trajectory quit exit")


def _write_line(output_stream: TextIO, line: str) -> None:
    """Write one newline-terminated line to the output stream."""
    output_stream.write(f"{line}\n")


def _write_observation(output_stream: TextIO, observation: Observation) -> None:
    """Write an observation's text and image paths to the output stream."""
    if observation.text is not None:
        _write_line(output_stream, observation.text)
    if observation.image_paths:
        image_paths = ", ".join(str(path) for path in observation.image_paths)
        _write_line(output_stream, f"Image paths: {image_paths}")


def _write_step_result(output_stream: TextIO, step_result: StepResult) -> None:
    """Write a step result summary to the output stream."""
    move_uci = step_result.info.get("move_uci")
    move_san = step_result.info.get("move_san")
    if move_uci is not None:
        _write_line(output_stream, f"Move UCI: {move_uci}")
    if move_san is not None:
        _write_line(output_stream, f"Move SAN: {move_san}")
    _write_line(output_stream, f"Reward: {step_result.reward}")
    _write_line(output_stream, f"Terminated: {step_result.terminated}")
    _write_line(output_stream, f"Truncated: {step_result.truncated}")
    _write_line(output_stream, f"Info: {_format_json(step_result.info)}")


def _write_trajectory(output_stream: TextIO, env: Environment[Any, Any]) -> None:
    """Write a short trajectory summary for the active episode."""
    _write_line(output_stream, f"Trajectory steps: {len(env.trajectory.steps)}")
    for turn_index, step in enumerate(env.trajectory.steps, start=1):
        info_text = _format_json(step.info)
        _write_line(
            output_stream,
            (
                f"{turn_index}. raw_action={step.raw_action!r} "
                f"reward={step.reward} terminated={step.terminated} "
                f"truncated={step.truncated} info={info_text}"
            ),
        )
