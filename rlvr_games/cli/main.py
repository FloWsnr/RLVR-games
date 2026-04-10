"""Command-line entrypoints for interactive RLVR sessions."""

from argparse import ArgumentParser, ArgumentTypeError, Namespace
import json
from pathlib import Path
import sys
from typing import Any, Sequence, TextIO

from rlvr_games.core.exceptions import InvalidActionError
from rlvr_games.core.protocol import Environment
from rlvr_games.core.rollout import build_action_context
from rlvr_games.core.types import (
    EpisodeConfig,
    InvalidActionMode,
    InvalidActionPolicy,
    Observation,
    RenderedImage,
    StepResult,
)
from rlvr_games.games.chess import (
    ChessBoardOrientation,
    ChessTextRendererKind,
    STANDARD_START_FEN,
    make_chess_env,
)
from rlvr_games.games.game2048 import (
    STANDARD_2048_SIZE,
    STANDARD_2048_TARGET,
    make_game2048_env,
    normalize_initial_board,
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
    play_subparsers = play_parser.add_subparsers(dest="game", required=True)

    chess_parser = play_subparsers.add_parser("chess")
    _add_common_play_arguments(chess_parser)
    chess_parser.add_argument("--fen", default=STANDARD_START_FEN)
    chess_parser.add_argument(
        "--renderer",
        choices=tuple(kind.value for kind in ChessTextRendererKind),
        default=ChessTextRendererKind.ASCII.value,
    )
    chess_parser.add_argument("--image-coordinates", action="store_true")
    chess_parser.add_argument(
        "--orientation",
        choices=tuple(orientation.value for orientation in ChessBoardOrientation),
        default=ChessBoardOrientation.WHITE.value,
    )
    game_2048_parser = play_subparsers.add_parser("2048")
    _add_common_play_arguments(game_2048_parser)
    game_2048_parser.add_argument(
        "--board",
        type=_parse_2048_board_argument,
    )
    game_2048_parser.add_argument(
        "--target-value",
        type=int,
        default=STANDARD_2048_TARGET,
    )

    return parser


def _add_common_play_arguments(parser: ArgumentParser) -> None:
    """Attach play-session arguments shared by supported games.

    Parameters
    ----------
    parser : ArgumentParser
        Parser to extend with common play-session arguments.
    """
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--max-transitions", type=int)
    parser.add_argument("--image-output-dir", type=Path)
    parser.add_argument("--image-size", type=int, default=360)
    parser.add_argument(
        "--invalid-action-policy",
        choices=tuple(mode.value for mode in InvalidActionMode),
        default=InvalidActionMode.RAISE.value,
    )
    parser.add_argument("--invalid-action-penalty", type=float)


def run_play_session(
    *,
    env: Environment[Any, Any],
    seed: int,
    image_output_dir: Path | None,
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
    image_output_dir : Path | None
        Optional directory where any rendered observation images should be
        persisted as PNG files for inspection.
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
    _write_observation(
        output_stream,
        observation,
        image_output_dir=image_output_dir,
    )
    if env.episode_finished:
        _write_line(output_stream, "Episode finished.")
        return 0

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

        if command == "state":
            _write_line(output_stream, f"State: {_format_json(observation.metadata)}")
            continue

        if command == "fen":
            fen = observation.metadata.get("fen")
            if fen is None:
                _write_line(output_stream, "FEN unavailable for this game.")
            else:
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
        _write_observation(
            output_stream,
            observation,
            image_output_dir=image_output_dir,
        )

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
        invalid_action_policy = _build_invalid_action_policy(args=args, parser=parser)
        env: Environment[Any, Any] = make_chess_env(
            initial_fen=args.fen,
            config=EpisodeConfig(
                max_attempts=args.max_attempts,
                max_transitions=args.max_transitions,
                invalid_action_policy=invalid_action_policy,
            ),
            text_renderer_kind=ChessTextRendererKind(args.renderer),
            include_images=args.image_output_dir is not None,
            image_size=args.image_size,
            image_coordinates=args.image_coordinates,
            orientation=ChessBoardOrientation(args.orientation),
        )
        return run_play_session(
            env=env,
            seed=args.seed,
            image_output_dir=args.image_output_dir,
            input_stream=sys.stdin,
            output_stream=sys.stdout,
        )

    if args.command == "play" and args.game == "2048":
        invalid_action_policy = _build_invalid_action_policy(args=args, parser=parser)
        initial_board = args.board
        board_size = STANDARD_2048_SIZE
        if initial_board is not None:
            board_size = len(initial_board)
        env = make_game2048_env(
            size=board_size,
            target_value=args.target_value,
            initial_board=initial_board,
            initial_score=0,
            initial_move_count=0,
            config=EpisodeConfig(
                max_attempts=args.max_attempts,
                max_transitions=args.max_transitions,
                invalid_action_policy=invalid_action_policy,
            ),
            include_images=args.image_output_dir is not None,
            image_size=args.image_size,
        )
        return run_play_session(
            env=env,
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


def _format_json(payload: dict[str, object]) -> str:
    """Serialize a metadata dictionary for CLI output."""
    return json.dumps(payload, default=str, sort_keys=True)


def _build_invalid_action_policy(
    *,
    args: Namespace,
    parser: ArgumentParser,
) -> InvalidActionPolicy:
    """Build the invalid-action policy implied by parsed CLI arguments.

    Parameters
    ----------
    args : Namespace
        Parsed CLI arguments.
    parser : ArgumentParser
        Parser used to raise argument errors when the configuration is
        inconsistent.

    Returns
    -------
    InvalidActionPolicy
        Validated invalid-action policy for the environment.
    """
    invalid_action_mode = InvalidActionMode(args.invalid_action_policy)
    if invalid_action_mode == InvalidActionMode.RAISE:
        if args.invalid_action_penalty is not None:
            parser.error(
                "--invalid-action-penalty requires a penalize invalid-action policy."
            )
        return InvalidActionPolicy(
            mode=invalid_action_mode,
            penalty=None,
        )

    if args.invalid_action_penalty is None:
        parser.error(
            "--invalid-action-penalty is required for penalize invalid-action policies."
        )

    return InvalidActionPolicy(
        mode=invalid_action_mode,
        penalty=args.invalid_action_penalty,
    )


def _write_help(output_stream: TextIO) -> None:
    """Print the supported interactive commands."""
    _write_line(output_stream, "Commands: help legal state fen trajectory quit exit")


def _write_line(output_stream: TextIO, line: str) -> None:
    """Write one newline-terminated line to the output stream."""
    output_stream.write(f"{line}\n")


def _write_observation(
    output_stream: TextIO,
    observation: Observation,
    *,
    image_output_dir: Path | None,
) -> None:
    """Write an observation's text and any persisted image paths.

    Parameters
    ----------
    output_stream : TextIO
        Stream receiving formatted observation output.
    observation : Observation
        Observation to print and optionally persist.
    image_output_dir : Path | None
        Optional directory where in-memory images should be saved as PNG files.
    """
    if observation.text is not None:
        _write_line(output_stream, observation.text)
    if observation.images and image_output_dir is not None:
        image_paths = ", ".join(
            str(path)
            for path in _persist_rendered_images(
                image_output_dir=image_output_dir,
                images=observation.images,
            )
        )
        _write_line(output_stream, f"Image paths: {image_paths}")


def _persist_rendered_images(
    *,
    image_output_dir: Path,
    images: tuple[RenderedImage, ...],
) -> tuple[Path, ...]:
    """Persist rendered images to PNG files and return their paths.

    Parameters
    ----------
    image_output_dir : Path
        Directory where PNG files should be written.
    images : tuple[RenderedImage, ...]
        In-memory images to persist.

    Returns
    -------
    tuple[Path, ...]
        Persisted PNG paths in the same order as the input images.
    """
    image_output_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    for image_index, rendered_image in enumerate(images):
        suffix = "" if image_index == 0 else f"-{image_index}"
        image_path = image_output_dir / f"{rendered_image.key}{suffix}.png"
        if not image_path.exists():
            rendered_image.image.save(image_path, format="PNG")
        image_paths.append(image_path)
    return tuple(image_paths)


def _write_step_result(output_stream: TextIO, step_result: StepResult) -> None:
    """Write a step result summary to the output stream."""
    _write_line(output_stream, f"Accepted: {step_result.accepted}")
    move_uci = step_result.info.get("move_uci")
    move_san = step_result.info.get("move_san")
    direction = step_result.info.get("direction")
    score_gain = step_result.info.get("score_gain")
    spawned_tile = step_result.info.get("spawned_tile")
    if move_uci is not None:
        _write_line(output_stream, f"Move UCI: {move_uci}")
    if move_san is not None:
        _write_line(output_stream, f"Move SAN: {move_san}")
    if direction is not None:
        _write_line(output_stream, f"Direction: {direction}")
    if score_gain is not None:
        _write_line(output_stream, f"Score gain: {score_gain}")
    if isinstance(spawned_tile, dict):
        _write_line(
            output_stream,
            (
                "Spawned tile: "
                f"value={spawned_tile.get('value')} "
                f"row={spawned_tile.get('row')} "
                f"col={spawned_tile.get('col')}"
            ),
        )
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
                f"accepted={step.accepted} "
                f"reward={step.reward} terminated={step.terminated} "
                f"truncated={step.truncated} info={info_text}"
            ),
        )


def _parse_2048_board_argument(raw_board: str) -> tuple[tuple[int, ...], ...]:
    """Parse a CLI 2048 board argument into a canonical nested tuple.

    Parameters
    ----------
    raw_board : str
        Board text in the form ``"2,0,0,0/0,2,0,0/..."``.

    Returns
    -------
    tuple[tuple[int, ...], ...]
        Parsed immutable board.

    Raises
    ------
    ArgumentTypeError
        If the board text cannot be parsed or validated.
    """
    try:
        rows = tuple(
            tuple(int(value.strip()) for value in row_text.split(","))
            for row_text in raw_board.split("/")
        )
    except ValueError as exc:
        raise ArgumentTypeError(
            "2048 boards must use comma-separated integers and '/' row separators."
        ) from exc

    try:
        return normalize_initial_board(board=rows)
    except ValueError as exc:
        raise ArgumentTypeError(str(exc)) from exc
