"""2048-specific CLI registration."""

from argparse import ArgumentParser, ArgumentTypeError, Namespace
from typing import Any

from rlvr_games.cli.common import build_episode_config
from rlvr_games.cli.specs import GameCliSpec
from rlvr_games.core.protocol import Environment
from rlvr_games.core.types import StepResult
from rlvr_games.games.game2048.factory import make_game2048_env
from rlvr_games.games.game2048.scenarios import (
    STANDARD_2048_SIZE,
    STANDARD_2048_TARGET,
    normalize_initial_board,
)


def register_game2048_arguments(parser: ArgumentParser) -> None:
    """Attach 2048-specific CLI arguments to a play subparser.

    Parameters
    ----------
    parser : ArgumentParser
        2048 play subparser to configure.
    """
    parser.add_argument(
        "--board",
        type=parse_2048_board_argument,
    )
    parser.add_argument(
        "--target-value",
        type=int,
        default=STANDARD_2048_TARGET,
    )


def build_game2048_environment(
    args: Namespace,
    parser: ArgumentParser,
) -> Environment[Any, Any]:
    """Construct a 2048 environment from parsed CLI arguments.

    Parameters
    ----------
    args : Namespace
        Parsed CLI arguments for a 2048 play session.
    parser : ArgumentParser
        Parser used to report invalid argument combinations.

    Returns
    -------
    Environment[Any, Any]
        Fully configured 2048 environment.
    """
    initial_board = args.board
    board_size = STANDARD_2048_SIZE
    if initial_board is not None:
        board_size = len(initial_board)

    return make_game2048_env(
        size=board_size,
        target_value=args.target_value,
        initial_board=initial_board,
        initial_score=0,
        initial_move_count=0,
        config=build_episode_config(args=args, parser=parser),
        include_images=args.image_output_dir is not None,
        image_size=args.image_size,
    )


def format_game2048_step_result(step_result: StepResult) -> tuple[str, ...]:
    """Render 2048-specific transition summary lines.

    Parameters
    ----------
    step_result : StepResult
        Step result whose transition metadata should be summarized.

    Returns
    -------
    tuple[str, ...]
        Human-readable summary lines derived from the 2048 transition info.
    """
    summary_lines: list[str] = []
    direction = step_result.info.get("direction")
    score_gain = step_result.info.get("score_gain")
    spawned_tile = step_result.info.get("spawned_tile")
    if direction is not None:
        summary_lines.append(f"Direction: {direction}")
    if score_gain is not None:
        summary_lines.append(f"Score gain: {score_gain}")
    if isinstance(spawned_tile, dict):
        summary_lines.append(
            (
                "Spawned tile: "
                f"value={spawned_tile.get('value')} "
                f"row={spawned_tile.get('row')} "
                f"col={spawned_tile.get('col')}"
            )
        )
    return tuple(summary_lines)


def parse_2048_board_argument(raw_board: str) -> tuple[tuple[int, ...], ...]:
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


GAME2048_CLI_SPEC = GameCliSpec(
    name="2048",
    register_arguments=register_game2048_arguments,
    build_environment=build_game2048_environment,
    format_step_result=format_game2048_step_result,
    interactive_commands=(),
)
