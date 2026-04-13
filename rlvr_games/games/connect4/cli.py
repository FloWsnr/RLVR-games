"""Connect 4-specific CLI registration."""

from argparse import ArgumentParser, ArgumentTypeError, Namespace
from typing import Any

from rlvr_games.cli.common import build_episode_config
from rlvr_games.cli.specs import GameCliSpec
from rlvr_games.core.protocol import Environment
from rlvr_games.core.types import StepResult
from rlvr_games.games.connect4.factory import make_connect4_env
from rlvr_games.games.connect4.rewards import TerminalOutcomeReward
from rlvr_games.games.connect4.scenarios import (
    DEFAULT_RANDOM_START_MAX_MOVES,
    FixedBoardScenario,
    RandomPositionScenario,
    STANDARD_CONNECT4_COLUMNS,
    STANDARD_CONNECT4_CONNECT_LENGTH,
    STANDARD_CONNECT4_ROWS,
    normalize_initial_board,
)
from rlvr_games.games.connect4.state import Board


def register_connect4_arguments(parser: ArgumentParser) -> None:
    """Attach Connect 4-specific CLI arguments to a play subparser.

    Parameters
    ----------
    parser : ArgumentParser
        Connect 4 play subparser to configure.
    """
    parser.add_argument("--rows", type=int, default=STANDARD_CONNECT4_ROWS)
    parser.add_argument("--columns", type=int, default=STANDARD_CONNECT4_COLUMNS)
    parser.add_argument(
        "--connect-length",
        type=int,
        default=STANDARD_CONNECT4_CONNECT_LENGTH,
    )
    parser.add_argument(
        "--max-start-moves", type=int, default=DEFAULT_RANDOM_START_MAX_MOVES
    )
    parser.add_argument("--board", type=parse_connect4_board_argument)


def build_connect4_environment(
    args: Namespace,
    parser: ArgumentParser,
) -> Environment[Any, Any]:
    """Construct a Connect 4 environment from parsed CLI arguments.

    Parameters
    ----------
    args : Namespace
        Parsed CLI arguments for a Connect 4 play session.
    parser : ArgumentParser
        Parser used to report invalid argument combinations.

    Returns
    -------
    Environment[Any, Any]
        Fully configured Connect 4 environment.
    """
    if args.max_start_moves < 0:
        parser.error("--max-start-moves must be non-negative for connect4.")

    if args.board is None:
        scenario = RandomPositionScenario(
            rows=args.rows,
            columns=args.columns,
            connect_length=args.connect_length,
            min_start_moves=0,
            max_start_moves=args.max_start_moves,
        )
    else:
        if args.max_start_moves != DEFAULT_RANDOM_START_MAX_MOVES:
            parser.error("--max-start-moves is only supported without --board.")
        scenario = FixedBoardScenario(
            initial_board=args.board,
            connect_length=args.connect_length,
        )

    return make_connect4_env(
        scenario=scenario,
        reward_fn=TerminalOutcomeReward(
            perspective="mover",
            win_reward=1.0,
            draw_reward=0.0,
            loss_reward=-1.0,
        ),
        config=build_episode_config(args=args, parser=parser),
        include_images=args.image_output_dir is not None,
        image_size=args.image_size,
    )


def format_connect4_step_result(step_result: StepResult) -> tuple[str, ...]:
    """Render Connect 4-specific transition summary lines.

    Parameters
    ----------
    step_result : StepResult
        Step result whose transition metadata should be summarized.

    Returns
    -------
    tuple[str, ...]
        Human-readable summary lines derived from the Connect 4 transition
        info.
    """
    summary_lines: list[str] = []
    player = step_result.info.get("player")
    column = step_result.info.get("column")
    row_from_bottom = step_result.info.get("row_from_bottom")
    winner = step_result.info.get("winner")
    if player is not None:
        summary_lines.append(f"Player: {player}")
    if column is not None:
        summary_lines.append(f"Column: {column}")
    if row_from_bottom is not None:
        summary_lines.append(f"Row from bottom: {row_from_bottom}")
    if winner is not None:
        summary_lines.append(f"Winner: {winner}")
    return tuple(summary_lines)


def parse_connect4_board_argument(raw_board: str) -> Board:
    """Parse a CLI Connect 4 board argument into a canonical nested tuple.

    Parameters
    ----------
    raw_board : str
        Board text in the form ``"......./......./......./...x..."``.

    Returns
    -------
    Board
        Parsed immutable board.

    Raises
    ------
    ArgumentTypeError
        If the board text cannot be parsed or validated.
    """
    rows = tuple(
        tuple(character.lower() for character in row_text.strip())
        for row_text in raw_board.split("/")
    )
    try:
        return normalize_initial_board(board=rows)
    except ValueError as exc:
        raise ArgumentTypeError(str(exc)) from exc


CONNECT4_CLI_SPEC = GameCliSpec(
    name="connect4",
    register_arguments=register_connect4_arguments,
    build_environment=build_connect4_environment,
    format_step_result=format_connect4_step_result,
    interactive_commands=(),
)
