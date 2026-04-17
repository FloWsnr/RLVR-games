"""Minesweeper-specific CLI registration."""

from argparse import ArgumentParser, ArgumentTypeError, Namespace
from enum import StrEnum
from typing import Any

from rlvr_games.cli.common import (
    COMMON_TASK_SPEC_DISALLOWED_ARGUMENT_NAMES,
    build_environment_from_task_spec_argument,
    build_episode_config,
)
from rlvr_games.cli.specs import GameCliSpec
from rlvr_games.core.protocol import Environment, RewardFn
from rlvr_games.core.types import StepResult
from rlvr_games.games.minesweeper.actions import MinesweeperAction
from rlvr_games.games.minesweeper.factory import make_minesweeper_env
from rlvr_games.games.minesweeper.rewards import OutcomeReward, SafeRevealCountReward
from rlvr_games.games.minesweeper.scenarios import (
    STANDARD_MINESWEEPER_COLUMNS,
    STANDARD_MINESWEEPER_MINE_COUNT,
    STANDARD_MINESWEEPER_ROWS,
    FixedBoardScenario,
    normalize_initial_board,
)
from rlvr_games.games.minesweeper.state import MinesweeperState


class MinesweeperRewardKind(StrEnum):
    """Supported Minesweeper reward policies exposed through the CLI."""

    OUTCOME = "outcome"
    REVEAL_COUNT_DENSE = "reveal-count-dense"


def register_minesweeper_arguments(parser: ArgumentParser) -> None:
    """Attach Minesweeper-specific CLI arguments to a play subparser."""
    parser.add_argument("--rows", type=int, default=STANDARD_MINESWEEPER_ROWS)
    parser.add_argument("--columns", type=int, default=STANDARD_MINESWEEPER_COLUMNS)
    parser.add_argument("--mines", type=int, default=STANDARD_MINESWEEPER_MINE_COUNT)
    parser.add_argument(
        "--reward",
        choices=tuple(kind.value for kind in MinesweeperRewardKind),
        default=MinesweeperRewardKind.OUTCOME.value,
    )
    parser.add_argument("--board", type=parse_minesweeper_board_argument)


def build_minesweeper_environment(
    args: Namespace,
    parser: ArgumentParser,
) -> Environment[Any, Any]:
    """Construct a Minesweeper environment from parsed CLI arguments."""
    task_spec_environment = build_environment_from_task_spec_argument(
        args=args,
        parser=parser,
        expected_game="minesweeper",
        disallowed_argument_names=COMMON_TASK_SPEC_DISALLOWED_ARGUMENT_NAMES
        + (
            "rows",
            "columns",
            "mines",
            "reward",
            "board",
        ),
    )
    if task_spec_environment is not None:
        return task_spec_environment

    initial_board = args.board
    rows = args.rows
    columns = args.columns
    mine_count = args.mines
    if initial_board is not None:
        scenario = FixedBoardScenario(hidden_board=initial_board)
        rows = len(scenario.hidden_board)
        columns = len(scenario.hidden_board[0])
        mine_count = sum(1 for row in scenario.hidden_board for cell in row if cell)
        initial_board = scenario.hidden_board

    return make_minesweeper_env(
        rows=rows,
        columns=columns,
        mine_count=mine_count,
        initial_board=initial_board,
        reward_fn=build_minesweeper_reward(args=args),
        config=build_episode_config(args=args, parser=parser),
        include_images=args.image_output_dir is not None,
        image_size=args.image_size,
    )


def build_minesweeper_reward(
    *,
    args: Namespace,
) -> RewardFn[MinesweeperState, MinesweeperAction]:
    """Construct a Minesweeper reward function from parsed CLI arguments."""
    reward_kind = MinesweeperRewardKind(args.reward)
    if reward_kind == MinesweeperRewardKind.OUTCOME:
        return OutcomeReward(win_reward=1.0, loss_reward=-1.0)
    return SafeRevealCountReward(mine_penalty=-1.0)


def format_minesweeper_step_result(step_result: StepResult) -> tuple[str, ...]:
    """Render Minesweeper-specific transition summary lines."""
    summary_lines: list[str] = []
    verb = step_result.info.get("verb")
    row = step_result.info.get("row")
    col = step_result.info.get("col")
    newly_revealed_count = step_result.info.get("newly_revealed_count")
    flagged_cell_count = step_result.info.get("flagged_cell_count")
    remaining_safe_cells = step_result.info.get("remaining_safe_cells")
    if verb is not None and row is not None and col is not None:
        summary_lines.append(f"Action: {verb} row={row} col={col}")
    if newly_revealed_count is not None:
        summary_lines.append(f"Newly revealed: {newly_revealed_count}")
    if flagged_cell_count is not None:
        summary_lines.append(f"Flags: {flagged_cell_count}")
    if remaining_safe_cells is not None:
        summary_lines.append(f"Remaining safe cells: {remaining_safe_cells}")
    return tuple(summary_lines)


def parse_minesweeper_board_argument(raw_board: str) -> tuple[tuple[bool, ...], ...]:
    """Parse a CLI Minesweeper board argument into a canonical mine layout.

    Parameters
    ----------
    raw_board : str
        Board text in the form ``".../.*./..."`` using ``"."`` and ``"*"``
        cell markers.

    Returns
    -------
    tuple[tuple[bool, ...], ...]
        Parsed immutable hidden mine layout.
    """
    rows = tuple(row_text.strip() for row_text in raw_board.split("/"))
    try:
        return normalize_initial_board(board=rows)
    except ValueError as exc:
        raise ArgumentTypeError(str(exc)) from exc


MINESWEEPER_CLI_SPEC = GameCliSpec(
    name="minesweeper",
    register_arguments=register_minesweeper_arguments,
    build_environment=build_minesweeper_environment,
    format_step_result=format_minesweeper_step_result,
    interactive_commands=(),
)
