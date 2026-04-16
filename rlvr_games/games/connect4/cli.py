"""Connect 4-specific CLI registration."""

from argparse import ArgumentParser, ArgumentTypeError, Namespace
from enum import StrEnum
from typing import Any

from rlvr_games.cli.common import build_episode_config
from rlvr_games.cli.specs import GameCliSpec
from rlvr_games.core.protocol import Environment, RewardFn
from rlvr_games.core.types import StepResult
from rlvr_games.games.connect4.actions import Connect4Action
from rlvr_games.games.connect4.factory import make_connect4_env
from rlvr_games.games.connect4.rewards import (
    SolverMoveScoreReward,
    TerminalOutcomeReward,
)
from rlvr_games.games.connect4.scenarios import (
    DEFAULT_RANDOM_START_MAX_MOVES,
    FixedBoardScenario,
    RandomPositionScenario,
    normalize_initial_board,
)
from rlvr_games.games.connect4.solver import BitBullySolver
from rlvr_games.games.connect4.state import Board, Connect4State
from rlvr_games.games.connect4.turns import Connect4SolverAutoAdvancePolicy
from rlvr_games.games.connect4.variant import validate_standard_connect4_dimensions


class Connect4RewardKind(StrEnum):
    """Supported Connect 4 reward policies exposed through the CLI."""

    TERMINAL = "terminal"
    SOLVER_MOVE_DENSE = "solver-move-dense"


class Connect4OpponentKind(StrEnum):
    """Supported Connect 4 opponent modes exposed through the CLI."""

    HUMAN = "human"
    SOLVER = "solver"


def register_connect4_arguments(parser: ArgumentParser) -> None:
    """Attach Connect 4-specific CLI arguments to a play subparser.

    Parameters
    ----------
    parser : ArgumentParser
        Connect 4 play subparser to configure.
    """
    parser.add_argument(
        "--max-start-moves", type=int, default=DEFAULT_RANDOM_START_MAX_MOVES
    )
    parser.add_argument(
        "--reward",
        choices=tuple(kind.value for kind in Connect4RewardKind),
        default=Connect4RewardKind.TERMINAL.value,
    )
    parser.add_argument(
        "--opponent",
        choices=tuple(kind.value for kind in Connect4OpponentKind),
        default=Connect4OpponentKind.HUMAN.value,
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
    scenario = build_connect4_scenario(args=args, parser=parser)
    solver: BitBullySolver | None = None
    if connect4_solver_requested(args=args):
        solver = BitBullySolver()

    return make_connect4_env(
        scenario=scenario,
        reward_fn=build_connect4_reward(
            args=args,
            solver=solver,
        ),
        config=build_episode_config(args=args, parser=parser),
        include_images=args.image_output_dir is not None,
        image_size=args.image_size,
        auto_advance_policy=build_connect4_auto_advance_policy(
            args=args,
            solver=solver,
        ),
    )


def build_connect4_scenario(
    *,
    args: Namespace,
    parser: ArgumentParser,
) -> RandomPositionScenario | FixedBoardScenario:
    """Construct a Connect 4 scenario from parsed CLI arguments.

    Parameters
    ----------
    args : Namespace
        Parsed CLI arguments for a Connect 4 play session.
    parser : ArgumentParser
        Parser used to report invalid argument combinations.

    Returns
    -------
    RandomPositionScenario | FixedBoardScenario
        Scenario implied by the parsed CLI arguments.
    """
    if args.max_start_moves < 0:
        parser.error("--max-start-moves must be non-negative for connect4.")

    if args.board is None:
        return RandomPositionScenario(
            min_start_moves=0,
            max_start_moves=args.max_start_moves,
        )

    if args.max_start_moves != DEFAULT_RANDOM_START_MAX_MOVES:
        parser.error("--max-start-moves is only supported without --board.")
    try:
        validate_standard_connect4_dimensions(
            rows=len(args.board),
            columns=len(args.board[0]),
        )
    except ValueError as exc:
        parser.error(str(exc))
    return FixedBoardScenario(initial_board=args.board)


def build_connect4_reward(
    *,
    args: Namespace,
    solver: BitBullySolver | None,
) -> RewardFn[Connect4State, Connect4Action]:
    """Construct a Connect 4 reward function from parsed CLI arguments.

    Parameters
    ----------
    args : Namespace
        Parsed CLI arguments for a Connect 4 play session.
    solver : BitBullySolver | None
        Shared BitBully solver when a solver-backed feature was requested.

    Returns
    -------
    RewardFn[Connect4State, Connect4Action]
        Reward function implied by the parsed CLI arguments.
    """
    reward_kind = Connect4RewardKind(args.reward)
    if reward_kind == Connect4RewardKind.TERMINAL:
        return TerminalOutcomeReward(
            perspective="mover",
            win_reward=1.0,
            draw_reward=0.0,
            loss_reward=-1.0,
        )

    if solver is None:
        raise ValueError("BitBully solver reward construction requires a solver.")
    return SolverMoveScoreReward(
        scorer=solver,
        perspective="mover",
    )


def build_connect4_auto_advance_policy(
    *,
    args: Namespace,
    solver: BitBullySolver | None,
) -> Connect4SolverAutoAdvancePolicy | None:
    """Construct the Connect 4 auto-advance policy implied by parsed args.

    Parameters
    ----------
    args : Namespace
        Parsed CLI arguments for a Connect 4 play session.
    solver : BitBullySolver | None
        Shared BitBully solver when a solver-backed feature was requested.

    Returns
    -------
    Connect4SolverAutoAdvancePolicy | None
        Auto-advance policy implied by the parsed CLI arguments.
    """
    opponent_kind = Connect4OpponentKind(args.opponent)
    if opponent_kind == Connect4OpponentKind.HUMAN:
        return None

    if solver is None:
        raise ValueError("BitBully solver opponent construction requires a solver.")
    return Connect4SolverAutoAdvancePolicy(move_selector=solver)


def connect4_solver_requested(*, args: Namespace) -> bool:
    """Return whether any Connect 4 CLI option requires BitBully."""
    return (
        Connect4RewardKind(args.reward) == Connect4RewardKind.SOLVER_MOVE_DENSE
        or Connect4OpponentKind(args.opponent) == Connect4OpponentKind.SOLVER
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
    transitions_payload = step_result.info.get("transitions")
    if isinstance(transitions_payload, tuple) and transitions_payload:
        summary_lines: list[str] = []
        multiple_transitions = len(transitions_payload) > 1
        for transition_payload in transitions_payload:
            if not isinstance(transition_payload, dict):
                continue
            source = transition_payload.get("source")
            transition_info = transition_payload.get("info")
            if not isinstance(source, str) or not isinstance(transition_info, dict):
                continue
            prefix = ""
            if multiple_transitions:
                prefix = f"{source.replace('_', ' ').title()} "
            _append_connect4_summary(
                summary_lines=summary_lines,
                transition_info=transition_info,
                prefix=prefix,
            )
        if summary_lines:
            return tuple(summary_lines)

    summary_lines = []
    _append_connect4_summary(
        summary_lines=summary_lines,
        transition_info=step_result.info,
        prefix="",
    )
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


def _append_connect4_summary(
    *,
    summary_lines: list[str],
    transition_info: dict[str, object],
    prefix: str,
) -> None:
    """Append one transition summary to the output line buffer."""
    player = transition_info.get("player")
    column = transition_info.get("column")
    row_from_bottom = transition_info.get("row_from_bottom")
    winner = transition_info.get("winner")
    if player is not None:
        summary_lines.append(f"{prefix}Player: {player}")
    if column is not None:
        summary_lines.append(f"{prefix}Column: {column}")
    if row_from_bottom is not None:
        summary_lines.append(f"{prefix}Row from bottom: {row_from_bottom}")
    if winner is not None:
        summary_lines.append(f"{prefix}Winner: {winner}")


CONNECT4_CLI_SPEC = GameCliSpec(
    name="connect4",
    register_arguments=register_connect4_arguments,
    build_environment=build_connect4_environment,
    format_step_result=format_connect4_step_result,
    interactive_commands=(),
)
