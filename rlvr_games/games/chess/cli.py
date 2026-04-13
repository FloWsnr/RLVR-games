"""Chess-specific CLI registration."""

from argparse import ArgumentParser, Namespace
from enum import StrEnum
from pathlib import Path
from typing import Any

from rlvr_games.cli.common import build_episode_config
from rlvr_games.cli.specs import GameCliSpec
from rlvr_games.core.protocol import Environment
from rlvr_games.core.rewards import ZeroReward
from rlvr_games.core.types import StepResult
from rlvr_games.datasets import DatasetSplit
from rlvr_games.games.chess.factory import (
    ChessBoardOrientation,
    ChessTextRendererKind,
    make_chess_env,
)
from rlvr_games.games.chess.scenarios import (
    ChessPuzzleDatasetScenario,
    STANDARD_START_FEN,
    StartingPositionScenario,
)


class ChessScenarioKind(StrEnum):
    """Supported chess scenario kinds exposed through the CLI."""

    STARTING_POSITION = "starting-position"
    LICHESS_PUZZLES = "lichess-puzzles"


def register_chess_arguments(parser: ArgumentParser) -> None:
    """Attach chess-specific CLI arguments to a play subparser.

    Parameters
    ----------
    parser : ArgumentParser
        Chess play subparser to configure.
    """
    parser.add_argument(
        "--scenario",
        choices=tuple(kind.value for kind in ChessScenarioKind),
        default=ChessScenarioKind.STARTING_POSITION.value,
    )
    parser.add_argument("--fen", default=STANDARD_START_FEN)
    parser.add_argument("--dataset-manifest", type=Path)
    parser.add_argument(
        "--dataset-split",
        choices=tuple(split.value for split in DatasetSplit),
        default=DatasetSplit.TRAIN.value,
    )
    parser.add_argument(
        "--renderer",
        choices=tuple(kind.value for kind in ChessTextRendererKind),
        default=ChessTextRendererKind.ASCII.value,
    )
    parser.add_argument("--image-coordinates", action="store_true")
    parser.add_argument(
        "--orientation",
        choices=tuple(orientation.value for orientation in ChessBoardOrientation),
        default=ChessBoardOrientation.WHITE.value,
    )


def build_chess_environment(
    args: Namespace,
    parser: ArgumentParser,
) -> Environment[Any, Any]:
    """Construct a chess environment from parsed CLI arguments.

    Parameters
    ----------
    args : Namespace
        Parsed CLI arguments for a chess play session.
    parser : ArgumentParser
        Parser used to report invalid argument combinations.

    Returns
    -------
    Environment[Any, Any]
        Fully configured chess environment.
    """
    return make_chess_env(
        scenario=build_chess_scenario(args=args, parser=parser),
        reward_fn=ZeroReward(),
        config=build_episode_config(args=args, parser=parser),
        text_renderer_kind=ChessTextRendererKind(args.renderer),
        include_images=args.image_output_dir is not None,
        image_size=args.image_size,
        image_coordinates=args.image_coordinates,
        orientation=ChessBoardOrientation(args.orientation),
    )


def build_chess_scenario(
    *,
    args: Namespace,
    parser: ArgumentParser,
) -> StartingPositionScenario | ChessPuzzleDatasetScenario:
    """Construct a chess scenario from parsed CLI arguments.

    Parameters
    ----------
    args : Namespace
        Parsed CLI arguments for a chess play session.
    parser : ArgumentParser
        Parser used to report invalid argument combinations.

    Returns
    -------
    StartingPositionScenario | ChessPuzzleDatasetScenario
        Scenario instance implied by the parsed CLI arguments.
    """
    scenario_kind = ChessScenarioKind(args.scenario)
    if scenario_kind == ChessScenarioKind.STARTING_POSITION:
        if args.dataset_manifest is not None:
            parser.error(
                "--dataset-manifest requires --scenario lichess-puzzles for chess."
            )
        return StartingPositionScenario(initial_fen=args.fen)

    if args.dataset_manifest is None:
        parser.error("--dataset-manifest is required for --scenario lichess-puzzles.")
    if args.fen != STANDARD_START_FEN:
        parser.error("--fen is only supported with --scenario starting-position.")

    return ChessPuzzleDatasetScenario(
        manifest_path=args.dataset_manifest,
        split=DatasetSplit(args.dataset_split),
    )


def format_chess_step_result(step_result: StepResult) -> tuple[str, ...]:
    """Render chess-specific transition summary lines.

    Parameters
    ----------
    step_result : StepResult
        Step result whose transition metadata should be summarized.

    Returns
    -------
    tuple[str, ...]
        Human-readable summary lines derived from the chess transition info.
    """
    summary_lines: list[str] = []
    move_uci = step_result.info.get("move_uci")
    move_san = step_result.info.get("move_san")
    if move_uci is not None:
        summary_lines.append(f"Move UCI: {move_uci}")
    if move_san is not None:
        summary_lines.append(f"Move SAN: {move_san}")
    return tuple(summary_lines)


CHESS_CLI_SPEC = GameCliSpec(
    name="chess",
    register_arguments=register_chess_arguments,
    build_environment=build_chess_environment,
    format_step_result=format_chess_step_result,
    interactive_commands=(),
)
