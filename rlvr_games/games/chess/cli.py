"""Chess-specific CLI registration."""

from argparse import ArgumentParser, Namespace
from enum import StrEnum
from pathlib import Path
from typing import Any

from rlvr_games.cli.common import build_episode_config
from rlvr_games.cli.specs import GameCliSpec
from rlvr_games.core.protocol import Environment, RewardFn
from rlvr_games.core.types import StepResult
from rlvr_games.datasets import DatasetSplit
from rlvr_games.games.chess.actions import ChessAction
from rlvr_games.games.chess.factory import (
    ChessBoardOrientation,
    ChessTextRendererKind,
    make_chess_env,
)
from rlvr_games.games.chess.rewards import (
    EngineEvalDenseReward,
    EngineEvalSparseReward,
    PuzzleOnlyMoveDenseReward,
    PuzzleOnlyMoveSparseReward,
)
from rlvr_games.games.chess.scenarios import (
    ChessPuzzleDatasetScenario,
    STANDARD_START_FEN,
    StartingPositionScenario,
)
from rlvr_games.games.chess.state import ChessState
from rlvr_games.games.chess.stockfish_runtime import StockfishEvaluator
from rlvr_games.games.chess.turns import (
    ChessEngineAutoAdvancePolicy,
    ChessPuzzleAutoAdvancePolicy,
    StockfishMoveSelector,
)


class ChessScenarioKind(StrEnum):
    """Supported chess scenario kinds exposed through the CLI."""

    STARTING_POSITION = "starting-position"
    LICHESS_PUZZLES = "lichess-puzzles"


class ChessRewardKind(StrEnum):
    """Supported chess reward policies exposed through the CLI."""

    PUZZLE_DENSE = "puzzle-dense"
    PUZZLE_SPARSE = "puzzle-sparse"
    ENGINE_EVAL_DENSE = "engine-eval-dense"
    ENGINE_EVAL_SPARSE = "engine-eval-sparse"


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
    parser.add_argument(
        "--reward",
        choices=tuple(kind.value for kind in ChessRewardKind),
        required=True,
    )
    parser.add_argument("--stockfish-path", type=Path)
    parser.add_argument("--engine-depth", type=int)
    parser.add_argument("--engine-mate-score", type=int)


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
    scenario = build_chess_scenario(args=args, parser=parser)
    reward_fn = build_chess_reward(args=args, parser=parser)
    return make_chess_env(
        scenario=scenario,
        reward_fn=reward_fn,
        config=build_episode_config(args=args, parser=parser),
        text_renderer_kind=ChessTextRendererKind(args.renderer),
        include_images=args.image_output_dir is not None,
        image_size=args.image_size,
        image_coordinates=args.image_coordinates,
        orientation=ChessBoardOrientation(args.orientation),
        auto_advance_policy=build_chess_auto_advance_policy(
            args=args,
            parser=parser,
            scenario=scenario,
        ),
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


def build_chess_reward(
    *,
    args: Namespace,
    parser: ArgumentParser,
) -> RewardFn[ChessState, ChessAction]:
    """Construct a chess reward function from parsed CLI arguments.

    Parameters
    ----------
    args : Namespace
        Parsed CLI arguments for a chess play session.
    parser : ArgumentParser
        Parser used to report invalid argument combinations.

    Returns
    -------
    RewardFn[ChessState, ChessAction]
        Reward function implied by the parsed CLI arguments.
    """
    scenario_kind = ChessScenarioKind(args.scenario)
    reward_kind = ChessRewardKind(args.reward)
    if reward_kind in (
        ChessRewardKind.PUZZLE_DENSE,
        ChessRewardKind.PUZZLE_SPARSE,
    ):
        if scenario_kind != ChessScenarioKind.LICHESS_PUZZLES:
            parser.error("Puzzle rewards require --scenario lichess-puzzles for chess.")
        if args.stockfish_path is not None:
            parser.error("--stockfish-path requires an engine eval reward.")
        if args.engine_depth is not None:
            parser.error("--engine-depth requires an engine eval reward.")
        if args.engine_mate_score is not None:
            parser.error("--engine-mate-score requires an engine eval reward.")
        if reward_kind == ChessRewardKind.PUZZLE_DENSE:
            return PuzzleOnlyMoveDenseReward(
                correct_move_reward=1.0,
                incorrect_move_reward=-1.0,
            )
        return PuzzleOnlyMoveSparseReward(
            success_reward=1.0,
            incorrect_move_reward=-1.0,
        )

    if scenario_kind != ChessScenarioKind.STARTING_POSITION:
        parser.error(
            "Engine eval rewards require --scenario starting-position for chess."
        )
    if args.engine_depth is None:
        parser.error("--engine-depth is required for engine eval rewards.")
    if args.engine_mate_score is None:
        parser.error("--engine-mate-score is required for engine eval rewards.")

    evaluator = _build_stockfish_evaluator(args=args)
    if reward_kind == ChessRewardKind.ENGINE_EVAL_DENSE:
        return EngineEvalDenseReward(
            evaluator=evaluator,
            perspective="mover",
        )
    return EngineEvalSparseReward(
        evaluator=evaluator,
        perspective="mover",
    )


def _build_stockfish_evaluator(*, args: Namespace) -> StockfishEvaluator:
    """Construct a Stockfish evaluator from parsed CLI arguments.

    Parameters
    ----------
    args : Namespace
        Parsed CLI arguments for a chess play session.

    Returns
    -------
    StockfishEvaluator
        Stockfish-backed evaluator configured from the parsed CLI arguments.
    """
    if args.engine_depth is None or args.engine_mate_score is None:
        raise ValueError(
            "Stockfish evaluator construction requires engine_depth and "
            "engine_mate_score."
        )
    if args.stockfish_path is not None:
        return StockfishEvaluator.from_engine_path(
            engine_path=args.stockfish_path,
            depth=args.engine_depth,
            mate_score=args.engine_mate_score,
        )
    return StockfishEvaluator.from_installed_binary(
        depth=args.engine_depth,
        mate_score=args.engine_mate_score,
    )


def build_chess_auto_advance_policy(
    *,
    args: Namespace,
    parser: ArgumentParser,
    scenario: StartingPositionScenario | ChessPuzzleDatasetScenario,
) -> ChessEngineAutoAdvancePolicy | ChessPuzzleAutoAdvancePolicy | None:
    """Construct the chess auto-advance policy implied by parsed arguments.

    Parameters
    ----------
    args : Namespace
        Parsed CLI arguments for a chess play session.
    parser : ArgumentParser
        Parser used to report invalid argument combinations.
    scenario : StartingPositionScenario | ChessPuzzleDatasetScenario
        Scenario already selected for the session.

    Returns
    -------
    ChessEngineAutoAdvancePolicy | ChessPuzzleAutoAdvancePolicy | None
        Auto-advance policy that should be used for the session.
    """
    if isinstance(scenario, ChessPuzzleDatasetScenario):
        return ChessPuzzleAutoAdvancePolicy()
    if args.engine_depth is None:
        parser.error("--engine-depth is required for engine-controlled chess play.")
    return ChessEngineAutoAdvancePolicy(
        move_selector=_build_stockfish_move_selector(args=args),
    )


def _build_stockfish_move_selector(*, args: Namespace) -> StockfishMoveSelector:
    """Construct a Stockfish move selector from parsed CLI arguments.

    Parameters
    ----------
    args : Namespace
        Parsed CLI arguments for a chess play session.

    Returns
    -------
    StockfishMoveSelector
        Stockfish-backed move selector configured from the parsed CLI
        arguments.
    """
    if args.engine_depth is None:
        raise ValueError("Stockfish move selector construction requires engine_depth.")
    if args.stockfish_path is not None:
        return StockfishMoveSelector.from_engine_path(
            engine_path=args.stockfish_path,
            depth=args.engine_depth,
        )
    return StockfishMoveSelector.from_installed_binary(depth=args.engine_depth)


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
    transitions_payload = step_result.info.get("transitions")
    if not isinstance(transitions_payload, tuple):
        transitions_payload = ()

    summary_lines: list[str] = []
    if transitions_payload:
        for transition_index, transition_payload in enumerate(transitions_payload):
            if not isinstance(transition_payload, dict):
                continue
            source = transition_payload.get("source")
            transition_info = transition_payload.get("info")
            if not isinstance(source, str) or not isinstance(transition_info, dict):
                continue
            move_uci = transition_info.get("move_uci")
            move_san = transition_info.get("move_san")
            if transition_index == 0 and len(transitions_payload) == 1:
                label_prefix = "Move"
            else:
                label_prefix = f"{source.replace('_', ' ').title()} Move"
            if move_uci is not None:
                summary_lines.append(f"{label_prefix} UCI: {move_uci}")
            if move_san is not None:
                summary_lines.append(f"{label_prefix} SAN: {move_san}")
        return tuple(summary_lines)

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
