"""Chess task-spec parsing and environment construction."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import chess

from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.task_spec_base import (
    TaskSpec,
    optional_bool,
    optional_float,
    optional_int,
    optional_mapping,
    optional_path,
    optional_string,
    parse_task_spec_header,
    reject_unknown_keys,
    required_float,
    required_int,
    required_string,
    require_mapping,
)
from rlvr_games.datasets import DatasetSplit
from rlvr_games.games.chess.actions import ChessAction
from rlvr_games.games.chess.factory import (
    ChessBoardOrientation,
    ChessTextRendererKind,
    make_chess_env,
)
from rlvr_games.games.chess.rewards import (
    ChessPerspective,
    ChessRewardPerspective,
    ChessStateEvaluator,
    EngineEvalDenseReward,
    EngineEvalSparseReward,
    PuzzleOnlyMoveDenseReward,
    PuzzleOnlyMoveSparseReward,
    TerminalOutcomeReward,
    UciEngineEvaluator,
)
from rlvr_games.games.chess.scenarios import (
    ChessPuzzleDatasetScenario,
    STANDARD_START_FEN,
    StartingPositionScenario,
)
from rlvr_games.games.chess.state import ChessState
from rlvr_games.games.chess.turns import (
    ChessEngineAutoAdvancePolicy,
    ChessPuzzleAutoAdvancePolicy,
    StockfishMoveSelector,
)


@dataclass(slots=True, frozen=True)
class ChessStartingPositionScenarioTaskSpec:
    """Task-spec variant for a fixed chess starting position."""

    initial_fen: str = STANDARD_START_FEN

    def __post_init__(self) -> None:
        """Validate the configured FEN string."""
        try:
            chess.Board(self.initial_fen)
        except ValueError as exc:
            raise ValueError(
                f"Invalid chess FEN for task spec: {self.initial_fen}"
            ) from exc


@dataclass(slots=True, frozen=True)
class ChessPuzzleDatasetScenarioTaskSpec:
    """Task-spec variant for a sampled chess puzzle dataset position."""

    manifest_path: Path
    split: DatasetSplit = DatasetSplit.TRAIN

    def __post_init__(self) -> None:
        """Validate that the puzzle manifest path exists."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Chess dataset manifest does not exist: {self.manifest_path}"
            )


@dataclass(slots=True, frozen=True)
class ChessTerminalOutcomeRewardTaskSpec:
    """Sparse terminal-outcome reward for chess."""

    perspective: ChessPerspective
    win_reward: float
    draw_reward: float
    loss_reward: float


@dataclass(slots=True, frozen=True)
class ChessPuzzleDenseRewardTaskSpec:
    """Dense per-move puzzle-line reward for chess."""

    correct_move_reward: float
    incorrect_move_reward: float


@dataclass(slots=True, frozen=True)
class ChessPuzzleSparseRewardTaskSpec:
    """Sparse terminal puzzle reward for chess."""

    success_reward: float
    incorrect_move_reward: float


@dataclass(slots=True, frozen=True)
class ChessEngineConfigTaskSpec:
    """Shared stockfish/UCI engine configuration for chess task specs."""

    path: Path
    depth: int
    mate_score: int

    def __post_init__(self) -> None:
        """Validate engine parameters."""
        if self.depth < 1:
            raise ValueError("Chess engine depth must be >= 1.")
        if self.mate_score < 1:
            raise ValueError("Chess engine mate_score must be >= 1.")


@dataclass(slots=True, frozen=True)
class ChessEngineEvalRewardTaskSpec:
    """Evaluator-backed dense or sparse chess reward."""

    kind: str
    perspective: ChessRewardPerspective
    engine: ChessEngineConfigTaskSpec


@dataclass(slots=True, frozen=True)
class ChessObservationTaskSpec:
    """Chess observation/rendering configuration."""

    text_renderer: ChessTextRendererKind = ChessTextRendererKind.ASCII
    include_images: bool = False
    image_size: int = 360
    image_coordinates: bool = False
    orientation: ChessBoardOrientation = ChessBoardOrientation.WHITE

    def __post_init__(self) -> None:
        """Validate observation parameters."""
        if self.image_size < 1:
            raise ValueError("Chess observation image_size must be >= 1.")


@dataclass(slots=True, frozen=True)
class ChessPuzzleAutoAdvanceTaskSpec:
    """Auto-advance configuration that replays canonical puzzle replies."""


@dataclass(slots=True, frozen=True)
class ChessStockfishAutoAdvanceTaskSpec:
    """Auto-advance configuration that uses Stockfish for replies."""

    path: Path
    depth: int

    def __post_init__(self) -> None:
        """Validate Stockfish move-selection parameters."""
        if self.depth < 1:
            raise ValueError("Chess auto-advance depth must be >= 1.")


@dataclass(slots=True, frozen=True)
class ChessControlTaskSpec:
    """Chess control and auto-advance configuration."""

    auto_advance: (
        ChessPuzzleAutoAdvanceTaskSpec | ChessStockfishAutoAdvanceTaskSpec | None
    ) = None


@dataclass(slots=True)
class ChessTaskSpec(TaskSpec):
    """Validated authored chess task specification."""

    scenario: ChessStartingPositionScenarioTaskSpec | ChessPuzzleDatasetScenarioTaskSpec
    reward: (
        ChessTerminalOutcomeRewardTaskSpec
        | ChessPuzzleDenseRewardTaskSpec
        | ChessPuzzleSparseRewardTaskSpec
        | ChessEngineEvalRewardTaskSpec
    )
    observation: ChessObservationTaskSpec = field(
        default_factory=ChessObservationTaskSpec
    )
    control: ChessControlTaskSpec = field(default_factory=ChessControlTaskSpec)

    @property
    def game(self) -> str:
        """Return the game name carried by this task spec."""
        return "chess"

    def __post_init__(self) -> None:
        """Validate cross-field chess task-spec combinations."""
        TaskSpec.__post_init__(self)
        if isinstance(
            self.reward,
            ChessPuzzleDenseRewardTaskSpec | ChessPuzzleSparseRewardTaskSpec,
        ) and not isinstance(self.scenario, ChessPuzzleDatasetScenarioTaskSpec):
            raise ValueError(
                "Chess puzzle rewards require the dataset_puzzle scenario."
            )
        if isinstance(self.control.auto_advance, ChessPuzzleAutoAdvanceTaskSpec) and (
            not isinstance(self.scenario, ChessPuzzleDatasetScenarioTaskSpec)
        ):
            raise ValueError(
                "Chess puzzle auto-advance requires the dataset_puzzle scenario."
            )


def chess_task_spec_from_mapping(
    *,
    payload: dict[str, object],
    base_dir: Path,
) -> ChessTaskSpec:
    """Parse a chess task specification from a raw mapping."""
    header = parse_task_spec_header(
        payload=payload,
        expected_game="chess",
        allowed_top_level_keys=(
            "schema_version",
            "id",
            "game",
            "scenario",
            "reward",
            "episode",
            "observation",
            "control",
            "metadata",
        ),
    )
    scenario = _parse_chess_scenario(
        payload=require_mapping(payload.get("scenario"), context="chess scenario"),
        base_dir=base_dir,
    )
    reward = _parse_chess_reward(
        payload=require_mapping(payload.get("reward"), context="chess reward"),
        base_dir=base_dir,
    )
    observation_payload = optional_mapping(
        payload,
        "observation",
        context="chess task specification",
    )
    control_payload = optional_mapping(
        payload,
        "control",
        context="chess task specification",
    )
    return ChessTaskSpec(
        schema_version=header.schema_version,
        task_id=header.task_id,
        episode_config=header.episode_config,
        metadata=header.metadata,
        scenario=scenario,
        reward=reward,
        observation=_parse_chess_observation(observation_payload),
        control=_parse_chess_control(payload=control_payload, base_dir=base_dir),
    )


def build_chess_environment_from_task_spec(
    *,
    task_spec: TaskSpec,
) -> TurnBasedEnv[ChessState, ChessAction]:
    """Build a chess environment from a validated chess task specification."""
    if not isinstance(task_spec, ChessTaskSpec):
        raise TypeError(
            "build_chess_environment_from_task_spec requires ChessTaskSpec."
        )

    return make_chess_env(
        scenario=_build_chess_scenario(task_spec.scenario),
        reward_fn=_build_chess_reward(task_spec.reward),
        config=task_spec.episode_config,
        text_renderer_kind=task_spec.observation.text_renderer,
        include_images=task_spec.observation.include_images,
        image_size=task_spec.observation.image_size,
        image_coordinates=task_spec.observation.image_coordinates,
        orientation=task_spec.observation.orientation,
        auto_advance_policy=_build_chess_auto_advance(task_spec.control),
    )


def _parse_chess_scenario(
    *,
    payload: dict[str, object],
    base_dir: Path,
) -> ChessStartingPositionScenarioTaskSpec | ChessPuzzleDatasetScenarioTaskSpec:
    kind = required_string(payload, "kind", context="chess scenario")
    if kind == "starting_position":
        reject_unknown_keys(
            payload,
            allowed_keys=("kind", "initial_fen"),
            context="chess scenario",
        )
        initial_fen = optional_string(payload, "initial_fen", context="chess scenario")
        return ChessStartingPositionScenarioTaskSpec(
            initial_fen=STANDARD_START_FEN if initial_fen is None else initial_fen
        )
    if kind == "dataset_puzzle":
        reject_unknown_keys(
            payload,
            allowed_keys=("kind", "manifest_path", "split"),
            context="chess scenario",
        )
        manifest_path = optional_path(
            payload,
            "manifest_path",
            base_dir=base_dir,
            context="chess scenario",
        )
        if manifest_path is None:
            raise ValueError("Chess dataset_puzzle scenario requires manifest_path.")
        split = optional_string(payload, "split", context="chess scenario")
        return ChessPuzzleDatasetScenarioTaskSpec(
            manifest_path=manifest_path,
            split=DatasetSplit.TRAIN if split is None else DatasetSplit(split),
        )
    raise ValueError(f"Unsupported chess scenario kind: {kind!r}.")


def _parse_chess_reward(
    *,
    payload: dict[str, object],
    base_dir: Path,
) -> (
    ChessTerminalOutcomeRewardTaskSpec
    | ChessPuzzleDenseRewardTaskSpec
    | ChessPuzzleSparseRewardTaskSpec
    | ChessEngineEvalRewardTaskSpec
):
    kind = required_string(payload, "kind", context="chess reward")
    if kind == "terminal_outcome":
        reject_unknown_keys(
            payload,
            allowed_keys=(
                "kind",
                "perspective",
                "win_reward",
                "draw_reward",
                "loss_reward",
            ),
            context="chess reward",
        )
        return ChessTerminalOutcomeRewardTaskSpec(
            perspective=_parse_chess_perspective(
                payload,
                key="perspective",
                context="chess reward",
            ),
            win_reward=required_float(payload, "win_reward", context="chess reward"),
            draw_reward=required_float(payload, "draw_reward", context="chess reward"),
            loss_reward=required_float(payload, "loss_reward", context="chess reward"),
        )
    if kind == "puzzle_dense":
        reject_unknown_keys(
            payload,
            allowed_keys=("kind", "correct_move_reward", "incorrect_move_reward"),
            context="chess reward",
        )
        return ChessPuzzleDenseRewardTaskSpec(
            correct_move_reward=(
                1.0
                if optional_float(
                    payload,
                    "correct_move_reward",
                    context="chess reward",
                )
                is None
                else cast(
                    float,
                    optional_float(
                        payload,
                        "correct_move_reward",
                        context="chess reward",
                    ),
                )
            ),
            incorrect_move_reward=(
                -1.0
                if optional_float(
                    payload,
                    "incorrect_move_reward",
                    context="chess reward",
                )
                is None
                else cast(
                    float,
                    optional_float(
                        payload,
                        "incorrect_move_reward",
                        context="chess reward",
                    ),
                )
            ),
        )
    if kind == "puzzle_sparse":
        reject_unknown_keys(
            payload,
            allowed_keys=("kind", "success_reward", "incorrect_move_reward"),
            context="chess reward",
        )
        return ChessPuzzleSparseRewardTaskSpec(
            success_reward=(
                1.0
                if optional_float(payload, "success_reward", context="chess reward")
                is None
                else cast(
                    float,
                    optional_float(
                        payload,
                        "success_reward",
                        context="chess reward",
                    ),
                )
            ),
            incorrect_move_reward=(
                -1.0
                if optional_float(
                    payload,
                    "incorrect_move_reward",
                    context="chess reward",
                )
                is None
                else cast(
                    float,
                    optional_float(
                        payload,
                        "incorrect_move_reward",
                        context="chess reward",
                    ),
                )
            ),
        )
    if kind in {"engine_eval_dense", "engine_eval_sparse"}:
        reject_unknown_keys(
            payload,
            allowed_keys=("kind", "perspective", "engine"),
            context="chess reward",
        )
        engine_payload = require_mapping(payload.get("engine"), context="chess engine")
        return ChessEngineEvalRewardTaskSpec(
            kind=kind,
            perspective=_parse_chess_reward_perspective(
                payload,
                key="perspective",
                context="chess reward",
            ),
            engine=_parse_chess_engine_config(
                payload=engine_payload, base_dir=base_dir
            ),
        )
    raise ValueError(f"Unsupported chess reward kind: {kind!r}.")


def _parse_chess_engine_config(
    *,
    payload: dict[str, object],
    base_dir: Path,
) -> ChessEngineConfigTaskSpec:
    reject_unknown_keys(
        payload,
        allowed_keys=("path", "depth", "mate_score"),
        context="chess engine",
    )
    path = optional_path(payload, "path", base_dir=base_dir, context="chess engine")
    if path is None:
        raise ValueError("Chess engine configuration requires path.")
    return ChessEngineConfigTaskSpec(
        path=path,
        depth=required_int(payload, "depth", context="chess engine"),
        mate_score=required_int(payload, "mate_score", context="chess engine"),
    )


def _parse_chess_observation(
    payload: dict[str, object] | None,
) -> ChessObservationTaskSpec:
    if payload is None:
        return ChessObservationTaskSpec()
    reject_unknown_keys(
        payload,
        allowed_keys=(
            "text_renderer",
            "include_images",
            "image_size",
            "image_coordinates",
            "orientation",
        ),
        context="chess observation",
    )
    text_renderer = optional_string(
        payload, "text_renderer", context="chess observation"
    )
    orientation = optional_string(payload, "orientation", context="chess observation")
    include_images = optional_bool(
        payload, "include_images", context="chess observation"
    )
    image_coordinates = optional_bool(
        payload,
        "image_coordinates",
        context="chess observation",
    )
    image_size = optional_int(payload, "image_size", context="chess observation")
    return ChessObservationTaskSpec(
        text_renderer=(
            ChessTextRendererKind.ASCII
            if text_renderer is None
            else ChessTextRendererKind(text_renderer)
        ),
        include_images=False if include_images is None else include_images,
        image_size=360 if image_size is None else image_size,
        image_coordinates=(False if image_coordinates is None else image_coordinates),
        orientation=(
            ChessBoardOrientation.WHITE
            if orientation is None
            else ChessBoardOrientation(orientation)
        ),
    )


def _parse_chess_control(
    *,
    payload: dict[str, object] | None,
    base_dir: Path,
) -> ChessControlTaskSpec:
    if payload is None:
        return ChessControlTaskSpec()
    reject_unknown_keys(
        payload,
        allowed_keys=("auto_advance",),
        context="chess control",
    )
    auto_advance_payload = optional_mapping(
        payload, "auto_advance", context="chess control"
    )
    if auto_advance_payload is None:
        return ChessControlTaskSpec()
    kind = required_string(auto_advance_payload, "kind", context="chess auto_advance")
    if kind == "puzzle_solution":
        reject_unknown_keys(
            auto_advance_payload,
            allowed_keys=("kind",),
            context="chess auto_advance",
        )
        return ChessControlTaskSpec(auto_advance=ChessPuzzleAutoAdvanceTaskSpec())
    if kind == "stockfish":
        reject_unknown_keys(
            auto_advance_payload,
            allowed_keys=("kind", "engine"),
            context="chess auto_advance",
        )
        engine_payload = require_mapping(
            auto_advance_payload.get("engine"),
            context="chess auto_advance engine",
        )
        reject_unknown_keys(
            engine_payload,
            allowed_keys=("path", "depth"),
            context="chess auto_advance engine",
        )
        path = optional_path(
            engine_payload,
            "path",
            base_dir=base_dir,
            context="chess auto_advance engine",
        )
        if path is None:
            raise ValueError("Chess stockfish auto_advance requires engine.path.")
        return ChessControlTaskSpec(
            auto_advance=ChessStockfishAutoAdvanceTaskSpec(
                path=path,
                depth=required_int(
                    engine_payload,
                    "depth",
                    context="chess auto_advance engine",
                ),
            )
        )
    raise ValueError(f"Unsupported chess auto_advance kind: {kind!r}.")


def _build_chess_scenario(
    scenario_spec: ChessStartingPositionScenarioTaskSpec
    | ChessPuzzleDatasetScenarioTaskSpec,
) -> StartingPositionScenario | ChessPuzzleDatasetScenario:
    if isinstance(scenario_spec, ChessStartingPositionScenarioTaskSpec):
        return StartingPositionScenario(initial_fen=scenario_spec.initial_fen)
    return ChessPuzzleDatasetScenario(
        manifest_path=scenario_spec.manifest_path,
        split=scenario_spec.split,
    )


def _build_chess_reward(
    reward_spec: (
        ChessTerminalOutcomeRewardTaskSpec
        | ChessPuzzleDenseRewardTaskSpec
        | ChessPuzzleSparseRewardTaskSpec
        | ChessEngineEvalRewardTaskSpec
    ),
) -> (
    TerminalOutcomeReward
    | PuzzleOnlyMoveDenseReward
    | PuzzleOnlyMoveSparseReward
    | EngineEvalDenseReward
    | EngineEvalSparseReward
):
    if isinstance(reward_spec, ChessTerminalOutcomeRewardTaskSpec):
        return TerminalOutcomeReward(
            perspective=reward_spec.perspective,
            win_reward=reward_spec.win_reward,
            draw_reward=reward_spec.draw_reward,
            loss_reward=reward_spec.loss_reward,
        )
    if isinstance(reward_spec, ChessPuzzleDenseRewardTaskSpec):
        return PuzzleOnlyMoveDenseReward(
            correct_move_reward=reward_spec.correct_move_reward,
            incorrect_move_reward=reward_spec.incorrect_move_reward,
        )
    if isinstance(reward_spec, ChessPuzzleSparseRewardTaskSpec):
        return PuzzleOnlyMoveSparseReward(
            success_reward=reward_spec.success_reward,
            incorrect_move_reward=reward_spec.incorrect_move_reward,
        )

    evaluator: ChessStateEvaluator = UciEngineEvaluator(
        engine_path=reward_spec.engine.path,
        depth=reward_spec.engine.depth,
        mate_score=reward_spec.engine.mate_score,
    )
    if reward_spec.kind == "engine_eval_dense":
        return EngineEvalDenseReward(
            evaluator=evaluator,
            perspective=reward_spec.perspective,
        )
    return EngineEvalSparseReward(
        evaluator=evaluator,
        perspective=reward_spec.perspective,
    )


def _build_chess_auto_advance(
    control_spec: ChessControlTaskSpec,
) -> ChessEngineAutoAdvancePolicy | ChessPuzzleAutoAdvancePolicy | None:
    auto_advance = control_spec.auto_advance
    if auto_advance is None:
        return None
    if isinstance(auto_advance, ChessPuzzleAutoAdvanceTaskSpec):
        return ChessPuzzleAutoAdvancePolicy()
    return ChessEngineAutoAdvancePolicy(
        move_selector=StockfishMoveSelector.from_engine_path(
            engine_path=auto_advance.path,
            depth=auto_advance.depth,
        )
    )


def _parse_chess_perspective(
    payload: dict[str, object],
    *,
    key: str,
    context: str,
) -> ChessPerspective:
    perspective = required_string(payload, key, context=context)
    if perspective not in ("white", "black"):
        raise ValueError(f"{context} field {key!r} must be 'white' or 'black'.")
    return cast(ChessPerspective, perspective)


def _parse_chess_reward_perspective(
    payload: dict[str, object],
    *,
    key: str,
    context: str,
) -> ChessRewardPerspective:
    perspective = required_string(payload, key, context=context)
    if perspective not in ("white", "black", "mover"):
        raise ValueError(
            f"{context} field {key!r} must be 'white', 'black', or 'mover'."
        )
    return cast(ChessRewardPerspective, perspective)


__all__ = [
    "ChessControlTaskSpec",
    "ChessObservationTaskSpec",
    "ChessTaskSpec",
    "build_chess_environment_from_task_spec",
    "chess_task_spec_from_mapping",
]
