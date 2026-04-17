"""Chess task-spec parsing and environment construction."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import chess
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    ValidationInfo,
    field_validator,
)

from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.task_spec_base import (
    NumericScalar,
    TaskSpec,
    TaskSpecModel,
    resolve_path_from_context,
    validate_task_spec_model,
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


class _ChessYamlModel(BaseModel):
    """Base model for authored chess YAML fragments."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ChessStartingPositionScenarioModel(_ChessYamlModel):
    """Authored starting-position scenario block."""

    kind: Literal["starting_position"] = "starting_position"
    initial_fen: str = STANDARD_START_FEN

    def to_runtime(self) -> ChessStartingPositionScenarioTaskSpec:
        """Convert the authored scenario into the runtime dataclass."""
        return ChessStartingPositionScenarioTaskSpec(initial_fen=self.initial_fen)


class ChessPuzzleDatasetScenarioModel(_ChessYamlModel):
    """Authored dataset-puzzle scenario block."""

    kind: Literal["dataset_puzzle"] = "dataset_puzzle"
    manifest_path: Path
    split: DatasetSplit = DatasetSplit.TRAIN

    @field_validator("manifest_path", mode="before")
    @classmethod
    def resolve_manifest_path(cls, value: object, info: ValidationInfo) -> object:
        """Resolve dataset manifest paths relative to the authored YAML file."""
        if not isinstance(value, str | Path):
            return value
        return resolve_path_from_context(
            raw_path=value,
            info=info,
            context="chess scenario field 'manifest_path'",
        )

    def to_runtime(self) -> ChessPuzzleDatasetScenarioTaskSpec:
        """Convert the authored scenario into the runtime dataclass."""
        return ChessPuzzleDatasetScenarioTaskSpec(
            manifest_path=self.manifest_path,
            split=self.split,
        )


ChessScenarioModel = Annotated[
    ChessStartingPositionScenarioModel | ChessPuzzleDatasetScenarioModel,
    Field(discriminator="kind"),
]


class ChessTerminalOutcomeRewardModel(_ChessYamlModel):
    """Authored terminal-outcome reward block."""

    kind: Literal["terminal_outcome"] = "terminal_outcome"
    perspective: ChessPerspective
    win_reward: NumericScalar
    draw_reward: NumericScalar
    loss_reward: NumericScalar

    def to_runtime(self) -> ChessTerminalOutcomeRewardTaskSpec:
        """Convert the authored reward into the runtime dataclass."""
        return ChessTerminalOutcomeRewardTaskSpec(
            perspective=self.perspective,
            win_reward=float(self.win_reward),
            draw_reward=float(self.draw_reward),
            loss_reward=float(self.loss_reward),
        )


class ChessPuzzleDenseRewardModel(_ChessYamlModel):
    """Authored dense puzzle reward block."""

    kind: Literal["puzzle_dense"] = "puzzle_dense"
    correct_move_reward: NumericScalar = 1.0
    incorrect_move_reward: NumericScalar = -1.0

    def to_runtime(self) -> ChessPuzzleDenseRewardTaskSpec:
        """Convert the authored reward into the runtime dataclass."""
        return ChessPuzzleDenseRewardTaskSpec(
            correct_move_reward=float(self.correct_move_reward),
            incorrect_move_reward=float(self.incorrect_move_reward),
        )


class ChessPuzzleSparseRewardModel(_ChessYamlModel):
    """Authored sparse puzzle reward block."""

    kind: Literal["puzzle_sparse"] = "puzzle_sparse"
    success_reward: NumericScalar = 1.0
    incorrect_move_reward: NumericScalar = -1.0

    def to_runtime(self) -> ChessPuzzleSparseRewardTaskSpec:
        """Convert the authored reward into the runtime dataclass."""
        return ChessPuzzleSparseRewardTaskSpec(
            success_reward=float(self.success_reward),
            incorrect_move_reward=float(self.incorrect_move_reward),
        )


class ChessEngineConfigModel(_ChessYamlModel):
    """Authored engine configuration block."""

    path: Path
    depth: StrictInt
    mate_score: StrictInt

    @field_validator("path", mode="before")
    @classmethod
    def resolve_engine_path(cls, value: object, info: ValidationInfo) -> object:
        """Resolve engine paths relative to the authored YAML file."""
        if not isinstance(value, str | Path):
            return value
        return resolve_path_from_context(
            raw_path=value,
            info=info,
            context="chess reward engine field 'path'",
        )

    def to_runtime(self) -> ChessEngineConfigTaskSpec:
        """Convert the authored engine block into the runtime dataclass."""
        return ChessEngineConfigTaskSpec(
            path=self.path,
            depth=self.depth,
            mate_score=self.mate_score,
        )


class ChessEngineEvalRewardModel(_ChessYamlModel):
    """Authored engine-eval reward block."""

    kind: Literal["engine_eval_dense", "engine_eval_sparse"]
    perspective: ChessRewardPerspective
    engine: ChessEngineConfigModel

    def to_runtime(self) -> ChessEngineEvalRewardTaskSpec:
        """Convert the authored reward into the runtime dataclass."""
        return ChessEngineEvalRewardTaskSpec(
            kind=self.kind,
            perspective=self.perspective,
            engine=self.engine.to_runtime(),
        )


ChessRewardModel = Annotated[
    ChessTerminalOutcomeRewardModel
    | ChessPuzzleDenseRewardModel
    | ChessPuzzleSparseRewardModel
    | ChessEngineEvalRewardModel,
    Field(discriminator="kind"),
]


class ChessObservationModel(_ChessYamlModel):
    """Authored observation block."""

    text_renderer: ChessTextRendererKind = ChessTextRendererKind.ASCII
    include_images: StrictBool = False
    image_size: StrictInt = 360
    image_coordinates: StrictBool = False
    orientation: ChessBoardOrientation = ChessBoardOrientation.WHITE

    def to_runtime(self) -> ChessObservationTaskSpec:
        """Convert the authored observation block into the runtime dataclass."""
        return ChessObservationTaskSpec(
            text_renderer=self.text_renderer,
            include_images=self.include_images,
            image_size=self.image_size,
            image_coordinates=self.image_coordinates,
            orientation=self.orientation,
        )


class ChessPuzzleAutoAdvanceModel(_ChessYamlModel):
    """Authored puzzle-solution auto-advance block."""

    kind: Literal["puzzle_solution"] = "puzzle_solution"

    def to_runtime(self) -> ChessPuzzleAutoAdvanceTaskSpec:
        """Convert the authored auto-advance block into the runtime dataclass."""
        return ChessPuzzleAutoAdvanceTaskSpec()


class ChessStockfishAutoAdvanceEngineModel(_ChessYamlModel):
    """Authored Stockfish move-selection configuration block."""

    path: Path
    depth: StrictInt

    @field_validator("path", mode="before")
    @classmethod
    def resolve_engine_path(cls, value: object, info: ValidationInfo) -> object:
        """Resolve Stockfish paths relative to the authored YAML file."""
        if not isinstance(value, str | Path):
            return value
        return resolve_path_from_context(
            raw_path=value,
            info=info,
            context="chess auto_advance engine field 'path'",
        )


class ChessStockfishAutoAdvanceModel(_ChessYamlModel):
    """Authored Stockfish auto-advance block."""

    kind: Literal["stockfish"] = "stockfish"
    engine: ChessStockfishAutoAdvanceEngineModel

    def to_runtime(self) -> ChessStockfishAutoAdvanceTaskSpec:
        """Convert the authored auto-advance block into the runtime dataclass."""
        return ChessStockfishAutoAdvanceTaskSpec(
            path=self.engine.path,
            depth=self.engine.depth,
        )


ChessAutoAdvanceModel = Annotated[
    ChessPuzzleAutoAdvanceModel | ChessStockfishAutoAdvanceModel,
    Field(discriminator="kind"),
]


class ChessControlModel(_ChessYamlModel):
    """Authored control block."""

    auto_advance: ChessAutoAdvanceModel | None = None

    def to_runtime(self) -> ChessControlTaskSpec:
        """Convert the authored control block into the runtime dataclass."""
        auto_advance = self.auto_advance
        return ChessControlTaskSpec(
            auto_advance=None if auto_advance is None else auto_advance.to_runtime()
        )


class ChessTaskSpecModel(TaskSpecModel):
    """Authored top-level chess task specification."""

    game: Literal["chess"] = "chess"
    scenario: ChessScenarioModel
    reward: ChessRewardModel
    observation: ChessObservationModel | None = None
    control: ChessControlModel | None = None

    def to_runtime(self) -> ChessTaskSpec:
        """Convert the authored model into the runtime task spec."""
        observation = self.observation
        if observation is None:
            observation = ChessObservationModel()
        control = self.control
        if control is None:
            control = ChessControlModel()
        return ChessTaskSpec(
            schema_version=self.schema_version,
            task_id=self.task_id,
            episode_config=self.episode_config(),
            metadata=self.metadata,
            scenario=self.scenario.to_runtime(),
            reward=self.reward.to_runtime(),
            observation=observation.to_runtime(),
            control=control.to_runtime(),
        )


def chess_task_spec_from_mapping(
    *,
    payload: dict[str, object],
    base_dir: Path,
) -> ChessTaskSpec:
    """Parse a chess task specification from a raw mapping."""
    task_spec = validate_task_spec_model(
        model_type=ChessTaskSpecModel,
        payload=payload,
        base_dir=base_dir,
    )
    return task_spec.to_runtime()


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


__all__ = [
    "ChessControlTaskSpec",
    "ChessObservationTaskSpec",
    "ChessTaskSpec",
    "build_chess_environment_from_task_spec",
    "chess_task_spec_from_mapping",
]
