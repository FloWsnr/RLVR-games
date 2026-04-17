"""2048 task-spec parsing and environment construction."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt

from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.task_spec_base import (
    TaskSpec,
    TaskSpecModel,
    validate_task_spec_model,
)
from rlvr_games.games.game2048.actions import Game2048Action
from rlvr_games.games.game2048.engine import is_power_of_two
from rlvr_games.games.game2048.factory import make_game2048_env
from rlvr_games.games.game2048.rewards import ScoreDeltaReward, TargetTileReward
from rlvr_games.games.game2048.scenarios import (
    STANDARD_2048_SIZE,
    STANDARD_2048_TARGET,
    STANDARD_START_TILE_COUNT,
    normalize_initial_board,
)
from rlvr_games.games.game2048.state import Game2048State


@dataclass(slots=True, frozen=True)
class Game2048RandomStartScenarioTaskSpec:
    """Task-spec variant for a random 2048 start board."""

    size: int = STANDARD_2048_SIZE
    target_value: int = STANDARD_2048_TARGET
    start_tile_count: int = STANDARD_START_TILE_COUNT

    def __post_init__(self) -> None:
        """Validate random-start scenario parameters."""
        if self.size < 2:
            raise ValueError("2048 boards must be at least 2x2.")
        if self.target_value < 2 or not is_power_of_two(self.target_value):
            raise ValueError("2048 target_value must be a power of two >= 2.")
        if self.start_tile_count <= 0:
            raise ValueError("2048 start_tile_count must be positive.")
        if self.start_tile_count > self.size * self.size:
            raise ValueError("2048 start_tile_count cannot exceed board capacity.")


@dataclass(slots=True, frozen=True)
class Game2048FixedBoardScenarioTaskSpec:
    """Task-spec variant for a fixed 2048 board position."""

    board: tuple[tuple[int, ...], ...]
    target_value: int = STANDARD_2048_TARGET
    initial_score: int = 0
    initial_move_count: int = 0

    def __post_init__(self) -> None:
        """Validate fixed-board scenario parameters."""
        if self.target_value < 2 or not is_power_of_two(self.target_value):
            raise ValueError("2048 target_value must be a power of two >= 2.")
        if self.initial_score < 0:
            raise ValueError("2048 initial_score must be non-negative.")
        if self.initial_move_count < 0:
            raise ValueError("2048 initial_move_count must be non-negative.")


@dataclass(slots=True, frozen=True)
class Game2048TargetTileRewardTaskSpec:
    """Sparse target-tile reward for 2048."""


@dataclass(slots=True, frozen=True)
class Game2048ScoreDeltaRewardTaskSpec:
    """Dense score-delta reward for 2048."""


@dataclass(slots=True, frozen=True)
class Game2048ObservationTaskSpec:
    """2048 observation configuration."""

    include_images: bool = False
    image_size: int = 360

    def __post_init__(self) -> None:
        """Validate observation parameters."""
        if self.image_size < 1:
            raise ValueError("2048 observation image_size must be >= 1.")


@dataclass(slots=True)
class Game2048TaskSpec(TaskSpec):
    """Validated authored 2048 task specification."""

    scenario: Game2048RandomStartScenarioTaskSpec | Game2048FixedBoardScenarioTaskSpec
    reward: Game2048TargetTileRewardTaskSpec | Game2048ScoreDeltaRewardTaskSpec
    observation: Game2048ObservationTaskSpec = field(
        default_factory=Game2048ObservationTaskSpec
    )

    @property
    def game(self) -> str:
        """Return the game name carried by this task spec."""
        return "game2048"


class _Game2048YamlModel(BaseModel):
    """Base model for authored 2048 YAML fragments."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Game2048RandomStartScenarioModel(_Game2048YamlModel):
    """Authored random-start scenario block."""

    kind: Literal["random_start"] = "random_start"
    size: StrictInt = STANDARD_2048_SIZE
    target_value: StrictInt = STANDARD_2048_TARGET
    start_tile_count: StrictInt = STANDARD_START_TILE_COUNT

    def to_runtime(self) -> Game2048RandomStartScenarioTaskSpec:
        """Convert the authored scenario into the runtime dataclass."""
        return Game2048RandomStartScenarioTaskSpec(
            size=self.size,
            target_value=self.target_value,
            start_tile_count=self.start_tile_count,
        )


class Game2048FixedBoardScenarioModel(_Game2048YamlModel):
    """Authored fixed-board scenario block."""

    kind: Literal["fixed_board"] = "fixed_board"
    board: tuple[tuple[StrictInt, ...], ...]
    target_value: StrictInt = STANDARD_2048_TARGET
    initial_score: StrictInt = 0
    initial_move_count: StrictInt = 0

    def to_runtime(self) -> Game2048FixedBoardScenarioTaskSpec:
        """Convert the authored scenario into the runtime dataclass."""
        return Game2048FixedBoardScenarioTaskSpec(
            board=normalize_initial_board(board=self.board),
            target_value=self.target_value,
            initial_score=self.initial_score,
            initial_move_count=self.initial_move_count,
        )


Game2048ScenarioModel = Annotated[
    Game2048RandomStartScenarioModel | Game2048FixedBoardScenarioModel,
    Field(discriminator="kind"),
]


class Game2048TargetTileRewardModel(_Game2048YamlModel):
    """Authored target-tile reward block."""

    kind: Literal["target_tile"] = "target_tile"

    def to_runtime(self) -> Game2048TargetTileRewardTaskSpec:
        """Convert the authored reward into the runtime dataclass."""
        return Game2048TargetTileRewardTaskSpec()


class Game2048ScoreDeltaRewardModel(_Game2048YamlModel):
    """Authored score-delta reward block."""

    kind: Literal["score_delta"] = "score_delta"

    def to_runtime(self) -> Game2048ScoreDeltaRewardTaskSpec:
        """Convert the authored reward into the runtime dataclass."""
        return Game2048ScoreDeltaRewardTaskSpec()


Game2048RewardModel = Annotated[
    Game2048TargetTileRewardModel | Game2048ScoreDeltaRewardModel,
    Field(discriminator="kind"),
]


class Game2048ObservationModel(_Game2048YamlModel):
    """Authored observation block."""

    include_images: StrictBool = False
    image_size: StrictInt = 360

    def to_runtime(self) -> Game2048ObservationTaskSpec:
        """Convert the authored observation block into the runtime dataclass."""
        return Game2048ObservationTaskSpec(
            include_images=self.include_images,
            image_size=self.image_size,
        )


class Game2048TaskSpecModel(TaskSpecModel):
    """Authored top-level 2048 task specification."""

    game: Literal["game2048"] = "game2048"
    scenario: Game2048ScenarioModel
    reward: Game2048RewardModel
    observation: Game2048ObservationModel | None = None

    def to_runtime(self) -> Game2048TaskSpec:
        """Convert the authored model into the runtime task spec."""
        observation = self.observation
        if observation is None:
            observation = Game2048ObservationModel()
        return Game2048TaskSpec(
            schema_version=self.schema_version,
            task_id=self.task_id,
            episode_config=self.episode_config(),
            metadata=self.metadata,
            scenario=self.scenario.to_runtime(),
            reward=self.reward.to_runtime(),
            observation=observation.to_runtime(),
        )


def game2048_task_spec_from_mapping(
    *,
    payload: dict[str, object],
    base_dir: Path,
) -> Game2048TaskSpec:
    """Parse a 2048 task specification from a raw mapping."""
    task_spec = validate_task_spec_model(
        model_type=Game2048TaskSpecModel,
        payload=payload,
        base_dir=base_dir,
    )
    return task_spec.to_runtime()


def build_game2048_environment_from_task_spec(
    *,
    task_spec: TaskSpec,
) -> TurnBasedEnv[Game2048State, Game2048Action]:
    """Build a 2048 environment from a validated task specification."""
    if not isinstance(task_spec, Game2048TaskSpec):
        raise TypeError(
            "build_game2048_environment_from_task_spec requires Game2048TaskSpec."
        )

    size = STANDARD_2048_SIZE
    target_value = STANDARD_2048_TARGET
    start_tile_count = STANDARD_START_TILE_COUNT
    initial_board: tuple[tuple[int, ...], ...] | None = None
    initial_score = 0
    initial_move_count = 0
    if isinstance(task_spec.scenario, Game2048RandomStartScenarioTaskSpec):
        size = task_spec.scenario.size
        target_value = task_spec.scenario.target_value
        start_tile_count = task_spec.scenario.start_tile_count
    else:
        initial_board = task_spec.scenario.board
        size = len(initial_board)
        target_value = task_spec.scenario.target_value
        initial_score = task_spec.scenario.initial_score
        initial_move_count = task_spec.scenario.initial_move_count

    return make_game2048_env(
        size=size,
        target_value=target_value,
        initial_board=initial_board,
        initial_score=initial_score,
        initial_move_count=initial_move_count,
        start_tile_count=start_tile_count,
        reward_fn=_build_game2048_reward(task_spec.reward),
        config=task_spec.episode_config,
        include_images=task_spec.observation.include_images,
        image_size=task_spec.observation.image_size,
    )


def _build_game2048_reward(
    reward_spec: Game2048TargetTileRewardTaskSpec | Game2048ScoreDeltaRewardTaskSpec,
) -> TargetTileReward | ScoreDeltaReward:
    if isinstance(reward_spec, Game2048TargetTileRewardTaskSpec):
        return TargetTileReward()
    return ScoreDeltaReward()


__all__ = [
    "Game2048TaskSpec",
    "build_game2048_environment_from_task_spec",
    "game2048_task_spec_from_mapping",
]
