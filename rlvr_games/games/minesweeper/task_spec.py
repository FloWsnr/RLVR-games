"""Minesweeper task-spec parsing and environment construction."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.task_spec_base import (
    NumericScalar,
    TaskSpec,
    TaskSpecModel,
    validate_task_spec_model,
)
from rlvr_games.games.minesweeper.actions import MinesweeperAction
from rlvr_games.games.minesweeper.factory import make_minesweeper_env
from rlvr_games.games.minesweeper.rewards import OutcomeReward, SafeRevealCountReward
from rlvr_games.games.minesweeper.scenarios import (
    STANDARD_MINESWEEPER_COLUMNS,
    STANDARD_MINESWEEPER_MINE_COUNT,
    STANDARD_MINESWEEPER_ROWS,
    normalize_initial_board,
)
from rlvr_games.games.minesweeper.state import MinesweeperState


@dataclass(slots=True, frozen=True)
class MinesweeperRandomBoardScenarioTaskSpec:
    """Task-spec variant for deferred random Minesweeper boards."""

    rows: int = STANDARD_MINESWEEPER_ROWS
    columns: int = STANDARD_MINESWEEPER_COLUMNS
    mine_count: int = STANDARD_MINESWEEPER_MINE_COUNT

    def __post_init__(self) -> None:
        """Validate random-board scenario parameters."""
        if self.rows <= 0:
            raise ValueError("Minesweeper rows must be positive.")
        if self.columns <= 0:
            raise ValueError("Minesweeper columns must be positive.")
        if self.mine_count < 0:
            raise ValueError("Minesweeper mine_count must be non-negative.")
        if self.mine_count >= self.rows * self.columns:
            raise ValueError("mine_count must leave at least one safe cell.")


@dataclass(slots=True, frozen=True)
class MinesweeperFixedBoardScenarioTaskSpec:
    """Task-spec variant for a fixed Minesweeper mine layout."""

    board: tuple[tuple[bool, ...], ...]


@dataclass(slots=True, frozen=True)
class MinesweeperOutcomeRewardTaskSpec:
    """Sparse outcome reward for Minesweeper."""

    win_reward: float
    loss_reward: float


@dataclass(slots=True, frozen=True)
class MinesweeperSafeRevealRewardTaskSpec:
    """Dense safe-reveal-count reward for Minesweeper."""

    mine_penalty: float


@dataclass(slots=True, frozen=True)
class MinesweeperObservationTaskSpec:
    """Minesweeper observation configuration."""

    include_images: bool = False
    image_size: int = 240

    def __post_init__(self) -> None:
        """Validate observation parameters."""
        if self.image_size < 1:
            raise ValueError("Minesweeper observation image_size must be >= 1.")


@dataclass(slots=True)
class MinesweeperTaskSpec(TaskSpec):
    """Validated authored Minesweeper task specification."""

    scenario: (
        MinesweeperRandomBoardScenarioTaskSpec | MinesweeperFixedBoardScenarioTaskSpec
    )
    reward: MinesweeperOutcomeRewardTaskSpec | MinesweeperSafeRevealRewardTaskSpec
    observation: MinesweeperObservationTaskSpec = field(
        default_factory=MinesweeperObservationTaskSpec
    )

    @property
    def game(self) -> str:
        """Return the game name carried by this task spec."""
        return "minesweeper"


class _MinesweeperYamlModel(BaseModel):
    """Base model for authored Minesweeper YAML fragments."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class MinesweeperRandomBoardScenarioModel(_MinesweeperYamlModel):
    """Authored random-board scenario block."""

    kind: Literal["random_board"] = "random_board"
    rows: StrictInt = STANDARD_MINESWEEPER_ROWS
    columns: StrictInt = STANDARD_MINESWEEPER_COLUMNS
    mine_count: StrictInt = STANDARD_MINESWEEPER_MINE_COUNT

    def to_runtime(self) -> MinesweeperRandomBoardScenarioTaskSpec:
        """Convert the authored scenario into the runtime dataclass."""
        return MinesweeperRandomBoardScenarioTaskSpec(
            rows=self.rows,
            columns=self.columns,
            mine_count=self.mine_count,
        )


class MinesweeperFixedBoardScenarioModel(_MinesweeperYamlModel):
    """Authored fixed-board scenario block."""

    kind: Literal["fixed_board"] = "fixed_board"
    board: tuple[StrictStr | tuple[StrictBool | StrictInt | StrictStr, ...], ...]

    def to_runtime(self) -> MinesweeperFixedBoardScenarioTaskSpec:
        """Convert the authored scenario into the runtime dataclass."""
        return MinesweeperFixedBoardScenarioTaskSpec(
            board=normalize_initial_board(board=self.board)
        )


MinesweeperScenarioModel = Annotated[
    MinesweeperRandomBoardScenarioModel | MinesweeperFixedBoardScenarioModel,
    Field(discriminator="kind"),
]


class MinesweeperOutcomeRewardModel(_MinesweeperYamlModel):
    """Authored outcome reward block."""

    kind: Literal["outcome"] = "outcome"
    win_reward: NumericScalar
    loss_reward: NumericScalar

    def to_runtime(self) -> MinesweeperOutcomeRewardTaskSpec:
        """Convert the authored reward into the runtime dataclass."""
        return MinesweeperOutcomeRewardTaskSpec(
            win_reward=float(self.win_reward),
            loss_reward=float(self.loss_reward),
        )


class MinesweeperSafeRevealRewardModel(_MinesweeperYamlModel):
    """Authored dense safe-reveal reward block."""

    kind: Literal["safe_reveal_count_dense"] = "safe_reveal_count_dense"
    mine_penalty: NumericScalar

    def to_runtime(self) -> MinesweeperSafeRevealRewardTaskSpec:
        """Convert the authored reward into the runtime dataclass."""
        return MinesweeperSafeRevealRewardTaskSpec(
            mine_penalty=float(self.mine_penalty)
        )


MinesweeperRewardModel = Annotated[
    MinesweeperOutcomeRewardModel | MinesweeperSafeRevealRewardModel,
    Field(discriminator="kind"),
]


class MinesweeperObservationModel(_MinesweeperYamlModel):
    """Authored observation block."""

    include_images: StrictBool = False
    image_size: StrictInt = 240

    def to_runtime(self) -> MinesweeperObservationTaskSpec:
        """Convert the authored observation block into the runtime dataclass."""
        return MinesweeperObservationTaskSpec(
            include_images=self.include_images,
            image_size=self.image_size,
        )


class MinesweeperTaskSpecModel(TaskSpecModel):
    """Authored top-level Minesweeper task specification."""

    game: Literal["minesweeper"] = "minesweeper"
    scenario: MinesweeperScenarioModel
    reward: MinesweeperRewardModel
    observation: MinesweeperObservationModel | None = None

    def to_runtime(self) -> MinesweeperTaskSpec:
        """Convert the authored model into the runtime task spec."""
        observation = self.observation
        if observation is None:
            observation = MinesweeperObservationModel()
        return MinesweeperTaskSpec(
            schema_version=self.schema_version,
            task_id=self.task_id,
            episode_config=self.episode_config(),
            metadata=self.metadata,
            scenario=self.scenario.to_runtime(),
            reward=self.reward.to_runtime(),
            observation=observation.to_runtime(),
        )


def minesweeper_task_spec_from_mapping(
    *,
    payload: dict[str, object],
    base_dir: Path,
) -> MinesweeperTaskSpec:
    """Parse a Minesweeper task specification from a raw mapping."""
    task_spec = validate_task_spec_model(
        model_type=MinesweeperTaskSpecModel,
        payload=payload,
        base_dir=base_dir,
    )
    return task_spec.to_runtime()


def build_minesweeper_environment_from_task_spec(
    *,
    task_spec: TaskSpec,
) -> TurnBasedEnv[MinesweeperState, MinesweeperAction]:
    """Build a Minesweeper environment from a validated task specification."""
    if not isinstance(task_spec, MinesweeperTaskSpec):
        raise TypeError(
            "build_minesweeper_environment_from_task_spec requires MinesweeperTaskSpec."
        )

    rows = STANDARD_MINESWEEPER_ROWS
    columns = STANDARD_MINESWEEPER_COLUMNS
    mine_count = STANDARD_MINESWEEPER_MINE_COUNT
    initial_board: tuple[tuple[bool, ...], ...] | None = None
    if isinstance(task_spec.scenario, MinesweeperRandomBoardScenarioTaskSpec):
        rows = task_spec.scenario.rows
        columns = task_spec.scenario.columns
        mine_count = task_spec.scenario.mine_count
    else:
        initial_board = task_spec.scenario.board
        rows = len(initial_board)
        columns = len(initial_board[0])
        mine_count = sum(1 for row in initial_board for cell in row if cell)

    return make_minesweeper_env(
        rows=rows,
        columns=columns,
        mine_count=mine_count,
        initial_board=initial_board,
        reward_fn=_build_minesweeper_reward(task_spec.reward),
        config=task_spec.episode_config,
        include_images=task_spec.observation.include_images,
        image_size=task_spec.observation.image_size,
    )


def _build_minesweeper_reward(
    reward_spec: MinesweeperOutcomeRewardTaskSpec | MinesweeperSafeRevealRewardTaskSpec,
) -> OutcomeReward | SafeRevealCountReward:
    if isinstance(reward_spec, MinesweeperOutcomeRewardTaskSpec):
        return OutcomeReward(
            win_reward=reward_spec.win_reward,
            loss_reward=reward_spec.loss_reward,
        )
    return SafeRevealCountReward(mine_penalty=reward_spec.mine_penalty)


__all__ = [
    "MinesweeperTaskSpec",
    "build_minesweeper_environment_from_task_spec",
    "minesweeper_task_spec_from_mapping",
]
