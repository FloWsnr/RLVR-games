"""Connect 4 task-spec parsing and environment construction."""

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
from rlvr_games.games.connect4.actions import Connect4Action
from rlvr_games.games.connect4.factory import make_connect4_env
from rlvr_games.games.connect4.rewards import (
    Connect4RewardPerspective,
    SolverMoveScoreReward,
    TerminalOutcomeReward,
)
from rlvr_games.games.connect4.scenarios import (
    DEFAULT_RANDOM_START_MAX_MOVES,
    FixedBoardScenario,
    RandomPositionScenario,
    STANDARD_CONNECT4_COLUMNS,
    STANDARD_CONNECT4_CONNECT_LENGTH,
    STANDARD_CONNECT4_ROWS,
    normalize_initial_board,
)
from rlvr_games.games.connect4.solver import BitBullySolver
from rlvr_games.games.connect4.state import Connect4State
from rlvr_games.games.connect4.turns import Connect4SolverAutoAdvancePolicy


@dataclass(slots=True, frozen=True)
class Connect4RandomPositionScenarioTaskSpec:
    """Task-spec variant for a random non-terminal Connect 4 opening."""

    rows: int = STANDARD_CONNECT4_ROWS
    columns: int = STANDARD_CONNECT4_COLUMNS
    connect_length: int = STANDARD_CONNECT4_CONNECT_LENGTH
    min_start_moves: int = 0
    max_start_moves: int = DEFAULT_RANDOM_START_MAX_MOVES

    def __post_init__(self) -> None:
        """Validate random-position scenario parameters."""
        if self.rows < 1:
            raise ValueError("Connect 4 scenarios require at least one row.")
        if self.columns < 1:
            raise ValueError("Connect 4 scenarios require at least one column.")
        if self.min_start_moves < 0:
            raise ValueError("Connect 4 min_start_moves must be non-negative.")
        if self.max_start_moves < 0:
            raise ValueError("Connect 4 max_start_moves must be non-negative.")
        if self.min_start_moves > self.max_start_moves:
            raise ValueError("Connect 4 min_start_moves cannot exceed max_start_moves.")
        if self.max_start_moves > self.rows * self.columns:
            raise ValueError("Connect 4 max_start_moves cannot exceed board capacity.")


@dataclass(slots=True, frozen=True)
class Connect4FixedBoardScenarioTaskSpec:
    """Task-spec variant for a fixed Connect 4 board position."""

    board: tuple[tuple[str, ...], ...]
    connect_length: int = STANDARD_CONNECT4_CONNECT_LENGTH


@dataclass(slots=True, frozen=True)
class Connect4TerminalOutcomeRewardTaskSpec:
    """Sparse terminal-outcome reward for Connect 4."""

    perspective: Connect4RewardPerspective
    win_reward: float
    draw_reward: float
    loss_reward: float


@dataclass(slots=True, frozen=True)
class Connect4SolverMoveRewardTaskSpec:
    """Dense BitBully move-score reward for Connect 4."""

    perspective: Connect4RewardPerspective


@dataclass(slots=True, frozen=True)
class Connect4ObservationTaskSpec:
    """Connect 4 observation configuration."""

    include_images: bool = False
    image_size: int = 360

    def __post_init__(self) -> None:
        """Validate observation parameters."""
        if self.image_size < 1:
            raise ValueError("Connect 4 observation image_size must be >= 1.")


@dataclass(slots=True, frozen=True)
class Connect4SolverAutoAdvanceTaskSpec:
    """BitBully auto-advance opponent configuration for Connect 4."""


@dataclass(slots=True, frozen=True)
class Connect4ControlTaskSpec:
    """Connect 4 control and auto-advance configuration."""

    auto_advance: Connect4SolverAutoAdvanceTaskSpec | None = None


@dataclass(slots=True)
class Connect4TaskSpec(TaskSpec):
    """Validated authored Connect 4 task specification."""

    scenario: (
        Connect4RandomPositionScenarioTaskSpec | Connect4FixedBoardScenarioTaskSpec
    )
    reward: Connect4TerminalOutcomeRewardTaskSpec | Connect4SolverMoveRewardTaskSpec
    observation: Connect4ObservationTaskSpec = field(
        default_factory=Connect4ObservationTaskSpec
    )
    control: Connect4ControlTaskSpec = field(default_factory=Connect4ControlTaskSpec)

    @property
    def game(self) -> str:
        """Return the game name carried by this task spec."""
        return "connect4"

    def __post_init__(self) -> None:
        """Validate cross-field Connect 4 task-spec combinations."""
        TaskSpec.__post_init__(self)
        solver_requested = (
            isinstance(
                self.reward,
                Connect4SolverMoveRewardTaskSpec,
            )
            or self.control.auto_advance is not None
        )
        if solver_requested and not _is_standard_connect4_scenario(self.scenario):
            raise ValueError(
                "BitBully-backed Connect 4 reward/auto-advance requires the "
                "standard 6x7 connect-4 game."
            )


class _Connect4YamlModel(BaseModel):
    """Base model for authored Connect 4 YAML fragments."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Connect4RandomPositionScenarioModel(_Connect4YamlModel):
    """Authored random-position scenario block."""

    kind: Literal["random_position"] = "random_position"
    rows: StrictInt = STANDARD_CONNECT4_ROWS
    columns: StrictInt = STANDARD_CONNECT4_COLUMNS
    connect_length: StrictInt = STANDARD_CONNECT4_CONNECT_LENGTH
    min_start_moves: StrictInt = 0
    max_start_moves: StrictInt = DEFAULT_RANDOM_START_MAX_MOVES

    def to_runtime(self) -> Connect4RandomPositionScenarioTaskSpec:
        """Convert the authored scenario into the runtime dataclass."""
        return Connect4RandomPositionScenarioTaskSpec(
            rows=self.rows,
            columns=self.columns,
            connect_length=self.connect_length,
            min_start_moves=self.min_start_moves,
            max_start_moves=self.max_start_moves,
        )


class Connect4FixedBoardScenarioModel(_Connect4YamlModel):
    """Authored fixed-board scenario block."""

    kind: Literal["fixed_board"] = "fixed_board"
    board: tuple[StrictStr | tuple[StrictStr, ...], ...]
    connect_length: StrictInt = STANDARD_CONNECT4_CONNECT_LENGTH

    def to_runtime(self) -> Connect4FixedBoardScenarioTaskSpec:
        """Convert the authored scenario into the runtime dataclass."""
        normalized_board = tuple(
            tuple(row) if isinstance(row, str) else row for row in self.board
        )
        return Connect4FixedBoardScenarioTaskSpec(
            board=normalize_initial_board(board=normalized_board),
            connect_length=self.connect_length,
        )


Connect4ScenarioModel = Annotated[
    Connect4RandomPositionScenarioModel | Connect4FixedBoardScenarioModel,
    Field(discriminator="kind"),
]


class Connect4TerminalOutcomeRewardModel(_Connect4YamlModel):
    """Authored terminal-outcome reward block."""

    kind: Literal["terminal_outcome"] = "terminal_outcome"
    perspective: Connect4RewardPerspective
    win_reward: NumericScalar
    draw_reward: NumericScalar
    loss_reward: NumericScalar

    def to_runtime(self) -> Connect4TerminalOutcomeRewardTaskSpec:
        """Convert the authored reward into the runtime dataclass."""
        return Connect4TerminalOutcomeRewardTaskSpec(
            perspective=self.perspective,
            win_reward=float(self.win_reward),
            draw_reward=float(self.draw_reward),
            loss_reward=float(self.loss_reward),
        )


class Connect4SolverMoveRewardModel(_Connect4YamlModel):
    """Authored BitBully move-score reward block."""

    kind: Literal["solver_move_dense"] = "solver_move_dense"
    perspective: Connect4RewardPerspective

    def to_runtime(self) -> Connect4SolverMoveRewardTaskSpec:
        """Convert the authored reward into the runtime dataclass."""
        return Connect4SolverMoveRewardTaskSpec(perspective=self.perspective)


Connect4RewardModel = Annotated[
    Connect4TerminalOutcomeRewardModel | Connect4SolverMoveRewardModel,
    Field(discriminator="kind"),
]


class Connect4ObservationModel(_Connect4YamlModel):
    """Authored observation block."""

    include_images: StrictBool = False
    image_size: StrictInt = 360

    def to_runtime(self) -> Connect4ObservationTaskSpec:
        """Convert the authored observation block into the runtime dataclass."""
        return Connect4ObservationTaskSpec(
            include_images=self.include_images,
            image_size=self.image_size,
        )


class Connect4SolverAutoAdvanceModel(_Connect4YamlModel):
    """Authored solver auto-advance block."""

    kind: Literal["solver"] = "solver"

    def to_runtime(self) -> Connect4SolverAutoAdvanceTaskSpec:
        """Convert the authored auto-advance block into the runtime dataclass."""
        return Connect4SolverAutoAdvanceTaskSpec()


class Connect4ControlModel(_Connect4YamlModel):
    """Authored control block."""

    auto_advance: Connect4SolverAutoAdvanceModel | None = None

    def to_runtime(self) -> Connect4ControlTaskSpec:
        """Convert the authored control block into the runtime dataclass."""
        auto_advance = self.auto_advance
        return Connect4ControlTaskSpec(
            auto_advance=None if auto_advance is None else auto_advance.to_runtime()
        )


class Connect4TaskSpecModel(TaskSpecModel):
    """Authored top-level Connect 4 task specification."""

    game: Literal["connect4"] = "connect4"
    scenario: Connect4ScenarioModel
    reward: Connect4RewardModel
    observation: Connect4ObservationModel | None = None
    control: Connect4ControlModel | None = None

    def to_runtime(self) -> Connect4TaskSpec:
        """Convert the authored model into the runtime task spec."""
        observation = self.observation
        if observation is None:
            observation = Connect4ObservationModel()
        control = self.control
        if control is None:
            control = Connect4ControlModel()
        return Connect4TaskSpec(
            schema_version=self.schema_version,
            task_id=self.task_id,
            episode_config=self.episode_config(),
            metadata=self.metadata,
            scenario=self.scenario.to_runtime(),
            reward=self.reward.to_runtime(),
            observation=observation.to_runtime(),
            control=control.to_runtime(),
        )


def connect4_task_spec_from_mapping(
    *,
    payload: dict[str, object],
    base_dir: Path,
) -> Connect4TaskSpec:
    """Parse a Connect 4 task specification from a raw mapping."""
    task_spec = validate_task_spec_model(
        model_type=Connect4TaskSpecModel,
        payload=payload,
        base_dir=base_dir,
    )
    return task_spec.to_runtime()


def build_connect4_environment_from_task_spec(
    *,
    task_spec: TaskSpec,
) -> TurnBasedEnv[Connect4State, Connect4Action]:
    """Build a Connect 4 environment from a validated task specification."""
    if not isinstance(task_spec, Connect4TaskSpec):
        raise TypeError(
            "build_connect4_environment_from_task_spec requires Connect4TaskSpec."
        )

    solver: BitBullySolver | None = None
    if isinstance(task_spec.reward, Connect4SolverMoveRewardTaskSpec) or (
        task_spec.control.auto_advance is not None
    ):
        solver = BitBullySolver()

    return make_connect4_env(
        scenario=_build_connect4_scenario(task_spec.scenario),
        reward_fn=_build_connect4_reward(task_spec.reward, solver=solver),
        config=task_spec.episode_config,
        include_images=task_spec.observation.include_images,
        image_size=task_spec.observation.image_size,
        auto_advance_policy=_build_connect4_auto_advance(
            task_spec.control,
            solver=solver,
        ),
    )


def _build_connect4_scenario(
    scenario_spec: Connect4RandomPositionScenarioTaskSpec
    | Connect4FixedBoardScenarioTaskSpec,
) -> RandomPositionScenario | FixedBoardScenario:
    if isinstance(scenario_spec, Connect4RandomPositionScenarioTaskSpec):
        return RandomPositionScenario(
            rows=scenario_spec.rows,
            columns=scenario_spec.columns,
            connect_length=scenario_spec.connect_length,
            min_start_moves=scenario_spec.min_start_moves,
            max_start_moves=scenario_spec.max_start_moves,
        )
    return FixedBoardScenario(
        initial_board=scenario_spec.board,
        connect_length=scenario_spec.connect_length,
    )


def _build_connect4_reward(
    reward_spec: Connect4TerminalOutcomeRewardTaskSpec
    | Connect4SolverMoveRewardTaskSpec,
    *,
    solver: BitBullySolver | None,
) -> TerminalOutcomeReward | SolverMoveScoreReward:
    if isinstance(reward_spec, Connect4TerminalOutcomeRewardTaskSpec):
        return TerminalOutcomeReward(
            perspective=reward_spec.perspective,
            win_reward=reward_spec.win_reward,
            draw_reward=reward_spec.draw_reward,
            loss_reward=reward_spec.loss_reward,
        )
    if solver is None:
        raise ValueError("Connect 4 solver reward construction requires BitBully.")
    return SolverMoveScoreReward(
        scorer=solver,
        perspective=reward_spec.perspective,
    )


def _build_connect4_auto_advance(
    control_spec: Connect4ControlTaskSpec,
    *,
    solver: BitBullySolver | None,
) -> Connect4SolverAutoAdvancePolicy | None:
    if control_spec.auto_advance is None:
        return None
    if solver is None:
        raise ValueError(
            "Connect 4 solver auto-advance construction requires BitBully."
        )
    return Connect4SolverAutoAdvancePolicy(move_selector=solver)


def _is_standard_connect4_scenario(
    scenario_spec: Connect4RandomPositionScenarioTaskSpec
    | Connect4FixedBoardScenarioTaskSpec,
) -> bool:
    if isinstance(scenario_spec, Connect4RandomPositionScenarioTaskSpec):
        return (
            scenario_spec.rows == STANDARD_CONNECT4_ROWS
            and scenario_spec.columns == STANDARD_CONNECT4_COLUMNS
            and scenario_spec.connect_length == STANDARD_CONNECT4_CONNECT_LENGTH
        )
    return (
        len(scenario_spec.board) == STANDARD_CONNECT4_ROWS
        and len(scenario_spec.board[0]) == STANDARD_CONNECT4_COLUMNS
        and scenario_spec.connect_length == STANDARD_CONNECT4_CONNECT_LENGTH
    )


__all__ = [
    "Connect4TaskSpec",
    "build_connect4_environment_from_task_spec",
    "connect4_task_spec_from_mapping",
]
