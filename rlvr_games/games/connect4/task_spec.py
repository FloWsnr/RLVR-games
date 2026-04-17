"""Connect 4 task-spec parsing and environment construction."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.task_spec_base import (
    TaskSpec,
    optional_bool,
    optional_int,
    optional_mapping,
    parse_task_spec_header,
    reject_unknown_keys,
    required_float,
    required_int,
    required_string,
    require_mapping,
    require_nested_sequence,
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


def connect4_task_spec_from_mapping(
    *,
    payload: dict[str, object],
    base_dir: Path,
) -> Connect4TaskSpec:
    """Parse a Connect 4 task specification from a raw mapping."""
    del base_dir
    header = parse_task_spec_header(
        payload=payload,
        expected_game="connect4",
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
    scenario = _parse_connect4_scenario(
        payload=require_mapping(payload.get("scenario"), context="connect4 scenario"),
    )
    reward = _parse_connect4_reward(
        payload=require_mapping(payload.get("reward"), context="connect4 reward"),
    )
    observation_payload = optional_mapping(
        payload,
        "observation",
        context="connect4 task specification",
    )
    control_payload = optional_mapping(
        payload,
        "control",
        context="connect4 task specification",
    )
    return Connect4TaskSpec(
        schema_version=header.schema_version,
        task_id=header.task_id,
        episode_config=header.episode_config,
        metadata=header.metadata,
        scenario=scenario,
        reward=reward,
        observation=_parse_connect4_observation(observation_payload),
        control=_parse_connect4_control(control_payload),
    )


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


def _parse_connect4_scenario(
    *,
    payload: dict[str, object],
) -> Connect4RandomPositionScenarioTaskSpec | Connect4FixedBoardScenarioTaskSpec:
    kind = required_string(payload, "kind", context="connect4 scenario")
    if kind == "random_position":
        reject_unknown_keys(
            payload,
            allowed_keys=(
                "kind",
                "rows",
                "columns",
                "connect_length",
                "min_start_moves",
                "max_start_moves",
            ),
            context="connect4 scenario",
        )
        return Connect4RandomPositionScenarioTaskSpec(
            rows=(
                STANDARD_CONNECT4_ROWS
                if optional_int(payload, "rows", context="connect4 scenario") is None
                else required_int(payload, "rows", context="connect4 scenario")
            ),
            columns=(
                STANDARD_CONNECT4_COLUMNS
                if optional_int(payload, "columns", context="connect4 scenario") is None
                else required_int(payload, "columns", context="connect4 scenario")
            ),
            connect_length=(
                STANDARD_CONNECT4_CONNECT_LENGTH
                if optional_int(
                    payload,
                    "connect_length",
                    context="connect4 scenario",
                )
                is None
                else required_int(
                    payload,
                    "connect_length",
                    context="connect4 scenario",
                )
            ),
            min_start_moves=(
                0
                if optional_int(
                    payload,
                    "min_start_moves",
                    context="connect4 scenario",
                )
                is None
                else required_int(
                    payload,
                    "min_start_moves",
                    context="connect4 scenario",
                )
            ),
            max_start_moves=(
                DEFAULT_RANDOM_START_MAX_MOVES
                if optional_int(
                    payload,
                    "max_start_moves",
                    context="connect4 scenario",
                )
                is None
                else required_int(
                    payload,
                    "max_start_moves",
                    context="connect4 scenario",
                )
            ),
        )
    if kind == "fixed_board":
        reject_unknown_keys(
            payload,
            allowed_keys=("kind", "board", "connect_length"),
            context="connect4 scenario",
        )
        board_payload = require_nested_sequence(
            payload.get("board"),
            context="connect4 fixed board",
        )
        return Connect4FixedBoardScenarioTaskSpec(
            board=normalize_initial_board(
                board=cast(tuple[tuple[str, ...], ...], board_payload)
            ),
            connect_length=(
                STANDARD_CONNECT4_CONNECT_LENGTH
                if optional_int(
                    payload,
                    "connect_length",
                    context="connect4 scenario",
                )
                is None
                else required_int(
                    payload,
                    "connect_length",
                    context="connect4 scenario",
                )
            ),
        )
    raise ValueError(f"Unsupported Connect 4 scenario kind: {kind!r}.")


def _parse_connect4_reward(
    *,
    payload: dict[str, object],
) -> Connect4TerminalOutcomeRewardTaskSpec | Connect4SolverMoveRewardTaskSpec:
    kind = required_string(payload, "kind", context="connect4 reward")
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
            context="connect4 reward",
        )
        return Connect4TerminalOutcomeRewardTaskSpec(
            perspective=_parse_connect4_reward_perspective(
                payload,
                key="perspective",
                context="connect4 reward",
            ),
            win_reward=required_float(
                payload,
                "win_reward",
                context="connect4 reward",
            ),
            draw_reward=required_float(
                payload,
                "draw_reward",
                context="connect4 reward",
            ),
            loss_reward=required_float(
                payload,
                "loss_reward",
                context="connect4 reward",
            ),
        )
    if kind == "solver_move_dense":
        reject_unknown_keys(
            payload,
            allowed_keys=("kind", "perspective"),
            context="connect4 reward",
        )
        perspective = _parse_connect4_reward_perspective(
            payload,
            key="perspective",
            context="connect4 reward",
        )
        return Connect4SolverMoveRewardTaskSpec(perspective=perspective)
    raise ValueError(f"Unsupported Connect 4 reward kind: {kind!r}.")


def _parse_connect4_observation(
    payload: dict[str, object] | None,
) -> Connect4ObservationTaskSpec:
    if payload is None:
        return Connect4ObservationTaskSpec()
    reject_unknown_keys(
        payload,
        allowed_keys=("include_images", "image_size"),
        context="connect4 observation",
    )
    include_images = optional_bool(
        payload,
        "include_images",
        context="connect4 observation",
    )
    image_size = optional_int(payload, "image_size", context="connect4 observation")
    return Connect4ObservationTaskSpec(
        include_images=False if include_images is None else include_images,
        image_size=360 if image_size is None else image_size,
    )


def _parse_connect4_control(
    payload: dict[str, object] | None,
) -> Connect4ControlTaskSpec:
    if payload is None:
        return Connect4ControlTaskSpec()
    reject_unknown_keys(
        payload,
        allowed_keys=("auto_advance",),
        context="connect4 control",
    )
    auto_advance_payload = optional_mapping(
        payload,
        "auto_advance",
        context="connect4 control",
    )
    if auto_advance_payload is None:
        return Connect4ControlTaskSpec()
    reject_unknown_keys(
        auto_advance_payload,
        allowed_keys=("kind",),
        context="connect4 auto_advance",
    )
    kind = required_string(
        auto_advance_payload,
        "kind",
        context="connect4 auto_advance",
    )
    if kind != "solver":
        raise ValueError(f"Unsupported Connect 4 auto_advance kind: {kind!r}.")
    return Connect4ControlTaskSpec(auto_advance=Connect4SolverAutoAdvanceTaskSpec())


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


def _parse_connect4_reward_perspective(
    payload: dict[str, object],
    *,
    key: str,
    context: str,
) -> Connect4RewardPerspective:
    perspective = required_string(payload, key, context=context)
    if perspective not in ("x", "o", "mover"):
        raise ValueError(f"{context} field {key!r} must be 'x', 'o', or 'mover'.")
    return cast(Connect4RewardPerspective, perspective)


__all__ = [
    "Connect4TaskSpec",
    "build_connect4_environment_from_task_spec",
    "connect4_task_spec_from_mapping",
]
