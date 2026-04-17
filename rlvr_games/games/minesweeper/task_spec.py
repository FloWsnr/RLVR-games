"""Minesweeper task-spec parsing and environment construction."""

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


def minesweeper_task_spec_from_mapping(
    *,
    payload: dict[str, object],
    base_dir: Path,
) -> MinesweeperTaskSpec:
    """Parse a Minesweeper task specification from a raw mapping."""
    del base_dir
    header = parse_task_spec_header(
        payload=payload,
        expected_game="minesweeper",
        allowed_top_level_keys=(
            "schema_version",
            "id",
            "game",
            "scenario",
            "reward",
            "episode",
            "observation",
            "metadata",
        ),
    )
    scenario = _parse_minesweeper_scenario(
        payload=require_mapping(
            payload.get("scenario"), context="minesweeper scenario"
        ),
    )
    reward = _parse_minesweeper_reward(
        payload=require_mapping(payload.get("reward"), context="minesweeper reward"),
    )
    observation_payload = optional_mapping(
        payload,
        "observation",
        context="minesweeper task specification",
    )
    return MinesweeperTaskSpec(
        schema_version=header.schema_version,
        task_id=header.task_id,
        episode_config=header.episode_config,
        metadata=header.metadata,
        scenario=scenario,
        reward=reward,
        observation=_parse_minesweeper_observation(observation_payload),
    )


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


def _parse_minesweeper_scenario(
    *,
    payload: dict[str, object],
) -> MinesweeperRandomBoardScenarioTaskSpec | MinesweeperFixedBoardScenarioTaskSpec:
    kind = required_string(payload, "kind", context="minesweeper scenario")
    if kind == "random_board":
        reject_unknown_keys(
            payload,
            allowed_keys=("kind", "rows", "columns", "mine_count"),
            context="minesweeper scenario",
        )
        return MinesweeperRandomBoardScenarioTaskSpec(
            rows=(
                STANDARD_MINESWEEPER_ROWS
                if optional_int(payload, "rows", context="minesweeper scenario") is None
                else required_int(payload, "rows", context="minesweeper scenario")
            ),
            columns=(
                STANDARD_MINESWEEPER_COLUMNS
                if optional_int(payload, "columns", context="minesweeper scenario")
                is None
                else required_int(payload, "columns", context="minesweeper scenario")
            ),
            mine_count=(
                STANDARD_MINESWEEPER_MINE_COUNT
                if optional_int(
                    payload,
                    "mine_count",
                    context="minesweeper scenario",
                )
                is None
                else required_int(
                    payload,
                    "mine_count",
                    context="minesweeper scenario",
                )
            ),
        )
    if kind == "fixed_board":
        reject_unknown_keys(
            payload,
            allowed_keys=("kind", "board"),
            context="minesweeper scenario",
        )
        return MinesweeperFixedBoardScenarioTaskSpec(
            board=normalize_initial_board(
                board=cast(
                    tuple[tuple[bool | str | int, ...], ...],
                    require_nested_sequence(
                        payload.get("board"),
                        context="minesweeper fixed board",
                    ),
                )
            )
        )
    raise ValueError(f"Unsupported Minesweeper scenario kind: {kind!r}.")


def _parse_minesweeper_reward(
    *,
    payload: dict[str, object],
) -> MinesweeperOutcomeRewardTaskSpec | MinesweeperSafeRevealRewardTaskSpec:
    kind = required_string(payload, "kind", context="minesweeper reward")
    if kind == "outcome":
        reject_unknown_keys(
            payload,
            allowed_keys=("kind", "win_reward", "loss_reward"),
            context="minesweeper reward",
        )
        return MinesweeperOutcomeRewardTaskSpec(
            win_reward=required_float(
                payload,
                "win_reward",
                context="minesweeper reward",
            ),
            loss_reward=required_float(
                payload,
                "loss_reward",
                context="minesweeper reward",
            ),
        )
    if kind == "safe_reveal_count_dense":
        reject_unknown_keys(
            payload,
            allowed_keys=("kind", "mine_penalty"),
            context="minesweeper reward",
        )
        return MinesweeperSafeRevealRewardTaskSpec(
            mine_penalty=required_float(
                payload,
                "mine_penalty",
                context="minesweeper reward",
            )
        )
    raise ValueError(f"Unsupported Minesweeper reward kind: {kind!r}.")


def _parse_minesweeper_observation(
    payload: dict[str, object] | None,
) -> MinesweeperObservationTaskSpec:
    if payload is None:
        return MinesweeperObservationTaskSpec()
    reject_unknown_keys(
        payload,
        allowed_keys=("include_images", "image_size"),
        context="minesweeper observation",
    )
    include_images = optional_bool(
        payload,
        "include_images",
        context="minesweeper observation",
    )
    image_size = optional_int(payload, "image_size", context="minesweeper observation")
    return MinesweeperObservationTaskSpec(
        include_images=False if include_images is None else include_images,
        image_size=240 if image_size is None else image_size,
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
