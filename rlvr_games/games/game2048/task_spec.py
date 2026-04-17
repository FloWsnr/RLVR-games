"""2048 task-spec parsing and environment construction."""

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
    required_int,
    required_string,
    require_mapping,
    require_nested_sequence,
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


def game2048_task_spec_from_mapping(
    *,
    payload: dict[str, object],
    base_dir: Path,
) -> Game2048TaskSpec:
    """Parse a 2048 task specification from a raw mapping."""
    del base_dir
    header = parse_task_spec_header(
        payload=payload,
        expected_game="game2048",
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
    scenario = _parse_game2048_scenario(
        payload=require_mapping(payload.get("scenario"), context="2048 scenario"),
    )
    reward = _parse_game2048_reward(
        payload=require_mapping(payload.get("reward"), context="2048 reward"),
    )
    observation_payload = optional_mapping(
        payload,
        "observation",
        context="2048 task specification",
    )
    return Game2048TaskSpec(
        schema_version=header.schema_version,
        task_id=header.task_id,
        episode_config=header.episode_config,
        metadata=header.metadata,
        scenario=scenario,
        reward=reward,
        observation=_parse_game2048_observation(observation_payload),
    )


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


def _parse_game2048_scenario(
    *,
    payload: dict[str, object],
) -> Game2048RandomStartScenarioTaskSpec | Game2048FixedBoardScenarioTaskSpec:
    kind = required_string(payload, "kind", context="2048 scenario")
    if kind == "random_start":
        reject_unknown_keys(
            payload,
            allowed_keys=("kind", "size", "target_value", "start_tile_count"),
            context="2048 scenario",
        )
        return Game2048RandomStartScenarioTaskSpec(
            size=(
                STANDARD_2048_SIZE
                if optional_int(payload, "size", context="2048 scenario") is None
                else required_int(payload, "size", context="2048 scenario")
            ),
            target_value=(
                STANDARD_2048_TARGET
                if optional_int(payload, "target_value", context="2048 scenario")
                is None
                else required_int(payload, "target_value", context="2048 scenario")
            ),
            start_tile_count=(
                STANDARD_START_TILE_COUNT
                if optional_int(
                    payload,
                    "start_tile_count",
                    context="2048 scenario",
                )
                is None
                else required_int(
                    payload,
                    "start_tile_count",
                    context="2048 scenario",
                )
            ),
        )
    if kind == "fixed_board":
        reject_unknown_keys(
            payload,
            allowed_keys=(
                "kind",
                "board",
                "target_value",
                "initial_score",
                "initial_move_count",
            ),
            context="2048 scenario",
        )
        return Game2048FixedBoardScenarioTaskSpec(
            board=normalize_initial_board(
                board=cast(
                    tuple[tuple[int, ...], ...],
                    require_nested_sequence(
                        payload.get("board"),
                        context="2048 fixed board",
                    ),
                )
            ),
            target_value=(
                STANDARD_2048_TARGET
                if optional_int(payload, "target_value", context="2048 scenario")
                is None
                else required_int(payload, "target_value", context="2048 scenario")
            ),
            initial_score=(
                0
                if optional_int(payload, "initial_score", context="2048 scenario")
                is None
                else required_int(payload, "initial_score", context="2048 scenario")
            ),
            initial_move_count=(
                0
                if optional_int(
                    payload,
                    "initial_move_count",
                    context="2048 scenario",
                )
                is None
                else required_int(
                    payload,
                    "initial_move_count",
                    context="2048 scenario",
                )
            ),
        )
    raise ValueError(f"Unsupported 2048 scenario kind: {kind!r}.")


def _parse_game2048_reward(
    *,
    payload: dict[str, object],
) -> Game2048TargetTileRewardTaskSpec | Game2048ScoreDeltaRewardTaskSpec:
    kind = required_string(payload, "kind", context="2048 reward")
    if kind == "target_tile":
        reject_unknown_keys(
            payload,
            allowed_keys=("kind",),
            context="2048 reward",
        )
        return Game2048TargetTileRewardTaskSpec()
    if kind == "score_delta":
        reject_unknown_keys(
            payload,
            allowed_keys=("kind",),
            context="2048 reward",
        )
        return Game2048ScoreDeltaRewardTaskSpec()
    raise ValueError(f"Unsupported 2048 reward kind: {kind!r}.")


def _parse_game2048_observation(
    payload: dict[str, object] | None,
) -> Game2048ObservationTaskSpec:
    if payload is None:
        return Game2048ObservationTaskSpec()
    reject_unknown_keys(
        payload,
        allowed_keys=("include_images", "image_size"),
        context="2048 observation",
    )
    include_images = optional_bool(
        payload, "include_images", context="2048 observation"
    )
    image_size = optional_int(payload, "image_size", context="2048 observation")
    return Game2048ObservationTaskSpec(
        include_images=False if include_images is None else include_images,
        image_size=360 if image_size is None else image_size,
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
