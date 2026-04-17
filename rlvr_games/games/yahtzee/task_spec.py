"""Yahtzee task-spec parsing and environment construction."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

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
)
from rlvr_games.games.yahtzee.actions import YahtzeeAction
from rlvr_games.games.yahtzee.chance import YahtzeeChanceModel
from rlvr_games.games.yahtzee.engine import (
    CATEGORY_ORDER,
    ZERO_DICE,
    empty_category_scores,
    filled_category_count,
    normalize_category_name,
)
from rlvr_games.games.yahtzee.factory import make_yahtzee_env
from rlvr_games.games.yahtzee.rewards import FinalScoreReward, ScoreDeltaReward
from rlvr_games.games.yahtzee.state import YahtzeeState


@dataclass(slots=True, frozen=True)
class YahtzeeStandardGameScenarioTaskSpec:
    """Task-spec variant for a standard new Yahtzee game."""


@dataclass(slots=True, frozen=True)
class YahtzeeFixedStateScenarioTaskSpec:
    """Task-spec variant for a fixed canonical Yahtzee state."""

    dice: tuple[int, int, int, int, int]
    rolls_used_in_turn: int
    awaiting_roll: bool
    category_scores: tuple[int | None, ...]
    rng_seed: int

    def __post_init__(self) -> None:
        """Validate fixed-state scenario parameters."""
        chance_model = YahtzeeChanceModel()
        YahtzeeState(
            dice=self.dice,
            rolls_used_in_turn=self.rolls_used_in_turn,
            turns_completed=filled_category_count(scores=self.category_scores),
            awaiting_roll=self.awaiting_roll,
            category_scores=self.category_scores,
            rng_state=chance_model.initial_rng_state(seed=self.rng_seed),
        )


@dataclass(slots=True, frozen=True)
class YahtzeeFinalScoreRewardTaskSpec:
    """Sparse final-score reward for Yahtzee."""


@dataclass(slots=True, frozen=True)
class YahtzeeScoreDeltaRewardTaskSpec:
    """Dense score-delta reward for Yahtzee."""


@dataclass(slots=True, frozen=True)
class YahtzeeObservationTaskSpec:
    """Yahtzee observation configuration."""

    include_images: bool = False
    image_size: int = 360

    def __post_init__(self) -> None:
        """Validate observation parameters."""
        if self.image_size < 1:
            raise ValueError("Yahtzee observation image_size must be >= 1.")


@dataclass(slots=True)
class YahtzeeTaskSpec(TaskSpec):
    """Validated authored Yahtzee task specification."""

    scenario: YahtzeeStandardGameScenarioTaskSpec | YahtzeeFixedStateScenarioTaskSpec
    reward: YahtzeeFinalScoreRewardTaskSpec | YahtzeeScoreDeltaRewardTaskSpec
    observation: YahtzeeObservationTaskSpec = field(
        default_factory=YahtzeeObservationTaskSpec
    )

    @property
    def game(self) -> str:
        """Return the game name carried by this task spec."""
        return "yahtzee"


def yahtzee_task_spec_from_mapping(
    *,
    payload: dict[str, object],
    base_dir: Path,
) -> YahtzeeTaskSpec:
    """Parse a Yahtzee task specification from a raw mapping."""
    del base_dir
    header = parse_task_spec_header(
        payload=payload,
        expected_game="yahtzee",
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
    scenario = _parse_yahtzee_scenario(
        payload=require_mapping(payload.get("scenario"), context="yahtzee scenario"),
    )
    reward = _parse_yahtzee_reward(
        payload=require_mapping(payload.get("reward"), context="yahtzee reward"),
    )
    observation_payload = optional_mapping(
        payload,
        "observation",
        context="yahtzee task specification",
    )
    return YahtzeeTaskSpec(
        schema_version=header.schema_version,
        task_id=header.task_id,
        episode_config=header.episode_config,
        metadata=header.metadata,
        scenario=scenario,
        reward=reward,
        observation=_parse_yahtzee_observation(observation_payload),
    )


def build_yahtzee_environment_from_task_spec(
    *,
    task_spec: TaskSpec,
) -> TurnBasedEnv[YahtzeeState, YahtzeeAction]:
    """Build a Yahtzee environment from a validated task specification."""
    if not isinstance(task_spec, YahtzeeTaskSpec):
        raise TypeError(
            "build_yahtzee_environment_from_task_spec requires YahtzeeTaskSpec."
        )

    initial_state: YahtzeeState | None = None
    if isinstance(task_spec.scenario, YahtzeeFixedStateScenarioTaskSpec):
        chance_model = YahtzeeChanceModel()
        initial_state = YahtzeeState(
            dice=task_spec.scenario.dice,
            rolls_used_in_turn=task_spec.scenario.rolls_used_in_turn,
            turns_completed=filled_category_count(
                scores=task_spec.scenario.category_scores
            ),
            awaiting_roll=task_spec.scenario.awaiting_roll,
            category_scores=task_spec.scenario.category_scores,
            rng_state=chance_model.initial_rng_state(seed=task_spec.scenario.rng_seed),
        )

    return make_yahtzee_env(
        initial_state=initial_state,
        reward_fn=_build_yahtzee_reward(task_spec.reward),
        config=task_spec.episode_config,
        include_images=task_spec.observation.include_images,
        image_size=task_spec.observation.image_size,
    )


def _parse_yahtzee_scenario(
    *,
    payload: dict[str, object],
) -> YahtzeeStandardGameScenarioTaskSpec | YahtzeeFixedStateScenarioTaskSpec:
    kind = required_string(payload, "kind", context="yahtzee scenario")
    if kind == "standard_game":
        reject_unknown_keys(
            payload,
            allowed_keys=("kind",),
            context="yahtzee scenario",
        )
        return YahtzeeStandardGameScenarioTaskSpec()
    if kind == "fixed_state":
        reject_unknown_keys(
            payload,
            allowed_keys=(
                "kind",
                "dice",
                "rolls_used_in_turn",
                "awaiting_roll",
                "category_scores",
                "rng_seed",
            ),
            context="yahtzee scenario",
        )
        awaiting_roll = payload.get("awaiting_roll")
        if not isinstance(awaiting_roll, bool):
            raise TypeError("yahtzee scenario field 'awaiting_roll' must be a bool.")
        dice_payload = payload.get("dice")
        dice: tuple[int, int, int, int, int]
        if dice_payload is None:
            if not awaiting_roll:
                raise ValueError(
                    "Yahtzee fixed_state requires dice when awaiting_roll is false."
                )
            dice = ZERO_DICE
        else:
            dice = _parse_yahtzee_dice(dice_payload)
        rolls_used_in_turn = optional_int(
            payload,
            "rolls_used_in_turn",
            context="yahtzee scenario",
        )
        if rolls_used_in_turn is None:
            rolls_used_in_turn = 0 if awaiting_roll else 1
        return YahtzeeFixedStateScenarioTaskSpec(
            dice=dice,
            rolls_used_in_turn=rolls_used_in_turn,
            awaiting_roll=awaiting_roll,
            category_scores=_parse_yahtzee_category_scores(
                payload.get("category_scores")
            ),
            rng_seed=(
                0
                if optional_int(payload, "rng_seed", context="yahtzee scenario") is None
                else required_int(payload, "rng_seed", context="yahtzee scenario")
            ),
        )
    raise ValueError(f"Unsupported Yahtzee scenario kind: {kind!r}.")


def _parse_yahtzee_reward(
    *,
    payload: dict[str, object],
) -> YahtzeeFinalScoreRewardTaskSpec | YahtzeeScoreDeltaRewardTaskSpec:
    kind = required_string(payload, "kind", context="yahtzee reward")
    if kind == "final_score":
        reject_unknown_keys(
            payload,
            allowed_keys=("kind",),
            context="yahtzee reward",
        )
        return YahtzeeFinalScoreRewardTaskSpec()
    if kind == "score_delta":
        reject_unknown_keys(
            payload,
            allowed_keys=("kind",),
            context="yahtzee reward",
        )
        return YahtzeeScoreDeltaRewardTaskSpec()
    raise ValueError(f"Unsupported Yahtzee reward kind: {kind!r}.")


def _parse_yahtzee_observation(
    payload: dict[str, object] | None,
) -> YahtzeeObservationTaskSpec:
    if payload is None:
        return YahtzeeObservationTaskSpec()
    reject_unknown_keys(
        payload,
        allowed_keys=("include_images", "image_size"),
        context="yahtzee observation",
    )
    include_images = optional_bool(
        payload, "include_images", context="yahtzee observation"
    )
    image_size = optional_int(payload, "image_size", context="yahtzee observation")
    return YahtzeeObservationTaskSpec(
        include_images=False if include_images is None else include_images,
        image_size=360 if image_size is None else image_size,
    )


def _parse_yahtzee_dice(payload: object) -> tuple[int, int, int, int, int]:
    if isinstance(payload, str) or not isinstance(payload, Sequence):
        raise TypeError("yahtzee scenario dice must be a sequence of five ints.")
    values = tuple(int(value) for value in payload)
    if len(values) != 5:
        raise ValueError("Yahtzee fixed_state dice must contain exactly five values.")
    return values  # type: ignore[return-value]


def _parse_yahtzee_category_scores(payload: object) -> tuple[int | None, ...]:
    if payload is None:
        return empty_category_scores()
    mapping = require_mapping(payload, context="yahtzee category_scores")
    normalized_scores: list[int | None] = [None for _ in CATEGORY_ORDER]
    for raw_category, raw_score in mapping.items():
        category = normalize_category_name(raw_name=raw_category)
        if category is None:
            raise ValueError(
                f"Unknown Yahtzee category in fixed_state scorecard: {raw_category!r}."
            )
        if raw_score is None:
            continue
        if isinstance(raw_score, bool) or not isinstance(raw_score, int):
            raise TypeError("Yahtzee fixed_state category scores must be ints or null.")
        if raw_score < 0:
            raise ValueError(
                "Yahtzee fixed_state category scores must be non-negative."
            )
        normalized_scores[CATEGORY_ORDER.index(category)] = raw_score
    return tuple(normalized_scores)


def _build_yahtzee_reward(
    reward_spec: YahtzeeFinalScoreRewardTaskSpec | YahtzeeScoreDeltaRewardTaskSpec,
) -> FinalScoreReward | ScoreDeltaReward:
    if isinstance(reward_spec, YahtzeeFinalScoreRewardTaskSpec):
        return FinalScoreReward()
    return ScoreDeltaReward()


__all__ = [
    "YahtzeeTaskSpec",
    "build_yahtzee_environment_from_task_spec",
    "yahtzee_task_spec_from_mapping",
]
