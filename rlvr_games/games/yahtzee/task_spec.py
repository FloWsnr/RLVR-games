"""Yahtzee task-spec parsing and environment construction."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr

from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.task_spec_base import (
    TaskSpec,
    TaskSpecModel,
    validate_task_spec_model,
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


class _YahtzeeYamlModel(BaseModel):
    """Base model for authored Yahtzee YAML fragments."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class YahtzeeStandardGameScenarioModel(_YahtzeeYamlModel):
    """Authored standard-game scenario block."""

    kind: Literal["standard_game"] = "standard_game"

    def to_runtime(self) -> YahtzeeStandardGameScenarioTaskSpec:
        """Convert the authored scenario into the runtime dataclass."""
        return YahtzeeStandardGameScenarioTaskSpec()


class YahtzeeFixedStateScenarioModel(_YahtzeeYamlModel):
    """Authored fixed-state scenario block."""

    kind: Literal["fixed_state"] = "fixed_state"
    dice: tuple[StrictInt, StrictInt, StrictInt, StrictInt, StrictInt] | None = None
    rolls_used_in_turn: StrictInt | None = None
    awaiting_roll: StrictBool
    category_scores: dict[StrictStr, StrictInt | None] = Field(default_factory=dict)
    rng_seed: StrictInt = 0

    def to_runtime(self) -> YahtzeeFixedStateScenarioTaskSpec:
        """Convert the authored scenario into the runtime dataclass."""
        dice = self.dice
        if dice is None:
            if not self.awaiting_roll:
                raise ValueError(
                    "Yahtzee fixed_state requires dice when awaiting_roll is false."
                )
            dice = ZERO_DICE
        rolls_used_in_turn = self.rolls_used_in_turn
        if rolls_used_in_turn is None:
            rolls_used_in_turn = 0 if self.awaiting_roll else 1
        return YahtzeeFixedStateScenarioTaskSpec(
            dice=dice,
            rolls_used_in_turn=rolls_used_in_turn,
            awaiting_roll=self.awaiting_roll,
            category_scores=_normalize_category_scores(self.category_scores),
            rng_seed=self.rng_seed,
        )


YahtzeeScenarioModel = Annotated[
    YahtzeeStandardGameScenarioModel | YahtzeeFixedStateScenarioModel,
    Field(discriminator="kind"),
]


class YahtzeeFinalScoreRewardModel(_YahtzeeYamlModel):
    """Authored final-score reward block."""

    kind: Literal["final_score"] = "final_score"

    def to_runtime(self) -> YahtzeeFinalScoreRewardTaskSpec:
        """Convert the authored reward into the runtime dataclass."""
        return YahtzeeFinalScoreRewardTaskSpec()


class YahtzeeScoreDeltaRewardModel(_YahtzeeYamlModel):
    """Authored score-delta reward block."""

    kind: Literal["score_delta"] = "score_delta"

    def to_runtime(self) -> YahtzeeScoreDeltaRewardTaskSpec:
        """Convert the authored reward into the runtime dataclass."""
        return YahtzeeScoreDeltaRewardTaskSpec()


YahtzeeRewardModel = Annotated[
    YahtzeeFinalScoreRewardModel | YahtzeeScoreDeltaRewardModel,
    Field(discriminator="kind"),
]


class YahtzeeObservationModel(_YahtzeeYamlModel):
    """Authored observation block."""

    include_images: StrictBool = False
    image_size: StrictInt = 360

    def to_runtime(self) -> YahtzeeObservationTaskSpec:
        """Convert the authored observation block into the runtime dataclass."""
        return YahtzeeObservationTaskSpec(
            include_images=self.include_images,
            image_size=self.image_size,
        )


class YahtzeeTaskSpecModel(TaskSpecModel):
    """Authored top-level Yahtzee task specification."""

    game: Literal["yahtzee"] = "yahtzee"
    scenario: YahtzeeScenarioModel
    reward: YahtzeeRewardModel
    observation: YahtzeeObservationModel | None = None

    def to_runtime(self) -> YahtzeeTaskSpec:
        """Convert the authored model into the runtime task spec."""
        observation = self.observation
        if observation is None:
            observation = YahtzeeObservationModel()
        return YahtzeeTaskSpec(
            schema_version=self.schema_version,
            task_id=self.task_id,
            episode_config=self.episode_config(),
            metadata=self.metadata,
            scenario=self.scenario.to_runtime(),
            reward=self.reward.to_runtime(),
            observation=observation.to_runtime(),
        )


def yahtzee_task_spec_from_mapping(
    *,
    payload: dict[str, object],
    base_dir: Path,
) -> YahtzeeTaskSpec:
    """Parse a Yahtzee task specification from a raw mapping."""
    task_spec = validate_task_spec_model(
        model_type=YahtzeeTaskSpecModel,
        payload=payload,
        base_dir=base_dir,
    )
    return task_spec.to_runtime()


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


def _normalize_category_scores(
    payload: dict[str, int | None],
) -> tuple[int | None, ...]:
    normalized_scores: list[int | None] = [None for _ in CATEGORY_ORDER]
    for raw_category, raw_score in payload.items():
        category = normalize_category_name(raw_name=raw_category)
        if category is None:
            raise ValueError(
                f"Unknown Yahtzee category in fixed_state scorecard: {raw_category!r}."
            )
        normalized_scores[CATEGORY_ORDER.index(category)] = raw_score
    if not payload:
        return empty_category_scores()
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
