"""Mastermind task-spec parsing and environment construction."""

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
from rlvr_games.games.mastermind.actions import MastermindAction
from rlvr_games.games.mastermind.engine import STANDARD_MASTERMIND_MIN_IMAGE_SIZE
from rlvr_games.games.mastermind.factory import make_mastermind_env
from rlvr_games.games.mastermind.rewards import (
    CandidateReductionDenseReward,
    TerminalOutcomeReward,
)
from rlvr_games.games.mastermind.scenarios import (
    FixedCodeScenario,
    StandardGameScenario,
    normalize_initial_code,
)
from rlvr_games.games.mastermind.state import MastermindState


@dataclass(slots=True, frozen=True)
class MastermindStandardGameScenarioTaskSpec:
    """Task-spec variant for one standard random Mastermind game."""


@dataclass(slots=True, frozen=True)
class MastermindFixedCodeScenarioTaskSpec:
    """Task-spec variant for one fixed secret code."""

    code: tuple[int, int, int, int]


@dataclass(slots=True, frozen=True)
class MastermindTerminalOutcomeRewardTaskSpec:
    """Sparse terminal-outcome reward for Mastermind."""

    win_reward: float
    loss_reward: float


@dataclass(slots=True, frozen=True)
class MastermindCandidateReductionRewardTaskSpec:
    """Dense candidate-reduction reward for Mastermind."""


@dataclass(slots=True, frozen=True)
class MastermindObservationTaskSpec:
    """Mastermind observation configuration."""

    include_images: bool = False
    image_size: int = 360

    def __post_init__(self) -> None:
        """Validate observation parameters."""
        if self.image_size < 1:
            raise ValueError("Mastermind observation image_size must be >= 1.")
        if self.include_images and self.image_size < STANDARD_MASTERMIND_MIN_IMAGE_SIZE:
            raise ValueError(
                "Mastermind image_size must be >= "
                f"{STANDARD_MASTERMIND_MIN_IMAGE_SIZE} when include_images is true."
            )


@dataclass(slots=True)
class MastermindTaskSpec(TaskSpec):
    """Validated authored Mastermind task specification."""

    scenario: (
        MastermindStandardGameScenarioTaskSpec | MastermindFixedCodeScenarioTaskSpec
    )
    reward: (
        MastermindTerminalOutcomeRewardTaskSpec
        | MastermindCandidateReductionRewardTaskSpec
    )
    observation: MastermindObservationTaskSpec = field(
        default_factory=MastermindObservationTaskSpec
    )

    @property
    def game(self) -> str:
        """Return the game name carried by this task spec."""
        return "mastermind"


class _MastermindYamlModel(BaseModel):
    """Base model for authored Mastermind YAML fragments."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class MastermindStandardGameScenarioModel(_MastermindYamlModel):
    """Authored standard-game scenario block."""

    kind: Literal["standard_game"] = "standard_game"

    def to_runtime(self) -> MastermindStandardGameScenarioTaskSpec:
        """Convert the authored scenario into the runtime dataclass."""
        return MastermindStandardGameScenarioTaskSpec()


class MastermindFixedCodeScenarioModel(_MastermindYamlModel):
    """Authored fixed-code scenario block."""

    kind: Literal["fixed_code"] = "fixed_code"
    code: tuple[StrictInt, StrictInt, StrictInt, StrictInt] | StrictStr

    def to_runtime(self) -> MastermindFixedCodeScenarioTaskSpec:
        """Convert the authored scenario into the runtime dataclass."""
        return MastermindFixedCodeScenarioTaskSpec(
            code=normalize_initial_code(code=self.code)
        )


MastermindScenarioModel = Annotated[
    MastermindStandardGameScenarioModel | MastermindFixedCodeScenarioModel,
    Field(discriminator="kind"),
]


class MastermindTerminalOutcomeRewardModel(_MastermindYamlModel):
    """Authored terminal-outcome reward block."""

    kind: Literal["terminal_outcome"] = "terminal_outcome"
    win_reward: NumericScalar
    loss_reward: NumericScalar

    def to_runtime(self) -> MastermindTerminalOutcomeRewardTaskSpec:
        """Convert the authored reward into the runtime dataclass."""
        return MastermindTerminalOutcomeRewardTaskSpec(
            win_reward=float(self.win_reward),
            loss_reward=float(self.loss_reward),
        )


class MastermindCandidateReductionRewardModel(_MastermindYamlModel):
    """Authored dense candidate-reduction reward block."""

    kind: Literal["candidate_reduction_dense"] = "candidate_reduction_dense"

    def to_runtime(self) -> MastermindCandidateReductionRewardTaskSpec:
        """Convert the authored reward into the runtime dataclass."""
        return MastermindCandidateReductionRewardTaskSpec()


MastermindRewardModel = Annotated[
    MastermindTerminalOutcomeRewardModel | MastermindCandidateReductionRewardModel,
    Field(discriminator="kind"),
]


class MastermindObservationModel(_MastermindYamlModel):
    """Authored observation block."""

    include_images: StrictBool = False
    image_size: StrictInt = 360

    def to_runtime(self) -> MastermindObservationTaskSpec:
        """Convert the authored observation block into the runtime dataclass."""
        return MastermindObservationTaskSpec(
            include_images=self.include_images,
            image_size=self.image_size,
        )


class MastermindTaskSpecModel(TaskSpecModel):
    """Authored top-level Mastermind task specification."""

    game: Literal["mastermind"] = "mastermind"
    scenario: MastermindScenarioModel
    reward: MastermindRewardModel
    observation: MastermindObservationModel | None = None

    def to_runtime(self) -> MastermindTaskSpec:
        """Convert the authored model into the runtime task spec."""
        observation = self.observation
        if observation is None:
            observation = MastermindObservationModel()
        return MastermindTaskSpec(
            schema_version=self.schema_version,
            task_id=self.task_id,
            episode_config=self.episode_config(),
            metadata=self.metadata,
            scenario=self.scenario.to_runtime(),
            reward=self.reward.to_runtime(),
            observation=observation.to_runtime(),
        )


def mastermind_task_spec_from_mapping(
    *,
    payload: dict[str, object],
    base_dir: Path,
) -> MastermindTaskSpec:
    """Parse a Mastermind task specification from a raw mapping."""
    task_spec = validate_task_spec_model(
        model_type=MastermindTaskSpecModel,
        payload=payload,
        base_dir=base_dir,
    )
    return task_spec.to_runtime()


def build_mastermind_environment_from_task_spec(
    *,
    task_spec: TaskSpec,
) -> TurnBasedEnv[MastermindState, MastermindAction]:
    """Build a Mastermind environment from a validated task specification."""
    if not isinstance(task_spec, MastermindTaskSpec):
        raise TypeError(
            "build_mastermind_environment_from_task_spec requires MastermindTaskSpec."
        )

    if isinstance(task_spec.scenario, MastermindStandardGameScenarioTaskSpec):
        scenario = StandardGameScenario()
    else:
        scenario = FixedCodeScenario(secret_code=task_spec.scenario.code)

    return make_mastermind_env(
        scenario=scenario,
        reward_fn=_build_mastermind_reward(task_spec.reward),
        config=task_spec.episode_config,
        include_images=task_spec.observation.include_images,
        image_size=task_spec.observation.image_size,
    )


def _build_mastermind_reward(
    reward_task_spec: (
        MastermindTerminalOutcomeRewardTaskSpec
        | MastermindCandidateReductionRewardTaskSpec
    ),
) -> TerminalOutcomeReward | CandidateReductionDenseReward:
    """Build one Mastermind reward function from a runtime reward task spec."""
    if isinstance(reward_task_spec, MastermindTerminalOutcomeRewardTaskSpec):
        return TerminalOutcomeReward(
            win_reward=reward_task_spec.win_reward,
            loss_reward=reward_task_spec.loss_reward,
        )
    return CandidateReductionDenseReward()


__all__ = [
    "MastermindCandidateReductionRewardTaskSpec",
    "MastermindFixedCodeScenarioTaskSpec",
    "MastermindObservationTaskSpec",
    "MastermindStandardGameScenarioTaskSpec",
    "MastermindTaskSpec",
    "MastermindTerminalOutcomeRewardTaskSpec",
    "build_mastermind_environment_from_task_spec",
    "mastermind_task_spec_from_mapping",
]
