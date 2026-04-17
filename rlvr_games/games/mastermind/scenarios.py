"""Mastermind scenario initializers."""

from dataclasses import dataclass
import random

from rlvr_games.core.trajectory import ScenarioReset
from rlvr_games.games.mastermind.engine import (
    ALL_STANDARD_CODES,
    MastermindCode,
    STANDARD_MASTERMIND_CODE_LENGTH,
    STANDARD_MASTERMIND_COLOR_COUNT,
    STANDARD_MASTERMIND_MAX_GUESSES,
    normalize_code,
)
from rlvr_games.games.mastermind.state import MastermindState


@dataclass(slots=True)
class StandardGameScenario:
    """Scenario that initializes one standard random Mastermind game."""

    def reset(self, *, seed: int) -> ScenarioReset[MastermindState]:
        """Create a fresh standard Mastermind episode."""
        rng = random.Random(seed)
        state = MastermindState(secret_code=rng.choice(ALL_STANDARD_CODES))
        return ScenarioReset(
            initial_state=state,
            reset_info={
                "scenario": "standard_game",
                "code_length": STANDARD_MASTERMIND_CODE_LENGTH,
                "color_count": STANDARD_MASTERMIND_COLOR_COUNT,
                "max_guesses": STANDARD_MASTERMIND_MAX_GUESSES,
            },
        )


@dataclass(slots=True)
class FixedCodeScenario:
    """Scenario that initializes Mastermind from one explicit secret code."""

    secret_code: MastermindCode

    def __post_init__(self) -> None:
        """Normalize the configured fixed secret code."""
        self.secret_code = normalize_code(code=self.secret_code)

    def reset(self, *, seed: int) -> ScenarioReset[MastermindState]:
        """Create a fresh fixed-code Mastermind episode."""
        del seed
        state = MastermindState(secret_code=self.secret_code)
        return ScenarioReset(
            initial_state=state,
            reset_info={
                "scenario": "fixed_code",
                "code_length": STANDARD_MASTERMIND_CODE_LENGTH,
                "color_count": STANDARD_MASTERMIND_COLOR_COUNT,
                "max_guesses": STANDARD_MASTERMIND_MAX_GUESSES,
            },
        )


def normalize_initial_code(
    *, code: MastermindCode | tuple[int, ...] | str
) -> MastermindCode:
    """Normalize a programmatic, CLI-provided, or YAML-provided secret code."""
    return normalize_code(code=code)
