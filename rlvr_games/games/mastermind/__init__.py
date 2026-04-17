"""Mastermind environment scaffolding."""

from rlvr_games.games.mastermind.actions import (
    MastermindAction,
    serialize_mastermind_action,
)
from rlvr_games.games.mastermind.backend import MastermindBackend
from rlvr_games.games.mastermind.engine import (
    ALL_STANDARD_CODES,
    Feedback,
    MastermindCode,
    STANDARD_MASTERMIND_CODE_LENGTH,
    STANDARD_MASTERMIND_COLOR_COUNT,
    STANDARD_MASTERMIND_MAX_GUESSES,
    consistent_code_count,
    format_code,
    format_feedback,
    is_consistent_with_history,
    normalize_code,
    score_guess,
)
from rlvr_games.games.mastermind.factory import make_mastermind_env
from rlvr_games.games.mastermind.render import (
    MastermindAsciiBoardFormatter,
    MastermindImageRenderer,
    MastermindObservationRenderer,
)
from rlvr_games.games.mastermind.rewards import (
    CandidateReductionDenseReward,
    TerminalOutcomeReward,
)
from rlvr_games.games.mastermind.scenarios import (
    FixedCodeScenario,
    StandardGameScenario,
    normalize_initial_code,
)
from rlvr_games.games.mastermind.state import (
    MastermindGuessRecord,
    MastermindOutcome,
    MastermindState,
    inspect_mastermind_state,
    public_mastermind_metadata,
)

__all__ = [
    "ALL_STANDARD_CODES",
    "CandidateReductionDenseReward",
    "Feedback",
    "FixedCodeScenario",
    "format_code",
    "format_feedback",
    "inspect_mastermind_state",
    "is_consistent_with_history",
    "make_mastermind_env",
    "MastermindAction",
    "MastermindAsciiBoardFormatter",
    "MastermindBackend",
    "MastermindCode",
    "MastermindGuessRecord",
    "MastermindImageRenderer",
    "MastermindObservationRenderer",
    "MastermindOutcome",
    "MastermindState",
    "normalize_code",
    "normalize_initial_code",
    "public_mastermind_metadata",
    "score_guess",
    "serialize_mastermind_action",
    "STANDARD_MASTERMIND_CODE_LENGTH",
    "STANDARD_MASTERMIND_COLOR_COUNT",
    "STANDARD_MASTERMIND_MAX_GUESSES",
    "StandardGameScenario",
    "TerminalOutcomeReward",
    "consistent_code_count",
]
