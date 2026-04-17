"""Canonical Mastermind state types."""

from dataclasses import dataclass, field
from typing import Sequence

from rlvr_games.games.mastermind.actions import serialize_mastermind_action
from rlvr_games.games.mastermind.engine import (
    ALL_STANDARD_CODES,
    Feedback,
    FeedbackHistoryEntry,
    MastermindCode,
    STANDARD_MASTERMIND_CODE_LENGTH,
    STANDARD_MASTERMIND_COLOR_COUNT,
    STANDARD_MASTERMIND_MAX_GUESSES,
    consistent_code_count,
    format_code,
    format_feedback,
    normalize_code,
    score_guess,
)

STANDARD_MASTERMIND_ACTIONS: tuple[str, ...] = tuple(
    serialize_mastermind_action(code=code) for code in ALL_STANDARD_CODES
)


@dataclass(slots=True, frozen=True)
class MastermindOutcome:
    """Terminal outcome summary for a canonical Mastermind state.

    Attributes
    ----------
    is_terminal : bool
        Whether the episode has ended.
    won : bool
        Whether the codebreaker cracked the code.
    termination : str | None
        Structured termination reason. Supported terminal values are
        ``"cracked"`` and ``"out_of_guesses"``.
    """

    is_terminal: bool
    won: bool
    termination: str | None = None

    def __post_init__(self) -> None:
        """Validate that the outcome flags are internally coherent."""
        if self.is_terminal and self.termination is None:
            raise ValueError("Terminal Mastermind outcomes require a termination.")
        if not self.is_terminal:
            if self.won or self.termination is not None:
                raise ValueError(
                    "Non-terminal Mastermind outcomes must not include terminal "
                    "metadata."
                )
            return
        if self.termination == "cracked":
            if not self.won:
                raise ValueError("Cracked Mastermind outcomes must be wins.")
            return
        if self.termination == "out_of_guesses":
            if self.won:
                raise ValueError("Out-of-guesses Mastermind outcomes cannot be wins.")
            return
        raise ValueError(
            "Mastermind termination must be 'cracked' or 'out_of_guesses'."
        )

    def metadata(self) -> dict[str, object]:
        """Return dictionary metadata for the outcome."""
        if not self.is_terminal:
            return {}
        return {
            "won": self.won,
            "termination": self.termination,
        }


@dataclass(init=False, slots=True, frozen=True)
class MastermindGuessRecord:
    """One public guess-and-feedback row in the Mastermind history."""

    guess: MastermindCode
    black_pegs: int
    white_pegs: int

    def __init__(
        self,
        *,
        guess: MastermindCode | Sequence[int | str] | str,
        black_pegs: int,
        white_pegs: int,
    ) -> None:
        """Normalize and validate one public Mastermind clue row."""
        feedback = Feedback(black_pegs=black_pegs, white_pegs=white_pegs)
        object.__setattr__(self, "guess", normalize_code(code=guess))
        object.__setattr__(self, "black_pegs", feedback.black_pegs)
        object.__setattr__(self, "white_pegs", feedback.white_pegs)

    @property
    def feedback(self) -> Feedback:
        """Return the structured feedback for this history row."""
        return Feedback(black_pegs=self.black_pegs, white_pegs=self.white_pegs)

    def metadata(self) -> dict[str, object]:
        """Return one JSON-like public metadata row."""
        return {
            "guess": self.guess,
            "guess_text": format_code(code=self.guess),
            "black_pegs": self.black_pegs,
            "white_pegs": self.white_pegs,
            "feedback": format_feedback(feedback=self.feedback),
        }


@dataclass(init=False, slots=True, frozen=True)
class MastermindState:
    """Canonical standard-variant Mastermind state.

    Attributes
    ----------
    secret_code : MastermindCode
        Hidden canonical secret code.
    guess_history : tuple[MastermindGuessRecord, ...]
        Public guess and feedback history in turn order.
    max_guesses : int
        Maximum number of accepted guesses before the game is lost.
    move_count : int
        Number of accepted guesses so far.
    guesses_remaining : int
        Remaining accepted guesses before the game ends.
    candidate_count : int
        Number of standard secret codes consistent with the public feedback
        history.
    legal_actions : tuple[str, ...]
        Canonical serialized legal guesses for the current state.
    outcome : MastermindOutcome
        Terminal outcome summary for the current state.
    """

    secret_code: MastermindCode
    guess_history: tuple[MastermindGuessRecord, ...]
    max_guesses: int
    move_count: int = field(init=False)
    guesses_remaining: int = field(init=False)
    candidate_count: int = field(init=False)
    legal_actions: tuple[str, ...] = field(init=False)
    outcome: MastermindOutcome = field(init=False)

    def __init__(
        self,
        *,
        secret_code: MastermindCode | Sequence[int | str] | str,
        guess_history: tuple[MastermindGuessRecord, ...] = (),
        max_guesses: int = STANDARD_MASTERMIND_MAX_GUESSES,
    ) -> None:
        """Create a canonical Mastermind state."""
        normalized_secret_code = normalize_code(code=secret_code)
        if max_guesses <= 0:
            raise ValueError("Mastermind max_guesses must be positive.")

        if len(guess_history) > max_guesses:
            raise ValueError(
                "Mastermind guess_history cannot exceed the configured max_guesses."
            )

        history_entries: list[FeedbackHistoryEntry] = []
        solved_turn_index: int | None = None
        for turn_index, record in enumerate(guess_history):
            actual_feedback = score_guess(
                secret_code=normalized_secret_code,
                guess=record.guess,
            )
            if record.feedback != actual_feedback:
                raise ValueError(
                    "Mastermind guess_history feedback must match the secret code."
                )
            if solved_turn_index is not None:
                raise ValueError(
                    "Mastermind guess_history cannot contain guesses after a solved "
                    "row."
                )
            if actual_feedback.black_pegs == STANDARD_MASTERMIND_CODE_LENGTH:
                solved_turn_index = turn_index
            history_entries.append((record.guess, record.feedback))

        candidate_count = consistent_code_count(history=history_entries)
        if candidate_count < 1:
            raise ValueError(
                "Mastermind guess_history must leave at least one consistent code."
            )

        move_count = len(guess_history)
        guesses_remaining = max_guesses - move_count
        if solved_turn_index is not None:
            outcome = MastermindOutcome(
                is_terminal=True,
                won=True,
                termination="cracked",
            )
            legal_actions: tuple[str, ...] = ()
        elif move_count == max_guesses:
            outcome = MastermindOutcome(
                is_terminal=True,
                won=False,
                termination="out_of_guesses",
            )
            legal_actions = ()
        else:
            outcome = MastermindOutcome(is_terminal=False, won=False)
            legal_actions = STANDARD_MASTERMIND_ACTIONS

        object.__setattr__(self, "secret_code", normalized_secret_code)
        object.__setattr__(self, "guess_history", guess_history)
        object.__setattr__(self, "max_guesses", max_guesses)
        object.__setattr__(self, "move_count", move_count)
        object.__setattr__(self, "guesses_remaining", guesses_remaining)
        object.__setattr__(self, "candidate_count", candidate_count)
        object.__setattr__(self, "legal_actions", legal_actions)
        object.__setattr__(self, "outcome", outcome)

    @property
    def is_terminal(self) -> bool:
        """Return whether the current state ends the episode."""
        return self.outcome.is_terminal

    @property
    def legal_action_count(self) -> int:
        """Return the number of legal actions for the current state."""
        return len(self.legal_actions)


def public_mastermind_metadata(state: MastermindState) -> dict[str, object]:
    """Return public-safe metadata derived from canonical Mastermind state."""
    metadata: dict[str, object] = {
        "guess_history": tuple(record.metadata() for record in state.guess_history),
        "code_length": STANDARD_MASTERMIND_CODE_LENGTH,
        "color_count": STANDARD_MASTERMIND_COLOR_COUNT,
        "max_guesses": state.max_guesses,
        "guesses_used": state.move_count,
        "guesses_remaining": state.guesses_remaining,
        "candidate_count": state.candidate_count,
        "is_terminal": state.is_terminal,
        "won": state.outcome.won,
    }
    if state.guess_history:
        metadata["last_feedback"] = state.guess_history[-1].metadata()
    if state.is_terminal:
        metadata["termination"] = state.outcome.termination
    return metadata


def inspect_mastermind_state(state: MastermindState) -> dict[str, object]:
    """Return a debug-oriented summary of canonical Mastermind state."""
    summary = public_mastermind_metadata(state)
    summary["secret_code"] = state.secret_code
    summary["secret_code_text"] = format_code(code=state.secret_code)
    summary["legal_action_count"] = state.legal_action_count
    return summary
