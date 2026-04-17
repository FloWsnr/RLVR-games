"""Core Mastermind code-space and feedback helpers."""

from collections import Counter
from dataclasses import dataclass
from itertools import product
from typing import Sequence, cast

STANDARD_MASTERMIND_CODE_LENGTH = 4
STANDARD_MASTERMIND_COLOR_COUNT = 6
STANDARD_MASTERMIND_MAX_GUESSES = 10
STANDARD_MASTERMIND_MIN_IMAGE_SIZE = 128

type MastermindCode = tuple[int, int, int, int]
type FeedbackHistoryEntry = tuple[MastermindCode, "Feedback"]


@dataclass(slots=True, frozen=True)
class Feedback:
    """Feedback summary for one Mastermind guess.

    Attributes
    ----------
    black_pegs : int
        Number of exact matches between the secret code and the guess.
    white_pegs : int
        Number of non-exact color matches between the secret code and the
        guess after exact matches are removed.
    """

    black_pegs: int
    white_pegs: int

    def __post_init__(self) -> None:
        """Validate that the feedback counts are internally coherent."""
        if self.black_pegs < 0 or self.white_pegs < 0:
            raise ValueError("Mastermind feedback counts must be non-negative.")
        if self.black_pegs > STANDARD_MASTERMIND_CODE_LENGTH:
            raise ValueError("Mastermind black_pegs cannot exceed the code length.")
        if self.white_pegs > STANDARD_MASTERMIND_CODE_LENGTH:
            raise ValueError("Mastermind white_pegs cannot exceed the code length.")
        if self.black_pegs + self.white_pegs > STANDARD_MASTERMIND_CODE_LENGTH:
            raise ValueError(
                "Mastermind black_pegs + white_pegs cannot exceed the code length."
            )
        if self.black_pegs == STANDARD_MASTERMIND_CODE_LENGTH and self.white_pegs != 0:
            raise ValueError("Solved Mastermind feedback cannot include white pegs.")
        if self.black_pegs == STANDARD_MASTERMIND_CODE_LENGTH - 1 and self.white_pegs:
            raise ValueError(
                "Mastermind feedback of three black pegs and one white peg is "
                "impossible."
            )


ALL_STANDARD_CODES: tuple[MastermindCode, ...] = tuple(
    cast(MastermindCode, code)
    for code in product(
        range(1, STANDARD_MASTERMIND_COLOR_COUNT + 1),
        repeat=STANDARD_MASTERMIND_CODE_LENGTH,
    )
)


def normalize_code(*, code: Sequence[int | str] | str) -> MastermindCode:
    """Normalize one Mastermind code into its canonical tuple form.

    Parameters
    ----------
    code : Sequence[int | str] | str
        Code-like value using digits from ``1`` to ``6``.

    Returns
    -------
    MastermindCode
        Immutable four-digit canonical code.
    """
    raw_tokens: list[str]
    if isinstance(code, str):
        normalized = code.strip().replace(",", " ")
        if not normalized:
            raise ValueError("Mastermind codes must be non-empty.")
        raw_tokens = normalized.split()
    else:
        raw_tokens = [str(token).strip() for token in code]

    if len(raw_tokens) == 1 and len(raw_tokens[0]) == STANDARD_MASTERMIND_CODE_LENGTH:
        compact = raw_tokens[0]
        if compact.isdigit():
            raw_tokens = list(compact)

    if len(raw_tokens) != STANDARD_MASTERMIND_CODE_LENGTH:
        raise ValueError(
            "Mastermind codes must contain exactly four digits from 1 to 6."
        )
    if not all(token.isdigit() for token in raw_tokens):
        raise ValueError("Mastermind codes must use only digits from 1 to 6.")

    digits = tuple(int(token) for token in raw_tokens)
    if any(digit < 1 or digit > STANDARD_MASTERMIND_COLOR_COUNT for digit in digits):
        raise ValueError("Mastermind digits must all be between 1 and 6.")
    return cast(MastermindCode, digits)


def format_code(*, code: MastermindCode) -> str:
    """Return one canonical Mastermind code as spaced digits."""
    return " ".join(str(digit) for digit in code)


def format_feedback(*, feedback: Feedback) -> str:
    """Return a compact text rendering of one feedback result."""
    return f"{feedback.black_pegs}B {feedback.white_pegs}W"


def score_guess(
    *,
    secret_code: MastermindCode,
    guess: MastermindCode,
) -> Feedback:
    """Score one guess against one secret code."""
    black_pegs = 0
    unmatched_secret: list[int] = []
    unmatched_guess: list[int] = []
    for secret_digit, guess_digit in zip(secret_code, guess, strict=True):
        if secret_digit == guess_digit:
            black_pegs += 1
        else:
            unmatched_secret.append(secret_digit)
            unmatched_guess.append(guess_digit)

    secret_counter = Counter(unmatched_secret)
    guess_counter = Counter(unmatched_guess)
    white_pegs = sum(
        min(secret_counter[digit], guess_counter[digit])
        for digit in range(1, STANDARD_MASTERMIND_COLOR_COUNT + 1)
    )
    return Feedback(black_pegs=black_pegs, white_pegs=white_pegs)


def is_consistent_with_history(
    *,
    code: MastermindCode,
    history: Sequence[FeedbackHistoryEntry],
) -> bool:
    """Return whether one candidate code matches every public clue so far."""
    for guess, feedback in history:
        if score_guess(secret_code=code, guess=guess) != feedback:
            return False
    return True


def consistent_code_count(*, history: Sequence[FeedbackHistoryEntry]) -> int:
    """Return the number of standard codes consistent with the public clues."""
    return sum(
        1
        for candidate_code in ALL_STANDARD_CODES
        if is_consistent_with_history(code=candidate_code, history=history)
    )
