"""Rule-verified backend for Mastermind."""

from rlvr_games.core.exceptions import InvalidActionError
from rlvr_games.core.types import ParseResult
from rlvr_games.games.mastermind.actions import MastermindAction
from rlvr_games.games.mastermind.engine import (
    STANDARD_MASTERMIND_CODE_LENGTH,
    format_code,
    format_feedback,
    score_guess,
)
from rlvr_games.games.mastermind.state import MastermindGuessRecord, MastermindState


class MastermindBackend:
    """Authoritative verifier for Mastermind parsing and transitions."""

    def parse_action(
        self,
        state: MastermindState,
        raw_action: str,
    ) -> ParseResult[MastermindAction]:
        """Parse and validate one raw Mastermind guess."""
        normalized = raw_action.strip().lower().replace(",", " ")
        if not normalized:
            return ParseResult(
                action=None,
                error=(
                    "Mastermind actions must be four digits from 1 to 6, for "
                    "example 'guess 1 1 2 2' or '1122'."
                ),
            )

        tokens = normalized.split()
        guess_tokens: list[str]
        if len(tokens) == 1 and len(tokens[0]) == STANDARD_MASTERMIND_CODE_LENGTH:
            guess_tokens = list(tokens[0])
        elif len(tokens) == STANDARD_MASTERMIND_CODE_LENGTH:
            guess_tokens = list(tokens)
        elif (
            len(tokens) == STANDARD_MASTERMIND_CODE_LENGTH + 1 and tokens[0] == "guess"
        ):
            guess_tokens = list(tokens[1:])
        else:
            return ParseResult(
                action=None,
                error=(
                    "Mastermind actions must use either 'guess <d1> <d2> <d3> <d4>' "
                    "or a bare four-digit code such as '1122'."
                ),
            )

        try:
            action = MastermindAction(code=guess_tokens)
        except ValueError as exc:
            return ParseResult(action=None, error=str(exc))

        if action.label not in state.legal_actions:
            return ParseResult(
                action=None,
                error=(
                    f"Mastermind guess {format_code(code=action.code)!r} is illegal "
                    "for the current state."
                ),
            )

        return ParseResult(action=action, error=None)

    def legal_actions(self, state: MastermindState) -> list[str]:
        """Enumerate legal model-facing actions for the current state."""
        return list(state.legal_actions)

    def apply_action(
        self,
        state: MastermindState,
        action: MastermindAction,
    ) -> tuple[MastermindState, dict[str, object]]:
        """Apply one verified Mastermind guess and return the next state."""
        if action.label not in state.legal_actions:
            raise InvalidActionError(
                f"Mastermind guess {format_code(code=action.code)!r} is illegal for "
                "the current state."
            )

        feedback = score_guess(secret_code=state.secret_code, guess=action.code)
        guess_record = MastermindGuessRecord(
            guess=action.code,
            black_pegs=feedback.black_pegs,
            white_pegs=feedback.white_pegs,
        )
        next_state = MastermindState(
            secret_code=state.secret_code,
            guess_history=state.guess_history + (guess_record,),
            max_guesses=state.max_guesses,
        )
        transition_info: dict[str, object] = {
            "guess": action.code,
            "guess_text": format_code(code=action.code),
            "black_pegs": feedback.black_pegs,
            "white_pegs": feedback.white_pegs,
            "feedback": format_feedback(feedback=feedback),
            "move_count": next_state.move_count,
            "guesses_remaining": next_state.guesses_remaining,
            "candidate_count": next_state.candidate_count,
            "is_terminal": next_state.is_terminal,
        }
        transition_info.update(next_state.outcome.metadata())
        return next_state, transition_info

    def is_terminal(self, state: MastermindState) -> bool:
        """Return whether the supplied Mastermind state ends the episode."""
        return state.is_terminal
