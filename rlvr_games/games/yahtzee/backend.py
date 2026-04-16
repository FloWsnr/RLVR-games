"""Rule-verified backend for Yahtzee."""

from rlvr_games.core.exceptions import InvalidActionError
from rlvr_games.core.types import ParseResult
from rlvr_games.games.yahtzee.actions import (
    YahtzeeAction,
    YahtzeeActionKind,
)
from rlvr_games.games.yahtzee.chance import YahtzeeChanceModel
from rlvr_games.games.yahtzee.engine import (
    ZERO_DICE,
    category_index,
    normalize_category_name,
    normalize_dice,
    score_category,
)
from rlvr_games.games.yahtzee.state import YahtzeeState, public_yahtzee_metadata

_REROLL_ALIASES = frozenset({"reroll", "roll", "rr"})
_SCORE_ALIASES = frozenset({"score", "s"})


class YahtzeeBackend:
    """Authoritative verifier for Yahtzee parsing and transitions."""

    def __init__(self, *, chance_model: YahtzeeChanceModel) -> None:
        """Initialize the backend with one deterministic chance model.

        Parameters
        ----------
        chance_model : YahtzeeChanceModel
            Deterministic RNG-backed dice roller used for opening rolls and
            rerolls.
        """
        self.chance_model = chance_model

    def parse_action(
        self,
        state: YahtzeeState,
        raw_action: str,
    ) -> ParseResult[YahtzeeAction]:
        """Parse and validate one raw model action.

        Parameters
        ----------
        state : YahtzeeState
            Canonical state used to interpret and validate the action.
        raw_action : str
            Raw action text emitted by an agent.

        Returns
        -------
        ParseResult[YahtzeeAction]
            Structured parse result describing the accepted action or the
            reason it was rejected.
        """
        normalized = raw_action.strip().lower().replace(",", " ")
        if not normalized:
            return ParseResult(
                action=None,
                error=(
                    "Yahtzee actions must be 'score <category>' or "
                    "'reroll <positions>'."
                ),
            )

        if state.awaiting_roll:
            return ParseResult(
                action=None,
                error="Yahtzee is waiting for an internal opening roll.",
            )

        tokens = normalized.split()
        first_token = tokens[0]
        if first_token in _REROLL_ALIASES:
            return self._parse_reroll_action(
                state=state,
                raw_action=raw_action,
                tokens=tokens,
            )
        if first_token in _SCORE_ALIASES:
            return self._parse_score_action(
                state=state,
                raw_action=raw_action,
                category_text=" ".join(tokens[1:]),
            )

        bare_category = normalize_category_name(raw_name=normalized)
        if bare_category is not None:
            action = YahtzeeAction(
                kind=YahtzeeActionKind.SCORE,
                category=bare_category,
            )
            if action.label not in state.legal_actions:
                return ParseResult(
                    action=None,
                    error=(
                        f"Yahtzee action {action.label!r} is illegal for the "
                        "current turn."
                    ),
                )
            return ParseResult(action=action, error=None)

        return ParseResult(
            action=None,
            error=(
                "Unknown Yahtzee action. Use 'score <category>' or "
                "'reroll <positions>'."
            ),
        )

    def legal_actions(self, state: YahtzeeState) -> list[str]:
        """Enumerate legal model-facing actions for the current state.

        Parameters
        ----------
        state : YahtzeeState
            Canonical state whose legal actions should be exposed.

        Returns
        -------
        list[str]
            Canonical serialized legal actions.
        """
        return list(state.legal_actions)

    def apply_opening_roll(
        self,
        state: YahtzeeState,
    ) -> tuple[YahtzeeState, dict[str, object]]:
        """Apply the reset-time opening roll for a fresh turn.

        Parameters
        ----------
        state : YahtzeeState
            Canonical state awaiting the opening roll.

        Returns
        -------
        tuple[YahtzeeState, dict[str, object]]
            Next state after rolling all five dice and transition metadata.
        """
        if state.awaiting_roll is False:
            raise InvalidActionError(
                "Yahtzee opening rolls require an awaiting-roll state."
            )
        if state.is_terminal:
            raise InvalidActionError(
                "Terminal Yahtzee states cannot take an opening roll."
            )

        roll_transition = self.chance_model.roll_all(rng_state=state.rng_state)
        next_state = YahtzeeState(
            dice=roll_transition.dice,
            rolls_used_in_turn=1,
            turns_completed=state.turns_completed,
            awaiting_roll=False,
            category_scores=state.category_scores,
            rng_state=roll_transition.rng_state,
        )
        info: dict[str, object] = {
            "event_kind": "opening_roll",
            "rolled_positions": tuple(
                position + 1 for position in roll_transition.rolled_positions
            ),
            "rolled_position_indices": roll_transition.rolled_positions,
            "dice_before": ZERO_DICE,
            "dice": next_state.dice,
            **public_yahtzee_metadata(next_state),
        }
        return next_state, info

    def apply_action(
        self,
        state: YahtzeeState,
        action: YahtzeeAction,
    ) -> tuple[YahtzeeState, dict[str, object]]:
        """Apply one verified Yahtzee action.

        Parameters
        ----------
        state : YahtzeeState
            Canonical state before applying the action.
        action : YahtzeeAction
            Canonical parsed action accepted by the backend.

        Returns
        -------
        tuple[YahtzeeState, dict[str, object]]
            Next canonical state and public-safe transition metadata.
        """
        if (
            action.kind != YahtzeeActionKind.OPENING_ROLL
            and action.label not in state.legal_actions
        ):
            raise InvalidActionError(
                f"Yahtzee action {action.label!r} is illegal for the current turn."
            )

        if action.kind == YahtzeeActionKind.OPENING_ROLL:
            return self.apply_opening_roll(state)
        if action.kind == YahtzeeActionKind.REROLL:
            return self._apply_reroll(state=state, action=action)
        if action.kind == YahtzeeActionKind.SCORE:
            return self._apply_score(state=state, action=action)
        raise InvalidActionError(f"Unsupported Yahtzee action kind: {action.kind!r}.")

    def is_terminal(self, state: YahtzeeState) -> bool:
        """Return whether the supplied Yahtzee state ends the episode.

        Parameters
        ----------
        state : YahtzeeState
            Canonical state to inspect.

        Returns
        -------
        bool
            `True` when the scorecard has been fully completed.
        """
        return state.is_terminal

    def _apply_reroll(
        self,
        *,
        state: YahtzeeState,
        action: YahtzeeAction,
    ) -> tuple[YahtzeeState, dict[str, object]]:
        """Apply one reroll action.

        Parameters
        ----------
        state : YahtzeeState
            Canonical state before the reroll.
        action : YahtzeeAction
            Canonical reroll action.

        Returns
        -------
        tuple[YahtzeeState, dict[str, object]]
            Next state and transition metadata.
        """
        roll_transition = self.chance_model.reroll(
            dice=normalize_dice(dice=state.dice),
            reroll_positions=action.reroll_positions,
            rng_state=state.rng_state,
        )
        next_state = YahtzeeState(
            dice=roll_transition.dice,
            rolls_used_in_turn=state.rolls_used_in_turn + 1,
            turns_completed=state.turns_completed,
            awaiting_roll=False,
            category_scores=state.category_scores,
            rng_state=roll_transition.rng_state,
        )
        info: dict[str, object] = {
            "action_kind": "reroll",
            "rerolled_positions": tuple(
                position + 1 for position in action.reroll_positions
            ),
            "rerolled_position_indices": action.reroll_positions,
            "dice_before": state.dice,
            "dice": next_state.dice,
            **public_yahtzee_metadata(next_state),
        }
        return next_state, info

    def _apply_score(
        self,
        *,
        state: YahtzeeState,
        action: YahtzeeAction,
    ) -> tuple[YahtzeeState, dict[str, object]]:
        """Apply one score action and advance to the next turn when needed.

        Parameters
        ----------
        state : YahtzeeState
            Canonical state before the score action.
        action : YahtzeeAction
            Canonical score action.

        Returns
        -------
        tuple[YahtzeeState, dict[str, object]]
            Next state and transition metadata.
        """
        if action.category is None:
            raise InvalidActionError("Yahtzee score actions require a category.")

        score_value = score_category(dice=state.dice, category=action.category)
        updated_scores = list(state.category_scores)
        updated_scores[category_index(category=action.category)] = score_value
        next_turns_completed = state.turns_completed + 1

        next_state = YahtzeeState(
            dice=state.dice
            if next_turns_completed == len(updated_scores)
            else ZERO_DICE,
            rolls_used_in_turn=(
                state.rolls_used_in_turn
                if next_turns_completed == len(updated_scores)
                else 0
            ),
            turns_completed=next_turns_completed,
            awaiting_roll=next_turns_completed != len(updated_scores),
            category_scores=tuple(updated_scores),
            rng_state=state.rng_state,
        )
        info: dict[str, object] = {
            "action_kind": "score",
            "scored_category": action.category.value,
            "score_value": score_value,
            "dice_scored": state.dice,
            "scorecard": public_yahtzee_metadata(next_state)["scorecard"],
            "available_categories": next_state.available_categories,
            "total_score": next_state.total_score,
            "turn_number": next_state.turn_number,
            "turns_completed": next_state.turns_completed,
            "next_turn_pending": not next_state.is_terminal,
            "is_terminal": next_state.is_terminal,
        }
        info.update(next_state.outcome.metadata())
        return next_state, info

    def _parse_reroll_action(
        self,
        *,
        state: YahtzeeState,
        raw_action: str,
        tokens: list[str],
    ) -> ParseResult[YahtzeeAction]:
        """Parse one reroll action.

        Parameters
        ----------
        state : YahtzeeState
            Canonical state used for legality validation.
        raw_action : str
            Original action text.
        tokens : list[str]
            Tokenized normalized action.

        Returns
        -------
        ParseResult[YahtzeeAction]
            Parsed reroll action or a rejection reason.
        """
        position_tokens = tokens[1:]
        if not position_tokens:
            return ParseResult(
                action=None,
                error="Yahtzee reroll actions must list one or more dice positions.",
            )

        if len(position_tokens) == 1 and position_tokens[0] == "all":
            positions = (0, 1, 2, 3, 4)
        else:
            if not all(token.isdigit() for token in position_tokens):
                return ParseResult(
                    action=None,
                    error=(
                        "Yahtzee reroll positions must be one-based integers or "
                        f"'all': {raw_action!r}."
                    ),
                )
            positions = tuple(sorted({int(token) - 1 for token in position_tokens}))
            if len(positions) != len(position_tokens):
                return ParseResult(
                    action=None,
                    error="Yahtzee reroll positions must not repeat dice.",
                )
            if any(position < 0 or position > 4 for position in positions):
                return ParseResult(
                    action=None,
                    error="Yahtzee reroll positions must be in the range 1..5.",
                )

        action = YahtzeeAction(
            kind=YahtzeeActionKind.REROLL,
            reroll_positions=positions,
        )
        if action.label not in state.legal_actions:
            return ParseResult(
                action=None,
                error=(
                    f"Yahtzee action {action.label!r} is illegal for the current turn."
                ),
            )
        return ParseResult(action=action, error=None)

    def _parse_score_action(
        self,
        *,
        state: YahtzeeState,
        raw_action: str,
        category_text: str,
    ) -> ParseResult[YahtzeeAction]:
        """Parse one score action.

        Parameters
        ----------
        state : YahtzeeState
            Canonical state used for legality validation.
        raw_action : str
            Original action text.
        category_text : str
            Trailing category text after the score verb.

        Returns
        -------
        ParseResult[YahtzeeAction]
            Parsed score action or a rejection reason.
        """
        if not category_text:
            return ParseResult(
                action=None,
                error="Yahtzee score actions must specify a category.",
            )

        category = normalize_category_name(raw_name=category_text)
        if category is None:
            return ParseResult(
                action=None,
                error=f"Unknown Yahtzee category: {raw_action!r}.",
            )

        action = YahtzeeAction(kind=YahtzeeActionKind.SCORE, category=category)
        if action.label not in state.legal_actions:
            return ParseResult(
                action=None,
                error=(
                    f"Yahtzee action {action.label!r} is illegal for the current turn."
                ),
            )
        return ParseResult(action=action, error=None)
