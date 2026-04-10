"""Rule-verified backend for 2048."""

from random import Random

from rlvr_games.core.exceptions import InvalidActionError
from rlvr_games.core.types import ParseResult
from rlvr_games.games.game2048.actions import Game2048Action, MoveDirection
from rlvr_games.games.game2048.engine import MergeSummary, apply_move, spawn_random_tile
from rlvr_games.games.game2048.state import Game2048State

_ACTION_ALIASES = {
    "u": MoveDirection.UP,
    "up": MoveDirection.UP,
    "r": MoveDirection.RIGHT,
    "right": MoveDirection.RIGHT,
    "d": MoveDirection.DOWN,
    "down": MoveDirection.DOWN,
    "l": MoveDirection.LEFT,
    "left": MoveDirection.LEFT,
}


class Game2048Backend:
    """Authoritative verifier for 2048 action parsing and transitions."""

    def parse_action(
        self,
        state: Game2048State,
        raw_action: str,
    ) -> ParseResult[Game2048Action]:
        """Parse and validate a raw model action as a legal 2048 move.

        Parameters
        ----------
        state : Game2048State
            Current canonical 2048 state.
        raw_action : str
            Model-produced direction string.

        Returns
        -------
        ParseResult[Game2048Action]
            Structured parse result containing either a canonical legal move or
            an explicit rejection message.
        """
        normalized_action = raw_action.strip().lower()
        if not normalized_action:
            return ParseResult(
                action=None,
                error="2048 actions must be a non-empty direction string.",
            )

        direction = _ACTION_ALIASES.get(normalized_action)
        if direction is None:
            legal_directions = ", ".join(direction.value for direction in MoveDirection)
            return ParseResult(
                action=None,
                error=(
                    f"2048 actions must be one of {legal_directions}: {raw_action!r}."
                ),
            )

        if direction.value not in state.legal_actions:
            legal_actions_text = ", ".join(state.legal_actions) or "none"
            return ParseResult(
                action=None,
                error=(
                    f"2048 direction {direction.value!r} is illegal for the current "
                    f"board. Legal actions: {legal_actions_text}."
                ),
            )

        return ParseResult(action=Game2048Action(direction=direction), error=None)

    def legal_actions(self, state: Game2048State) -> list[str]:
        """Enumerate legal model-facing actions for the current board.

        Parameters
        ----------
        state : Game2048State
            Canonical 2048 state to inspect.

        Returns
        -------
        list[str]
            Legal direction labels in canonical order.
        """
        return list(state.legal_actions)

    def apply_action(
        self,
        state: Game2048State,
        action: Game2048Action,
    ) -> tuple[Game2048State, dict[str, object]]:
        """Apply a verified 2048 move and return the resulting transition.

        Parameters
        ----------
        state : Game2048State
            Canonical state before the move.
        action : Game2048Action
            Parsed direction to apply.

        Returns
        -------
        tuple[Game2048State, dict[str, object]]
            Updated canonical state and verifier metadata describing the move,
            merge score, spawned tile, and terminal outcome.

        Raises
        ------
        InvalidActionError
            If the supplied direction is not legal for the state.
        """
        if action.label not in state.legal_actions:
            legal_actions_text = ", ".join(state.legal_actions) or "none"
            raise InvalidActionError(
                f"2048 direction {action.label!r} is illegal for the current board. "
                f"Legal actions: {legal_actions_text}."
            )

        move_summary = apply_move(board=state.board, direction=action.direction)
        if not move_summary.moved:
            raise InvalidActionError(
                f"2048 direction {action.label!r} does not change the current board."
            )

        rng = Random()
        rng.setstate(state.rng_state)
        spawned_board, spawned_tile = spawn_random_tile(
            board=move_summary.board, rng=rng
        )
        next_state = Game2048State(
            board=spawned_board,
            score=state.score + move_summary.score_gain,
            move_count=state.move_count + 1,
            target_value=state.target_value,
            rng_state=rng.getstate(),
        )

        transition_info: dict[str, object] = {
            "direction": action.label,
            "board": next_state.board,
            "score": next_state.score,
            "score_gain": move_summary.score_gain,
            "move_count": next_state.move_count,
            "spawned_tile": {
                "row": spawned_tile.row,
                "col": spawned_tile.col,
                "value": spawned_tile.value,
            },
            "merge_count": len(move_summary.merges),
            "merges": tuple(
                _merge_metadata(merge=merge) for merge in move_summary.merges
            ),
            "max_tile": next_state.max_tile,
            "empty_cell_count": next_state.empty_cell_count,
            "legal_action_count": next_state.legal_action_count,
            "is_terminal": next_state.is_terminal,
            "won": next_state.outcome.won if next_state.is_terminal else False,
            "target_value": next_state.target_value,
        }
        transition_info.update(next_state.outcome.metadata())
        return next_state, transition_info

    def is_terminal(self, state: Game2048State) -> bool:
        """Return whether the supplied 2048 state ends the episode.

        Parameters
        ----------
        state : Game2048State
            Canonical 2048 state to inspect.

        Returns
        -------
        bool
            `True` when the target tile has been reached or no moves remain.
        """
        return state.is_terminal


def _merge_metadata(*, merge: MergeSummary) -> dict[str, object]:
    """Serialize merge metadata for trajectory-safe storage.

    Parameters
    ----------
    merge : MergeSummary
        Merge event to serialize.

    Returns
    -------
    dict[str, object]
        Dictionary containing destination coordinates, merged value, and the
        two source coordinates.
    """
    return {
        "row": merge.row,
        "col": merge.col,
        "value": merge.value,
        "sources": merge.sources,
    }
