"""Rule-verified backend for Connect 4."""

from rlvr_games.core.exceptions import InvalidActionError
from rlvr_games.core.types import ParseResult
from rlvr_games.games.connect4.actions import Connect4Action
from rlvr_games.games.connect4.state import Connect4State, drop_piece


class Connect4Backend:
    """Authoritative verifier for Connect 4 parsing and transitions."""

    def parse_action(
        self,
        state: Connect4State,
        raw_action: str,
    ) -> ParseResult[Connect4Action]:
        """Parse and validate a raw model action as a legal Connect 4 move.

        Parameters
        ----------
        state : Connect4State
            Current canonical Connect 4 state.
        raw_action : str
            Model-produced one-based column label.

        Returns
        -------
        ParseResult[Connect4Action]
            Structured parse result containing either a canonical legal action
            or an explicit rejection message.
        """
        normalized_action = raw_action.strip().lower()
        if not normalized_action:
            return ParseResult(
                action=None,
                error="Connect 4 actions must be a non-empty column label.",
            )

        try:
            human_column = int(normalized_action)
        except ValueError:
            return ParseResult(
                action=None,
                error=(
                    "Connect 4 actions must be integer column labels such as "
                    f"'1' or '{state.columns}': {raw_action!r}."
                ),
            )

        if human_column < 1 or human_column > state.columns:
            return ParseResult(
                action=None,
                error=(
                    "Connect 4 columns must be within the current board width: "
                    f"{raw_action!r}."
                ),
            )

        action = Connect4Action(column=human_column - 1)
        if action.label not in state.legal_actions:
            return ParseResult(
                action=None,
                error=(
                    f"Connect 4 column {action.label!r} is illegal for the current "
                    "board."
                ),
            )

        return ParseResult(action=action, error=None)

    def legal_actions(self, state: Connect4State) -> list[str]:
        """Enumerate legal model-facing actions for the current position.

        Parameters
        ----------
        state : Connect4State
            Canonical Connect 4 state to inspect.

        Returns
        -------
        list[str]
            Legal one-based column labels in canonical left-to-right order.
        """
        return list(state.legal_actions)

    def apply_action(
        self,
        state: Connect4State,
        action: Connect4Action,
    ) -> tuple[Connect4State, dict[str, object]]:
        """Apply a verified Connect 4 move and return the resulting transition.

        Parameters
        ----------
        state : Connect4State
            Canonical state before the move.
        action : Connect4Action
            Parsed column action to apply.

        Returns
        -------
        tuple[Connect4State, dict[str, object]]
            Updated canonical state and verifier metadata describing the move,
            placement coordinate, and any terminal outcome.

        Raises
        ------
        InvalidActionError
            If the supplied column is not legal for the state.
        """
        if action.label not in state.legal_actions:
            raise InvalidActionError(
                f"Connect 4 column {action.label!r} is illegal for the current board."
            )

        next_board, placed_coordinate = drop_piece(
            board=state.board,
            column=action.column,
            player=state.current_player,
        )
        next_state = Connect4State(
            board=next_board,
            connect_length=state.connect_length,
        )

        row_index, column_index = placed_coordinate
        transition_info: dict[str, object] = {
            "player": state.current_player,
            "column": action.column + 1,
            "column_index": column_index,
            "row_index": row_index,
            "row_from_bottom": state.rows - row_index,
            "move_count": next_state.move_count,
            "is_terminal": next_state.is_terminal,
        }
        transition_info.update(next_state.outcome.metadata())
        return next_state, transition_info

    def is_terminal(self, state: Connect4State) -> bool:
        """Return whether the supplied Connect 4 state ends the episode.

        Parameters
        ----------
        state : Connect4State
            Canonical Connect 4 state to inspect.

        Returns
        -------
        bool
            `True` when the position is a win or draw.
        """
        return state.is_terminal
