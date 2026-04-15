"""Rule-verified backend for Minesweeper."""

from rlvr_games.core.exceptions import InvalidActionError
from rlvr_games.core.types import ParseResult
from rlvr_games.games.minesweeper.actions import (
    MinesweeperAction,
    MinesweeperVerb,
    VERB_ALIASES,
)
from rlvr_games.games.minesweeper.engine import (
    adjacent_mine_counts,
    place_random_mines,
    reveal_cells,
)
from rlvr_games.games.minesweeper.state import MinesweeperState


class MinesweeperBackend:
    """Authoritative verifier for Minesweeper parsing and transitions."""

    def parse_action(
        self,
        state: MinesweeperState,
        raw_action: str,
    ) -> ParseResult[MinesweeperAction]:
        """Parse and validate a raw model action as a legal Minesweeper move.

        Parameters
        ----------
        state : MinesweeperState
            Current canonical Minesweeper state.
        raw_action : str
            Model-produced action string.

        Returns
        -------
        ParseResult[MinesweeperAction]
            Structured parse result containing either a canonical legal action
            or an explicit rejection message.
        """
        normalized = raw_action.strip().lower()
        if not normalized:
            return ParseResult(
                action=None,
                error=(
                    "Minesweeper actions must be 'reveal <row> <col>', "
                    "'flag <row> <col>', or 'unflag <row> <col>'."
                ),
            )

        tokens = normalized.replace(",", " ").split()
        if len(tokens) == 2 and all(token.isdigit() for token in tokens):
            verb = MinesweeperVerb.REVEAL
            coordinate_tokens = tokens
        elif len(tokens) == 3:
            verb = VERB_ALIASES.get(tokens[0])
            coordinate_tokens = tokens[1:]
            if verb is None:
                return ParseResult(
                    action=None,
                    error=(
                        "Unknown Minesweeper action verb. Use reveal, flag, or "
                        f"unflag: {raw_action!r}."
                    ),
                )
        else:
            return ParseResult(
                action=None,
                error=(
                    "Minesweeper actions must use either '<row> <col>' or "
                    "'<verb> <row> <col>'."
                ),
            )

        if not all(token.isdigit() for token in coordinate_tokens):
            return ParseResult(
                action=None,
                error=(
                    "Minesweeper coordinates must be one-based positive integers: "
                    f"{raw_action!r}."
                ),
            )

        row = int(coordinate_tokens[0]) - 1
        col = int(coordinate_tokens[1]) - 1
        if not (0 <= row < state.rows and 0 <= col < state.columns):
            return ParseResult(
                action=None,
                error=(
                    "Minesweeper coordinates are outside the current board: "
                    f"{raw_action!r} for a {state.rows}x{state.columns} board."
                ),
            )

        action = MinesweeperAction(verb=verb, row=row, col=col)
        if action.label not in state.legal_actions:
            legal_actions_text = ", ".join(state.legal_actions[:12])
            if len(state.legal_actions) > 12:
                legal_actions_text += ", ..."
            if not legal_actions_text:
                legal_actions_text = "none"
            return ParseResult(
                action=None,
                error=(
                    f"Minesweeper action {action.label!r} is illegal for the current "
                    f"board. Legal actions: {legal_actions_text}."
                ),
            )

        return ParseResult(action=action, error=None)

    def legal_actions(self, state: MinesweeperState) -> list[str]:
        """Enumerate legal model-facing actions for the current board."""
        return list(state.legal_actions)

    def apply_action(
        self,
        state: MinesweeperState,
        action: MinesweeperAction,
    ) -> tuple[MinesweeperState, dict[str, object]]:
        """Apply a verified Minesweeper action and return the next state.

        Parameters
        ----------
        state : MinesweeperState
            Canonical state before the transition.
        action : MinesweeperAction
            Parsed action to apply.

        Returns
        -------
        tuple[MinesweeperState, dict[str, object]]
            Updated canonical state and transition metadata.
        """
        if action.label not in state.legal_actions:
            raise InvalidActionError(
                f"Minesweeper action {action.label!r} is illegal for the current board."
            )

        generated_mines = False
        hidden_board = state.hidden_board
        adjacent_counts = state.adjacent_counts
        if hidden_board is None:
            if state.placement_seed is None:
                raise ValueError(
                    "Deferred Minesweeper layouts require a placement_seed."
                )
            if action.verb == MinesweeperVerb.REVEAL:
                hidden_board = place_random_mines(
                    rows=state.rows,
                    columns=state.columns,
                    mine_count=state.mine_count,
                    safe_cell=(action.row, action.col),
                    seed=state.placement_seed,
                )
                adjacent_counts = adjacent_mine_counts(board=hidden_board)
                generated_mines = True

        if action.verb == MinesweeperVerb.REVEAL:
            if hidden_board is None or adjacent_counts is None:
                raise ValueError(
                    "Reveal actions require a resolved Minesweeper hidden board."
                )
            next_revealed, newly_revealed = reveal_cells(
                board=hidden_board,
                counts=adjacent_counts,
                revealed=state.revealed,
                flagged=state.flagged,
                start=(action.row, action.col),
            )
            next_state = MinesweeperState(
                rows=state.rows,
                columns=state.columns,
                mine_count=state.mine_count,
                hidden_board=hidden_board,
                revealed=next_revealed,
                flagged=state.flagged,
                move_count=state.move_count + 1,
                placement_seed=None,
            )
            hit_mine = bool(hidden_board[action.row][action.col])
            info: dict[str, object] = {
                "verb": action.verb.value,
                "row": action.row + 1,
                "col": action.col + 1,
                "row_index": action.row,
                "col_index": action.col,
                "generated_mines": generated_mines,
                "newly_revealed_count": len(newly_revealed),
                "newly_revealed": tuple(
                    {
                        "row": row_index + 1,
                        "col": col_index + 1,
                        "row_index": row_index,
                        "col_index": col_index,
                        "adjacent_mines": adjacent_counts[row_index][col_index],
                    }
                    for row_index, col_index in newly_revealed
                ),
                "adjacent_mines": adjacent_counts[action.row][action.col],
                "exploded": hit_mine,
                "move_count": next_state.move_count,
                "revealed_cell_count": next_state.revealed_cell_count,
                "flagged_cell_count": next_state.flagged_cell_count,
                "remaining_safe_cells": next_state.remaining_safe_cells,
                "pending_mine_layout": next_state.has_pending_mines,
                "is_terminal": next_state.is_terminal,
            }
            info.update(next_state.outcome.metadata())
            return next_state, info

        next_flagged = [list(row) for row in state.flagged]
        next_flagged[action.row][action.col] = action.verb == MinesweeperVerb.FLAG
        next_state = MinesweeperState(
            rows=state.rows,
            columns=state.columns,
            mine_count=state.mine_count,
            hidden_board=hidden_board,
            revealed=state.revealed,
            flagged=tuple(tuple(row) for row in next_flagged),
            move_count=state.move_count + 1,
            placement_seed=state.placement_seed,
        )
        info = {
            "verb": action.verb.value,
            "row": action.row + 1,
            "col": action.col + 1,
            "row_index": action.row,
            "col_index": action.col,
            "generated_mines": False,
            "newly_revealed_count": 0,
            "move_count": next_state.move_count,
            "revealed_cell_count": next_state.revealed_cell_count,
            "flagged_cell_count": next_state.flagged_cell_count,
            "remaining_safe_cells": next_state.remaining_safe_cells,
            "pending_mine_layout": next_state.has_pending_mines,
            "is_terminal": next_state.is_terminal,
        }
        info.update(next_state.outcome.metadata())
        return next_state, info

    def is_terminal(self, state: MinesweeperState) -> bool:
        """Return whether the supplied Minesweeper state ends the episode."""
        return state.is_terminal
