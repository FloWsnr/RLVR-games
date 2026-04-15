"""Canonical Minesweeper state types."""

from dataclasses import dataclass, field

from rlvr_games.games.minesweeper.actions import (
    MinesweeperVerb,
    serialize_minesweeper_action,
)
from rlvr_games.games.minesweeper.engine import (
    AdjacentMineBoard,
    BoolBoard,
    MineBoard,
    adjacent_mine_counts,
    count_mines,
    make_boolean_board,
    mine_board_text_rows,
    normalize_mine_board,
)


@dataclass(slots=True, frozen=True)
class MinesweeperOutcome:
    """Terminal outcome summary for a canonical Minesweeper state.

    Attributes
    ----------
    is_terminal : bool
        Whether the episode has ended.
    won : bool
        Whether all safe cells were revealed.
    termination : str | None
        Structured terminal reason. Supported values are ``"mine"`` and
        ``"cleared"``.
    exploded_cell : tuple[int, int] | None
        Zero-based exploded cell coordinates when the episode ended by mine.
    """

    is_terminal: bool
    won: bool
    termination: str | None = None
    exploded_cell: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        """Validate that terminal metadata is internally coherent."""
        if self.is_terminal and self.termination is None:
            raise ValueError(
                "Terminal Minesweeper outcomes require a termination reason."
            )
        if not self.is_terminal:
            if (
                self.won
                or self.termination is not None
                or self.exploded_cell is not None
            ):
                raise ValueError(
                    "Non-terminal Minesweeper outcomes must not include terminal "
                    "metadata."
                )
            return

        if self.termination == "mine":
            if self.won:
                raise ValueError("Mine termination cannot also be a win.")
            if self.exploded_cell is None:
                raise ValueError("Mine termination requires an exploded cell.")
            return

        if self.termination == "cleared":
            if not self.won:
                raise ValueError("Cleared termination requires a winning outcome.")
            if self.exploded_cell is not None:
                raise ValueError("Cleared termination cannot include an exploded cell.")
            return

        raise ValueError("Minesweeper termination must be one of 'mine' or 'cleared'.")

    def metadata(self) -> dict[str, object]:
        """Return dictionary metadata for the outcome."""
        if not self.is_terminal:
            return {}

        metadata: dict[str, object] = {
            "won": self.won,
            "termination": self.termination,
        }
        if self.exploded_cell is not None:
            metadata["exploded_cell"] = {
                "row": self.exploded_cell[0] + 1,
                "col": self.exploded_cell[1] + 1,
                "row_index": self.exploded_cell[0],
                "col_index": self.exploded_cell[1],
            }
        return metadata


@dataclass(init=False, slots=True, frozen=True)
class MinesweeperState:
    """Canonical Minesweeper state.

    Attributes
    ----------
    rows : int
        Board row count.
    columns : int
        Board column count.
    mine_count : int
        Total number of mines in the hidden layout.
    hidden_board : MineBoard | None
        Hidden mine layout. `None` means the random layout has not been placed
        yet and will be generated on the first reveal action.
    revealed : BoolBoard
        Revealed-cell mask.
    flagged : BoolBoard
        Flagged-cell mask.
    move_count : int
        Number of accepted actions already applied.
    placement_seed : int | None
        Seed used for deferred mine placement when `hidden_board` is `None`.
    adjacent_counts : AdjacentMineBoard | None
        Cached adjacent-mine counts when the hidden layout has been resolved.
    legal_actions : tuple[str, ...]
        Canonical legal action labels for the current state.
    outcome : MinesweeperOutcome
        Terminal outcome summary for the current state.
    """

    rows: int
    columns: int
    mine_count: int
    hidden_board: MineBoard | None
    revealed: BoolBoard
    flagged: BoolBoard
    move_count: int
    placement_seed: int | None
    adjacent_counts: AdjacentMineBoard | None = field(init=False)
    legal_actions: tuple[str, ...] = field(init=False)
    revealed_cell_count: int = field(init=False)
    revealed_safe_count: int = field(init=False)
    flagged_cell_count: int = field(init=False)
    hidden_cell_count: int = field(init=False)
    outcome: MinesweeperOutcome = field(init=False)

    def __init__(
        self,
        *,
        rows: int,
        columns: int,
        mine_count: int,
        hidden_board: MineBoard | None,
        revealed: BoolBoard | None = None,
        flagged: BoolBoard | None = None,
        move_count: int,
        placement_seed: int | None,
    ) -> None:
        """Create a canonical Minesweeper state.

        Parameters
        ----------
        rows : int
            Board row count.
        columns : int
            Board column count.
        mine_count : int
            Total mine count for the hidden layout.
        hidden_board : MineBoard | None
            Hidden mine layout or `None` when placement is deferred until the
            first reveal.
        revealed : BoolBoard | None
            Revealed-cell mask. Defaults to all `False`.
        flagged : BoolBoard | None
            Flagged-cell mask. Defaults to all `False`.
        move_count : int
            Number of accepted actions already applied.
        placement_seed : int | None
            Seed used for deferred placement when `hidden_board` is `None`.
        """
        if rows <= 0:
            raise ValueError("Minesweeper boards must have at least one row.")
        if columns <= 0:
            raise ValueError("Minesweeper boards must have at least one column.")
        if mine_count < 0:
            raise ValueError("Minesweeper mine_count must be non-negative.")
        if mine_count >= rows * columns:
            raise ValueError("mine_count must leave at least one safe cell.")
        if move_count < 0:
            raise ValueError("Minesweeper move_count must be non-negative.")

        current_revealed = (
            make_boolean_board(rows=rows, columns=columns, fill=False)
            if revealed is None
            else _normalize_mask(
                mask=revealed,
                rows=rows,
                columns=columns,
                name="revealed",
            )
        )
        current_flagged = (
            make_boolean_board(rows=rows, columns=columns, fill=False)
            if flagged is None
            else _normalize_mask(
                mask=flagged,
                rows=rows,
                columns=columns,
                name="flagged",
            )
        )
        for row_index in range(rows):
            for col_index in range(columns):
                if (
                    current_revealed[row_index][col_index]
                    and current_flagged[row_index][col_index]
                ):
                    raise ValueError(
                        "Minesweeper cells cannot be both revealed and flagged."
                    )

        current_hidden_board: MineBoard | None = None
        current_adjacent_counts: AdjacentMineBoard | None = None
        if hidden_board is None:
            if placement_seed is None:
                raise ValueError(
                    "Deferred Minesweeper layouts require a placement_seed."
                )
            if any(cell for row in current_revealed for cell in row):
                raise ValueError(
                    "Deferred Minesweeper layouts cannot already have revealed cells."
                )
        else:
            current_hidden_board = normalize_mine_board(board=hidden_board)
            if len(current_hidden_board) != rows:
                raise ValueError("Minesweeper hidden_board row count must match rows.")
            if len(current_hidden_board[0]) != columns:
                raise ValueError(
                    "Minesweeper hidden_board column count must match columns."
                )
            if count_mines(board=current_hidden_board) != mine_count:
                raise ValueError(
                    "Minesweeper hidden_board mine count must match mine_count."
                )
            if placement_seed is not None:
                raise ValueError(
                    "Resolved Minesweeper layouts must not keep a placement_seed."
                )
            current_adjacent_counts = adjacent_mine_counts(board=current_hidden_board)

        revealed_cell_count = sum(1 for row in current_revealed for cell in row if cell)
        flagged_cell_count = sum(1 for row in current_flagged for cell in row if cell)
        hidden_cell_count = (rows * columns) - revealed_cell_count

        exploded_cell: tuple[int, int] | None = None
        revealed_safe_count = 0
        if current_hidden_board is not None:
            for row_index in range(rows):
                for col_index in range(columns):
                    if not current_revealed[row_index][col_index]:
                        continue
                    if current_hidden_board[row_index][col_index]:
                        if exploded_cell is not None:
                            raise ValueError(
                                "Minesweeper states may reveal at most one mine."
                            )
                        exploded_cell = (row_index, col_index)
                    else:
                        revealed_safe_count += 1

        safe_cell_count = (rows * columns) - mine_count
        if exploded_cell is not None:
            outcome = MinesweeperOutcome(
                is_terminal=True,
                won=False,
                termination="mine",
                exploded_cell=exploded_cell,
            )
        elif (
            current_hidden_board is not None and revealed_safe_count == safe_cell_count
        ):
            outcome = MinesweeperOutcome(
                is_terminal=True,
                won=True,
                termination="cleared",
                exploded_cell=None,
            )
        else:
            outcome = MinesweeperOutcome(
                is_terminal=False,
                won=False,
                termination=None,
                exploded_cell=None,
            )

        legal_actions = ()
        if not outcome.is_terminal:
            legal_actions = _build_legal_actions(
                rows=rows,
                columns=columns,
                revealed=current_revealed,
                flagged=current_flagged,
            )

        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "mine_count", mine_count)
        object.__setattr__(self, "hidden_board", current_hidden_board)
        object.__setattr__(self, "revealed", current_revealed)
        object.__setattr__(self, "flagged", current_flagged)
        object.__setattr__(self, "move_count", move_count)
        object.__setattr__(self, "placement_seed", placement_seed)
        object.__setattr__(self, "adjacent_counts", current_adjacent_counts)
        object.__setattr__(self, "legal_actions", legal_actions)
        object.__setattr__(self, "revealed_cell_count", revealed_cell_count)
        object.__setattr__(self, "revealed_safe_count", revealed_safe_count)
        object.__setattr__(self, "flagged_cell_count", flagged_cell_count)
        object.__setattr__(self, "hidden_cell_count", hidden_cell_count)
        object.__setattr__(self, "outcome", outcome)

    @property
    def has_pending_mines(self) -> bool:
        """Return whether the hidden mine layout is still deferred."""
        return self.hidden_board is None

    @property
    def legal_action_count(self) -> int:
        """Return the number of legal actions."""
        return len(self.legal_actions)

    @property
    def remaining_safe_cells(self) -> int:
        """Return the number of safe cells that remain unrevealed."""
        return (self.rows * self.columns) - self.mine_count - self.revealed_safe_count

    @property
    def is_terminal(self) -> bool:
        """Return whether the current state ends the episode."""
        return self.outcome.is_terminal


def public_minesweeper_board(state: MinesweeperState) -> tuple[tuple[str, ...], ...]:
    """Return the public-facing board view for a Minesweeper state.

    Parameters
    ----------
    state : MinesweeperState
        Canonical state to render.

    Returns
    -------
    tuple[tuple[str, ...], ...]
        Public cell labels that do not reveal hidden mines mid-episode.
    """
    public_rows: list[tuple[str, ...]] = []
    for row_index in range(state.rows):
        public_row: list[str] = []
        for col_index in range(state.columns):
            if state.revealed[row_index][col_index]:
                if (
                    state.hidden_board is not None
                    and state.hidden_board[row_index][col_index]
                ):
                    public_row.append("*")
                    continue

                if state.adjacent_counts is None:
                    raise ValueError(
                        "Resolved Minesweeper counts are required for revealed cells."
                    )
                adjacent_count = state.adjacent_counts[row_index][col_index]
                public_row.append("." if adjacent_count == 0 else str(adjacent_count))
                continue

            if state.flagged[row_index][col_index]:
                public_row.append("F")
                continue

            if (
                state.is_terminal
                and state.hidden_board is not None
                and state.hidden_board[row_index][col_index]
            ):
                public_row.append("M")
                continue

            public_row.append("#")
        public_rows.append(tuple(public_row))
    return tuple(public_rows)


def public_minesweeper_metadata(state: MinesweeperState) -> dict[str, object]:
    """Return model-safe observation metadata for a Minesweeper state.

    Parameters
    ----------
    state : MinesweeperState
        Canonical state to summarize.

    Returns
    -------
    dict[str, object]
        Observation metadata that excludes hidden mine positions.
    """
    metadata: dict[str, object] = {
        "rows": state.rows,
        "columns": state.columns,
        "mine_count": state.mine_count,
        "move_count": state.move_count,
        "visible_board": public_minesweeper_board(state=state),
        "revealed_cell_count": state.revealed_cell_count,
        "revealed_safe_count": state.revealed_safe_count,
        "flagged_cell_count": state.flagged_cell_count,
        "hidden_cell_count": state.hidden_cell_count,
        "remaining_safe_cells": state.remaining_safe_cells,
        "legal_action_count": state.legal_action_count,
        "pending_mine_layout": state.has_pending_mines,
        "is_terminal": state.is_terminal,
        "won": state.outcome.won,
    }
    metadata.update(state.outcome.metadata())
    return metadata


def inspect_minesweeper_state(state: MinesweeperState) -> dict[str, object]:
    """Return a debug-oriented Minesweeper state summary.

    Parameters
    ----------
    state : MinesweeperState
        Canonical state to inspect.

    Returns
    -------
    dict[str, object]
        Debug summary that includes the hidden mine layout when available.
    """
    metadata = public_minesweeper_metadata(state=state)
    metadata.update(
        {
            "hidden_board": (
                None
                if state.hidden_board is None
                else mine_board_text_rows(board=state.hidden_board)
            ),
            "adjacent_counts": state.adjacent_counts,
            "revealed": state.revealed,
            "flagged": state.flagged,
            "placement_seed": state.placement_seed,
        }
    )
    return metadata


def _normalize_mask(
    *,
    mask: BoolBoard,
    rows: int,
    columns: int,
    name: str,
) -> BoolBoard:
    """Validate and normalize a boolean mask."""
    if len(mask) != rows:
        raise ValueError(f"Minesweeper {name} mask row count must match rows.")
    normalized_rows: list[tuple[bool, ...]] = []
    for row in mask:
        if len(row) != columns:
            raise ValueError(
                f"Minesweeper {name} mask column count must match columns."
            )
        normalized_row: list[bool] = []
        for cell in row:
            if not isinstance(cell, bool):
                raise ValueError(f"Minesweeper {name} mask cells must be bool.")
            normalized_row.append(cell)
        normalized_rows.append(tuple(normalized_row))
    return tuple(normalized_rows)


def _build_legal_actions(
    *,
    rows: int,
    columns: int,
    revealed: BoolBoard,
    flagged: BoolBoard,
) -> tuple[str, ...]:
    """Return the canonical legal action list for a non-terminal state."""
    reveal_actions: list[str] = []
    flag_actions: list[str] = []
    unflag_actions: list[str] = []
    for row_index in range(rows):
        for col_index in range(columns):
            if revealed[row_index][col_index]:
                continue
            if flagged[row_index][col_index]:
                unflag_actions.append(
                    serialize_minesweeper_action(
                        verb=MinesweeperVerb.UNFLAG,
                        row=row_index,
                        col=col_index,
                    )
                )
                continue
            reveal_actions.append(
                serialize_minesweeper_action(
                    verb=MinesweeperVerb.REVEAL,
                    row=row_index,
                    col=col_index,
                )
            )
            flag_actions.append(
                serialize_minesweeper_action(
                    verb=MinesweeperVerb.FLAG,
                    row=row_index,
                    col=col_index,
                )
            )
    return tuple(reveal_actions + flag_actions + unflag_actions)
