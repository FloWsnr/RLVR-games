"""Automatic turn-resolution helpers for chess environments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

import chess.engine

from rlvr_games.core.protocol import AutoAdvancePolicy, GameBackend
from rlvr_games.core.types import AutoAction, EpisodeBoundary
from rlvr_games.games.chess.actions import ChessAction
from rlvr_games.games.chess.rewards import (
    ChessPerspective,
    puzzle_solution_moves_uci,
    puzzle_solution_progress_index,
)
from rlvr_games.games.chess.state import ChessState
from rlvr_games.games.chess.stockfish_runtime import (
    resolve_stockfish_binary_path,
    validate_stockfish_binary_path,
)


class ChessMoveSelector(Protocol):
    """Protocol for selecting one automatic chess move."""

    def select_action(
        self,
        *,
        state: ChessState,
        backend: GameBackend[ChessState, ChessAction],
    ) -> AutoAction[ChessAction]:
        """Return one selected automatic move for the supplied state."""
        ...


@dataclass(slots=True)
class UciEngineMoveSelector:
    """Select chess moves from a local UCI-compatible engine.

    Attributes
    ----------
    engine_path : Path
        Filesystem path to the UCI engine binary.
    depth : int
        Fixed search depth used for each move selection.
    source : str
        Transition source label stored in auto-advanced trajectory metadata.
    """

    engine_path: Path
    depth: int
    source: str = "opponent"
    _engine: chess.engine.SimpleEngine = field(init=False, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        """Start the backing engine process and validate configuration.

        Raises
        ------
        ValueError
            If `depth` is not strictly positive.
        FileNotFoundError
            If `engine_path` does not exist.
        """
        if self.depth < 1:
            raise ValueError("UciEngineMoveSelector depth must be >= 1.")
        self.engine_path = self.engine_path.expanduser().resolve()
        if not self.engine_path.exists():
            raise FileNotFoundError(
                f"UCI engine binary does not exist: {self.engine_path}"
            )
        self._engine = chess.engine.SimpleEngine.popen_uci(str(self.engine_path))

    def select_action(
        self,
        *,
        state: ChessState,
        backend: GameBackend[ChessState, ChessAction],
    ) -> AutoAction[ChessAction]:
        """Return one engine-selected legal move for `state`.

        Parameters
        ----------
        state : ChessState
            Canonical chess state to analyse.
        backend : GameBackend[ChessState, ChessAction]
            Backend used to validate the selected UCI move.

        Returns
        -------
        AutoAction[ChessAction]
            Parsed engine-selected move ready to be auto-applied.

        Raises
        ------
        ValueError
            If the engine does not return a legal move for the supplied state.
        """
        result = self._engine.play(state.board, chess.engine.Limit(depth=self.depth))
        move = result.move
        if move is None or move not in state.board.legal_moves:
            raise ValueError("UCI engine did not return a legal move.")

        raw_action = move.uci()
        parse_result = backend.parse_action(state, raw_action)
        if parse_result.error is not None:
            raise ValueError(
                "UCI engine produced a move that backend parsing rejected: "
                f"{raw_action}."
            )
        return AutoAction(
            source=self.source,
            raw_action=raw_action,
            action=parse_result.require_action(),
        )

    def close(self) -> None:
        """Stop the backing engine process.

        Returns
        -------
        None
            This method is idempotent.
        """
        if self._closed:
            return
        self._engine.quit()
        self._closed = True


class StockfishMoveSelector(UciEngineMoveSelector):
    """Concrete Stockfish-backed move selector with binary resolution helpers."""

    @classmethod
    def from_engine_path(
        cls,
        *,
        engine_path: Path,
        depth: int,
    ) -> "StockfishMoveSelector":
        """Construct a move selector from one explicit Stockfish path."""
        return cls(
            engine_path=validate_stockfish_binary_path(engine_path=engine_path),
            depth=depth,
        )

    @classmethod
    def from_installed_binary(
        cls,
        *,
        depth: int,
    ) -> "StockfishMoveSelector":
        """Construct a move selector from the configured local Stockfish."""
        return cls(
            engine_path=resolve_stockfish_binary_path(),
            depth=depth,
        )


@dataclass(slots=True)
class ChessEngineAutoAdvancePolicy(AutoAdvancePolicy[ChessState, ChessAction]):
    """Auto-advance chess turns by letting an engine play the opponent moves."""

    move_selector: ChessMoveSelector
    _agent_side: ChessPerspective = field(init=False, default="white", repr=False)

    def reset(self, *, initial_state: ChessState) -> None:
        """Record which side is controlled by the agent for this episode."""
        self._agent_side = cast(ChessPerspective, initial_state.side_to_move)

    def is_agent_turn(self, *, state: ChessState) -> bool:
        """Return whether `state` is controlled by the agent."""
        return state.side_to_move == self._agent_side

    def select_internal_action(
        self,
        *,
        state: ChessState,
        backend: GameBackend[ChessState, ChessAction],
    ) -> AutoAction[ChessAction] | None:
        """Return one engine-selected opponent move."""
        return self.move_selector.select_action(state=state, backend=backend)

    def episode_boundary(self, *, state: ChessState) -> EpisodeBoundary | None:
        """Return no additional task boundary for engine-played games."""
        del state
        return None

    def close(self) -> None:
        """Close the underlying move selector when it owns resources."""
        close_method = getattr(self.move_selector, "close", None)
        if callable(close_method):
            close_method()


@dataclass(slots=True)
class ChessPuzzleAutoAdvancePolicy(AutoAdvancePolicy[ChessState, ChessAction]):
    """Auto-advance puzzle turns by replaying the canonical solution replies."""

    _agent_side: ChessPerspective = field(init=False, default="white", repr=False)

    def reset(self, *, initial_state: ChessState) -> None:
        """Record the puzzle side controlled by the agent."""
        self._agent_side = cast(ChessPerspective, initial_state.side_to_move)

    def is_agent_turn(self, *, state: ChessState) -> bool:
        """Return whether `state` is controlled by the agent."""
        return state.side_to_move == self._agent_side

    def select_internal_action(
        self,
        *,
        state: ChessState,
        backend: GameBackend[ChessState, ChessAction],
    ) -> AutoAction[ChessAction] | None:
        """Return the next canonical puzzle reply when one exists."""
        progress_index = puzzle_solution_progress_index(state=state)
        if progress_index is None:
            return None

        solution_moves_uci = puzzle_solution_moves_uci(state=state)
        if progress_index >= len(solution_moves_uci):
            return None

        raw_action = solution_moves_uci[progress_index]
        parse_result = backend.parse_action(state, raw_action)
        if parse_result.error is not None:
            raise ValueError(
                f"Puzzle solution move was rejected by backend parsing: {raw_action}."
            )
        return AutoAction(
            source="opponent",
            raw_action=raw_action,
            action=parse_result.require_action(),
        )

    def episode_boundary(self, *, state: ChessState) -> EpisodeBoundary | None:
        """Return puzzle-completion or off-path termination when applicable."""
        progress_index = puzzle_solution_progress_index(state=state)
        if progress_index is None:
            return EpisodeBoundary(
                terminated=True,
                truncated=False,
                info={"episode_completion_reason": "puzzle_off_path"},
            )

        solution_moves_uci = puzzle_solution_moves_uci(state=state)
        if progress_index >= len(solution_moves_uci):
            return EpisodeBoundary(
                terminated=True,
                truncated=False,
                info={"episode_completion_reason": "puzzle_solution_complete"},
            )
        return None


__all__ = [
    "ChessEngineAutoAdvancePolicy",
    "ChessMoveSelector",
    "ChessPuzzleAutoAdvancePolicy",
    "StockfishMoveSelector",
    "UciEngineMoveSelector",
]
