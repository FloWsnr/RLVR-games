"""Reward helpers for chess."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, cast

import chess
import chess.engine

from rlvr_games.games.chess.actions import ChessAction
from rlvr_games.games.chess.state import ChessState

ChessPerspective = Literal["white", "black"]
ChessRewardPerspective = Literal["white", "black", "mover"]


class ChessStateEvaluator(Protocol):
    """Protocol for scalar evaluation of chess positions."""

    def evaluate(
        self,
        *,
        state: ChessState,
        perspective: ChessPerspective,
    ) -> float:
        """Return a scalar evaluation for `state` from `perspective`.

        Parameters
        ----------
        state : ChessState
            Canonical chess state to evaluate.
        perspective : {"white", "black"}
            Side whose point of view determines the sign of the evaluation.

        Returns
        -------
        float
            Scalar position value where larger numbers are better for
            `perspective`.
        """
        ...


def resolve_reward_perspective(
    *,
    previous_state: ChessState,
    perspective: ChessRewardPerspective,
) -> ChessPerspective:
    """Resolve a reward perspective into a concrete chess side label.

    Parameters
    ----------
    previous_state : ChessState
        Canonical state before the accepted move.
    perspective : {"white", "black", "mover"}
        Configured reward perspective.

    Returns
    -------
    {"white", "black"}
        Concrete side label used for evaluator queries.
    """
    if perspective == "mover":
        return cast(ChessPerspective, previous_state.side_to_move)
    return perspective


@dataclass(slots=True)
class UciEngineEvaluator:
    """Evaluate positions with a local UCI-compatible chess engine.

    Attributes
    ----------
    engine_path : Path
        Filesystem path to the UCI engine binary.
    depth : int
        Fixed search depth used for each evaluation.
    mate_score : int
        Centipawn-equivalent value used to map mate scores onto a finite scale.
    """

    engine_path: Path
    depth: int
    mate_score: int
    _engine: chess.engine.SimpleEngine = field(init=False, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)
    _cache: dict[tuple[str, ChessPerspective], float] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Start the backing engine process and validate configuration.

        Raises
        ------
        ValueError
            If `depth` or `mate_score` is not positive.
        FileNotFoundError
            If `engine_path` does not exist.
        """
        if self.depth < 1:
            raise ValueError("UciEngineEvaluator depth must be >= 1.")
        if self.mate_score < 1:
            raise ValueError("UciEngineEvaluator mate_score must be >= 1.")
        self.engine_path = self.engine_path.expanduser().resolve()
        if not self.engine_path.exists():
            raise FileNotFoundError(
                f"UCI engine binary does not exist: {self.engine_path}"
            )
        self._engine = chess.engine.SimpleEngine.popen_uci(str(self.engine_path))
        self._cache = {}

    def evaluate(
        self,
        *,
        state: ChessState,
        perspective: ChessPerspective,
    ) -> float:
        """Return a cached engine evaluation for one canonical state.

        Parameters
        ----------
        state : ChessState
            Canonical chess state to evaluate.
        perspective : {"white", "black"}
            Side whose point of view determines the sign of the evaluation.

        Returns
        -------
        float
            Engine score mapped onto a finite centipawn-like scale.

        Raises
        ------
        ValueError
            If the engine does not report a score for the analysed position.
        """
        cache_key = (state.fen, perspective)
        cached_evaluation = self._cache.get(cache_key)
        if cached_evaluation is not None:
            return cached_evaluation

        perspective_color = chess.WHITE
        if perspective == "black":
            perspective_color = chess.BLACK

        info = self._engine.analyse(state.board, chess.engine.Limit(depth=self.depth))
        score = info.get("score")
        if score is None:
            raise ValueError("UCI engine analysis did not return a score.")

        evaluation = score.pov(perspective_color).score(mate_score=self.mate_score)
        if evaluation is None:
            raise ValueError("UCI engine score could not be converted to a scalar.")

        scalar_evaluation = float(evaluation)
        self._cache[cache_key] = scalar_evaluation
        return scalar_evaluation

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

    def __enter__(self) -> "UciEngineEvaluator":
        """Return the evaluator for context-manager use."""
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        """Close the backing engine when leaving a context manager."""
        del exc_type
        del exc
        del traceback
        self.close()


@dataclass(slots=True, frozen=True)
class TerminalOutcomeReward:
    """Sparse reward based on the terminal game outcome for one side.

    Attributes
    ----------
    perspective : {"white", "black"}
        Side whose outcome should determine the reward sign.
    win_reward : float
        Reward returned when `perspective` wins the game.
    draw_reward : float
        Reward returned for terminal drawn positions.
    loss_reward : float
        Reward returned when `perspective` loses the game.
    """

    perspective: ChessPerspective
    win_reward: float
    draw_reward: float
    loss_reward: float

    def __post_init__(self) -> None:
        """Validate that the configured side label is supported.

        Raises
        ------
        ValueError
            If `perspective` is not `"white"` or `"black"`.
        """
        if self.perspective not in ("white", "black"):
            raise ValueError("Chess reward perspective must be 'white' or 'black'.")

    def evaluate(
        self,
        *,
        previous_state: ChessState,
        action: ChessAction,
        next_state: ChessState,
        transition_info: dict[str, object],
    ) -> float:
        """Return a sparse terminal reward from the configured perspective.

        Parameters
        ----------
        previous_state : ChessState
            Canonical state before the move. It is ignored by this sparse
            terminal reward.
        action : ChessAction
            Parsed move that was applied. It is ignored by this sparse terminal
            reward.
        next_state : ChessState
            Canonical state after the move.
        transition_info : dict[str, object]
            Transition metadata emitted by the backend. It is ignored by this
            sparse terminal reward.

        Returns
        -------
        float
            `0.0` for non-terminal positions, otherwise the configured win,
            draw, or loss reward for `perspective`.
        """
        del action
        del transition_info
        if not next_state.outcome.is_terminal:
            return 0.0
        if next_state.outcome.winner is None:
            return self.draw_reward
        if next_state.outcome.winner == self.perspective:
            return self.win_reward
        return self.loss_reward


@dataclass(slots=True, frozen=True)
class PuzzleOnlyMoveDenseReward:
    """Dense reward for following the canonical puzzle solution line.

    Attributes
    ----------
    correct_move_reward : float
        Reward returned when the played move matches the next expected puzzle
        move.
    incorrect_move_reward : float
        Reward returned when the played move deviates from the expected puzzle
        line or the position is already off the solution path.
    """

    correct_move_reward: float
    incorrect_move_reward: float

    def evaluate(
        self,
        *,
        previous_state: ChessState,
        action: ChessAction,
        next_state: ChessState,
        transition_info: dict[str, object],
    ) -> float:
        """Return dense reward for the next puzzle move in the solution line.

        Parameters
        ----------
        previous_state : ChessState
            Canonical state before the move.
        action : ChessAction
            Parsed move that was applied.
        next_state : ChessState
            Canonical state after the move. It is ignored by this reward.
        transition_info : dict[str, object]
            Transition metadata emitted by the backend. It is ignored by this
            reward.

        Returns
        -------
        float
            Configured reward for a correct or incorrect solution move.
        """
        del next_state
        del transition_info
        progress_index = puzzle_solution_progress_index(state=previous_state)
        solution_moves_uci = puzzle_solution_moves_uci(state=previous_state)
        if progress_index is None:
            return self.incorrect_move_reward
        if progress_index >= len(solution_moves_uci):
            return 0.0
        if action.uci == solution_moves_uci[progress_index]:
            return self.correct_move_reward
        return self.incorrect_move_reward


@dataclass(slots=True, frozen=True)
class PuzzleOnlyMoveSparseReward:
    """Sparse reward for completing the canonical puzzle solution line.

    Attributes
    ----------
    success_reward : float
        Reward returned when the final expected move in the puzzle line is
        played from the correct preceding position.
    incorrect_move_reward : float
        Reward returned when the played move deviates from the expected puzzle
        line or the position is already off the solution path.
    """

    success_reward: float
    incorrect_move_reward: float

    def evaluate(
        self,
        *,
        previous_state: ChessState,
        action: ChessAction,
        next_state: ChessState,
        transition_info: dict[str, object],
    ) -> float:
        """Return sparse reward only when the final puzzle move is solved.

        Parameters
        ----------
        previous_state : ChessState
            Canonical state before the move.
        action : ChessAction
            Parsed move that was applied.
        next_state : ChessState
            Canonical state after the move. It is ignored by this reward.
        transition_info : dict[str, object]
            Transition metadata emitted by the backend. It is ignored by this
            reward.

        Returns
        -------
        float
            `success_reward` for the final correct move, `0.0` for intermediate
            correct moves, otherwise `incorrect_move_reward`.
        """
        del next_state
        del transition_info
        progress_index = puzzle_solution_progress_index(state=previous_state)
        solution_moves_uci = puzzle_solution_moves_uci(state=previous_state)
        if progress_index is None:
            return self.incorrect_move_reward
        if progress_index >= len(solution_moves_uci):
            return 0.0
        if action.uci != solution_moves_uci[progress_index]:
            return self.incorrect_move_reward
        if progress_index == len(solution_moves_uci) - 1:
            return self.success_reward
        return 0.0


@dataclass(slots=True, frozen=True)
class EngineEvalDenseReward:
    """Dense reward based on per-move changes in evaluator score.

    Attributes
    ----------
    evaluator : ChessStateEvaluator
        Position evaluator whose scalar output defines the reward scale.
    perspective : {"white", "black", "mover"}
        Side whose point of view determines the reward sign. `"mover"`
        evaluates each transition from the side that just acted.
    """

    evaluator: ChessStateEvaluator
    perspective: ChessRewardPerspective

    def evaluate(
        self,
        *,
        previous_state: ChessState,
        action: ChessAction,
        next_state: ChessState,
        transition_info: dict[str, object],
    ) -> float:
        """Return the evaluator delta induced by the accepted move.

        Parameters
        ----------
        previous_state : ChessState
            Canonical state before the move.
        action : ChessAction
            Parsed move that was applied. It is ignored by this reward.
        next_state : ChessState
            Canonical state after the move.
        transition_info : dict[str, object]
            Transition metadata emitted by the backend. It is ignored by this
            reward.

        Returns
        -------
        float
            Evaluator score delta from `previous_state` to `next_state`.
        """
        del action
        del transition_info
        concrete_perspective = resolve_reward_perspective(
            previous_state=previous_state,
            perspective=self.perspective,
        )
        previous_eval = self.evaluator.evaluate(
            state=previous_state,
            perspective=concrete_perspective,
        )
        next_eval = self.evaluator.evaluate(
            state=next_state,
            perspective=concrete_perspective,
        )
        return next_eval - previous_eval

    def close(self) -> None:
        """Close the underlying evaluator when it owns external resources.

        Returns
        -------
        None
            This method is a no-op for pure in-memory evaluators.
        """
        close_method = getattr(self.evaluator, "close", None)
        if callable(close_method):
            close_method()


@dataclass(slots=True, frozen=True)
class EngineEvalSparseReward:
    """Sparse reward that pays out only on terminal states via an evaluator.

    Attributes
    ----------
    evaluator : ChessStateEvaluator
        Position evaluator whose scalar output defines the reward scale.
    perspective : {"white", "black", "mover"}
        Side whose point of view determines the reward sign. `"mover"`
        evaluates terminal states from the side that just acted.
    """

    evaluator: ChessStateEvaluator
    perspective: ChessRewardPerspective

    def evaluate(
        self,
        *,
        previous_state: ChessState,
        action: ChessAction,
        next_state: ChessState,
        transition_info: dict[str, object],
    ) -> float:
        """Return evaluator output only when the new state is terminal.

        Parameters
        ----------
        previous_state : ChessState
            Canonical state before the move.
        action : ChessAction
            Parsed move that was applied. It is ignored by this reward.
        next_state : ChessState
            Canonical state after the move.
        transition_info : dict[str, object]
            Transition metadata emitted by the backend. It is ignored by this
            reward.

        Returns
        -------
        float
            `0.0` for non-terminal states, otherwise the evaluator score for
            `next_state` from `perspective`.
        """
        del action
        del transition_info
        if not next_state.outcome.is_terminal:
            return 0.0
        concrete_perspective = resolve_reward_perspective(
            previous_state=previous_state,
            perspective=self.perspective,
        )
        return self.evaluator.evaluate(
            state=next_state,
            perspective=concrete_perspective,
        )

    def close(self) -> None:
        """Close the underlying evaluator when it owns external resources.

        Returns
        -------
        None
            This method is a no-op for pure in-memory evaluators.
        """
        close_method = getattr(self.evaluator, "close", None)
        if callable(close_method):
            close_method()


def puzzle_solution_moves_uci(*, state: ChessState) -> tuple[str, ...]:
    """Return the canonical puzzle solution line carried in state metadata.

    Parameters
    ----------
    state : ChessState
        Chess state whose metadata should describe a puzzle solution line.

    Returns
    -------
    tuple[str, ...]
        Canonical UCI solution moves from the presented puzzle position.

    Raises
    ------
    ValueError
        If `state` does not carry the required puzzle metadata.
    """
    metadata = state.metadata
    if metadata.get("task_type") != "puzzle":
        raise ValueError(
            "Puzzle rewards require state.metadata['task_type'] == 'puzzle'."
        )

    solution_moves_payload = metadata.get("solution_moves_uci")
    if not isinstance(solution_moves_payload, tuple):
        raise ValueError("Puzzle rewards require tuple solution_moves_uci metadata.")
    if not all(isinstance(move_uci, str) for move_uci in solution_moves_payload):
        raise ValueError("Puzzle solution_moves_uci metadata must contain strings.")
    return solution_moves_payload


def puzzle_solution_progress_index(*, state: ChessState) -> int | None:
    """Return how many puzzle solution moves the current state has matched.

    Parameters
    ----------
    state : ChessState
        Chess state whose metadata should describe a puzzle solution line.

    Returns
    -------
    int | None
        Number of leading solution moves matched by `state`, or `None` when the
        state is not on the canonical solution path.

    Raises
    ------
    ValueError
        If `state` does not carry the required puzzle metadata.
    """
    metadata = state.metadata
    presented_fen = metadata.get("presented_fen")
    if not isinstance(presented_fen, str):
        raise ValueError("Puzzle rewards require presented_fen metadata.")

    solution_moves_uci = puzzle_solution_moves_uci(state=state)
    board = chess.Board(presented_fen)
    if board.fen() == state.fen:
        return 0

    for progress_index, move_uci in enumerate(solution_moves_uci, start=1):
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            raise ValueError("Puzzle solution contains an illegal move sequence.")
        board.push(move)
        if board.fen() == state.fen:
            return progress_index
    return None


__all__ = [
    "ChessPerspective",
    "ChessRewardPerspective",
    "ChessStateEvaluator",
    "EngineEvalDenseReward",
    "EngineEvalSparseReward",
    "PuzzleOnlyMoveDenseReward",
    "PuzzleOnlyMoveSparseReward",
    "TerminalOutcomeReward",
    "UciEngineEvaluator",
    "puzzle_solution_moves_uci",
    "puzzle_solution_progress_index",
    "resolve_reward_perspective",
]
