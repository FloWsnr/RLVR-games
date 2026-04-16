"""Chess reward tests."""

import chess

from rlvr_games.core import (
    DefaultObservationMessageAdapter,
    DefaultObservationMessagePolicy,
)
from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.types import EpisodeConfig
from rlvr_games.games.chess import (
    AsciiBoardFormatter,
    ChessAction,
    ChessBackend,
    ChessPerspective,
    ChessObservationRenderer,
    ChessState,
    EngineEvalDenseReward,
    EngineEvalSparseReward,
    PuzzleOnlyMoveDenseReward,
    PuzzleOnlyMoveSparseReward,
    StartingPositionScenario,
    TerminalOutcomeReward,
)
from rlvr_games.games.chess.scenarios import STANDARD_START_FEN
from rlvr_games.games.chess.state import inspect_chess_state, repetition_key_from_board


def make_renderer() -> ChessObservationRenderer:
    """Construct a text-only chess observation renderer.

    Returns
    -------
    ChessObservationRenderer
        Observation renderer with the standard ASCII board formatter.
    """
    return ChessObservationRenderer(
        board_formatter=AsciiBoardFormatter(orientation=chess.WHITE),
        image_renderer=None,
    )


def make_message_adapter() -> DefaultObservationMessageAdapter:
    """Return a generic observation message adapter for direct env tests."""
    return DefaultObservationMessageAdapter(policy=DefaultObservationMessagePolicy())


def make_puzzle_state(
    *,
    fen: str,
    presented_fen: str,
    solution_moves_uci: tuple[str, ...],
) -> ChessState:
    """Construct a puzzle state carrying the metadata needed for rewards.

    Parameters
    ----------
    fen : str
        Canonical FEN for the current state.
    presented_fen : str
        Starting FEN shown to the player before any puzzle solution move.
    solution_moves_uci : tuple[str, ...]
        Canonical puzzle solution line from `presented_fen`.

    Returns
    -------
    ChessState
        Chess state carrying puzzle metadata for reward evaluation.
    """
    board = chess.Board(fen)
    return ChessState.from_board(
        board=board,
        repetition_counts={repetition_key_from_board(board): 1},
        metadata={
            "task_type": "puzzle",
            "presented_fen": presented_fen,
            "solution_moves_uci": solution_moves_uci,
        },
    )


def fools_mate_transition() -> tuple[
    ChessState, ChessAction, ChessState, dict[str, object]
]:
    """Return the final accepted transition of the Fool's Mate sequence.

    Returns
    -------
    tuple[ChessState, object, ChessState, dict[str, object]]
        Previous state, parsed action, next state, and backend transition info
        for the terminal `Qh4#` move.
    """
    backend = ChessBackend()
    state = ChessState(fen=STANDARD_START_FEN)
    for raw_action in ("f2f3", "e7e5", "g2g4"):
        action = backend.parse_action(state, raw_action).require_action()
        state, _ = backend.apply_action(state, action)
    final_action = backend.parse_action(state, "d8h4").require_action()
    next_state, transition_info = backend.apply_action(state, final_action)
    return state, final_action, next_state, transition_info


class StubEvaluator:
    """Minimal evaluator stub for reward tests."""

    def __init__(self, values: dict[tuple[str, str], float]) -> None:
        """Store predetermined scalar evaluations.

        Parameters
        ----------
        values : dict[tuple[str, str], float]
            Mapping keyed by `(fen, perspective)`.
        """
        self.values = dict(values)

    def evaluate(
        self,
        *,
        state: ChessState,
        perspective: ChessPerspective,
    ) -> float:
        """Return the configured scalar evaluation.

        Parameters
        ----------
        state : ChessState
            Canonical state to evaluate.
        perspective : str
            Side whose point of view determines the sign.

        Returns
        -------
        float
            Configured scalar evaluation.
        """
        return self.values[(state.fen, perspective)]


class CloseableStubEvaluator:
    """Evaluator stub that records whether ``close()`` was called."""

    def __init__(self) -> None:
        """Initialize the close-tracking evaluator."""
        self.close_call_count = 0

    def evaluate(
        self,
        *,
        state: ChessState,
        perspective: ChessPerspective,
    ) -> float:
        """Return a dummy scalar evaluation.

        Parameters
        ----------
        state : ChessState
            Canonical state to evaluate. It is ignored by this stub.
        perspective : {"white", "black"}
            Side whose point of view determines the sign. It is ignored by this
            stub.

        Returns
        -------
        float
            Constant dummy evaluation.
        """
        del state
        del perspective
        return 0.0

    def close(self) -> None:
        """Record one close call from the reward wrapper."""
        self.close_call_count += 1


def test_terminal_outcome_reward_uses_the_configured_perspective() -> None:
    previous_state, action, next_state, transition_info = fools_mate_transition()

    black_reward = TerminalOutcomeReward(
        perspective="black",
        win_reward=1.0,
        draw_reward=0.25,
        loss_reward=-1.0,
    ).evaluate(
        previous_state=previous_state,
        action=action,
        next_state=next_state,
        transition_info=transition_info,
    )
    white_reward = TerminalOutcomeReward(
        perspective="white",
        win_reward=1.0,
        draw_reward=0.25,
        loss_reward=-1.0,
    ).evaluate(
        previous_state=previous_state,
        action=action,
        next_state=next_state,
        transition_info=transition_info,
    )

    assert black_reward == 1.0
    assert white_reward == -1.0


def test_terminal_outcome_reward_returns_draw_reward_for_threefold_repetition() -> None:
    env = TurnBasedEnv(
        backend=ChessBackend(),
        scenario=StartingPositionScenario(initial_fen=STANDARD_START_FEN),
        renderer=make_renderer(),
        inspect_canonical_state_fn=inspect_chess_state,
        reward_fn=TerminalOutcomeReward(
            perspective="white",
            win_reward=1.0,
            draw_reward=0.25,
            loss_reward=-1.0,
        ),
        config=EpisodeConfig(),
        observation_message_adapter=make_message_adapter(),
    )
    env.reset(seed=5)

    for raw_action in (
        "g1f3",
        "g8f6",
        "f3g1",
        "f6g8",
        "g1f3",
        "g8f6",
        "f3g1",
    ):
        result = env.step(raw_action)
        assert result.reward == 0.0

    final_result = env.step("f6g8")

    assert final_result.reward == 0.25


def test_puzzle_only_move_dense_reward_distinguishes_correct_and_incorrect_moves() -> (
    None
):
    backend = ChessBackend()
    presented_fen = STANDARD_START_FEN
    previous_state = make_puzzle_state(
        fen=presented_fen,
        presented_fen=presented_fen,
        solution_moves_uci=("e2e4", "e7e5"),
    )
    correct_action = backend.parse_action(previous_state, "e2e4").require_action()
    correct_next_state, correct_info = backend.apply_action(
        previous_state, correct_action
    )
    incorrect_action = backend.parse_action(previous_state, "d2d4").require_action()
    incorrect_next_state, incorrect_info = backend.apply_action(
        previous_state,
        incorrect_action,
    )
    reward_fn = PuzzleOnlyMoveDenseReward(
        correct_move_reward=0.5,
        incorrect_move_reward=-0.25,
    )

    correct_reward = reward_fn.evaluate(
        previous_state=previous_state,
        action=correct_action,
        next_state=correct_next_state,
        transition_info=correct_info,
    )
    incorrect_reward = reward_fn.evaluate(
        previous_state=previous_state,
        action=incorrect_action,
        next_state=incorrect_next_state,
        transition_info=incorrect_info,
    )

    assert correct_reward == 0.5
    assert incorrect_reward == -0.25


def test_puzzle_only_move_sparse_reward_only_pays_out_on_the_final_solution_move() -> (
    None
):
    backend = ChessBackend()
    presented_fen = STANDARD_START_FEN
    initial_state = make_puzzle_state(
        fen=presented_fen,
        presented_fen=presented_fen,
        solution_moves_uci=("e2e4", "e7e5"),
    )
    first_action = backend.parse_action(initial_state, "e2e4").require_action()
    intermediate_state, first_info = backend.apply_action(initial_state, first_action)
    final_action = backend.parse_action(intermediate_state, "e7e5").require_action()
    final_state, final_info = backend.apply_action(intermediate_state, final_action)
    reward_fn = PuzzleOnlyMoveSparseReward(
        success_reward=1.0,
        incorrect_move_reward=-0.5,
    )

    first_reward = reward_fn.evaluate(
        previous_state=initial_state,
        action=first_action,
        next_state=intermediate_state,
        transition_info=first_info,
    )
    final_reward = reward_fn.evaluate(
        previous_state=intermediate_state,
        action=final_action,
        next_state=final_state,
        transition_info=final_info,
    )

    assert first_reward == 0.0
    assert final_reward == 1.0


def test_engine_eval_dense_reward_returns_the_evaluation_delta() -> None:
    backend = ChessBackend()
    previous_state = ChessState(fen=STANDARD_START_FEN)
    action = backend.parse_action(previous_state, "e2e4").require_action()
    next_state, transition_info = backend.apply_action(previous_state, action)
    reward = EngineEvalDenseReward(
        evaluator=StubEvaluator(
            {
                (previous_state.fen, "white"): 15.0,
                (next_state.fen, "white"): 55.0,
            }
        ),
        perspective="white",
    ).evaluate(
        previous_state=previous_state,
        action=action,
        next_state=next_state,
        transition_info=transition_info,
    )

    assert reward == 40.0


def test_engine_eval_dense_reward_can_use_the_mover_perspective() -> None:
    backend = ChessBackend()
    start_state = ChessState(fen=STANDARD_START_FEN)
    white_action = backend.parse_action(start_state, "e2e4").require_action()
    previous_state, _ = backend.apply_action(start_state, white_action)
    black_action = backend.parse_action(previous_state, "c7c5").require_action()
    next_state, transition_info = backend.apply_action(previous_state, black_action)
    reward = EngineEvalDenseReward(
        evaluator=StubEvaluator(
            {
                (previous_state.fen, "black"): -20.0,
                (next_state.fen, "black"): 35.0,
                (previous_state.fen, "white"): 999.0,
                (next_state.fen, "white"): 999.0,
            }
        ),
        perspective="mover",
    ).evaluate(
        previous_state=previous_state,
        action=black_action,
        next_state=next_state,
        transition_info=transition_info,
    )

    assert reward == 55.0


def test_engine_eval_sparse_reward_only_pays_out_on_terminal_positions() -> None:
    previous_state, action, next_state, transition_info = fools_mate_transition()
    nonterminal_backend = ChessBackend()
    start_state = ChessState(fen=STANDARD_START_FEN)
    nonterminal_action = nonterminal_backend.parse_action(
        start_state, "e2e4"
    ).require_action()
    nonterminal_next_state, nonterminal_info = nonterminal_backend.apply_action(
        start_state,
        nonterminal_action,
    )
    evaluator = StubEvaluator(
        {
            (next_state.fen, "black"): 900.0,
            (nonterminal_next_state.fen, "black"): 42.0,
        }
    )
    reward_fn = EngineEvalSparseReward(evaluator=evaluator, perspective="black")

    nonterminal_reward = reward_fn.evaluate(
        previous_state=start_state,
        action=nonterminal_action,
        next_state=nonterminal_next_state,
        transition_info=nonterminal_info,
    )
    terminal_reward = reward_fn.evaluate(
        previous_state=previous_state,
        action=action,
        next_state=next_state,
        transition_info=transition_info,
    )

    assert nonterminal_reward == 0.0
    assert terminal_reward == 900.0


def test_engine_eval_sparse_reward_can_use_the_mover_perspective() -> None:
    previous_state, action, next_state, transition_info = fools_mate_transition()
    reward = EngineEvalSparseReward(
        evaluator=StubEvaluator(
            {
                (next_state.fen, "black"): 900.0,
                (next_state.fen, "white"): -900.0,
            }
        ),
        perspective="mover",
    ).evaluate(
        previous_state=previous_state,
        action=action,
        next_state=next_state,
        transition_info=transition_info,
    )

    assert reward == 900.0


def test_engine_eval_dense_reward_close_closes_the_underlying_evaluator() -> None:
    evaluator = CloseableStubEvaluator()
    reward = EngineEvalDenseReward(evaluator=evaluator, perspective="mover")

    reward.close()

    assert evaluator.close_call_count == 1


def test_engine_eval_sparse_reward_close_closes_the_underlying_evaluator() -> None:
    evaluator = CloseableStubEvaluator()
    reward = EngineEvalSparseReward(evaluator=evaluator, perspective="mover")

    reward.close()

    assert evaluator.close_call_count == 1
