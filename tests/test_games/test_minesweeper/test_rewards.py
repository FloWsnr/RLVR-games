"""Minesweeper reward tests."""

from rlvr_games.games.minesweeper import (
    MinesweeperBackend,
    MinesweeperState,
    OutcomeReward,
    SafeRevealCountReward,
    normalize_initial_board,
)

FIXED_BOARD = ("*..", "...", "..*")


def test_outcome_reward_returns_configured_win_reward_for_clear() -> None:
    backend = MinesweeperBackend()
    reward_fn = OutcomeReward(win_reward=3.0, loss_reward=-2.0)
    start_state = MinesweeperState(
        rows=3,
        columns=3,
        mine_count=2,
        hidden_board=normalize_initial_board(board=FIXED_BOARD),
        move_count=0,
        placement_seed=None,
    )
    mid_state, _ = backend.apply_action(
        start_state,
        backend.parse_action(start_state, "reveal 1 3").require_action(),
    )
    action = backend.parse_action(mid_state, "reveal 3 1").require_action()
    next_state, transition_info = backend.apply_action(mid_state, action)

    reward = reward_fn.evaluate(
        previous_state=mid_state,
        action=action,
        next_state=next_state,
        transition_info=transition_info,
    )

    assert next_state.is_terminal is True
    assert next_state.outcome.won is True
    assert reward == 3.0


def test_safe_reveal_count_reward_returns_newly_revealed_cells() -> None:
    backend = MinesweeperBackend()
    reward_fn = SafeRevealCountReward(mine_penalty=-5.0)
    state = MinesweeperState(
        rows=3,
        columns=3,
        mine_count=2,
        hidden_board=normalize_initial_board(board=FIXED_BOARD),
        move_count=0,
        placement_seed=None,
    )
    action = backend.parse_action(state, "reveal 1 3").require_action()
    next_state, transition_info = backend.apply_action(state, action)

    reward = reward_fn.evaluate(
        previous_state=state,
        action=action,
        next_state=next_state,
        transition_info=transition_info,
    )

    assert reward == 4.0
