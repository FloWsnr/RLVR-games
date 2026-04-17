"""Mastermind reward tests."""

from rlvr_games.games.mastermind import (
    CandidateReductionDenseReward,
    MastermindBackend,
    MastermindState,
    TerminalOutcomeReward,
)


def test_terminal_reward_returns_configured_win_value_for_cracked_code() -> None:
    backend = MastermindBackend()
    reward_fn = TerminalOutcomeReward(win_reward=3.0, loss_reward=-2.0)
    state = MastermindState(secret_code=(1, 1, 2, 2))
    action = backend.parse_action(state, "1122").require_action()
    next_state, transition_info = backend.apply_action(state, action)

    reward = reward_fn.evaluate(
        previous_state=state,
        action=action,
        next_state=next_state,
        transition_info=transition_info,
    )

    assert next_state.is_terminal is True
    assert reward == 3.0


def test_candidate_reduction_reward_returns_one_for_solved_transition() -> None:
    backend = MastermindBackend()
    reward_fn = CandidateReductionDenseReward()
    state = MastermindState(secret_code=(1, 1, 2, 2))
    action = backend.parse_action(state, "1122").require_action()
    next_state, transition_info = backend.apply_action(state, action)

    reward = reward_fn.evaluate(
        previous_state=state,
        action=action,
        next_state=next_state,
        transition_info=transition_info,
    )

    assert reward == 1.0


def test_candidate_reduction_reward_matches_normalized_pruning() -> None:
    backend = MastermindBackend()
    reward_fn = CandidateReductionDenseReward()
    state = MastermindState(secret_code=(1, 1, 2, 2))
    action = backend.parse_action(state, "1111").require_action()
    next_state, transition_info = backend.apply_action(state, action)

    reward = reward_fn.evaluate(
        previous_state=state,
        action=action,
        next_state=next_state,
        transition_info=transition_info,
    )

    expected_reward = (
        state.candidate_count - next_state.candidate_count
    ) / state.candidate_count
    assert reward == expected_reward
