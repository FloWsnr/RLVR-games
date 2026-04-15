"""Action context tests."""

from rlvr_games.core import EpisodeConfig
from rlvr_games.core.rollout import build_action_context

from tests.test_core.support import CounterBackend, make_counter_env


def test_build_action_context_uses_current_turn_index() -> None:
    env = make_counter_env(
        backend=CounterBackend(),
        config=EpisodeConfig(max_transitions=3),
    )
    env.reset(seed=7)

    initial_context = build_action_context(env=env)
    env.step("1")
    next_context = build_action_context(env=env)

    assert initial_context.turn_index == 0
    assert next_context.turn_index == 1
