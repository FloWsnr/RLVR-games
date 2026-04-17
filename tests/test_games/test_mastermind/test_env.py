"""Mastermind environment integration tests."""

from rlvr_games.core import TextMessagePart, build_action_context
from rlvr_games.core.env import TurnBasedEnv
from rlvr_games.core.types import EpisodeConfig
from rlvr_games.games.mastermind import (
    MastermindAction,
    MastermindState,
    TerminalOutcomeReward,
    FixedCodeScenario,
    make_mastermind_env,
)


def make_env() -> TurnBasedEnv[MastermindState, MastermindAction]:
    """Return a standard fixed-code Mastermind environment."""
    return make_mastermind_env(
        scenario=FixedCodeScenario(secret_code=(1, 1, 2, 2)),
        reward_fn=TerminalOutcomeReward(win_reward=1.0, loss_reward=-1.0),
        config=EpisodeConfig(),
        include_images=False,
        image_size=320,
    )


def test_cracking_the_code_terminates_with_reward_and_public_metadata() -> None:
    env = make_env()
    observation, info = env.reset(seed=1)

    result = env.step("1122")

    assert info["scenario"] == "fixed_code"
    assert "secret_code" not in info
    assert "Mastermind board:" in (observation.text or "")
    assert result.reward == 1.0
    assert result.terminated is True
    assert result.info["termination"] == "cracked"
    assert result.observation.metadata["won"] is True
    assert "secret_code" not in result.observation.metadata


def test_env_records_trajectory_and_keeps_secret_code_debug_only() -> None:
    env = make_env()
    _, info = env.reset(seed=5)

    result = env.step("1111")

    assert "secret_code" not in info
    assert result.accepted is True
    assert len(env.trajectory.steps) == 1
    assert env.trajectory.steps[0].action is not None
    assert env.trajectory.steps[0].action.label == "guess 1 1 1 1"
    assert env.inspect_canonical_state()["secret_code"] == (1, 1, 2, 2)
    assert "secret_code" not in result.info
    assert env.trajectory.steps[0].debug_info["secret_code"] == (1, 1, 2, 2)
    assert env.trajectory.steps[0].transitions[0].debug_info["secret_code"] == (
        1,
        1,
        2,
        2,
    )


def test_default_message_adapter_uses_standard_guess_reminder() -> None:
    env = make_env()
    observation, _ = env.reset(seed=9)

    messages = env.messages_for_observation(
        observation,
        action_context=build_action_context(env=env),
    )

    text_part = messages[0].content[0]
    assert isinstance(text_part, TextMessagePart)
    assert "`guess 1 1 2 2`" in text_part.text
