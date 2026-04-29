"""Tests for the cart inference scalar session."""

from dataclasses import replace
from typing import Mapping

from rlvr_physics.core.submissions import TaskSubmission
from rlvr_physics.tasks.physics.cart_inference.instances import (
    build_cart_inference_instance,
)
from rlvr_physics.tasks.physics.cart_inference.renderers import CART_TEXT_RENDERER
from rlvr_physics.tasks.physics.cart_inference.rewards import CartRewardConfig
from rlvr_physics.tasks.physics.cart_inference.sessions import CartInferenceSession
from tests.tasks.physics.cart_inference.conftest import (
    CART_INSTANCE_SEED,
    CART_SESSION_SEED,
    CartInferenceFixture,
)


def test_session_accepts_structured_final_answer_action(
    cart_task_fixture: CartInferenceFixture,
) -> None:
    fixture = cart_task_fixture

    result = fixture.session.submit(
        TaskSubmission.action(
            '{"action": "final_answer", "arguments": {"x": %s}}'
            % fixture.exact_target_position_m
        )
    )

    assert fixture.reset.turn.submission_modes == ("action",)
    assert fixture.renderer_name == CART_TEXT_RENDERER
    assert result.accepted
    assert result.terminal
    assert result.reward == 1.0
    assert "tolerance_abs_m" not in result.public_info
    assert "tolerance_abs_m" in result.debug_info


def test_session_reset_returns_identity_and_debug_metadata(
    cart_task_fixture: CartInferenceFixture,
) -> None:
    fixture = cart_task_fixture

    assert fixture.reset.public_info["task_id"] == fixture.instance.task_id
    assert fixture.reset.public_info["kind"] == fixture.instance.kind
    assert fixture.reset.public_info["domain"] == fixture.instance.domain
    assert fixture.reset.public_info["rollout_seed"] == CART_SESSION_SEED
    assert fixture.reset.public_info["renderer"] == fixture.renderer_name
    assert fixture.reset.public_info["limits"] == {
        "budget_limits": {
            "turns": fixture.config.turn_budget,
            "actions": fixture.config.action_budget,
            "final_answers": fixture.config.final_answer_budget,
        },
        "token_budget": fixture.config.token_budget,
    }
    assert (
        fixture.reset.debug_info["acceleration_mps2"]
        == fixture.instance.privileged_payload["acceleration_mps2"]
    )


def test_measurement_step_returns_public_measurement_metadata(
    cart_task_fixture: CartInferenceFixture,
) -> None:
    result = cart_task_fixture.session.submit(
        TaskSubmission.action(
            '{"action": "measure_position", "arguments": {"time": 5}}'
        )
    )

    measurement_info = result.public_info["measurement"]
    assert isinstance(measurement_info, Mapping)
    assert measurement_info["time_s"] == 5.0
    assert isinstance(measurement_info["measured_position_m"], float)
    assert "true_position_m" not in measurement_info
    assert "noise_m" not in measurement_info


def test_measurement_metadata_survives_immediate_truncation(
    cart_task_fixture: CartInferenceFixture,
) -> None:
    instance = replace(
        cart_task_fixture.instance,
        budget_limits={"turns": 2, "actions": 1, "final_answers": 1},
    )
    session = CartInferenceSession(
        instance, CART_TEXT_RENDERER, cart_task_fixture.config.reward
    )
    session.reset(seed=CART_SESSION_SEED)
    session.submit(TaskSubmission.action("unparseable"))

    result = session.submit(
        TaskSubmission.action(
            '{"action": "measure_position", "arguments": {"time": 5}}'
        )
    )

    assert result.truncated
    assert result.accepted
    assert result.observation is None
    assert result.public_info["budget_usage"] == {
        "turns": 2,
        "actions": 1,
        "final_answers": 0,
    }
    assert result.public_info["budget_remaining"] == {
        "turns": 0,
        "actions": 0,
        "final_answers": 1,
    }
    measurement_info = result.public_info["measurement"]
    assert isinstance(measurement_info, Mapping)
    assert measurement_info["time_s"] == 5.0
    assert isinstance(measurement_info["measured_position_m"], float)


def test_session_rejects_raw_final_text_for_cart_task(
    cart_task_fixture: CartInferenceFixture,
) -> None:
    result = cart_task_fixture.session.submit(TaskSubmission.final_text("x = 0 m"))

    assert not result.accepted
    assert not result.terminal
    assert result.observation is not None
    assert result.public_info["reason"] == "unsupported submission kind: final_text"


def test_session_uses_reward_config_for_accepted_measurement(
    cart_task_fixture: CartInferenceFixture,
) -> None:
    reward_config = CartRewardConfig(
        correct_final_answer_reward=1.0,
        incorrect_final_answer_reward=0.0,
        partial_credit_window_tolerances=10.0,
        accepted_measurement_reward=0.125,
        invalid_submission_reward=-0.25,
        budget_exceeded_reward=-0.5,
        session_already_done_reward=0.0,
    )
    config = replace(cart_task_fixture.config, reward=reward_config)
    instance = build_cart_inference_instance(seed=CART_INSTANCE_SEED, config=config)
    session = CartInferenceSession(instance, CART_TEXT_RENDERER, reward_config)
    session.reset(seed=CART_SESSION_SEED)

    result = session.submit(
        TaskSubmission.action(
            '{"action": "measure_position", "arguments": {"time": 5}}'
        )
    )

    assert result.accepted
    assert result.reward == 0.125
    assert result.reward_result.public_info["reward_event"] == "accepted_action"


def test_session_uses_reward_config_for_invalid_submission(
    cart_task_fixture: CartInferenceFixture,
) -> None:
    reward_config = CartRewardConfig(
        correct_final_answer_reward=1.0,
        incorrect_final_answer_reward=0.0,
        partial_credit_window_tolerances=10.0,
        accepted_measurement_reward=0.125,
        invalid_submission_reward=-0.25,
        budget_exceeded_reward=-0.5,
        session_already_done_reward=0.0,
    )
    config = replace(cart_task_fixture.config, reward=reward_config)
    instance = build_cart_inference_instance(seed=CART_INSTANCE_SEED, config=config)
    session = CartInferenceSession(instance, CART_TEXT_RENDERER, reward_config)
    session.reset(seed=CART_SESSION_SEED)

    result = session.submit(TaskSubmission.action("unparseable"))

    assert not result.accepted
    assert result.reward == -0.25
    assert result.reward_result.public_info["reward_event"] == "invalid_submission"
