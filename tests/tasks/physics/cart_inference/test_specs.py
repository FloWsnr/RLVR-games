"""Tests for cart inference public specs."""

import pytest

from rlvr_physics.tasks.physics.cart_inference.specs import (
    CART_INFERENCE_DOMAIN,
    CART_INFERENCE_KIND,
    CartInferenceConfig,
    cart_inference_spec,
)
from rlvr_physics.tasks.physics.cart_inference.renderers import CART_TEXT_RENDERER
from tests.tasks.physics.cart_inference.conftest import CartInferenceFixture


def test_cart_inference_spec_advertises_public_contract(
    cart_task_fixture: CartInferenceFixture,
) -> None:
    fixture = cart_task_fixture
    spec = fixture.task.spec

    assert spec.kind == CART_INFERENCE_KIND
    assert spec.domain == CART_INFERENCE_DOMAIN
    assert spec.source.source_type == "cart_inference_generator"
    assert spec.source.parameters["target_time_s"] == fixture.config.target_time_s
    assert "answer_tolerance_abs_m" not in spec.source.parameters
    assert {renderer.renderer_type for renderer in spec.renderers} == {
        CART_TEXT_RENDERER
    }
    assert spec.renderers[0].renderer_type == fixture.renderer_name
    assert spec.verifier.parameters["absolute_tolerance_source"] == (
        "privileged_instance_payload"
    )
    assert "absolute_tolerance_m" not in spec.verifier.parameters
    assert spec.reward.reward_type == "cart_inference_event_rewards"
    assert spec.reward.parameters["accepted_measurement_reward"] == (
        fixture.config.reward.accepted_measurement_reward
    )
    assert spec.budget_limits["turns"] == fixture.config.turn_budget
    assert spec.budget_limits["actions"] == fixture.config.action_budget
    assert spec.budget_limits["final_answers"] == (fixture.config.final_answer_budget)


def test_cart_inference_config_requires_final_answer_turn(
    cart_config: CartInferenceConfig,
) -> None:
    invalid_config = CartInferenceConfig(
        min_measurement_time_s=cart_config.min_measurement_time_s,
        max_measurement_time_s=cart_config.max_measurement_time_s,
        target_time_s=cart_config.target_time_s,
        measurement_noise_abs_m=cart_config.measurement_noise_abs_m,
        answer_tolerance_abs_m=cart_config.answer_tolerance_abs_m,
        turn_budget=cart_config.turn_budget,
        timeout_seconds=cart_config.timeout_seconds,
        token_budget=cart_config.token_budget,
        action_budget=cart_config.turn_budget,
        final_answer_budget=cart_config.final_answer_budget,
        reward=cart_config.reward,
    )

    with pytest.raises(ValueError, match="action_budget"):
        cart_inference_spec(invalid_config)


def test_cart_inference_config_has_one_terminal_final_answer_attempt(
    cart_config: CartInferenceConfig,
) -> None:
    invalid_config = CartInferenceConfig(
        min_measurement_time_s=cart_config.min_measurement_time_s,
        max_measurement_time_s=cart_config.max_measurement_time_s,
        target_time_s=cart_config.target_time_s,
        measurement_noise_abs_m=cart_config.measurement_noise_abs_m,
        answer_tolerance_abs_m=cart_config.answer_tolerance_abs_m,
        turn_budget=cart_config.turn_budget,
        timeout_seconds=cart_config.timeout_seconds,
        token_budget=cart_config.token_budget,
        action_budget=cart_config.action_budget,
        final_answer_budget=2,
        reward=cart_config.reward,
    )

    with pytest.raises(ValueError, match="final_answer_budget must be 1"):
        cart_inference_spec(invalid_config)


def test_cart_inference_config_rejects_non_finite_numeric_values(
    cart_config: CartInferenceConfig,
) -> None:
    """Cart configs reject non-finite public numeric fields."""

    invalid_config = CartInferenceConfig(
        min_measurement_time_s=cart_config.min_measurement_time_s,
        max_measurement_time_s=float("inf"),
        target_time_s=cart_config.target_time_s,
        measurement_noise_abs_m=cart_config.measurement_noise_abs_m,
        answer_tolerance_abs_m=cart_config.answer_tolerance_abs_m,
        turn_budget=cart_config.turn_budget,
        timeout_seconds=cart_config.timeout_seconds,
        token_budget=cart_config.token_budget,
        action_budget=cart_config.action_budget,
        final_answer_budget=cart_config.final_answer_budget,
        reward=cart_config.reward,
    )

    with pytest.raises(ValueError, match="max_measurement_time_s must be finite"):
        cart_inference_spec(invalid_config)
