"""Tests for cart inference public specs."""

import pytest

from rlvr_physics.tasks.physics.cart_inference.specs import (
    CART_INFERENCE_DOMAIN,
    CART_INFERENCE_KIND,
    CartInferenceConfig,
    cart_inference_spec,
)
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
    assert spec.renderers[0].renderer_type == fixture.renderer_name
    assert spec.verifier.parameters["absolute_tolerance_m"] == (
        fixture.config.answer_tolerance_abs_m
    )
    assert spec.reward.reward_type == "threshold_with_linear_partial_credit"
    assert spec.action_budget == fixture.config.action_budget


def test_cart_inference_config_requires_final_answer_turn(
    cart_config: CartInferenceConfig,
) -> None:
    invalid_config = CartInferenceConfig(
        min_measurement_time_s=cart_config.min_measurement_time_s,
        max_measurement_time_s=cart_config.max_measurement_time_s,
        target_time_s=cart_config.target_time_s,
        measurement_noise_abs_m=cart_config.measurement_noise_abs_m,
        answer_tolerance_abs_m=cart_config.answer_tolerance_abs_m,
        max_turns=cart_config.max_turns,
        timeout_seconds=cart_config.timeout_seconds,
        token_budget=cart_config.token_budget,
        action_budget=cart_config.max_turns,
    )

    with pytest.raises(ValueError, match="action_budget"):
        cart_inference_spec(invalid_config)
