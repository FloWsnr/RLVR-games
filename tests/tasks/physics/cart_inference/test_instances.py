"""Tests for cart inference instance construction."""

from typing import Mapping, cast

from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.core.payloads import to_plain_data
from rlvr_physics.tasks.physics.cart_inference.backbone import state_from_instance
from rlvr_physics.tasks.physics.cart_inference.instances import (
    build_cart_inference_instance,
)
from rlvr_physics.tasks.physics.cart_inference.specs import (
    CART_INFERENCE_DOMAIN,
    CART_INFERENCE_KIND,
    CartInferenceConfig,
)


def test_build_cart_inference_instance_is_deterministic(
    cart_config: CartInferenceConfig,
    cart_instance: TaskInstance,
) -> None:
    repeated = build_cart_inference_instance(
        seed=cart_instance.seed, config=cart_config
    )

    assert repeated.task_id == cart_instance.task_id
    assert repeated.content_hash() == cart_instance.content_hash()
    assert cart_instance.kind == CART_INFERENCE_KIND
    assert cart_instance.domain == CART_INFERENCE_DOMAIN


def test_cart_inference_instance_public_view_hides_privileged_state(
    cart_config: CartInferenceConfig,
    cart_instance: TaskInstance,
) -> None:
    public_view = cart_instance.public_view()
    payload = cast(Mapping[str, object], public_view["payload"])
    limits = cast(Mapping[str, object], public_view["limits"])
    public_text = str(to_plain_data(public_view))

    assert payload["target_time_s"] == cart_config.target_time_s
    assert limits["action_budget"] == cart_config.action_budget
    assert "acceleration_mps2" not in public_text
    assert "exact_target_position_m" not in public_text


def test_cart_inference_instance_builds_authoritative_state(
    cart_config: CartInferenceConfig,
    cart_instance: TaskInstance,
) -> None:
    state = state_from_instance(cart_instance)

    assert state.target_time_s == cart_config.target_time_s
    assert state.answer_tolerance_abs_m == cart_config.answer_tolerance_abs_m
    assert state.exact_target_position_m == (
        state.initial_position_m
        + state.initial_velocity_mps * state.target_time_s
        + 0.5 * state.acceleration_mps2 * state.target_time_s * state.target_time_s
    )
