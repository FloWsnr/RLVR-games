"""Tests for the cart inference configured task builder."""

from rlvr_physics.tasks.physics.cart_inference import (
    DEFAULT_CONFIG,
    cart_inference_task,
)
from rlvr_physics.tasks.physics.cart_inference.sessions import CartInferenceSession
from rlvr_physics.tasks.physics.cart_inference.specs import (
    CART_INFERENCE_DOMAIN,
    CART_INFERENCE_KIND,
)


def test_cart_inference_task_builds_instances_and_sessions() -> None:
    task = cart_inference_task(DEFAULT_CONFIG)
    instance = task.build_instance(seed=123)
    session = task.create_session(instance)

    assert task.spec.kind == CART_INFERENCE_KIND
    assert task.spec.domain == CART_INFERENCE_DOMAIN
    assert instance.kind == CART_INFERENCE_KIND
    assert instance.domain == CART_INFERENCE_DOMAIN
    assert isinstance(session, CartInferenceSession)
