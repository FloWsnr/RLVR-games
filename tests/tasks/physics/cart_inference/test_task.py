"""Tests for the cart inference configured task builder."""

from rlvr_physics.core.factory import ConfiguredTask
from rlvr_physics.tasks.physics.cart_inference.sessions import CartInferenceSession
from rlvr_physics.tasks.physics.cart_inference.specs import (
    CART_INFERENCE_DOMAIN,
    CART_INFERENCE_KIND,
)
from tests.tasks.physics.cart_inference.conftest import CART_INSTANCE_SEED


def test_cart_inference_task_builds_instances_and_sessions(
    cart_task: ConfiguredTask,
) -> None:
    task = cart_task
    instance = task.build_instance(seed=CART_INSTANCE_SEED)
    session = task.create_session(instance)

    assert task.spec.kind == CART_INFERENCE_KIND
    assert task.spec.domain == CART_INFERENCE_DOMAIN
    assert instance.kind == CART_INFERENCE_KIND
    assert instance.domain == CART_INFERENCE_DOMAIN
    assert isinstance(session, CartInferenceSession)
