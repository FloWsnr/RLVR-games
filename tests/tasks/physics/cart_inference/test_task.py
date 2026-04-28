"""Tests for the cart inference configured task builder."""

from rlvr_physics.core.factory import ConfiguredTask
from rlvr_physics.core.rendering import PNG_MIME_TYPE
from rlvr_physics.tasks.physics.cart_inference import (
    CART_IMAGE_RENDERER,
    cart_inference_task,
)
from rlvr_physics.tasks.physics.cart_inference.renderers import CART_TEXT_RENDERER
from rlvr_physics.tasks.physics.cart_inference.sessions import CartInferenceSession
from rlvr_physics.tasks.physics.cart_inference.specs import (
    CART_INFERENCE_DOMAIN,
    CART_INFERENCE_KIND,
    CartInferenceConfig,
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


def test_cart_inference_task_can_build_image_renderer_sessions(
    cart_config: CartInferenceConfig,
) -> None:
    task = cart_inference_task(cart_config, renderer_type=CART_IMAGE_RENDERER)
    instance = task.build_instance(seed=CART_INSTANCE_SEED)
    session = task.create_session(instance)

    reset = session.reset(seed=456)

    assert reset.turn.observation.renderer_name == CART_IMAGE_RENDERER


def test_cart_inference_spec_advertises_text_and_image_renderers(
    cart_task: ConfiguredTask,
) -> None:
    renderer_types = {renderer.renderer_type for renderer in cart_task.spec.renderers}

    assert renderer_types == {CART_TEXT_RENDERER, CART_IMAGE_RENDERER}


def test_cart_inference_spec_advertises_png_image_renderer(
    cart_task: ConfiguredTask,
) -> None:
    image_renderer = next(
        renderer
        for renderer in cart_task.spec.renderers
        if renderer.renderer_type == CART_IMAGE_RENDERER
    )

    assert image_renderer.parameters["mime_type"] == PNG_MIME_TYPE
