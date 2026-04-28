"""Shared fixtures for cart inference task tests."""

from dataclasses import dataclass

import pytest

from rlvr_physics.core.factory import ConfiguredTask
from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.core.session import TaskResetResult
from rlvr_physics.tasks.physics.cart_inference.backbone import CartInferenceBackbone
from rlvr_physics.tasks.physics.cart_inference.instances import (
    build_cart_inference_instance,
)
from rlvr_physics.tasks.physics.cart_inference.sessions import CartInferenceSession
from rlvr_physics.tasks.physics.cart_inference.specs import (
    DEFAULT_CONFIG,
    CartInferenceConfig,
)
from rlvr_physics.tasks.physics.cart_inference.task import cart_inference_task

CART_INSTANCE_SEED = 123
CART_SESSION_SEED = 456


@dataclass(frozen=True)
class CartInferenceFixture:
    """Reusable cart task setup for task-module tests."""

    config: CartInferenceConfig
    task: ConfiguredTask
    instance: TaskInstance
    session: CartInferenceSession
    reset: TaskResetResult
    renderer_name: str
    exact_target_position_m: float


@pytest.fixture
def cart_config() -> CartInferenceConfig:
    """Return the default cart inference configuration."""

    return DEFAULT_CONFIG


@pytest.fixture
def cart_task(cart_config: CartInferenceConfig) -> ConfiguredTask:
    """Return a configured cart inference task."""

    return cart_inference_task(cart_config)


@pytest.fixture
def cart_instance(cart_config: CartInferenceConfig) -> TaskInstance:
    """Return a deterministic cart inference instance."""

    return build_cart_inference_instance(seed=CART_INSTANCE_SEED, config=cart_config)


@pytest.fixture
def cart_backbone(cart_instance: TaskInstance) -> CartInferenceBackbone:
    """Return an authoritative cart inference backbone."""

    return CartInferenceBackbone(cart_instance)


@pytest.fixture
def cart_task_fixture(
    cart_config: CartInferenceConfig,
    cart_task: ConfiguredTask,
) -> CartInferenceFixture:
    """Return a configured cart task with reset session and renderer info."""

    instance = cart_task.build_instance(seed=CART_INSTANCE_SEED)
    session = CartInferenceSession(instance)
    reset = session.reset(seed=CART_SESSION_SEED)
    exact_position = instance.privileged_payload["exact_target_position_m"]
    if not isinstance(exact_position, int | float) or isinstance(exact_position, bool):
        raise TypeError("exact_target_position_m must be numeric")
    return CartInferenceFixture(
        config=cart_config,
        task=cart_task,
        instance=instance,
        session=session,
        reset=reset,
        renderer_name=reset.turn.observation.renderer_name,
        exact_target_position_m=float(exact_position),
    )
