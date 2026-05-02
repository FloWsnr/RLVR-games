"""Tests for the cart inference configured task builder."""

from dataclasses import replace

from rlvr_physics.core.factory import ConfiguredTask
from rlvr_physics.core.submissions import TaskSubmission
from rlvr_physics.tasks.physics.cart_inference import cart_inference_task
from rlvr_physics.tasks.physics.cart_inference.instances import (
    build_cart_inference_instance,
)
from rlvr_physics.tasks.physics.cart_inference.renderers import CART_TEXT_RENDERER
from rlvr_physics.tasks.physics.cart_inference.rewards import CartRewardConfig
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


def test_cart_inference_task_session_uses_configured_reward_policy(
    cart_config: CartInferenceConfig,
) -> None:
    reward_config = CartRewardConfig(
        correct_final_answer_reward=1.0,
        incorrect_final_answer_reward=0.0,
        partial_credit_window_tolerances=10.0,
        accepted_measurement_reward=0.25,
        invalid_submission_reward=-0.25,
        budget_exceeded_reward=-0.5,
        session_already_done_reward=0.0,
    )
    task_config = replace(cart_config, reward=reward_config)
    task = cart_inference_task(task_config)
    instance = build_cart_inference_instance(
        seed=CART_INSTANCE_SEED, config=cart_config
    )
    session = task.create_session(instance)
    session.reset(seed=456)

    result = session.submit(
        TaskSubmission.action(
            '{"action": "measure_position", "arguments": {"time": 5}}'
        )
    )

    assert result.reward == 0.25
    assert task.spec.reward.parameters["accepted_measurement_reward"] == 0.25


def test_cart_inference_spec_advertises_text_renderer(
    cart_task: ConfiguredTask,
) -> None:
    renderer_types = {renderer.renderer_type for renderer in cart_task.spec.renderers}

    assert renderer_types == {CART_TEXT_RENDERER}
