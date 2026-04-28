"""Tests for the cart inference backbone."""

import pytest

from rlvr_physics.core.session import TaskSubmission
from rlvr_physics.tasks.physics.cart_inference.backbone import (
    ActionBudgetExceeded,
    CartInferenceBackbone,
)
from rlvr_physics.tasks.physics.cart_inference.instances import (
    build_cart_inference_instance,
)
from rlvr_physics.tasks.physics.cart_inference.rewards import (
    reward_final_answer,
)
from rlvr_physics.tasks.physics.cart_inference.specs import DEFAULT_CONFIG


def test_backbone_measurement_updates_budget_and_resets() -> None:
    instance = build_cart_inference_instance(seed=123, config=DEFAULT_CONFIG)
    backbone = CartInferenceBackbone(instance)
    action = backbone.parse_action(TaskSubmission.action("measure_position(2.0)"))

    measurement = backbone.measure(action)

    assert measurement.measurement_index == 0
    assert backbone.measurements_used == 1
    assert backbone.measurements_remaining == DEFAULT_CONFIG.action_budget - 1

    backbone.reset_rollout()
    repeated_measurement = backbone.measure(action)

    assert repeated_measurement.measurement_index == 0
    assert backbone.measurements_used == 1


def test_backbone_enforces_measurement_action_budget() -> None:
    instance = build_cart_inference_instance(seed=123, config=DEFAULT_CONFIG)
    backbone = CartInferenceBackbone(instance)
    action = backbone.parse_action(TaskSubmission.action("measure_position(2.0)"))

    for _ in range(DEFAULT_CONFIG.action_budget):
        backbone.measure(action)

    with pytest.raises(ActionBudgetExceeded, match="action_budget_exceeded"):
        backbone.measure(action)


def test_backbone_evaluates_final_answer_action() -> None:
    instance = build_cart_inference_instance(seed=123, config=DEFAULT_CONFIG)
    backbone = CartInferenceBackbone(instance)
    exact_position_m = instance.privileged_payload["exact_target_position_m"]
    submission = TaskSubmission.action(
        f'{{"action": "final_answer", "x": {exact_position_m}}}'
    )

    action = backbone.parse_action(submission)
    submitted_position_m = backbone.final_answer_from_action(action)
    evaluation = backbone.evaluate_final_answer(submitted_position_m)
    reward = reward_final_answer(evaluation)

    assert evaluation.correct
    assert reward.reward == 1.0
