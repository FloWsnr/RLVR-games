"""Tests for the cart inference scalar session."""

from rlvr_physics.core.session import TaskSubmission
from rlvr_physics.tasks.physics.cart_inference import (
    DEFAULT_CONFIG,
    CartInferenceSession,
    build_cart_inference_instance,
)


def test_session_accepts_structured_final_answer_action() -> None:
    instance = build_cart_inference_instance(seed=123, config=DEFAULT_CONFIG)
    session = CartInferenceSession(instance)
    reset = session.reset(seed=456)
    exact_position_m = instance.privileged_payload["exact_target_position_m"]

    result = session.submit(
        TaskSubmission.action(f'{{"action": "final_answer", "x": {exact_position_m}}}')
    )

    assert reset.turn.submission_modes == ("action",)
    assert result.accepted
    assert result.terminal
    assert result.reward == 1.0


def test_session_rejects_raw_final_text_for_cart_task() -> None:
    instance = build_cart_inference_instance(seed=123, config=DEFAULT_CONFIG)
    session = CartInferenceSession(instance)
    session.reset(seed=456)

    result = session.submit(TaskSubmission.final_text("x = 0 m"))

    assert not result.accepted
    assert not result.terminal
    assert result.observation is not None
    assert result.public_info["reason"] == "unsupported submission kind: final_text"
