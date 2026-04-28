"""Tests for the cart inference scalar session."""

from rlvr_physics.core.session import TaskSubmission
from rlvr_physics.tasks.physics.cart_inference.renderers import CART_TEXT_RENDERER
from tests.tasks.physics.cart_inference.conftest import CartInferenceFixture


def test_session_accepts_structured_final_answer_action(
    cart_task_fixture: CartInferenceFixture,
) -> None:
    fixture = cart_task_fixture

    result = fixture.session.submit(
        TaskSubmission.action(
            f'{{"action": "final_answer", "x": {fixture.exact_target_position_m}}}'
        )
    )

    assert fixture.reset.turn.submission_modes == ("action",)
    assert fixture.renderer_name == CART_TEXT_RENDERER
    assert result.accepted
    assert result.terminal
    assert result.reward == 1.0


def test_session_rejects_raw_final_text_for_cart_task(
    cart_task_fixture: CartInferenceFixture,
) -> None:
    result = cart_task_fixture.session.submit(TaskSubmission.final_text("x = 0 m"))

    assert not result.accepted
    assert not result.terminal
    assert result.observation is not None
    assert result.public_info["reason"] == "unsupported submission kind: final_text"
