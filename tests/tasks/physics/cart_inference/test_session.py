"""Tests for the cart inference scalar session."""

from dataclasses import replace
from typing import Mapping

from rlvr_physics.core.session import TaskSubmission
from rlvr_physics.tasks.physics.cart_inference.renderers import CART_TEXT_RENDERER
from rlvr_physics.tasks.physics.cart_inference.sessions import CartInferenceSession
from tests.tasks.physics.cart_inference.conftest import (
    CART_SESSION_SEED,
    CartInferenceFixture,
)


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
    assert "tolerance_abs_m" not in result.public_info
    assert "tolerance_abs_m" in result.debug_info


def test_session_reset_returns_identity_and_debug_metadata(
    cart_task_fixture: CartInferenceFixture,
) -> None:
    fixture = cart_task_fixture

    assert fixture.reset.public_info["task_id"] == fixture.instance.task_id
    assert fixture.reset.public_info["kind"] == fixture.instance.kind
    assert fixture.reset.public_info["domain"] == fixture.instance.domain
    assert fixture.reset.public_info["rollout_seed"] == CART_SESSION_SEED
    assert fixture.reset.public_info["renderer"] == fixture.renderer_name
    assert fixture.reset.public_info["limits"] == fixture.instance.public_limits()
    assert (
        fixture.reset.debug_info["acceleration_mps2"]
        == fixture.instance.privileged_payload["acceleration_mps2"]
    )


def test_measurement_step_returns_public_measurement_metadata(
    cart_task_fixture: CartInferenceFixture,
) -> None:
    result = cart_task_fixture.session.submit(
        TaskSubmission.action("measure_position(5)")
    )

    measurement_info = result.public_info["measurement"]
    assert isinstance(measurement_info, Mapping)
    assert measurement_info["time_s"] == 5.0
    assert isinstance(measurement_info["measured_position_m"], float)
    assert "true_position_m" not in measurement_info
    assert "noise_m" not in measurement_info


def test_measurement_metadata_survives_immediate_truncation(
    cart_task_fixture: CartInferenceFixture,
) -> None:
    instance = replace(cart_task_fixture.instance, max_turns=1, action_budget=1)
    session = CartInferenceSession(instance, CART_TEXT_RENDERER)
    session.reset(seed=CART_SESSION_SEED)

    result = session.submit(TaskSubmission.action("measure_position(5)"))

    assert result.truncated
    assert result.observation is None
    measurement_info = result.public_info["measurement"]
    assert isinstance(measurement_info, Mapping)
    assert measurement_info["time_s"] == 5.0
    assert isinstance(measurement_info["measured_position_m"], float)


def test_session_rejects_raw_final_text_for_cart_task(
    cart_task_fixture: CartInferenceFixture,
) -> None:
    result = cart_task_fixture.session.submit(TaskSubmission.final_text("x = 0 m"))

    assert not result.accepted
    assert not result.terminal
    assert result.observation is not None
    assert result.public_info["reason"] == "unsupported submission kind: final_text"
