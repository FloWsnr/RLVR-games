"""Tests for cart inference renderers."""

from importlib import resources

import pytest

from rlvr_physics.core.submissions import TaskSubmission
from rlvr_physics.tasks.physics.cart_inference.instances import (
    build_cart_inference_instance,
)
from rlvr_physics.tasks.physics.cart_inference.renderers import (
    CART_TEXT_RENDERER,
    CartMeasurementView,
    CartRenderContext,
    render_cart_observation,
    validate_cart_renderer_type,
)
from rlvr_physics.tasks.physics.cart_inference.prompting import (
    cart_initial_feedback,
    cart_text_prompt_template,
    render_prompt_template,
)
from rlvr_physics.tasks.physics.cart_inference.sessions import CartInferenceSession
from rlvr_physics.tasks.physics.cart_inference.specs import DEFAULT_CONFIG

_CART_PACKAGE = "rlvr_physics.tasks.physics.cart_inference"


def test_text_renderer_reports_current_measurement_only() -> None:
    instance = build_cart_inference_instance(seed=123, config=DEFAULT_CONFIG)
    session = CartInferenceSession(instance, CART_TEXT_RENDERER, DEFAULT_CONFIG.reward)
    session.reset(seed=456)

    first_result = session.submit(
        TaskSubmission.action(
            '{"action": "measure_position", "arguments": {"time": 5}}'
        )
    )
    second_result = session.submit(
        TaskSubmission.action(
            '{"action": "measure_position", "arguments": {"time": 6}}'
        )
    )

    assert first_result.observation is not None
    assert "t=5 s" in first_result.observation.observation.text()
    assert second_result.observation is not None
    observation = second_result.observation.observation
    assert observation.renderer_name == CART_TEXT_RENDERER
    assert "Measurement history:" not in observation.text()
    assert "Current measurement:" in observation.text()
    assert "t=6 s" in observation.text()
    assert "t=5 s" not in observation.text()
    assert "actions used: 2 / 3" in observation.text()
    assert "actions remaining: 1" in observation.text()
    assert "final answer attempts used: 0 / 1" in observation.text()
    assert "final answer attempts remaining: 1" in observation.text()
    assert '{"action":"measure_position","arguments":{"time":10}}' in (
        observation.text()
    )


def test_cart_prompt_templates_are_task_local_files() -> None:
    assert cart_text_prompt_template() == _cart_prompt_file_text("text_observation.md")
    assert cart_initial_feedback() == _cart_prompt_file_text("initial_feedback.md")


def test_prompt_template_renderer_allows_literal_json_braces() -> None:
    template = 'Submit JSON like {"x": 1.23}; x0={{initial_position_m}} m.'

    rendered = render_prompt_template(template, {"initial_position_m": "0.5"})

    assert rendered == 'Submit JSON like {"x": 1.23}; x0=0.5 m.'


def test_prompt_template_renderer_rejects_unknown_markers() -> None:
    with pytest.raises(ValueError, match="missing_value"):
        render_prompt_template("x0={{missing_value}} m", {})


def test_render_context_exposes_only_public_state_fields() -> None:
    context = CartRenderContext(
        initial_position_m=1.25,
        initial_velocity_mps=-0.75,
        target_time_s=12.0,
        min_measurement_time_s=0.0,
        max_measurement_time_s=10.0,
        measurement_noise_abs_m=0.02,
        feedback="A cart moves on a horizontal track.",
        current_measurement=None,
        actions_used=0,
        action_budget=3,
        actions_remaining=3,
        final_answers_used=0,
        final_answer_budget=1,
        final_answers_remaining=1,
    )

    assert not hasattr(context, "acceleration_mps2")
    assert not hasattr(context, "exact_target_position_m")
    assert not hasattr(context, "measurement_noise_seed")
    assert not hasattr(context, "answer_tolerance_abs_m")


def test_renderer_context_accepts_current_measurement() -> None:
    context = CartRenderContext(
        initial_position_m=0.0,
        initial_velocity_mps=1.0,
        target_time_s=20.0,
        min_measurement_time_s=0.0,
        max_measurement_time_s=18.0,
        measurement_noise_abs_m=0.02,
        feedback="Measurement at t=5s: x=5.1 m.",
        current_measurement=CartMeasurementView(time_s=5.0, measured_position_m=5.1),
        actions_used=1,
        action_budget=3,
        actions_remaining=2,
        final_answers_used=0,
        final_answer_budget=1,
        final_answers_remaining=1,
    )

    observation = render_cart_observation(CART_TEXT_RENDERER, context)

    assert observation.renderer_name == CART_TEXT_RENDERER
    assert "t=5 s" in observation.text()


def test_cart_renderer_rejects_image_renderer_for_now() -> None:
    with pytest.raises(ValueError, match="unsupported cart inference renderer"):
        validate_cart_renderer_type("cart_inference.image")


def _cart_prompt_file_text(filename: str) -> str:
    """Return normalized text from a cart task prompt asset.

    Parameters
    ----------
    filename:
        Prompt asset filename inside the cart task ``prompts`` directory.

    Returns
    -------
    str
        Normalized prompt asset text.
    """

    return (
        resources.files(_CART_PACKAGE)
        .joinpath("prompts", filename)
        .read_text(encoding="utf-8")
        .rstrip()
    )
