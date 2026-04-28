"""Tests for cart inference renderers."""

from struct import unpack

from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.core.rendering import ImageContent, PNG_MIME_TYPE, TextContent
from rlvr_physics.core.session import TaskSubmission
from rlvr_physics.tasks.physics.cart_inference.instances import (
    build_cart_inference_instance,
)
from rlvr_physics.tasks.physics.cart_inference.renderers import (
    CART_IMAGE_RENDERER,
    CART_TEXT_RENDERER,
    CartMeasurementView,
    CartRenderContext,
    _render_cart_svg,
    render_cart_observation,
)
from rlvr_physics.tasks.physics.cart_inference.sessions import CartInferenceSession
from rlvr_physics.tasks.physics.cart_inference.specs import DEFAULT_CONFIG


def test_text_renderer_reports_current_measurement_only() -> None:
    instance = build_cart_inference_instance(seed=123, config=DEFAULT_CONFIG)
    session = CartInferenceSession(instance, CART_TEXT_RENDERER)
    session.reset(seed=456)

    first_result = session.submit(TaskSubmission.action("measure_position(5)"))
    second_result = session.submit(TaskSubmission.action("measure_position(6)"))

    assert first_result.observation is not None
    assert "t=5 s" in first_result.observation.observation.text()
    assert second_result.observation is not None
    observation = second_result.observation.observation
    assert observation.renderer_name == CART_TEXT_RENDERER
    assert "Measurement history:" not in observation.text()
    assert "Current measurement:" in observation.text()
    assert "t=6 s" in observation.text()
    assert "t=5 s" not in observation.text()
    assert "measurements used: 2 / 3" in observation.text()
    assert "measurements remaining: 1" in observation.text()


def test_image_renderer_returns_png_image_with_text_fallback() -> None:
    instance = build_cart_inference_instance(seed=123, config=DEFAULT_CONFIG)
    session = CartInferenceSession(instance, CART_IMAGE_RENDERER)

    reset = session.reset(seed=456)

    observation = reset.turn.observation
    assert observation.renderer_name == CART_IMAGE_RENDERER
    assert len(observation.contents) == 2
    assert isinstance(observation.contents[0], ImageContent)
    assert isinstance(observation.contents[1], TextContent)
    assert observation.contents[0].mime_type == PNG_MIME_TYPE
    assert observation.contents[0].data.startswith(b"\x89PNG\r\n\x1a\n")
    assert _png_size(observation.contents[0].data) == (960, 640)
    assert "Initial state:" not in observation.contents[0].alt_text
    assert "Initial state:" in observation.text()


def test_image_renderer_omits_privileged_state() -> None:
    instance = _hidden_sentinel_instance()
    session = CartInferenceSession(instance, CART_IMAGE_RENDERER)

    reset = session.reset(seed=456)

    observation = reset.turn.observation
    assert isinstance(observation.contents[0], ImageContent)
    svg_text = _render_cart_svg(
        CartRenderContext(
            initial_position_m=1.25,
            initial_velocity_mps=-0.75,
            target_time_s=12.0,
            min_measurement_time_s=0.0,
            max_measurement_time_s=10.0,
            measurement_noise_abs_m=0.02,
            feedback="A cart moves on a horizontal track.",
            current_measurement=None,
            measurements_used=0,
            action_budget=3,
            measurements_remaining=3,
        )
    )
    hidden_fragments = (
        "9.876543",
        "9.87654",
        "444.444",
        "987654321",
        "0.123456",
    )
    for fragment in hidden_fragments:
        assert fragment not in svg_text
        assert fragment not in observation.contents[0].alt_text
        assert fragment not in observation.text()


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
        measurements_used=0,
        action_budget=3,
        measurements_remaining=3,
    )

    assert not hasattr(context, "acceleration_mps2")
    assert not hasattr(context, "exact_target_position_m")
    assert not hasattr(context, "measurement_noise_seed")
    assert not hasattr(context, "answer_tolerance_abs_m")


def test_image_renderer_reports_current_measurement_only() -> None:
    instance = build_cart_inference_instance(seed=123, config=DEFAULT_CONFIG)
    session = CartInferenceSession(instance, CART_IMAGE_RENDERER)
    session.reset(seed=456)

    first_result = session.submit(TaskSubmission.action("measure_position(5)"))
    second_result = session.submit(TaskSubmission.action("measure_position(6)"))

    assert first_result.observation is not None
    first_observation = first_result.observation.observation
    assert isinstance(first_observation.contents[0], ImageContent)
    assert "t=5 s" in first_observation.text()
    assert second_result.observation is not None
    observation = second_result.observation.observation
    assert isinstance(observation.contents[0], ImageContent)
    assert first_observation.contents[0].data != observation.contents[0].data
    assert observation.contents[0].data.startswith(b"\x89PNG\r\n\x1a\n")
    assert "Current measurement:" in observation.text()
    assert "t=6 s" in observation.text()
    assert "t=5 s" not in observation.text()


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
        measurements_used=1,
        action_budget=3,
        measurements_remaining=2,
    )

    observation = render_cart_observation(CART_IMAGE_RENDERER, context)

    assert isinstance(observation.contents[0], ImageContent)
    assert "t=5 s" in observation.text()
    assert observation.contents[0].mime_type == PNG_MIME_TYPE
    assert observation.contents[0].data.startswith(b"\x89PNG\r\n\x1a\n")


def _hidden_sentinel_instance() -> TaskInstance:
    """Return a cart instance with distinctive privileged sentinel values."""

    return TaskInstance(
        task_id="cart-hidden-sentinel",
        kind="physics.cart_inference.v1",
        domain="physics",
        seed=777,
        public_payload={
            "initial_position_m": 1.25,
            "initial_velocity_mps": -0.75,
            "target_time_s": 12.0,
            "measurement_time_range_s": {"min": 0.0, "max": 10.0},
            "measurement_noise_abs_m": 0.02,
            "required_answer": {"field": "x", "units": "m"},
        },
        privileged_payload={
            "acceleration_mps2": 9.876543,
            "answer_tolerance_abs_m": 0.123456,
            "exact_target_position_m": 444.444,
            "measurement_noise_seed": 987654321,
        },
        max_turns=5,
        action_budget=3,
    )


def _png_size(data: bytes) -> tuple[int, int]:
    """Return the width and height from a PNG IHDR chunk."""

    return unpack(">II", data[16:24])
