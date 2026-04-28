"""Tests for cart inference renderers."""

from rlvr_physics.core.rendering import ImageContent, TextContent
from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.core.session import TaskSubmission
from rlvr_physics.tasks.physics.cart_inference.instances import (
    build_cart_inference_instance,
)
from rlvr_physics.tasks.physics.cart_inference.renderers import (
    CART_IMAGE_RENDERER,
    CART_TEXT_RENDERER,
    MAX_IMAGE_HISTORY_ROWS,
    CartMeasurementView,
    CartPublicStateView,
    CartRenderContext,
    render_cart_observation,
)
from rlvr_physics.tasks.physics.cart_inference.sessions import CartInferenceSession
from rlvr_physics.tasks.physics.cart_inference.specs import DEFAULT_CONFIG


def test_text_renderer_includes_measurement_history() -> None:
    instance = build_cart_inference_instance(seed=123, config=DEFAULT_CONFIG)
    session = CartInferenceSession(instance, CART_TEXT_RENDERER)
    session.reset(seed=456)

    result = session.submit(TaskSubmission.action("measure_position(5)"))

    assert result.observation is not None
    observation = result.observation.observation
    assert observation.renderer_name == CART_TEXT_RENDERER
    assert "Measurement history:" in observation.text()
    assert "t=5 s" in observation.text()
    assert "measurements remaining: 2" in observation.text()


def test_image_renderer_returns_svg_image_with_text_fallback() -> None:
    instance = build_cart_inference_instance(seed=123, config=DEFAULT_CONFIG)
    session = CartInferenceSession(instance, CART_IMAGE_RENDERER)

    reset = session.reset(seed=456)

    observation = reset.turn.observation
    assert observation.renderer_name == CART_IMAGE_RENDERER
    assert len(observation.contents) == 2
    assert isinstance(observation.contents[0], ImageContent)
    assert isinstance(observation.contents[1], TextContent)
    assert observation.contents[0].mime_type == "image/svg+xml"
    assert observation.contents[0].data.startswith(b"<svg")
    assert "Initial state:" in observation.text()


def test_image_renderer_omits_privileged_state() -> None:
    instance = _hidden_sentinel_instance()
    session = CartInferenceSession(instance, CART_IMAGE_RENDERER)

    reset = session.reset(seed=456)

    observation = reset.turn.observation
    assert isinstance(observation.contents[0], ImageContent)
    hidden_fragments = (
        "9.876543",
        "9.87654",
        "444.444",
        "987654321",
        "0.123456",
    )
    rendered_payloads = (
        observation.contents[0].data.decode("utf-8"),
        observation.contents[0].alt_text,
        observation.text(),
    )
    for payload in rendered_payloads:
        for fragment in hidden_fragments:
            assert fragment not in payload


def test_render_context_uses_public_state_view() -> None:
    context = CartRenderContext(
        state=CartPublicStateView(
            initial_position_m=1.25,
            initial_velocity_mps=-0.75,
            target_time_s=12.0,
            min_measurement_time_s=0.0,
            max_measurement_time_s=10.0,
            measurement_noise_abs_m=0.02,
        ),
        feedback="A cart moves on a horizontal track.",
        measurements=(),
        action_budget=3,
        measurements_remaining=3,
    )

    assert not hasattr(context.state, "acceleration_mps2")
    assert not hasattr(context.state, "exact_target_position_m")
    assert not hasattr(context.state, "measurement_noise_seed")
    assert not hasattr(context.state, "answer_tolerance_abs_m")


def test_image_renderer_caps_history_panel_rows() -> None:
    context = CartRenderContext(
        state=CartPublicStateView(
            initial_position_m=0.0,
            initial_velocity_mps=1.0,
            target_time_s=20.0,
            min_measurement_time_s=0.0,
            max_measurement_time_s=18.0,
            measurement_noise_abs_m=0.02,
        ),
        feedback="A cart moves on a horizontal track.",
        measurements=tuple(
            CartMeasurementView(time_s=float(index), measured_position_m=float(index))
            for index in range(MAX_IMAGE_HISTORY_ROWS + 4)
        ),
        action_budget=MAX_IMAGE_HISTORY_ROWS + 4,
        measurements_remaining=0,
    )

    observation = render_cart_observation(CART_IMAGE_RENDERER, context)

    assert isinstance(observation.contents[0], ImageContent)
    image_text = observation.contents[0].data.decode("utf-8")
    assert "... 4 earlier omitted" in image_text
    assert "1. t=0 s" not in image_text
    assert "5. t=4 s" in image_text


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
