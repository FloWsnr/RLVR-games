"""Text renderer for the cart inference task."""

from dataclasses import dataclass

from rlvr_physics.core.rendering import RenderedObservation, text_observation
from rlvr_physics.tasks.physics.cart_inference.backbone import (
    FINAL_ANSWER_ACTION,
    MEASURE_POSITION_ACTION,
)
from rlvr_physics.tasks.physics.cart_inference.prompting import (
    cart_text_prompt_template,
    render_prompt_template,
)

CART_TEXT_RENDERER = "cart_inference.text"


@dataclass(frozen=True)
class CartMeasurementView:
    """Public current-turn measurement available to renderers.

    Parameters
    ----------
    time_s:
        Measurement time in seconds.
    measured_position_m:
        Public noisy measured position in meters.
    """

    time_s: float
    measured_position_m: float


@dataclass(frozen=True)
class CartRenderContext:
    """Public current-turn cart state used by renderers.

    Parameters
    ----------
    initial_position_m:
        Public initial cart position in meters.
    initial_velocity_mps:
        Public initial cart velocity in meters per second.
    target_time_s:
        Public target prediction time in seconds.
    min_measurement_time_s:
        Public minimum valid measurement time in seconds.
    max_measurement_time_s:
        Public maximum valid measurement time in seconds.
    measurement_noise_abs_m:
        Public absolute bound on deterministic measurement noise.
    feedback:
        Latest public feedback shown to the model.
    current_measurement:
        Public measurement produced for this turn, when one was just accepted.
    actions_used:
        Number of public non-final task actions already used in this rollout.
    action_budget:
        Total public non-final task action budget.
    actions_remaining:
        Number of public non-final task actions still available.
    final_answers_used:
        Number of final-answer attempts already used in this rollout.
    final_answer_budget:
        Total public final-answer attempt budget.
    final_answers_remaining:
        Number of public final-answer attempts still available.
    """

    initial_position_m: float
    initial_velocity_mps: float
    target_time_s: float
    min_measurement_time_s: float
    max_measurement_time_s: float
    measurement_noise_abs_m: float
    feedback: str
    current_measurement: CartMeasurementView | None
    actions_used: int
    action_budget: int
    actions_remaining: int
    final_answers_used: int
    final_answer_budget: int
    final_answers_remaining: int


def validate_cart_renderer_type(renderer_type: str) -> None:
    """Validate a cart inference renderer identifier.

    Parameters
    ----------
    renderer_type:
        Renderer identifier requested by a configured task or session.

    Raises
    ------
    ValueError
        Raised when ``renderer_type`` is not supported by this task.
    """

    if renderer_type != CART_TEXT_RENDERER:
        raise ValueError(f"unsupported cart inference renderer: {renderer_type}")


def render_cart_observation(
    renderer_type: str, context: CartRenderContext
) -> RenderedObservation:
    """Render one cart inference observation.

    Parameters
    ----------
    renderer_type:
        Supported renderer identifier.
    context:
        Public cart rollout state to render.

    Returns
    -------
    RenderedObservation
        Text-only observation for the requested renderer.
    """

    validate_cart_renderer_type(renderer_type)
    return text_observation(CART_TEXT_RENDERER, render_cart_text(context))


def render_cart_text(context: CartRenderContext) -> str:
    """Build the text observation for one cart inference turn.

    Parameters
    ----------
    context:
        Public cart rollout state to render.

    Returns
    -------
    str
        Model-facing text prompt.
    """

    return render_prompt_template(
        cart_text_prompt_template(),
        {
            "feedback": context.feedback,
            "initial_position_m": _fmt(context.initial_position_m),
            "initial_velocity_mps": _fmt(context.initial_velocity_mps),
            "measure_position_action": MEASURE_POSITION_ACTION,
            "min_measurement_time_s": _fmt(context.min_measurement_time_s),
            "max_measurement_time_s": _fmt(context.max_measurement_time_s),
            "measurement_noise_abs_m": _fmt(context.measurement_noise_abs_m),
            "actions_used": context.actions_used,
            "action_budget": context.action_budget,
            "actions_remaining": context.actions_remaining,
            "final_answers_used": context.final_answers_used,
            "final_answer_budget": context.final_answer_budget,
            "final_answers_remaining": context.final_answers_remaining,
            "current_measurement_line": _render_current_measurement_line(context),
            "target_time_s": _fmt(context.target_time_s),
            "final_answer_action": FINAL_ANSWER_ACTION,
        },
    )


def _render_current_measurement_line(context: CartRenderContext) -> str:
    """Build the current measurement line for the text prompt.

    Parameters
    ----------
    context:
        Public cart rollout state to render.

    Returns
    -------
    str
        Current measurement line, or a no-measurement placeholder.
    """

    if context.current_measurement is not None:
        measurement = context.current_measurement
        return (
            f"- t={_fmt(measurement.time_s)} s, "
            f"x={_fmt(measurement.measured_position_m)} m"
        )
    return "- none on this turn"


def _fmt(value: float) -> str:
    """Return a compact numeric string.

    Parameters
    ----------
    value:
        Numeric value to render.

    Returns
    -------
    str
        Compact decimal representation.
    """

    return f"{value:.6g}"
