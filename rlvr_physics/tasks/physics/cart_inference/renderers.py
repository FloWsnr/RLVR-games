"""Renderers for the cart inference task."""

from dataclasses import dataclass
from html import escape

from rlvr_physics.core.rendering import (
    ImageContent,
    RenderedObservation,
    TextContent,
    text_observation,
)
from rlvr_physics.tasks.physics.cart_inference.backbone import (
    FINAL_ANSWER_ACTION,
    MEASURE_POSITION_ACTION,
)

CART_TEXT_RENDERER = "cart_inference.text"
CART_IMAGE_RENDERER = "cart_inference.image"
MAX_IMAGE_HISTORY_ROWS = 8


@dataclass(frozen=True)
class CartPublicStateView:
    """Public cart state available to renderers.

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

    Attributes
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
    """

    initial_position_m: float
    initial_velocity_mps: float
    target_time_s: float
    min_measurement_time_s: float
    max_measurement_time_s: float
    measurement_noise_abs_m: float


@dataclass(frozen=True)
class CartMeasurementView:
    """Public measurement record available to renderers.

    Parameters
    ----------
    time_s:
        Measurement time in seconds.
    measured_position_m:
        Public noisy measured position in meters.

    Attributes
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
    """Public cart rollout state used by renderers.

    Parameters
    ----------
    state:
        Public cart state view.
    feedback:
        Latest public feedback shown to the model.
    measurements:
        Public measurement history for this rollout.
    action_budget:
        Total public measurement action budget.
    measurements_remaining:
        Number of public measurements still available.

    Attributes
    ----------
    state:
        Public cart state view.
    feedback:
        Latest public feedback shown to the model.
    measurements:
        Public measurement history for this rollout.
    action_budget:
        Total public measurement action budget.
    measurements_remaining:
        Number of public measurements still available.
    """

    state: CartPublicStateView
    feedback: str
    measurements: tuple[CartMeasurementView, ...]
    action_budget: int
    measurements_remaining: int


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

    if renderer_type not in {CART_TEXT_RENDERER, CART_IMAGE_RENDERER}:
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
        Text-only or image-plus-text observation for the requested renderer.
    """

    validate_cart_renderer_type(renderer_type)
    if renderer_type == CART_TEXT_RENDERER:
        return text_observation(CART_TEXT_RENDERER, render_cart_text(context))
    return render_cart_image(context)


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

    state = context.state
    lines = [
        context.feedback,
        "",
        "Initial state:",
        f"- position x0 = {_fmt(state.initial_position_m)} m",
        f"- velocity v0 = {_fmt(state.initial_velocity_mps)} m/s",
        "",
        "Measurement access:",
        (
            f"- request {MEASURE_POSITION_ACTION}(time) with "
            f"{_fmt(state.min_measurement_time_s)} <= time <= "
            f"{_fmt(state.max_measurement_time_s)} seconds"
        ),
        f"- measurement noise is bounded by +/- {_fmt(state.measurement_noise_abs_m)} m",
        (f"- measurements used: {len(context.measurements)} / {context.action_budget}"),
        f"- measurements remaining: {context.measurements_remaining}",
        "",
        "Measurement history:",
    ]
    if context.measurements:
        for index, measurement in enumerate(context.measurements, start=1):
            lines.append(
                f"- {index}. t={_fmt(measurement.time_s)} s, "
                f"x={_fmt(measurement.measured_position_m)} m"
            )
    else:
        lines.append("- none yet")

    lines.extend(
        [
            "",
            (
                f"Predict the cart position at t={_fmt(state.target_time_s)} s. "
                f"Submit {FINAL_ANSWER_ACTION}(x) with x in meters."
            ),
        ]
    )
    return "\n".join(lines)


def render_cart_image(context: CartRenderContext) -> RenderedObservation:
    """Build an SVG image observation for one cart inference turn.

    Parameters
    ----------
    context:
        Public cart rollout state to render.

    Returns
    -------
    RenderedObservation
        Observation with an SVG image block followed by text fallback.
    """

    alt_text = render_cart_text(context)
    svg = _render_cart_svg(context)
    return RenderedObservation(
        renderer_name=CART_IMAGE_RENDERER,
        contents=(
            ImageContent(
                data=svg.encode("utf-8"),
                mime_type="image/svg+xml",
                alt_text=alt_text,
            ),
            TextContent(text=alt_text),
        ),
    )


def _render_cart_svg(context: CartRenderContext) -> str:
    """Return a deterministic SVG visualization for public cart state."""

    width = 960
    height = 640
    elements = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" role="img">'
        ),
        "<defs>",
        (
            '<marker id="arrow" markerWidth="8" markerHeight="8" refX="7" '
            'refY="4" orient="auto">'
        ),
        '<path d="M 0 0 L 8 4 L 0 8 z" fill="#1f6feb"/>',
        "</marker>",
        "</defs>",
        '<rect x="0" y="0" width="960" height="640" fill="#f8fafc"/>',
        _text(32.0, 42.0, "Cart inference", 22, "#0f172a", "700"),
        _text(
            32.0,
            66.0,
            "Public state only: infer the target position from measurements.",
            13,
            "#475569",
            "400",
        ),
    ]
    elements.extend(_track_panel(context))
    elements.extend(_timeline_panel(context))
    elements.extend(_chart_panel(context))
    elements.extend(_data_panel(context))
    elements.append("</svg>")
    return "".join(elements)


def _track_panel(context: CartRenderContext) -> list[str]:
    """Render the initial cart state panel."""

    state = context.state
    panel_x = 32.0
    panel_y = 88.0
    panel_w = 590.0
    panel_h = 128.0
    track_y = panel_y + 78.0
    track_left = panel_x + 42.0
    track_right = panel_x + panel_w - 42.0
    position_min = min(-3.0, state.initial_position_m - 1.0)
    position_max = max(3.0, state.initial_position_m + 1.0)
    cart_center = _scale(
        state.initial_position_m,
        position_min,
        position_max,
        track_left,
        track_right,
    )
    cart_x = _clamp(cart_center - 28.0, track_left, track_right - 56.0)
    arrow_length = 42.0 + min(abs(state.initial_velocity_mps), 2.0) * 30.0
    arrow_start = cart_x + 28.0
    arrow_end = arrow_start + arrow_length
    if state.initial_velocity_mps < 0.0:
        arrow_end = arrow_start - arrow_length
    arrow_end = _clamp(arrow_end, track_left, track_right)

    return [
        _panel(panel_x, panel_y, panel_w, panel_h),
        _text(
            panel_x + 18.0, panel_y + 28.0, "Initial cart state", 15, "#0f172a", "700"
        ),
        (
            f'<line x1="{track_left:.1f}" y1="{track_y:.1f}" '
            f'x2="{track_right:.1f}" y2="{track_y:.1f}" stroke="#334155" '
            'stroke-width="3"/>'
        ),
        (
            f'<line x1="{track_left:.1f}" y1="{track_y + 14.0:.1f}" '
            f'x2="{track_left:.1f}" y2="{track_y - 14.0:.1f}" '
            'stroke="#64748b" stroke-width="2"/>'
        ),
        (
            f'<line x1="{track_right:.1f}" y1="{track_y + 14.0:.1f}" '
            f'x2="{track_right:.1f}" y2="{track_y - 14.0:.1f}" '
            'stroke="#64748b" stroke-width="2"/>'
        ),
        (
            f'<rect x="{cart_x:.1f}" y="{track_y - 42.0:.1f}" width="56" '
            'height="32" rx="4" fill="#e0f2fe" stroke="#0369a1" '
            'stroke-width="2"/>'
        ),
        _wheel(cart_x + 14.0, track_y - 8.0),
        _wheel(cart_x + 42.0, track_y - 8.0),
        (
            f'<line x1="{arrow_start:.1f}" y1="{track_y - 58.0:.1f}" '
            f'x2="{arrow_end:.1f}" y2="{track_y - 58.0:.1f}" '
            'stroke="#1f6feb" stroke-width="4" marker-end="url(#arrow)"/>'
        ),
        _text(
            cart_x,
            track_y - 50.0,
            f"v0={_fmt(state.initial_velocity_mps)} m/s",
            12,
            "#1f6feb",
            "700",
        ),
        _text(
            cart_x,
            track_y - 48.0 + 72.0,
            f"x0={_fmt(state.initial_position_m)} m",
            12,
            "#0f172a",
            "700",
        ),
        _text(
            track_left,
            track_y + 34.0,
            f"{_fmt(position_min)} m",
            11,
            "#64748b",
            "400",
        ),
        _text(
            track_right,
            track_y + 34.0,
            f"{_fmt(position_max)} m",
            11,
            "#64748b",
            "400",
            "end",
        ),
    ]


def _timeline_panel(context: CartRenderContext) -> list[str]:
    """Render the public time window panel."""

    state = context.state
    panel_x = 32.0
    panel_y = 236.0
    panel_w = 590.0
    panel_h = 96.0
    axis_left = panel_x + 42.0
    axis_right = panel_x + panel_w - 42.0
    axis_y = panel_y + 58.0
    target_time = state.target_time_s
    window_left = _scale(
        state.min_measurement_time_s, 0.0, target_time, axis_left, axis_right
    )
    window_right = _scale(
        state.max_measurement_time_s, 0.0, target_time, axis_left, axis_right
    )
    target_x = axis_right

    elements = [
        _panel(panel_x, panel_y, panel_w, panel_h),
        _text(panel_x + 18.0, panel_y + 28.0, "Timeline", 15, "#0f172a", "700"),
        (
            f'<rect x="{window_left:.1f}" y="{axis_y - 18.0:.1f}" '
            f'width="{window_right - window_left:.1f}" height="36" '
            'fill="#dcfce7" stroke="#16a34a" stroke-width="1"/>'
        ),
        _text(
            (window_left + window_right) / 2.0,
            axis_y - 24.0,
            "valid measurement window",
            11,
            "#15803d",
            "700",
            "middle",
        ),
        (
            f'<line x1="{axis_left:.1f}" y1="{axis_y:.1f}" '
            f'x2="{axis_right:.1f}" y2="{axis_y:.1f}" stroke="#334155" '
            'stroke-width="2"/>'
        ),
        _tick(axis_left, axis_y, "0 s", "middle"),
    ]
    if state.min_measurement_time_s > 0.0:
        elements.append(
            _tick(
                window_left,
                axis_y,
                f"{_fmt(state.min_measurement_time_s)} s",
                "middle",
            )
        )
    elements.extend(
        [
            _tick(
                window_right,
                axis_y,
                f"{_fmt(state.max_measurement_time_s)} s",
                "middle",
            ),
            (
                f'<line x1="{target_x:.1f}" y1="{axis_y - 30.0:.1f}" '
                f'x2="{target_x:.1f}" y2="{axis_y + 30.0:.1f}" '
                'stroke="#dc2626" stroke-width="2" stroke-dasharray="5 4"/>'
            ),
            _text(
                target_x,
                axis_y - 36.0,
                f"target T={_fmt(target_time)} s",
                12,
                "#dc2626",
                "700",
                "end",
            ),
        ]
    )
    return elements


def _chart_panel(context: CartRenderContext) -> list[str]:
    """Render the public time-position measurement chart."""

    state = context.state
    panel_x = 32.0
    panel_y = 356.0
    panel_w = 590.0
    panel_h = 244.0
    chart_left = panel_x + 54.0
    chart_right = panel_x + panel_w - 34.0
    chart_top = panel_y + 46.0
    chart_bottom = panel_y + panel_h - 42.0
    measurement_values = [item.measured_position_m for item in context.measurements]
    y_min, y_max = _chart_range(
        state.initial_position_m,
        tuple(measurement_values),
        state.measurement_noise_abs_m,
    )
    window_left = _scale(
        state.min_measurement_time_s,
        0.0,
        state.target_time_s,
        chart_left,
        chart_right,
    )
    window_right = _scale(
        state.max_measurement_time_s,
        0.0,
        state.target_time_s,
        chart_left,
        chart_right,
    )

    elements = [
        _panel(panel_x, panel_y, panel_w, panel_h),
        _text(
            panel_x + 18.0, panel_y + 28.0, "Public measurements", 15, "#0f172a", "700"
        ),
        (
            f'<rect x="{window_left:.1f}" y="{chart_top:.1f}" '
            f'width="{window_right - window_left:.1f}" '
            f'height="{chart_bottom - chart_top:.1f}" fill="#f0fdf4"/>'
        ),
        (
            f'<line x1="{chart_left:.1f}" y1="{chart_bottom:.1f}" '
            f'x2="{chart_right:.1f}" y2="{chart_bottom:.1f}" '
            'stroke="#334155" stroke-width="2"/>'
        ),
        (
            f'<line x1="{chart_left:.1f}" y1="{chart_bottom:.1f}" '
            f'x2="{chart_left:.1f}" y2="{chart_top:.1f}" '
            'stroke="#334155" stroke-width="2"/>'
        ),
        (
            f'<line x1="{chart_right:.1f}" y1="{chart_bottom:.1f}" '
            f'x2="{chart_right:.1f}" y2="{chart_top:.1f}" '
            'stroke="#dc2626" stroke-width="2" stroke-dasharray="5 4"/>'
        ),
        _text(chart_right, chart_top - 8.0, "target time", 11, "#dc2626", "700", "end"),
        _text(chart_left, chart_bottom + 24.0, "0 s", 11, "#64748b", "400", "middle"),
        _text(
            chart_right,
            chart_bottom + 24.0,
            f"{_fmt(state.target_time_s)} s",
            11,
            "#64748b",
            "400",
            "middle",
        ),
        _text(
            chart_left - 8.0,
            chart_top + 4.0,
            f"{_fmt(y_max)} m",
            11,
            "#64748b",
            "400",
            "end",
        ),
        _text(
            chart_left - 8.0,
            chart_bottom + 4.0,
            f"{_fmt(y_min)} m",
            11,
            "#64748b",
            "400",
            "end",
        ),
        _text(
            (chart_left + chart_right) / 2.0,
            chart_bottom + 35.0,
            "time",
            11,
            "#475569",
            "400",
            "middle",
        ),
    ]
    elements.append(
        _point(
            chart_left,
            _scale_y(state.initial_position_m, y_min, y_max, chart_top, chart_bottom),
            "#16a34a",
            "initial",
        )
    )
    for index, measurement in enumerate(context.measurements, start=1):
        point_x = _scale(
            measurement.time_s,
            0.0,
            state.target_time_s,
            chart_left,
            chart_right,
        )
        point_y = _scale_y(
            measurement.measured_position_m, y_min, y_max, chart_top, chart_bottom
        )
        high_y = _scale_y(
            measurement.measured_position_m + state.measurement_noise_abs_m,
            y_min,
            y_max,
            chart_top,
            chart_bottom,
        )
        low_y = _scale_y(
            measurement.measured_position_m - state.measurement_noise_abs_m,
            y_min,
            y_max,
            chart_top,
            chart_bottom,
        )
        elements.extend(
            [
                (
                    f'<line x1="{point_x:.1f}" y1="{high_y:.1f}" '
                    f'x2="{point_x:.1f}" y2="{low_y:.1f}" '
                    'stroke="#f97316" stroke-width="2"/>'
                ),
                (
                    f'<line x1="{point_x - 5.0:.1f}" y1="{high_y:.1f}" '
                    f'x2="{point_x + 5.0:.1f}" y2="{high_y:.1f}" '
                    'stroke="#f97316" stroke-width="2"/>'
                ),
                (
                    f'<line x1="{point_x - 5.0:.1f}" y1="{low_y:.1f}" '
                    f'x2="{point_x + 5.0:.1f}" y2="{low_y:.1f}" '
                    'stroke="#f97316" stroke-width="2"/>'
                ),
                _point(point_x, point_y, "#f97316", str(index)),
            ]
        )
    if not context.measurements:
        elements.append(
            _text(
                (chart_left + chart_right) / 2.0,
                (chart_top + chart_bottom) / 2.0,
                "No measurements yet",
                14,
                "#64748b",
                "700",
                "middle",
            )
        )
    return elements


def _data_panel(context: CartRenderContext) -> list[str]:
    """Render exact public values and action hints."""

    state = context.state
    history_rows, omitted_count = _image_history_rows(context.measurements)
    panel_x = 650.0
    panel_y = 88.0
    panel_w = 278.0
    panel_h = 512.0
    lines = [
        "Public data",
        f"x0: {_fmt(state.initial_position_m)} m",
        f"v0: {_fmt(state.initial_velocity_mps)} m/s",
        f"Target T: {_fmt(state.target_time_s)} s",
        (
            f"Measure t: {_fmt(state.min_measurement_time_s)} to "
            f"{_fmt(state.max_measurement_time_s)} s"
        ),
        f"Noise bound: +/- {_fmt(state.measurement_noise_abs_m)} m",
        f"Measurements: {len(context.measurements)} / {context.action_budget}",
        f"Remaining: {context.measurements_remaining}",
        "",
        "Actions",
        f"{MEASURE_POSITION_ACTION}(time)",
        f"{FINAL_ANSWER_ACTION}(x)",
        "",
        "History",
    ]
    if history_rows:
        if omitted_count > 0:
            lines.append(f"... {omitted_count} earlier omitted")
        for index, measurement in history_rows:
            lines.append(
                f"{index}. t={_fmt(measurement.time_s)} s, "
                f"x={_fmt(measurement.measured_position_m)} m"
            )
    else:
        lines.append("none yet")

    elements = [_panel(panel_x, panel_y, panel_w, panel_h)]
    y = panel_y + 30.0
    for index, line in enumerate(lines):
        if line == "":
            y += 10.0
            continue
        weight = "700" if index in {0, 9, 13} else "400"
        color = "#0f172a" if weight == "700" else "#334155"
        elements.append(_text(panel_x + 18.0, y, line, 13, color, weight))
        y += 22.0
    return elements


def _image_history_rows(
    measurements: tuple[CartMeasurementView, ...],
) -> tuple[tuple[tuple[int, CartMeasurementView], ...], int]:
    """Return bounded measurement rows for the fixed-size image panel."""

    indexed_rows = tuple(enumerate(measurements, start=1))
    if len(indexed_rows) <= MAX_IMAGE_HISTORY_ROWS:
        return indexed_rows, 0
    omitted_count = len(indexed_rows) - MAX_IMAGE_HISTORY_ROWS
    return indexed_rows[-MAX_IMAGE_HISTORY_ROWS:], omitted_count


def _panel(x: float, y: float, width: float, height: float) -> str:
    """Return a rounded panel rectangle."""

    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" '
        f'height="{height:.1f}" rx="8" fill="#ffffff" '
        'stroke="#cbd5e1" stroke-width="1"/>'
    )


def _wheel(cx: float, cy: float) -> str:
    """Return one cart wheel SVG element."""

    return (
        f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="#1e293b" '
        'stroke="#020617" stroke-width="1"/>'
    )


def _tick(x: float, y: float, label: str, anchor: str) -> str:
    """Return one labeled tick mark."""

    return (
        f'<line x1="{x:.1f}" y1="{y - 8.0:.1f}" x2="{x:.1f}" '
        f'y2="{y + 8.0:.1f}" stroke="#64748b" stroke-width="2"/>'
        + _text(x, y + 26.0, label, 11, "#64748b", "400", anchor)
    )


def _point(x: float, y: float, color: str, label: str) -> str:
    """Return one labeled chart point."""

    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" '
        'stroke="#ffffff" stroke-width="2"/>'
        + _text(x + 9.0, y - 8.0, label, 10, color, "700")
    )


def _text(
    x: float,
    y: float,
    value: str,
    size: int,
    color: str,
    weight: str,
    anchor: str = "start",
) -> str:
    """Return escaped SVG text."""

    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}" '
        f'text-anchor="{anchor}">{escape(value)}</text>'
    )


def _chart_range(
    initial_position_m: float,
    measured_positions_m: tuple[float, ...],
    measurement_noise_abs_m: float,
) -> tuple[float, float]:
    """Return public y-axis bounds for the measurement chart."""

    values = [initial_position_m]
    for value in measured_positions_m:
        values.extend(
            [value - measurement_noise_abs_m, value, value + measurement_noise_abs_m]
        )
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        margin = max(1.0, measurement_noise_abs_m * 4.0)
    else:
        margin = max((maximum - minimum) * 0.15, measurement_noise_abs_m * 2.0, 0.25)
    return minimum - margin, maximum + margin


def _scale(
    value: float,
    input_min: float,
    input_max: float,
    output_min: float,
    output_max: float,
) -> float:
    """Scale one value between numeric ranges."""

    if input_max == input_min:
        return (output_min + output_max) / 2.0
    unit = (value - input_min) / (input_max - input_min)
    return output_min + unit * (output_max - output_min)


def _scale_y(
    value: float,
    input_min: float,
    input_max: float,
    output_top: float,
    output_bottom: float,
) -> float:
    """Scale one y value into SVG coordinates."""

    return output_bottom - _scale(
        value, input_min, input_max, 0.0, output_bottom - output_top
    )


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp one float to an inclusive range."""

    return min(max(value, minimum), maximum)


def _fmt(value: float) -> str:
    """Format public numeric values compactly."""

    return f"{value:.6g}"
