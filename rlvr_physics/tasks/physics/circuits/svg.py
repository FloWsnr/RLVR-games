"""SVG schematic drawing from deterministic circuit layouts."""

from cairosvg import svg2png
from html import escape
from typing import Mapping

from rlvr_physics.tasks.physics.circuits.layout import (
    Layout,
    NetLabel,
    Point,
    WireSegment,
    _plan_layout,
    pin_label_position,
    pin_position,
    plan_layout,
)
from rlvr_physics.tasks.physics.circuits.model import Circuit, PartSpec
from rlvr_physics.tasks.physics.circuits.symbol_assets import (
    draw_asset_part,
    svg_namespace_attributes,
)


def draw_svg(
    circuit: Circuit,
    catalog: Mapping[str, PartSpec],
    layout: Layout | None = None,
    *,
    show_pin_labels: bool = False,
) -> str:
    """Draw a circuit layout as deterministic SVG text.

    Parameters
    ----------
    circuit:
        Circuit being rendered.
    catalog:
        Component catalog.
    layout:
        Precomputed deterministic layout. When omitted, the force-directed
        layout planner runs before rendering. If pin labels are shown and no
        layout is supplied, routing reserves space for the pin labels. Supplied
        layouts are rendered verbatim, so callers that pass both a layout and
        ``show_pin_labels=True`` are responsible for label clearance.
    show_pin_labels:
        Whether to draw external pin labels. SVG-symbol rendering hides these
        labels by default to keep schematic images readable.

    Returns
    -------
    str
        SVG document text.
    """

    if layout is not None:
        planned_layout = layout
    elif show_pin_labels:
        planned_layout = _plan_layout(circuit, catalog, route_pin_labels=True)
    else:
        planned_layout = plan_layout(circuit, catalog)
    parts = circuit.part_by_ref()
    lines = [
        (
            f"<svg {svg_namespace_attributes()} "
            f'width="{planned_layout.size.width:.0f}" '
            f'height="{planned_layout.size.height:.0f}" '
            f'viewBox="0 0 {planned_layout.size.width:.0f} '
            f'{planned_layout.size.height:.0f}">'
        ),
        "<style>",
        ".wire{stroke:#1f2937;stroke-width:1.35;fill:none;stroke-linecap:round}",
        ".symbol{stroke:#111827;stroke-width:1.25;fill:none;stroke-linecap:round;stroke-linejoin:round}",
        ".symbol-fill{fill:#ffffff}",
        ".symbol-solid{stroke:#111827;stroke-width:1.0;fill:#111827;stroke-linejoin:round}",
        ".symbol-mask{fill:#ffffff}",
        ".symbol-asset{overflow:hidden}",
        ".label{font:11px monospace;fill:#111827}",
        ".label-halo{font:11px monospace;fill:none;stroke:#ffffff;stroke-width:3;stroke-linejoin:round}",
        ".net-label{font:10px monospace;font-weight:700;fill:#374151}",
        ".net-label-halo{font:10px monospace;font-weight:700;fill:none;stroke:#ffffff;stroke-width:3;stroke-linejoin:round}",
        ".pin{stroke:#6b7280;stroke-width:1.25;fill:none}",
        ".junction{fill:#111827}",
        "</style>",
        (
            f'<rect class="background" x="0" y="0" '
            f'width="{planned_layout.size.width:.0f}" '
            f'height="{planned_layout.size.height:.0f}" '
            f'fill="#ffffff"/>'
        ),
    ]
    for wire in planned_layout.wires:
        for segment in _rendered_wire_segments(wire.segments):
            lines.append(
                f'<line class="wire" x1="{segment.start.x:.1f}" '
                f'y1="{segment.start.y:.1f}" x2="{segment.end.x:.1f}" '
                f'y2="{segment.end.y:.1f}"/>'
            )
    for part in planned_layout.parts:
        spec = catalog[parts[part.ref].kind]
        lines.extend(draw_asset_part(part, spec, parts[part.ref].value))
    for point in _terminal_points(circuit, catalog, planned_layout):
        lines.append(
            f'<circle class="junction" cx="{point.x:.1f}" cy="{point.y:.1f}" r="1.1"/>'
        )
    for wire in planned_layout.wires:
        for point in _junction_points(wire.segments):
            lines.append(
                f'<circle class="junction" cx="{point.x:.1f}" cy="{point.y:.1f}" r="1.6"/>'
            )
    for label in planned_layout.net_labels:
        lines.extend(_draw_net_label(label))
    for part in planned_layout.parts:
        spec = catalog[parts[part.ref].kind]
        if show_pin_labels:
            for pin in spec.pins:
                pt = pin_position(part, spec, pin.name)
                label_position = pin_label_position(pt, pin.side, pin.name)
                lines.append(
                    f'<text class="label" x="{label_position.x:.1f}" '
                    f'y="{label_position.y:.1f}">{escape(pin.name)}</text>'
                )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _draw_net_label(label: NetLabel) -> tuple[str, str]:
    """Return SVG fragments for one local net label."""

    escaped_text = escape(label.text)
    return (
        f'<text class="net-label-halo" x="{label.position.x:.1f}" '
        f'y="{label.position.y:.1f}">{escaped_text}</text>',
        f'<text class="net-label" x="{label.position.x:.1f}" '
        f'y="{label.position.y:.1f}">{escaped_text}</text>',
    )


def draw_png(
    circuit: Circuit,
    catalog: Mapping[str, PartSpec],
    layout: Layout | None = None,
    *,
    show_pin_labels: bool = False,
) -> bytes:
    """Draw a circuit as PNG bytes.

    Parameters
    ----------
    circuit:
        Circuit being rendered.
    catalog:
        Component catalog.
    layout:
        Precomputed deterministic layout. When omitted, the force-directed
        layout planner runs before rendering.
    show_pin_labels:
        Whether to draw external pin labels.

    Returns
    -------
    bytes
        Encoded PNG image bytes.
    """

    return to_png(
        draw_svg(
            circuit,
            catalog,
            layout,
            show_pin_labels=show_pin_labels,
        )
    )


def _rendered_wire_segments(
    segments: tuple[WireSegment, ...],
) -> tuple[WireSegment, ...]:
    """Return collinear wire segments merged for SVG drawing."""

    horizontal: dict[float, list[tuple[float, float]]] = {}
    vertical: dict[float, list[tuple[float, float]]] = {}
    passthrough: list[WireSegment] = []
    for segment in segments:
        if segment.start.y == segment.end.y:
            y = segment.start.y
            horizontal.setdefault(y, []).append(
                (
                    min(segment.start.x, segment.end.x),
                    max(segment.start.x, segment.end.x),
                )
            )
        elif segment.start.x == segment.end.x:
            x = segment.start.x
            vertical.setdefault(x, []).append(
                (
                    min(segment.start.y, segment.end.y),
                    max(segment.start.y, segment.end.y),
                )
            )
        else:
            passthrough.append(segment)

    rendered: list[WireSegment] = []
    for y, intervals in sorted(horizontal.items()):
        for start, end in _merged_intervals(intervals):
            rendered.append(WireSegment(Point(start, y), Point(end, y)))
    for x, intervals in sorted(vertical.items()):
        for start, end in _merged_intervals(intervals):
            rendered.append(WireSegment(Point(x, start), Point(x, end)))
    rendered.extend(passthrough)
    return tuple(rendered)


def _merged_intervals(
    intervals: list[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    """Return sorted intervals with overlapping ranges collapsed."""

    merged: list[tuple[float, float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return tuple(merged)


def _terminal_points(
    circuit: Circuit,
    catalog: Mapping[str, PartSpec],
    layout: Layout,
) -> tuple[Point, ...]:
    """Return rendered pin terminal points for connected pins."""

    parts = circuit.part_by_ref()
    placed = layout.part_by_ref()
    points: list[Point] = []
    for connection in circuit.connections:
        if len(circuit.connections_for_net(connection.net)) < 2:
            continue
        part = parts[connection.ref]
        points.append(
            pin_position(
                placed[connection.ref],
                catalog[part.kind],
                connection.pin,
            )
        )
    return tuple(points)


def _junction_points(segments: tuple[WireSegment, ...]) -> tuple[Point, ...]:
    """Return same-net wire junction points for SVG drawing."""

    counts: dict[tuple[float, float], int] = {}
    for segment in segments:
        for point in (segment.start, segment.end):
            key = _point_key(point)
            counts[key] = counts.get(key, 0) + 1
    junctions = {key for key, count in counts.items() if count >= 3}
    for first_index, first in enumerate(segments):
        for second in segments[first_index + 1 :]:
            for point in _segment_intersections(first, second):
                if not (
                    _segment_endpoint_matches(first, point)
                    and _segment_endpoint_matches(second, point)
                ):
                    junctions.add(point)
    return tuple(Point(x, y) for x, y in sorted(junctions))


def _segment_endpoint_matches(segment: WireSegment, point: tuple[float, float]) -> bool:
    """Return whether a rounded point is an endpoint of a segment."""

    return point in {_point_key(segment.start), _point_key(segment.end)}


def _segment_intersections(
    first: WireSegment, second: WireSegment
) -> tuple[tuple[float, float], ...]:
    """Return simple axis-aligned segment intersections."""

    first_vertical = first.start.x == first.end.x
    second_vertical = second.start.x == second.end.x
    if first_vertical == second_vertical:
        return ()
    vertical = first if first_vertical else second
    horizontal = second if first_vertical else first
    x = vertical.start.x
    y = horizontal.start.y
    if min(horizontal.start.x, horizontal.end.x) <= x <= max(
        horizontal.start.x, horizontal.end.x
    ) and min(vertical.start.y, vertical.end.y) <= y <= max(
        vertical.start.y, vertical.end.y
    ):
        return (_point_key(Point(x, y)),)
    return ()


def _point_key(point: Point) -> tuple[float, float]:
    """Return a rounded point key for rendered geometry."""

    return (round(point.x, 6), round(point.y, 6))


def to_png(svg: str) -> bytes:
    """Rasterize SVG text to PNG bytes.

    Parameters
    ----------
    svg:
        SVG document text.

    Returns
    -------
    bytes
        Encoded PNG image bytes.
    """

    png = svg2png(bytestring=svg.encode("utf-8"))
    if not isinstance(png, bytes):
        raise ValueError("SVG rasterization did not return PNG bytes")
    return png
