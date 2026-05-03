"""SVG schematic drawing from deterministic circuit layouts."""

from cairosvg import svg2png
from html import escape
from typing import Mapping

from rlvr_physics.tasks.physics.circuits.layout import (
    Layout,
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
        layout planner runs before rendering.
    show_pin_labels:
        Whether to draw external pin labels. SVG-symbol rendering hides these
        labels by default to keep schematic images readable.

    Returns
    -------
    str
        SVG document text.
    """

    planned_layout = layout if layout is not None else plan_layout(circuit, catalog)
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
        ".wire{stroke:#1f2937;stroke-width:1.6;fill:none;stroke-linecap:round}",
        ".symbol{stroke:#111827;stroke-width:1.2;fill:none;stroke-linecap:round;stroke-linejoin:round}",
        ".symbol-fill{fill:#ffffff}",
        ".symbol-solid{stroke:#111827;stroke-width:1.0;fill:#111827;stroke-linejoin:round}",
        ".symbol-asset{overflow:hidden}",
        ".label{font:12px monospace;fill:#111827}",
        ".pin{stroke:#6b7280;stroke-width:1.4;fill:none}",
        "</style>",
        (
            f'<rect class="background" x="0" y="0" '
            f'width="{planned_layout.size.width:.0f}" '
            f'height="{planned_layout.size.height:.0f}" '
            f'fill="#ffffff"/>'
        ),
    ]
    for wire in planned_layout.wires:
        for segment in wire.segments:
            lines.append(
                f'<line class="wire" x1="{segment.start.x:.1f}" '
                f'y1="{segment.start.y:.1f}" x2="{segment.end.x:.1f}" '
                f'y2="{segment.end.y:.1f}"/>'
            )
    for part in planned_layout.parts:
        spec = catalog[parts[part.ref].kind]
        lines.extend(draw_asset_part(part, spec, parts[part.ref].value))
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
