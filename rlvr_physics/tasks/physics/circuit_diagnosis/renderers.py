"""Renderers for the circuit diagnosis task."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import svg

from rlvr_physics.core.rendering import (
    RenderedObservation,
    image_observation,
    text_observation,
)
from rlvr_physics.tasks._shared.rendering import rasterize_svg
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone import (
    CircuitComponent,
    CircuitDefinition,
    ReplacementSpec,
    SourceSetting,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.prompting import (
    circuit_image_prompt_template,
    circuit_text_prompt_template,
    render_prompt_template,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.specs import (
    CIRCUIT_IMAGE_RENDERER,
    CIRCUIT_TEXT_RENDERER,
    validate_circuit_renderer_type,
)

SVG_FONT_FAMILY = "Arial, sans-serif"


@dataclass(frozen=True)
class CircuitRenderContext:
    """Public current-turn circuit state used by renderers.

    Parameters
    ----------
    definition:
        Public nominal circuit description.
    node_positions:
        Renderer-facing schematic coordinates for public nodes.
    feedback:
        Latest public feedback shown to the model.
    source_setting:
        Current bench source setting, when connected.
    repairs:
        Accepted public repair overlays.
    budget_status:
        Public budget status lines.
    """

    definition: CircuitDefinition
    node_positions: Mapping[str, tuple[float, float]]
    feedback: str
    source_setting: SourceSetting | None
    repairs: Mapping[str, ReplacementSpec]
    budget_status: str


def render_circuit_observation(
    renderer_type: str, context: CircuitRenderContext
) -> RenderedObservation:
    """Render one circuit diagnosis observation."""

    validate_circuit_renderer_type(renderer_type)
    if renderer_type == CIRCUIT_TEXT_RENDERER:
        return text_observation(CIRCUIT_TEXT_RENDERER, render_circuit_text(context))
    if renderer_type == CIRCUIT_IMAGE_RENDERER:
        return render_circuit_image(context)
    raise AssertionError(f"validated renderer was not handled: {renderer_type}")


def render_circuit_text(context: CircuitRenderContext) -> str:
    """Build the text-only observation for one circuit diagnosis turn."""

    return _render_prompt(
        circuit_text_prompt_template(),
        context,
        netlist=_render_netlist(context.definition),
    )


def render_circuit_image(context: CircuitRenderContext) -> RenderedObservation:
    """Build a PNG schematic observation for one circuit diagnosis turn."""

    text = _render_prompt(circuit_image_prompt_template(), context, netlist="")
    svg_payload = _render_circuit_svg(context)
    raster_image = rasterize_svg(svg_payload)
    return image_observation(
        renderer_name=CIRCUIT_IMAGE_RENDERER,
        data=raster_image.data,
        mime_type=raster_image.mime_type,
        alt_text="Schematic of the nominal DC circuit with labeled nodes and components.",
        text=text,
    )


def _render_prompt(template: str, context: CircuitRenderContext, netlist: str) -> str:
    """Render a circuit prompt template."""

    return render_prompt_template(
        template,
        {
            "feedback": context.feedback,
            "netlist": netlist,
            "target_behavior": _render_target_behavior(context.definition),
            "source_setting": _render_source_setting(context.source_setting),
            "repair_state": _render_repair_state(context.repairs),
            "budget_status": context.budget_status,
        },
    )


def _render_netlist(definition: CircuitDefinition) -> str:
    """Return a compact text netlist for the public circuit."""

    lines = [
        f"- nodes: {', '.join(definition.nodes)}",
        f"- ground: {definition.ground_node}",
    ]
    for component in definition.components:
        lines.append(
            "- "
            f"{component.component_id}: {component.kind} "
            f"{component.node_a} -> {component.node_b}; "
            f"{_component_parameter_text(component)}"
        )
    return "\n".join(lines)


def _render_target_behavior(definition: CircuitDefinition) -> str:
    """Return public target behavior text."""

    lines: list[str] = [definition.description]
    if definition.target_source is None:
        lines.append("- target source: use the circuit's internal source")
    else:
        source = definition.target_source
        lines.append(
            "- target source: "
            f"{source.node_plus} relative to {source.node_minus} = "
            f"{_fmt(source.voltage_V)} V"
        )
    for check in definition.target_checks:
        if check.kind == "voltage_between":
            lines.append(
                "- "
                f"{check.check_id}: V({_param(check.parameters, 'node_a')}, "
                f"{_param(check.parameters, 'node_b')}) in "
                f"[{_fmt(_num(check.parameters, 'min_V'))}, "
                f"{_fmt(_num(check.parameters, 'max_V'))}] V"
            )
        elif check.kind == "current_range":
            lines.append(
                "- "
                f"{check.check_id}: I({_param(check.parameters, 'component')}) in "
                f"[{_fmt(_num(check.parameters, 'min_A'))}, "
                f"{_fmt(_num(check.parameters, 'max_A'))}] A"
            )
        elif check.kind == "power_max":
            lines.append(
                "- "
                f"{check.check_id}: P({_param(check.parameters, 'component')}) <= "
                f"{_fmt(_num(check.parameters, 'max_W'))} W"
            )
    return "\n".join(lines)


def _render_source_setting(source_setting: SourceSetting | None) -> str:
    """Return text for the current bench source."""

    if source_setting is None:
        return "- no external bench source connected"
    return (
        "- "
        f"{source_setting.node_plus} relative to {source_setting.node_minus} = "
        f"{_fmt(source_setting.voltage_V)} V"
    )


def _render_repair_state(repairs: Mapping[str, ReplacementSpec]) -> str:
    """Return text for accepted repair overlays."""

    if len(repairs) == 0:
        return "- no components replaced"
    lines: list[str] = []
    for component_id in sorted(repairs):
        repair = repairs[component_id]
        parameters = ", ".join(
            f"{key}={_fmt_value(value)}" for key, value in repair.parameters.items()
        )
        lines.append(f"- {component_id}: {repair.kind}, {parameters}")
    return "\n".join(lines)


def _render_circuit_svg(context: CircuitRenderContext) -> str:
    """Return a deterministic SVG schematic for the public circuit."""

    definition = context.definition
    width = 960
    height = 640
    elements: list[svg.Element] = [
        svg.Rect(x=0.0, y=0.0, width=width, height=height, fill="#f8fafc"),
        _text(34.0, 44.0, "Circuit diagnosis", 23, "#0f172a", "700"),
        _text(34.0, 68.0, definition.description, 13, "#475569", "400"),
    ]
    for component in definition.components:
        elements.extend(
            _component_elements(definition, context.node_positions, component)
        )
    for node_name, position in context.node_positions.items():
        elements.extend(_node_elements(definition, node_name, position))
    elements.extend(_side_panel(context))
    return str(
        svg.SVG(
            width=width,
            height=height,
            viewBox=svg.ViewBoxSpec(0, 0, width, height),
            extra={"role": "img"},
            elements=elements,
        )
    )


def _component_elements(
    definition: CircuitDefinition,
    node_positions: Mapping[str, tuple[float, float]],
    component: CircuitComponent,
) -> list[svg.Element]:
    """Return SVG elements for one component."""

    del definition
    x1, y1 = node_positions[component.node_a]
    x2, y2 = node_positions[component.node_b]
    mid_x = (x1 + x2) / 2.0
    mid_y = (y1 + y2) / 2.0
    elements: list[svg.Element] = [
        svg.Line(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            stroke="#64748b",
            stroke_width=3,
        )
    ]
    if component.kind == "resistor":
        elements.extend(_resistor_symbol(x1, y1, x2, y2))
    elif component.kind == "capacitor":
        elements.extend(_capacitor_symbol(mid_x, mid_y))
    elif component.kind == "diode":
        elements.extend(_diode_symbol(mid_x, mid_y))
    elif component.kind == "switch":
        elements.extend(_switch_symbol(mid_x, mid_y))
    elif component.kind == "voltage_source":
        elements.extend(_source_symbol(mid_x, mid_y))
    elements.append(
        _label_box(
            mid_x,
            mid_y - 28.0,
            f"{component.component_id} {_component_short_value(component)}",
        )
    )
    return elements


def _node_elements(
    definition: CircuitDefinition, node_name: str, position: tuple[float, float]
) -> list[svg.Element]:
    """Return SVG elements for one labeled node."""

    x, y = position
    elements: list[svg.Element] = [
        svg.Circle(cx=x, cy=y, r=7.0, fill="#0f172a"),
        _text(x + 12.0, y - 10.0, node_name, 14, "#0f172a", "700"),
    ]
    if node_name == definition.ground_node:
        elements.extend(_ground_symbol(x, y + 18.0))
    return elements


def _side_panel(context: CircuitRenderContext) -> list[svg.Element]:
    """Return the schematic side panel with target and feedback."""

    x = 710.0
    y = 96.0
    width = 214.0
    height = 488.0
    return [
        svg.Rect(
            x=x,
            y=y,
            width=width,
            height=height,
            rx=6,
            fill="#ffffff",
            stroke="#cbd5e1",
            stroke_width=1,
        ),
        _text(x + 16.0, y + 30.0, "Target", 16, "#0f172a", "700"),
        *_wrapped_text(
            x + 16.0,
            y + 54.0,
            _compact_target_text(context.definition),
            176.0,
            12,
            "#334155",
        ),
        _text(x + 16.0, y + 236.0, "Latest feedback", 16, "#0f172a", "700"),
        *_wrapped_text(
            x + 16.0,
            y + 260.0,
            context.feedback,
            176.0,
            12,
            "#334155",
        ),
    ]


def _resistor_symbol(x1: float, y1: float, x2: float, y2: float) -> list[svg.Element]:
    """Return a simple resistor symbol centered on a connection."""

    mid_x = (x1 + x2) / 2.0
    mid_y = (y1 + y2) / 2.0
    return [
        svg.Rect(
            x=mid_x - 34.0,
            y=mid_y - 13.0,
            width=68.0,
            height=26.0,
            rx=3,
            fill="#fef3c7",
            stroke="#92400e",
            stroke_width=2,
        )
    ]


def _capacitor_symbol(mid_x: float, mid_y: float) -> list[svg.Element]:
    """Return a capacitor symbol."""

    return [
        svg.Line(
            x1=mid_x - 10.0,
            y1=mid_y - 24.0,
            x2=mid_x - 10.0,
            y2=mid_y + 24.0,
            stroke="#0369a1",
            stroke_width=4,
        ),
        svg.Line(
            x1=mid_x + 10.0,
            y1=mid_y - 24.0,
            x2=mid_x + 10.0,
            y2=mid_y + 24.0,
            stroke="#0369a1",
            stroke_width=4,
        ),
    ]


def _diode_symbol(mid_x: float, mid_y: float) -> list[svg.Element]:
    """Return a diode symbol."""

    elements: list[svg.Element] = [
        svg.Polygon(
            points=[
                svg.Point(mid_x - 28.0, mid_y - 22.0),
                svg.Point(mid_x - 28.0, mid_y + 22.0),
                svg.Point(mid_x + 10.0, mid_y),
            ],
            fill="#dbeafe",
            stroke="#1d4ed8",
            stroke_width=2,
        ),
        svg.Line(
            x1=mid_x + 16.0,
            y1=mid_y - 24.0,
            x2=mid_x + 16.0,
            y2=mid_y + 24.0,
            stroke="#1d4ed8",
            stroke_width=4,
        ),
    ]
    return elements


def _switch_symbol(mid_x: float, mid_y: float) -> list[svg.Element]:
    """Return a switch symbol."""

    return [
        svg.Circle(cx=mid_x - 28.0, cy=mid_y, r=4.0, fill="#334155"),
        svg.Circle(cx=mid_x + 28.0, cy=mid_y, r=4.0, fill="#334155"),
        svg.Line(
            x1=mid_x - 24.0,
            y1=mid_y - 2.0,
            x2=mid_x + 24.0,
            y2=mid_y - 18.0,
            stroke="#334155",
            stroke_width=4,
        ),
    ]


def _source_symbol(mid_x: float, mid_y: float) -> list[svg.Element]:
    """Return a voltage source symbol."""

    return [
        svg.Circle(
            cx=mid_x,
            cy=mid_y,
            r=28.0,
            fill="#f0fdf4",
            stroke="#15803d",
            stroke_width=2,
        ),
        _text(mid_x, mid_y - 4.0, "+", 16, "#15803d", "700", "middle"),
        _text(mid_x, mid_y + 16.0, "-", 16, "#15803d", "700", "middle"),
    ]


def _ground_symbol(x: float, y: float) -> list[svg.Element]:
    """Return a ground symbol."""

    return [
        svg.Line(
            x1=x - 18.0, y1=y, x2=x + 18.0, y2=y, stroke="#334155", stroke_width=3
        ),
        svg.Line(
            x1=x - 12.0,
            y1=y + 8.0,
            x2=x + 12.0,
            y2=y + 8.0,
            stroke="#334155",
            stroke_width=3,
        ),
        svg.Line(
            x1=x - 6.0,
            y1=y + 16.0,
            x2=x + 6.0,
            y2=y + 16.0,
            stroke="#334155",
            stroke_width=3,
        ),
    ]


def _label_box(x: float, y: float, label: str) -> svg.Element:
    """Return a label in a schematic callout."""

    return svg.G(
        elements=[
            svg.Rect(
                x=x - 54.0,
                y=y - 16.0,
                width=108.0,
                height=24.0,
                rx=4,
                fill="#ffffff",
                stroke="#cbd5e1",
                stroke_width=1,
            ),
            _text(x, y, label, 12, "#0f172a", "700", "middle"),
        ]
    )


def _text(
    x: float,
    y: float,
    text: str,
    size: int,
    fill: str,
    weight: Literal["400", "700"],
    anchor: Literal["start", "middle", "end"] = "start",
) -> svg.Text:
    """Return an SVG text element."""

    return svg.Text(
        x=x,
        y=y,
        text=text,
        font_size=size,
        font_family=SVG_FONT_FAMILY,
        font_weight=weight,
        fill=fill,
        text_anchor=anchor,
    )


def _wrapped_text(
    x: float, y: float, text: str, width: float, size: int, fill: str
) -> list[svg.Element]:
    """Return simple wrapped SVG text lines."""

    del width
    words = text.replace("\n", " ").split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if current == "" else f"{current} {word}"
        if len(candidate) > 28:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current != "":
        lines.append(current)
    return [
        _text(x, y + index * (size + 5), line, size, fill, "400")
        for index, line in enumerate(lines[:12])
    ]


def _compact_target_text(definition: CircuitDefinition) -> str:
    """Return a compact target summary for the image side panel."""

    if definition.target_source is None:
        source = "internal source"
    else:
        target_source = definition.target_source
        source = (
            f"{target_source.node_plus}-{target_source.node_minus} "
            f"{_fmt(target_source.voltage_V)} V"
        )
    checks = ", ".join(check.check_id for check in definition.target_checks)
    return f"Use {source}; satisfy {checks}."


def _component_parameter_text(component: CircuitComponent) -> str:
    """Return public component parameter text."""

    if component.kind == "resistor":
        return f"value={_fmt(_num(component.parameters, 'value_ohm'))} ohm"
    if component.kind == "capacitor":
        return f"value={_fmt(_num(component.parameters, 'value_F'))} F"
    if component.kind == "diode":
        return (
            f"anode={component.node_a}, cathode={component.node_b}, "
            f"forward_drop={_fmt(_num(component.parameters, 'forward_drop_V'))} V"
        )
    if component.kind == "switch":
        state = "closed" if bool(component.parameters["closed"]) else "open"
        return f"nominal_state={state}"
    if component.kind == "voltage_source":
        return f"value={_fmt(_num(component.parameters, 'voltage_V'))} V"
    return ", ".join(
        f"{key}={_fmt_value(value)}" for key, value in component.parameters.items()
    )


def _component_short_value(component: CircuitComponent) -> str:
    """Return compact component value text for image labels."""

    if component.kind == "resistor":
        return f"{_fmt(_num(component.parameters, 'value_ohm'))} ohm"
    if component.kind == "capacitor":
        return f"{_fmt(_num(component.parameters, 'value_F'))} F"
    if component.kind == "diode":
        return f"{_fmt(_num(component.parameters, 'forward_drop_V'))} V"
    if component.kind == "switch":
        return "closed" if bool(component.parameters["closed"]) else "open"
    if component.kind == "voltage_source":
        return f"{_fmt(_num(component.parameters, 'voltage_V'))} V"
    return component.kind


def _param(values: Mapping[str, object], name: str) -> str:
    """Return a string parameter for rendering."""

    value = values[name]
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _num(values: Mapping[str, object], name: str) -> float:
    """Return a numeric parameter for rendering."""

    value = values[name]
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    if isinstance(value, int | float):
        return float(value)
    raise TypeError(f"{name} must be numeric")


def _fmt_value(value: object) -> str:
    """Return compact rendered text for a value."""

    if isinstance(value, int | float) and not isinstance(value, bool):
        return _fmt(float(value))
    return str(value)


def _fmt(value: float) -> str:
    """Return a compact numeric string."""

    return f"{value:.6g}"
