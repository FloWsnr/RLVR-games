"""Editable exported SVG assets for circuit renderers."""

from dataclasses import dataclass
from functools import cache
from html import escape
from importlib.resources import files
from math import cos, radians, sin
from typing import Mapping, Optional
from xml.etree import ElementTree

from rlvr_physics.tasks.physics.circuits.layout import (
    Bounds,
    PlacedPart,
    Point,
    component_label_position,
    pin_position,
)
from rlvr_physics.tasks.physics.circuits.model import PartSpec, PinSide

_SVG_NS = "http://www.w3.org/2000/svg"
_NORMALIZED_STROKE_WIDTH = "1.25"
_SYMBOL_MASK_INSET = 1.5

ElementTree.register_namespace("", _SVG_NS)


@dataclass(frozen=True)
class SourcePoint:
    """One point in a symbol SVG coordinate system."""

    x: float
    y: float


@dataclass(frozen=True)
class SourceViewBox:
    """SVG symbol source coordinate bounds."""

    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class SymbolAsset:
    """Loaded editable exported SVG symbol asset.

    Parameters
    ----------
    key:
        Stable renderer key for the asset.
    view_box:
        Asset source coordinate bounds.
    anchors:
        Named terminal anchors in asset coordinates. Pin names are used first,
        then side names such as ``left`` and ``top``.
    fragments:
        Serialized SVG child fragments from the asset file.
    rotation_degrees:
        Rotation applied around the destination part center after scaling.
    """

    key: str
    view_box: SourceViewBox
    anchors: tuple[tuple[str, SourcePoint], ...]
    fragments: tuple[str, ...]
    rotation_degrees: float = 0.0

    def anchor(self, name: str) -> SourcePoint | None:
        """Return a named source anchor if this asset defines it."""

        for anchor_name, point in self.anchors:
            if anchor_name == name:
                return point
        return None


def draw_asset_part(part: PlacedPart, spec: PartSpec, value: str) -> list[str]:
    """Draw one placed part using only exported SVG assets.

    Parameters
    ----------
    part:
        Placed part to draw.
    spec:
        Static component specification.
    value:
        Rendered value label.

    Returns
    -------
    list[str]
        SVG fragments for the part.

    Raises
    ------
    ValueError
        Raised when no asset is registered for the part.
    """

    asset = _asset_for_rendered_part(part.kind, spec, value)
    if asset is None:
        raise ValueError(f"no SVG asset registered for part kind: {part.kind}")
    terminals = _asset_terminals(part, spec, asset)
    lines = [
        f'<g id="{escape(part.ref)}" class="circuit-symbol">',
        f"<title>{escape(part.ref)} {escape(spec.display_name)}</title>",
        (
            f'<rect class="symbol-mask" x="{part.bounds.x + _SYMBOL_MASK_INSET:.1f}" '
            f'y="{part.bounds.y + _SYMBOL_MASK_INSET:.1f}" '
            f'width="{part.bounds.width - 2.0 * _SYMBOL_MASK_INSET:.1f}" '
            f'height="{part.bounds.height - 2.0 * _SYMBOL_MASK_INSET:.1f}"/>'
        ),
    ]
    lines.extend(_lead_lines(part, spec, terminals))
    lines.extend(_asset_symbol(part, asset))
    label_text = f"{part.ref} {value}".strip()
    label_position = component_label_position(part, spec, value)
    lines.append(
        f'<text class="label-halo" x="{label_position.x:.1f}" '
        f'y="{label_position.y:.1f}">{escape(label_text)}</text>'
    )
    lines.append(
        f'<text class="label" x="{label_position.x:.1f}" '
        f'y="{label_position.y:.1f}">'
        f"{escape(label_text)}</text>"
    )
    lines.append("</g>")
    return lines


def svg_namespace_attributes() -> str:
    """Return namespace attributes required by the schematic SVG document."""

    return 'xmlns="http://www.w3.org/2000/svg"'


def asset_for_part(part_kind: str, spec: PartSpec) -> SymbolAsset | None:
    """Return the preferred editable SVG asset for a part kind."""

    asset_spec = _asset_spec_for_part(part_kind, spec, "")
    if asset_spec is None:
        return None
    return _load_asset(asset_spec)


def _asset_for_rendered_part(
    part_kind: str, spec: PartSpec, value: str
) -> SymbolAsset | None:
    """Return the asset variant for one rendered part instance."""

    asset_spec = _asset_spec_for_part(part_kind, spec, value)
    if asset_spec is None:
        return None
    return _load_asset(asset_spec)


def _asset_spec_for_part(
    part_kind: str, spec: PartSpec, value: str
) -> Optional["_AssetSpec"]:
    """Return static asset metadata for one part and display value."""

    if part_kind == "ideal_switch" and _is_closed_switch_value(value):
        return _SPST_SWITCH_CLOSED
    asset_spec = _ASSETS_BY_KIND.get(part_kind)
    if asset_spec is None:
        asset_spec = _ASSETS_BY_ICON.get(spec.icon)
    return asset_spec


def _is_closed_switch_value(value: str) -> bool:
    """Return whether a switch display value requests closed contacts."""

    return value.strip().lower() == "closed"


def anchor_name_for_side(side: PinSide) -> str:
    """Return the fallback asset anchor name for a visual pin side."""

    if side is PinSide.LEFT:
        return "left"
    if side is PinSide.RIGHT:
        return "right"
    if side is PinSide.TOP:
        return "top"
    return "bottom"


def symbol_fragments_for_scale(asset: SymbolAsset, scale: float) -> tuple[str, ...]:
    """Return asset fragments with scale-compensated stroke widths."""

    fragments: list[str] = []
    for fragment in asset.fragments:
        root = ElementTree.fromstring(fragment)
        _normalize_asset_tree(root, scale)
        fragments.append(ElementTree.tostring(root, encoding="unicode"))
    return tuple(fragments)


def _asset_symbol(part: PlacedPart, asset: SymbolAsset) -> list[str]:
    """Return SVG fragments placing one exported symbol asset."""

    view_box = asset.view_box
    frame, scale = _asset_frame(part, asset)
    rotation = ""
    if asset.rotation_degrees:
        rotation = (
            f' transform="rotate({asset.rotation_degrees:.1f} '
            f'{part.center.x:.1f} {part.center.y:.1f})"'
        )
    return [
        f'<g class="symbol-asset" data-symbol="{escape(asset.key)}"{rotation}>',
        (
            f'<g transform="translate({frame.x:.1f} {frame.y:.1f}) '
            f"scale({scale:.6f}) "
            f'translate({-view_box.x:.1f} {-view_box.y:.1f})">'
        ),
        *symbol_fragments_for_scale(asset, scale),
        "</g>",
        "</g>",
    ]


def _asset_terminals(
    part: PlacedPart,
    spec: PartSpec,
    asset: SymbolAsset,
) -> dict[str, Point]:
    """Return absolute terminals for a placed SVG asset."""

    terminals: dict[str, Point] = {}
    for pin in spec.pins:
        source = asset.anchor(pin.name)
        if source is None:
            source = asset.anchor(anchor_name_for_side(pin.side))
        if source is not None:
            terminals[pin.name] = _asset_point(part, asset, source)
    return terminals


def _asset_point(
    part: PlacedPart,
    asset: SymbolAsset,
    source: SourcePoint,
) -> Point:
    """Map an asset source point into placed schematic coordinates."""

    view_box = asset.view_box
    frame, scale = _asset_frame(part, asset)
    point = Point(
        frame.x + (source.x - view_box.x) * scale,
        frame.y + (source.y - view_box.y) * scale,
    )
    if not asset.rotation_degrees:
        return point
    angle = radians(asset.rotation_degrees)
    dx = point.x - part.center.x
    dy = point.y - part.center.y
    return Point(
        part.center.x + dx * cos(angle) - dy * sin(angle),
        part.center.y + dx * sin(angle) + dy * cos(angle),
    )


def _asset_frame(part: PlacedPart, asset: SymbolAsset) -> tuple[Bounds, float]:
    """Return the drawn asset frame and uniform source-to-destination scale."""

    view_box = asset.view_box
    target_width = part.bounds.width
    target_height = part.bounds.height
    if abs(asset.rotation_degrees) % 180.0 == 90.0:
        target_width, target_height = target_height, target_width
    scale = min(
        target_width / view_box.width,
        target_height / view_box.height,
    )
    drawn_width = view_box.width * scale
    drawn_height = view_box.height * scale
    return (
        Bounds(
            x=part.bounds.x + (part.bounds.width - drawn_width) / 2.0,
            y=part.bounds.y + (part.bounds.height - drawn_height) / 2.0,
            width=drawn_width,
            height=drawn_height,
        ),
        scale,
    )


def _lead_lines(
    part: PlacedPart,
    spec: PartSpec,
    terminals: Mapping[str, Point],
) -> list[str]:
    """Draw lead lines from layout pins to symbol asset terminals."""

    lines: list[str] = []
    for pin in spec.pins:
        terminal = terminals.get(pin.name)
        if terminal is None:
            continue
        anchor = pin_position(part, spec, pin.name)
        if anchor == terminal:
            continue
        if anchor.x == terminal.x or anchor.y == terminal.y:
            lines.append(_line(anchor, terminal))
            continue
        elbow = Point(terminal.x, anchor.y)
        lines.append(_line(anchor, elbow))
        lines.append(_line(elbow, terminal))
    return lines


def _line(start: Point, end: Point) -> str:
    """Return an SVG lead line fragment."""

    return (
        f'<line class="symbol" x1="{start.x:.1f}" y1="{start.y:.1f}" '
        f'x2="{end.x:.1f}" y2="{end.y:.1f}"/>'
    )


@cache
def _load_asset(asset_spec: "_AssetSpec") -> SymbolAsset:
    """Load and parse one editable SVG symbol asset."""

    asset_path = files("rlvr_physics.tasks.physics.circuits.assets").joinpath(
        asset_spec.filename
    )
    root = ElementTree.fromstring(asset_path.read_text(encoding="utf-8"))
    view_box = _parse_view_box(root)
    anchors = tuple(_asset_anchors(asset_spec, view_box))
    fragments = tuple(
        ElementTree.tostring(child, encoding="unicode")
        for child in list(root)
        if _local_name(child.tag) in {"defs", "g"}
    )
    return SymbolAsset(
        key=asset_spec.key,
        view_box=view_box,
        anchors=anchors,
        fragments=fragments,
        rotation_degrees=asset_spec.rotation_degrees,
    )


def _parse_view_box(root: ElementTree.Element) -> SourceViewBox:
    """Parse source bounds from an SVG root."""

    raw_view_box = root.attrib.get("viewBox")
    if raw_view_box is None:
        width = _parse_number(root.attrib["width"])
        height = _parse_number(root.attrib["height"])
        return SourceViewBox(x=0.0, y=0.0, width=width, height=height)
    parts = tuple(float(part) for part in raw_view_box.replace(",", " ").split())
    if len(parts) != 4:
        raise ValueError(f"invalid symbol viewBox: {raw_view_box!r}")
    return SourceViewBox(
        x=parts[0],
        y=parts[1],
        width=parts[2],
        height=parts[3],
    )


def _asset_anchors(
    asset_spec: "_AssetSpec", view_box: SourceViewBox
) -> tuple[tuple[str, SourcePoint], ...]:
    """Return named source anchors for an exported asset."""

    anchors: list[tuple[str, SourcePoint]] = []
    for anchor_name, point_name in asset_spec.anchor_names:
        anchors.append((anchor_name, _anchor_point(view_box, point_name)))
    return tuple(anchors)


def _anchor_point(view_box: SourceViewBox, point_name: str) -> SourcePoint:
    """Return a source anchor point from side or side-slot metadata."""

    if ":" not in point_name:
        if point_name == "left":
            return SourcePoint(view_box.x, view_box.y + view_box.height / 2.0)
        if point_name == "right":
            return SourcePoint(
                view_box.x + view_box.width, view_box.y + view_box.height / 2.0
            )
        if point_name == "top":
            return SourcePoint(view_box.x + view_box.width / 2.0, view_box.y)
        if point_name == "bottom":
            return SourcePoint(
                view_box.x + view_box.width / 2.0, view_box.y + view_box.height
            )
        raise ValueError(f"unknown symbol anchor point: {point_name!r}")

    side, slot_text = point_name.split(":", maxsplit=1)
    index_text, count_text = slot_text.split("/", maxsplit=1)
    index = int(index_text)
    count = int(count_text)
    slot = float(index) / float(count + 1)
    if side == "left":
        return SourcePoint(view_box.x, view_box.y + view_box.height * slot)
    if side == "right":
        return SourcePoint(
            view_box.x + view_box.width, view_box.y + view_box.height * slot
        )
    if side == "top":
        return SourcePoint(view_box.x + view_box.width * slot, view_box.y)
    if side == "bottom":
        return SourcePoint(
            view_box.x + view_box.width * slot, view_box.y + view_box.height
        )
    raise ValueError(f"unknown symbol anchor side: {side!r}")


def _parse_number(raw_value: str) -> float:
    """Parse a plain SVG numeric attribute."""

    number_text = raw_value.strip()
    for suffix in ("px", "pt", "mm"):
        if number_text.endswith(suffix):
            number_text = number_text.removesuffix(suffix)
            break
    return float(number_text)


def _normalize_asset_tree(root: ElementTree.Element, scale: float) -> None:
    """Normalize exported asset strokes for scaled schematic rendering."""

    for element in root.iter():
        if _local_name(element.tag) in {
            "path",
            "line",
            "polyline",
            "polygon",
            "rect",
            "circle",
            "ellipse",
        }:
            _normalize_stroked_element(element, scale)


def _normalize_stroked_element(element: ElementTree.Element, scale: float) -> None:
    """Keep stroke thickness constant after symbol scaling."""

    style = _parse_style(element.attrib.get("style", ""))
    stroke = style.get("stroke", element.attrib.get("stroke"))
    has_stroke = stroke is not None and stroke != "none"
    if (
        not has_stroke
        and "stroke-width" not in style
        and "stroke-width" not in element.attrib
    ):
        return
    if style.get("stroke") == "none" or element.attrib.get("stroke") == "none":
        return
    stroke_width = str(float(_NORMALIZED_STROKE_WIDTH) / scale)
    if style:
        style["stroke-width"] = stroke_width
        element.set("style", _format_style(style))
    else:
        element.set("stroke-width", stroke_width)


def _parse_style(raw_style: str) -> dict[str, str]:
    """Parse an inline SVG style attribute into a mapping."""

    style: dict[str, str] = {}
    for item in raw_style.split(";"):
        if ":" not in item:
            continue
        key, value = item.split(":", maxsplit=1)
        style[key.strip()] = value.strip()
    return style


def _format_style(style: dict[str, str]) -> str:
    """Return a deterministic SVG style attribute."""

    return ";".join(f"{key}:{value}" for key, value in style.items())


def _local_name(tag: str) -> str:
    """Return an XML tag local name without namespace."""

    if "}" not in tag:
        return tag
    return tag.rsplit("}", maxsplit=1)[1]


@dataclass(frozen=True)
class _AssetSpec:
    """Static placement metadata for one exported SVG file."""

    key: str
    filename: str
    anchor_names: tuple[tuple[str, str], ...]
    rotation_degrees: float = 0.0


_HORIZONTAL_TWO_PIN = (
    ("left", "bottom"),
    ("right", "top"),
    ("1", "bottom"),
    ("2", "top"),
    ("net", "bottom"),
    ("rail", "top"),
    ("a", "bottom"),
    ("k", "top"),
)

_VERTICAL_TWO_PIN = (
    ("top", "top"),
    ("bottom", "bottom"),
    ("p", "top"),
    ("n", "bottom"),
)

_DIRECT_HORIZONTAL_TWO_PIN = (
    ("left", "left"),
    ("right", "right"),
    ("1", "left"),
    ("2", "right"),
    ("net", "left"),
    ("rail", "right"),
    ("a", "left"),
    ("k", "right"),
)

_VARIABLE_RESISTOR = _AssetSpec(
    key="variable_resistor",
    filename="variable_resistor.svg",
    anchor_names=_HORIZONTAL_TWO_PIN,
    rotation_degrees=90.0,
)

_RESISTOR = _AssetSpec(
    key="resistor",
    filename="resistor.svg",
    anchor_names=_HORIZONTAL_TWO_PIN,
    rotation_degrees=90.0,
)

_CAPACITOR = _AssetSpec(
    key="capacitor",
    filename="capacitor.svg",
    anchor_names=_HORIZONTAL_TWO_PIN,
    rotation_degrees=90.0,
)

_INDUCTOR = _AssetSpec(
    key="inductor",
    filename="inductor.svg",
    anchor_names=_HORIZONTAL_TWO_PIN,
    rotation_degrees=90.0,
)

_LED = _AssetSpec(
    key="led",
    filename="led.svg",
    anchor_names=_HORIZONTAL_TWO_PIN,
    rotation_degrees=90.0,
)

_DIODE = _AssetSpec(
    key="diode",
    filename="diode.svg",
    anchor_names=_HORIZONTAL_TWO_PIN,
    rotation_degrees=90.0,
)

_ZENER = _AssetSpec(
    key="zener",
    filename="zener.svg",
    anchor_names=_HORIZONTAL_TWO_PIN,
    rotation_degrees=90.0,
)

_SPST_SWITCH = _AssetSpec(
    key="spst_switch",
    filename="spst_switch.svg",
    anchor_names=_HORIZONTAL_TWO_PIN,
    rotation_degrees=90.0,
)

_SPST_SWITCH_CLOSED = _AssetSpec(
    key="spst_switch_closed",
    filename="spst_switch_closed.svg",
    anchor_names=_HORIZONTAL_TWO_PIN,
    rotation_degrees=90.0,
)

_DC_SOURCE = _AssetSpec(
    key="dc_voltage_source",
    filename="dc_voltage_source.svg",
    anchor_names=_VERTICAL_TWO_PIN,
)

_AC_SOURCE = _AssetSpec(
    key="ac_voltage_source",
    filename="ac_voltage_source.svg",
    anchor_names=_VERTICAL_TWO_PIN,
)

_CURRENT_SOURCE = _AssetSpec(
    key="current_source_dc",
    filename="current_source_dc.svg",
    anchor_names=_VERTICAL_TWO_PIN,
)

_GROUND = _AssetSpec(
    key="ground",
    filename="ground.svg",
    anchor_names=(("0", "top"), ("top", "top")),
)

_NPN = _AssetSpec(
    key="npn",
    filename="npn.svg",
    anchor_names=(("b", "left"), ("c", "top"), ("e", "bottom")),
)

_PNP = _AssetSpec(
    key="pnp",
    filename="pnp.svg",
    anchor_names=(("b", "left"), ("e", "top"), ("c", "bottom")),
)

_NMOS = _AssetSpec(
    key="nmos",
    filename="nmos.svg",
    anchor_names=(("g", "left"), ("d", "top"), ("s", "bottom")),
)

_PMOS = _AssetSpec(
    key="pmos",
    filename="pmos.svg",
    anchor_names=(("g", "left"), ("s", "top"), ("d", "bottom")),
)

_OPAMP = _AssetSpec(
    key="op_amp",
    filename="op_amp.svg",
    anchor_names=(
        ("noninv", "left:1/2"),
        ("inv", "left:2/2"),
        ("out", "right"),
        ("vpos", "top"),
        ("vneg", "bottom"),
    ),
)

_TRANSFORMER = _AssetSpec(
    key="transformer",
    filename="transformer.svg",
    anchor_names=(
        ("p1", "left:1/2"),
        ("p2", "left:2/2"),
        ("s1", "right:1/2"),
        ("s2", "right:2/2"),
    ),
)

_CONTROLLED_SOURCE = _AssetSpec(
    key="controlled_source",
    filename="controlled_source.svg",
    anchor_names=(
        ("p", "right:1/2"),
        ("n", "right:2/2"),
        ("cp", "left:1/2"),
        ("cn", "left:2/2"),
    ),
)

_CONTROLLED_CURRENT_SOURCE = _AssetSpec(
    key="controlled_current_source",
    filename="controlled_current_source.svg",
    anchor_names=(
        ("p", "right:1/2"),
        ("n", "right:2/2"),
        ("cp", "left:1/2"),
        ("cn", "left:2/2"),
    ),
)

_RELAY = _AssetSpec(
    key="relay",
    filename="relay.svg",
    anchor_names=(
        ("coil_p", "left:1/2"),
        ("coil_n", "left:2/2"),
        ("com", "bottom"),
        ("no", "right:1/2"),
        ("nc", "right:2/2"),
    ),
)

_GENERIC_IC = _AssetSpec(
    key="generic_ic",
    filename="generic_ic.svg",
    anchor_names=(
        ("in1", "left:1/2"),
        ("in2", "left:2/2"),
        ("out1", "right"),
        ("vcc", "top"),
        ("gnd", "bottom"),
    ),
)

_CONNECTOR = _AssetSpec(
    key="connector",
    filename="connector.svg",
    anchor_names=(("1", "left:1/2"), ("2", "left:2/2")),
)

_METER = _AssetSpec(
    key="meter",
    filename="meter.svg",
    anchor_names=(
        ("p", "top"),
        ("n", "bottom"),
        ("left", "left"),
        ("right", "right"),
    ),
)

_LAMP = _AssetSpec(
    key="lamp",
    filename="lamp.svg",
    anchor_names=_DIRECT_HORIZONTAL_TWO_PIN,
)

_MOTOR = _AssetSpec(
    key="motor",
    filename="motor.svg",
    anchor_names=_DIRECT_HORIZONTAL_TWO_PIN,
)

_LOGIC = _AssetSpec(
    key="and_gate",
    filename="logic.svg",
    anchor_names=(
        ("in1", "left:1/2"),
        ("in2", "left:2/2"),
        ("out", "right"),
        ("vcc", "top"),
        ("gnd", "bottom"),
    ),
)

_OR_GATE = _AssetSpec(
    key="or_gate",
    filename="or_gate.svg",
    anchor_names=(
        ("in1", "left:1/2"),
        ("in2", "left:2/2"),
        ("out", "right"),
        ("vcc", "top"),
        ("gnd", "bottom"),
    ),
)

_NOT_GATE = _AssetSpec(
    key="not_gate",
    filename="not_gate.svg",
    anchor_names=(
        ("in1", "left"),
        ("out", "right"),
        ("vcc", "top"),
        ("gnd", "bottom"),
    ),
)

_ASSETS_BY_KIND = {
    "bjt_npn": _NPN,
    "bjt_pnp": _PNP,
    "current_source_dc": _CURRENT_SOURCE,
    "diode": _DIODE,
    "jfet_n": _NMOS,
    "jfet_p": _PMOS,
    "led": _LED,
    "mosfet_n": _NMOS,
    "mosfet_p": _PMOS,
    "not_gate": _NOT_GATE,
    "or_gate": _OR_GATE,
    "vccs": _CONTROLLED_CURRENT_SOURCE,
    "voltage_source_dc": _DC_SOURCE,
    "zener": _ZENER,
}

_ASSETS_BY_ICON = {
    "capacitor": _CAPACITOR,
    "connector": _CONNECTOR,
    "controlled_source": _CONTROLLED_SOURCE,
    "ground": _GROUND,
    "ic": _GENERIC_IC,
    "inductor": _INDUCTOR,
    "lamp": _LAMP,
    "logic": _LOGIC,
    "meter": _METER,
    "motor": _MOTOR,
    "opamp": _OPAMP,
    "relay": _RELAY,
    "resistor": _RESISTOR,
    "source": _DC_SOURCE,
    "switch": _SPST_SWITCH,
    "transformer": _TRANSFORMER,
}
