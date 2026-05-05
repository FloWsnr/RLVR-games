"""Editable exported SVG assets for circuit renderers."""

from dataclasses import dataclass
from functools import cache
from html import escape
from importlib.resources import files
from math import cos, radians, sin
import re
from typing import Mapping
from xml.etree import ElementTree

from rlvr_physics.tasks.physics.circuits.layout import (
    Bounds,
    PlacedPart,
    Point,
    component_label_bounds_for_symbol_bounds,
    component_label_position_from_bounds,
    pin_position,
)
from rlvr_physics.tasks.physics.circuits.model import PartInstance, PartSpec

_SVG_NS = "http://www.w3.org/2000/svg"
_NORMALIZED_STROKE_WIDTH = "1.25"
_SYMBOL_MASK_INSET = 1.5
_SWITCH_CLOSED_MAX_RESISTANCE_OHM = 1.0e6
_TRANSFORM_PATTERN = re.compile(r"([A-Za-z]+)\(([^)]*)\)")

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
class _SvgTransform:
    """Two-dimensional SVG affine transform matrix."""

    a: float
    b: float
    c: float
    d: float
    e: float
    f: float

    @classmethod
    def identity(cls) -> "_SvgTransform":
        """Return an identity affine transform."""

        return cls(a=1.0, b=0.0, c=0.0, d=1.0, e=0.0, f=0.0)

    def compose(self, other: "_SvgTransform") -> "_SvgTransform":
        """Return this transform followed by ``other`` in child coordinates."""

        return _SvgTransform(
            a=self.a * other.a + self.c * other.b,
            b=self.b * other.a + self.d * other.b,
            c=self.a * other.c + self.c * other.d,
            d=self.b * other.c + self.d * other.d,
            e=self.a * other.e + self.c * other.f + self.e,
            f=self.b * other.e + self.d * other.f + self.f,
        )

    def apply(self, point: SourcePoint) -> SourcePoint:
        """Return ``point`` transformed by this matrix."""

        return SourcePoint(
            x=self.a * point.x + self.c * point.y + self.e,
            y=self.b * point.x + self.d * point.y + self.f,
        )


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
        Named terminal anchors in asset coordinates. Names must match catalog
        pin names.
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


def draw_asset_part(
    part: PlacedPart,
    spec: PartSpec,
    instance: PartInstance,
    *,
    draw_leads: bool = True,
) -> list[str]:
    """Draw one placed part using only exported SVG assets.

    Parameters
    ----------
    part:
        Placed part to draw.
    spec:
        Static component specification.
    instance:
        Canonical component instance to render.
    draw_leads:
        Whether to draw corrective lead lines from layout side slots to asset
        terminals. Full schematic rendering routes wires to asset terminals
        directly and disables these corrective leads.

    Returns
    -------
    list[str]
        SVG fragments for the part.

    Raises
    ------
    ValueError
        Raised when no asset is registered for the part.
    """

    asset = _asset_for_rendered_part(part.kind, spec, instance)
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
    if draw_leads:
        lines.extend(_lead_lines(part, spec, terminals))
    lines.extend(_asset_symbol(part, asset))
    label_text = f"{part.ref} {instance.value}".strip()
    label_position = component_label_position_from_bounds(
        _component_label_bounds_for_asset(part, spec, instance, asset)
    )
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

    asset_spec = _asset_spec_for_part(part_kind, spec, None)
    if asset_spec is None:
        return None
    return _load_asset(asset_spec)


def asset_render_bounds_for_part(
    part: PlacedPart,
    spec: PartSpec,
    instance: PartInstance,
) -> Bounds:
    """Return bounds of the rendered SVG asset frame for one placed part.

    Parameters
    ----------
    part:
        Placed part to inspect.
    spec:
        Static component specification.
    instance:
        Canonical component instance to render.

    Returns
    -------
    Bounds
        Axis-aligned bounds of the drawn asset frame after scaling and
        rotation.
    """

    asset = _asset_for_rendered_part(part.kind, spec, instance)
    if asset is None:
        return part.bounds
    return _asset_render_bounds(part, asset)


def asset_component_label_bounds(
    part: PlacedPart,
    spec: PartSpec,
    instance: PartInstance,
) -> Bounds:
    """Return component-label bounds anchored to the rendered SVG asset.

    Parameters
    ----------
    part:
        Placed part whose label is being rendered.
    spec:
        Static component specification.
    instance:
        Canonical component instance to render.

    Returns
    -------
    Bounds
        Approximate rendered bounds for the component label.
    """

    asset = _asset_for_rendered_part(part.kind, spec, instance)
    if asset is None:
        return component_label_bounds_for_symbol_bounds(
            part,
            spec,
            instance.value,
            part.bounds,
        )
    return _component_label_bounds_for_asset(part, spec, instance, asset)


def asset_terminals_for_part(
    part: PlacedPart,
    spec: PartSpec,
    instance: PartInstance,
) -> Mapping[str, Point]:
    """Return rendered SVG terminal coordinates keyed by pin name.

    Parameters
    ----------
    part:
        Placed part to inspect.
    spec:
        Static component specification.
    instance:
        Canonical component instance to render.
    """

    asset = _asset_for_rendered_part(part.kind, spec, instance)
    if asset is None:
        return {}
    return _asset_terminals(part, spec, asset)


def _asset_for_rendered_part(
    part_kind: str, spec: PartSpec, instance: PartInstance
) -> SymbolAsset | None:
    """Return the asset variant for one rendered part instance."""

    asset_spec = _asset_spec_for_part(part_kind, spec, instance)
    if asset_spec is None:
        return None
    return _load_asset(asset_spec)


def _asset_spec_for_part(
    part_kind: str, spec: PartSpec, instance: PartInstance | None
) -> "_AssetSpec | None":
    """Return static asset metadata for one part and display value."""

    if (
        part_kind == "ideal_switch"
        and instance is not None
        and _is_closed_switch(instance)
    ):
        return _SPST_SWITCH_CLOSED
    asset_spec = _ASSETS_BY_KIND.get(part_kind)
    if asset_spec is None:
        asset_spec = _ASSETS_BY_ICON.get(spec.icon)
    return asset_spec


def _is_closed_switch(instance: PartInstance) -> bool:
    """Return whether an ideal switch instance should render closed."""

    resistance = _numeric_parameter(instance, "state_resistance_ohm")
    if resistance is not None:
        return resistance <= _SWITCH_CLOSED_MAX_RESISTANCE_OHM
    state = instance.metadata.get("state")
    if isinstance(state, str):
        return state.strip().lower() == "closed"
    return False


def _numeric_parameter(instance: PartInstance, name: str) -> float | None:
    """Return a numeric instance parameter if present and parseable."""

    value = instance.parameters.get(name)
    if isinstance(value, (float, int)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


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
    return _rotate_point(point, part.center, radians(asset.rotation_degrees))


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


def _asset_render_bounds(part: PlacedPart, asset: SymbolAsset) -> Bounds:
    """Return the axis-aligned bounds of one transformed asset frame."""

    frame, _ = _asset_frame(part, asset)
    if not asset.rotation_degrees:
        return frame

    angle = radians(asset.rotation_degrees)
    corners = (
        Point(frame.x, frame.y),
        Point(frame.right, frame.y),
        Point(frame.right, frame.bottom),
        Point(frame.x, frame.bottom),
    )
    rotated = tuple(_rotate_point(point, part.center, angle) for point in corners)
    min_x = min(point.x for point in rotated)
    min_y = min(point.y for point in rotated)
    max_x = max(point.x for point in rotated)
    max_y = max(point.y for point in rotated)
    return Bounds(x=min_x, y=min_y, width=max_x - min_x, height=max_y - min_y)


def _rotate_point(point: Point, center: Point, angle: float) -> Point:
    """Return ``point`` rotated around ``center`` by ``angle`` radians."""

    dx = point.x - center.x
    dy = point.y - center.y
    return Point(
        center.x + dx * cos(angle) - dy * sin(angle),
        center.y + dx * sin(angle) + dy * cos(angle),
    )


def _component_label_bounds_for_asset(
    part: PlacedPart,
    spec: PartSpec,
    instance: PartInstance,
    asset: SymbolAsset,
) -> Bounds:
    """Return component-label bounds anchored to a loaded SVG asset."""

    return component_label_bounds_for_symbol_bounds(
        part,
        spec,
        instance.value,
        _asset_render_bounds(part, asset),
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
    anchors = tuple(_explicit_asset_anchors(root))
    fragments = tuple(
        ElementTree.tostring(child, encoding="unicode")
        for child in list(root)
        if _is_asset_fragment(child)
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


def _is_asset_fragment(element: ElementTree.Element) -> bool:
    """Return whether a root child should be emitted as symbol artwork."""

    if _local_name(element.tag) not in {"defs", "g"}:
        return False
    return element.attrib.get("data-rlvr-role") != "pin-anchors"


def _explicit_asset_anchors(
    root: ElementTree.Element,
) -> tuple[tuple[str, SourcePoint], ...]:
    """Return SVG-declared pin anchors in root source coordinates."""

    anchors = tuple(_iter_explicit_asset_anchors(root, _SvgTransform.identity()))
    seen_names: set[str] = set()
    unique_anchors: list[tuple[str, SourcePoint]] = []
    for name, point in anchors:
        if name in seen_names:
            raise ValueError(f"duplicate symbol pin anchor: {name}")
        seen_names.add(name)
        unique_anchors.append((name, point))
    return tuple(unique_anchors)


def _iter_explicit_asset_anchors(
    element: ElementTree.Element, transform: _SvgTransform
) -> tuple[tuple[str, SourcePoint], ...]:
    """Return explicit anchors declared on ``element`` and its descendants."""

    current_transform = transform.compose(
        _parse_transform(element.attrib.get("transform", ""))
    )
    anchor = _explicit_anchor(element)
    anchors: list[tuple[str, SourcePoint]] = []
    if anchor is not None:
        name, point = anchor
        anchors.append((name, current_transform.apply(point)))
    for child in list(element):
        anchors.extend(_iter_explicit_asset_anchors(child, current_transform))
    return tuple(anchors)


def _explicit_anchor(
    element: ElementTree.Element,
) -> tuple[str, SourcePoint] | None:
    """Return the SVG-declared anchor on one element if present."""

    name = _explicit_anchor_name(element)
    if name is None:
        return None
    return (name, _explicit_anchor_point(element))


def _explicit_anchor_name(element: ElementTree.Element) -> str | None:
    """Return a pin name encoded on an SVG anchor element."""

    for attribute_name in ("data-pin", "data-rlvr-pin"):
        raw_name = element.attrib.get(attribute_name)
        if raw_name:
            return raw_name
    element_id = element.attrib.get("id")
    if (
        element_id is not None
        and element_id.startswith("pin-")
        and _local_name(element.tag) not in {"defs", "g"}
    ):
        return element_id.removeprefix("pin-")
    return None


def _explicit_anchor_point(element: ElementTree.Element) -> SourcePoint:
    """Return the local coordinate for an SVG anchor element."""

    if _local_name(element.tag) in {"circle", "ellipse"}:
        return SourcePoint(
            x=_parse_number(element.attrib["cx"]),
            y=_parse_number(element.attrib["cy"]),
        )
    if "x" in element.attrib and "y" in element.attrib:
        return SourcePoint(
            x=_parse_number(element.attrib["x"]),
            y=_parse_number(element.attrib["y"]),
        )
    raise ValueError(
        f"unsupported symbol pin anchor element: {_local_name(element.tag)!r}"
    )


def _parse_transform(raw_transform: str) -> _SvgTransform:
    """Parse an SVG transform attribute into an affine matrix."""

    transform = _SvgTransform.identity()
    for match in _TRANSFORM_PATTERN.finditer(raw_transform):
        transform = transform.compose(
            _transform_function(match.group(1), _parse_number_list(match.group(2)))
        )
    return transform


def _transform_function(name: str, values: tuple[float, ...]) -> _SvgTransform:
    """Return the affine matrix for one SVG transform function."""

    if name == "translate":
        if len(values) == 1:
            return _SvgTransform(a=1.0, b=0.0, c=0.0, d=1.0, e=values[0], f=0.0)
        if len(values) == 2:
            return _SvgTransform(a=1.0, b=0.0, c=0.0, d=1.0, e=values[0], f=values[1])
    if name == "scale":
        if len(values) == 1:
            return _SvgTransform(a=values[0], b=0.0, c=0.0, d=values[0], e=0.0, f=0.0)
        if len(values) == 2:
            return _SvgTransform(a=values[0], b=0.0, c=0.0, d=values[1], e=0.0, f=0.0)
    if name == "matrix" and len(values) == 6:
        return _SvgTransform(
            a=values[0],
            b=values[1],
            c=values[2],
            d=values[3],
            e=values[4],
            f=values[5],
        )
    if name == "rotate" and len(values) in {1, 3}:
        return _rotation_transform(values)
    raise ValueError(f"unsupported SVG transform: {name}({values})")


def _rotation_transform(values: tuple[float, ...]) -> _SvgTransform:
    """Return the affine matrix for an SVG rotate transform."""

    angle = radians(values[0])
    rotation = _SvgTransform(
        a=cos(angle),
        b=sin(angle),
        c=-sin(angle),
        d=cos(angle),
        e=0.0,
        f=0.0,
    )
    if len(values) == 1:
        return rotation
    move_to_center = _SvgTransform(a=1.0, b=0.0, c=0.0, d=1.0, e=values[1], f=values[2])
    move_from_center = _SvgTransform(
        a=1.0, b=0.0, c=0.0, d=1.0, e=-values[1], f=-values[2]
    )
    return move_to_center.compose(rotation).compose(move_from_center)


def _parse_number_list(raw_value: str) -> tuple[float, ...]:
    """Parse an SVG comma-or-space separated numeric list."""

    return tuple(
        float(part) for part in re.split(r"[,\s]+", raw_value.strip()) if part.strip()
    )


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
    rotation_degrees: float = 0.0


_VARIABLE_RESISTOR = _AssetSpec(
    key="variable_resistor",
    filename="variable_resistor.svg",
    rotation_degrees=90.0,
)

_RESISTOR = _AssetSpec(
    key="resistor",
    filename="resistor.svg",
    rotation_degrees=90.0,
)

_PULLUP_RESISTOR = _AssetSpec(
    key="pullup_resistor",
    filename="pullup_resistor.svg",
)

_PULLDOWN_RESISTOR = _AssetSpec(
    key="pulldown_resistor",
    filename="pulldown_resistor.svg",
)

_CAPACITOR = _AssetSpec(
    key="capacitor",
    filename="capacitor.svg",
    rotation_degrees=90.0,
)

_INDUCTOR = _AssetSpec(
    key="inductor",
    filename="inductor.svg",
    rotation_degrees=90.0,
)

_LED = _AssetSpec(
    key="led",
    filename="led.svg",
    rotation_degrees=90.0,
)

_DIODE = _AssetSpec(
    key="diode",
    filename="diode.svg",
    rotation_degrees=90.0,
)

_ZENER = _AssetSpec(
    key="zener",
    filename="zener.svg",
    rotation_degrees=90.0,
)

_SPST_SWITCH = _AssetSpec(
    key="spst_switch",
    filename="spst_switch.svg",
    rotation_degrees=90.0,
)

_SPST_SWITCH_CLOSED = _AssetSpec(
    key="spst_switch_closed",
    filename="spst_switch_closed.svg",
    rotation_degrees=90.0,
)

_DC_SOURCE = _AssetSpec(
    key="dc_voltage_source",
    filename="dc_voltage_source.svg",
)

_AC_SOURCE = _AssetSpec(
    key="ac_voltage_source",
    filename="ac_voltage_source.svg",
)

_CURRENT_SOURCE = _AssetSpec(
    key="current_source_dc",
    filename="current_source_dc.svg",
)

_GROUND = _AssetSpec(
    key="ground",
    filename="ground.svg",
)

_NPN = _AssetSpec(
    key="npn",
    filename="npn.svg",
)

_PNP = _AssetSpec(
    key="pnp",
    filename="pnp.svg",
)

_NMOS = _AssetSpec(
    key="nmos",
    filename="nmos.svg",
)

_PMOS = _AssetSpec(
    key="pmos",
    filename="pmos.svg",
)

_OPAMP = _AssetSpec(
    key="op_amp",
    filename="op_amp.svg",
)

_TRANSFORMER = _AssetSpec(
    key="transformer",
    filename="transformer.svg",
)

_CONTROLLED_SOURCE = _AssetSpec(
    key="controlled_source",
    filename="controlled_source.svg",
)

_CONTROLLED_CURRENT_SOURCE = _AssetSpec(
    key="controlled_current_source",
    filename="controlled_current_source.svg",
)

_RELAY = _AssetSpec(
    key="relay",
    filename="relay.svg",
)

_GENERIC_IC = _AssetSpec(
    key="generic_ic",
    filename="generic_ic.svg",
)

_CONNECTOR = _AssetSpec(
    key="connector",
    filename="connector.svg",
)

_VOLTMETER = _AssetSpec(
    key="meter",
    filename="meter.svg",
)

_AMMETER = _AssetSpec(
    key="ammeter",
    filename="ammeter.svg",
)

_LAMP = _AssetSpec(
    key="lamp",
    filename="lamp.svg",
)

_MOTOR = _AssetSpec(
    key="motor",
    filename="motor.svg",
)

_LOGIC = _AssetSpec(
    key="and_gate",
    filename="logic.svg",
)

_OR_GATE = _AssetSpec(
    key="or_gate",
    filename="or_gate.svg",
)

_NOT_GATE = _AssetSpec(
    key="not_gate",
    filename="not_gate.svg",
)

_ASSETS_BY_KIND = {
    "ammeter": _AMMETER,
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
    "pulldown_resistor": _PULLDOWN_RESISTOR,
    "pullup_resistor": _PULLUP_RESISTOR,
    "vccs": _CONTROLLED_CURRENT_SOURCE,
    "voltage_source_dc": _DC_SOURCE,
    "voltmeter": _VOLTMETER,
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
    "meter": _VOLTMETER,
    "motor": _MOTOR,
    "opamp": _OPAMP,
    "relay": _RELAY,
    "resistor": _RESISTOR,
    "source": _DC_SOURCE,
    "switch": _SPST_SWITCH,
    "transformer": _TRANSFORMER,
}
