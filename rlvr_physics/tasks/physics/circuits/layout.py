"""Deterministic schematic layout planner for circuit renderers."""

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import combinations
from math import hypot
from typing import Mapping

from rlvr_physics.tasks.physics.circuits.model import (
    Circuit,
    ComponentFamily,
    Connection,
    PartInstance,
    PartSpec,
    PinSide,
)

GRID_SPACING = 20.0
CANVAS_MARGIN = 36.0
LAYER_SPACING = 210.0
ROW_SPACING = 118.0
ROUTING_PADDING = 22.0
ROUTING_CLEARANCE = 12.0
PIN_ESCAPE = 54.0
PIN_APPROACH_CLEARANCE = 18.0
PIN_APPROACH_BOUNDARY_CLEARANCE = 2.0
HIGH_FANOUT_NET_THRESHOLD = 4
NET_LABEL_CLEARANCE = 2.0
LABEL_CHAR_WIDTH = 6.8
LABEL_HEIGHT = 12.0
LABEL_ASCENT = 10.0
LABEL_GAP = 5.0
FORCE_ITERATIONS = 180
MAX_FORCE_STEP = 30.0
GEOMETRY_EPSILON = 1.0e-6


@dataclass(frozen=True)
class Point:
    """Two-dimensional layout point."""

    x: float
    y: float

    def translate(self, dx: float, dy: float) -> "Point":
        """Return this point translated by ``dx`` and ``dy``."""

        return Point(self.x + dx, self.y + dy)


@dataclass(frozen=True)
class Size:
    """Two-dimensional layout size."""

    width: float
    height: float


@dataclass(frozen=True)
class Bounds:
    """Axis-aligned layout bounds."""

    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        """Return the right edge coordinate."""

        return self.x + self.width

    @property
    def bottom(self) -> float:
        """Return the bottom edge coordinate."""

        return self.y + self.height

    @property
    def center(self) -> Point:
        """Return the center point."""

        return Point(self.x + self.width / 2.0, self.y + self.height / 2.0)

    def overlaps(self, other: "Bounds") -> bool:
        """Return whether two bounds overlap."""

        return not (
            self.x + self.width <= other.x
            or other.x + other.width <= self.x
            or self.y + self.height <= other.y
            or other.y + other.height <= self.y
        )

    def expanded(self, x_padding: float, y_padding: float) -> "Bounds":
        """Return bounds expanded by horizontal and vertical padding."""

        return Bounds(
            x=self.x - x_padding,
            y=self.y - y_padding,
            width=self.width + 2.0 * x_padding,
            height=self.height + 2.0 * y_padding,
        )

    def translate(self, dx: float, dy: float) -> "Bounds":
        """Return bounds translated by ``dx`` and ``dy``."""

        return Bounds(
            x=self.x + dx,
            y=self.y + dy,
            width=self.width,
            height=self.height,
        )


@dataclass(frozen=True)
class PlacedPart:
    """One placed part in schematic coordinates."""

    ref: str
    kind: str
    center: Point
    size: Size

    @property
    def bounds(self) -> Bounds:
        """Return axis-aligned bounds for this placed part."""

        return Bounds(
            x=self.center.x - self.size.width / 2.0,
            y=self.center.y - self.size.height / 2.0,
            width=self.size.width,
            height=self.size.height,
        )

    def translate(self, dx: float, dy: float) -> "PlacedPart":
        """Return this part translated by ``dx`` and ``dy``."""

        return PlacedPart(
            ref=self.ref,
            kind=self.kind,
            center=self.center.translate(dx, dy),
            size=self.size,
        )


@dataclass(frozen=True)
class WireSegment:
    """One orthogonal wire segment."""

    start: Point
    end: Point


@dataclass(frozen=True)
class WirePath:
    """Rendered path for one net.

    Nets rendered with local net labels may contain disconnected per-pin stubs;
    the matching :class:`NetLabel` entries provide the shared-net notation.
    """

    net: str
    segments: tuple[WireSegment, ...]


@dataclass(frozen=True)
class NetLabel:
    """Local label tying one wire stub to a shared net."""

    net: str
    text: str
    side: PinSide
    anchor: Point
    position: Point

    def translate(self, dx: float, dy: float) -> "NetLabel":
        """Return this label translated by ``dx`` and ``dy``."""

        return NetLabel(
            net=self.net,
            text=self.text,
            side=self.side,
            anchor=self.anchor.translate(dx, dy),
            position=self.position.translate(dx, dy),
        )


@dataclass(frozen=True)
class Layout:
    """Complete deterministic schematic layout.

    The ``net_labels`` field stores local labels used to tie disconnected
    stubs into the same logical net without drawing a long shared wire.
    """

    parts: tuple[PlacedPart, ...]
    wires: tuple[WirePath, ...]
    size: Size
    net_labels: tuple[NetLabel, ...] = ()

    def part_by_ref(self) -> Mapping[str, PlacedPart]:
        """Return a lookup from reference designator to placed part."""

        return {part.ref: part for part in self.parts}


@dataclass(frozen=True)
class _PinApproachBound:
    """Reserved routing lane leading into one component pin."""

    ref: str
    pin: str
    side: PinSide
    bounds: Bounds


def plan_layout(circuit: Circuit, catalog: Mapping[str, PartSpec]) -> Layout:
    """Plan deterministic force-directed component placement.

    Parameters
    ----------
    circuit:
        Circuit to place.
    catalog:
        Component catalog.

    Returns
    -------
    Layout
        Schematic layout with placed components, routed wire paths, and
        optional local net labels.
    """

    return _plan_layout(circuit, catalog, route_pin_labels=False)


def _plan_layout(
    circuit: Circuit, catalog: Mapping[str, PartSpec], *, route_pin_labels: bool
) -> Layout:
    """Plan deterministic component placement with renderer-specific routing."""

    initial_parts = _initial_placement(circuit, catalog)
    placed = _force_directed_parts(circuit, catalog, initial_parts)
    placed = _snap_parts(placed)
    placed = _clear_overlaps(circuit, catalog, placed)
    wires, net_labels = _route_wires(
        circuit,
        catalog,
        {part.ref: part for part in placed},
        route_pin_labels=route_pin_labels,
    )
    placed, wires, net_labels, size = _normalize_layout(
        circuit,
        catalog,
        placed,
        wires,
        net_labels,
    )
    return Layout(
        parts=tuple(sorted(placed, key=lambda part: part.ref)),
        wires=wires,
        size=size,
        net_labels=net_labels,
    )


def visual_bounds(placed_part: PlacedPart, spec: PartSpec, value: str) -> Bounds:
    """Return rendered bounds including labels and pin labels.

    Parameters
    ----------
    placed_part:
        Placed part to inspect.
    spec:
        Static component specification.
    value:
        Rendered part value text.

    Returns
    -------
    Bounds
        Bounds covering the drawn symbol plus model-visible labels.
    """

    boxes = [placed_part.bounds]
    boxes.append(component_label_bounds(placed_part, spec, value))
    for pin in spec.pins:
        anchor = pin_position(placed_part, spec, pin.name)
        boxes.append(_pin_label_bounds(anchor, pin.side, pin.name))
    return _union_bounds(boxes)


def placement_bounds(placed_part: PlacedPart, spec: PartSpec, value: str) -> Bounds:
    """Return bounds used by placement and routing clearance.

    Parameters
    ----------
    placed_part:
        Placed part to inspect.
    spec:
        Static component specification.
    value:
        Rendered part value text.

    Returns
    -------
    Bounds
        Visual bounds expanded with routing clearance.
    """

    boxes = [
        visual_bounds(placed_part, spec, value).expanded(
            ROUTING_PADDING, ROUTING_PADDING
        )
    ]
    boxes.extend(
        bounds.expanded(ROUTING_CLEARANCE, ROUTING_CLEARANCE)
        for bounds in _pin_approach_bounds(placed_part, spec)
    )
    return _union_bounds(boxes)


def pin_position(placed_part: PlacedPart, spec: PartSpec, pin_name: str) -> Point:
    """Return absolute pin anchor position for a placed part."""

    pin = spec.pin(pin_name)
    same_side = [item for item in spec.pins if item.side is pin.side]
    side_index = same_side.index(pin)
    side_count = len(same_side)
    bounds = placed_part.bounds
    if pin.side is PinSide.LEFT:
        return Point(bounds.x, bounds.y + _slot(side_index, side_count, bounds.height))
    if pin.side is PinSide.RIGHT:
        return Point(
            bounds.x + bounds.width,
            bounds.y + _slot(side_index, side_count, bounds.height),
        )
    if pin.side is PinSide.TOP:
        return Point(bounds.x + _slot(side_index, side_count, bounds.width), bounds.y)
    return Point(
        bounds.x + _slot(side_index, side_count, bounds.width),
        bounds.y + bounds.height,
    )


def pin_label_position(anchor: Point, side: PinSide, label: str) -> Point:
    """Return the SVG baseline point for a pin label.

    Parameters
    ----------
    anchor:
        Pin anchor point.
    side:
        Side of the symbol where the pin is drawn.
    label:
        Pin label text.

    Returns
    -------
    Point
        SVG text baseline point.
    """

    if side is PinSide.LEFT:
        return Point(
            anchor.x - PIN_ESCAPE - LABEL_GAP - _text_width(label),
            anchor.y - LABEL_GAP,
        )
    if side is PinSide.RIGHT:
        return Point(anchor.x + PIN_ESCAPE + LABEL_GAP, anchor.y - LABEL_GAP)
    if side is PinSide.TOP:
        return Point(anchor.x + LABEL_GAP, anchor.y - LABEL_GAP)
    return Point(anchor.x + LABEL_GAP, anchor.y + LABEL_GAP + LABEL_HEIGHT)


def component_label_bounds(
    placed_part: PlacedPart, spec: PartSpec, value: str
) -> Bounds:
    """Return the label bounds for one placed component."""

    label_text = f"{placed_part.ref} {value}".strip()
    label_width = max(_text_width(label_text), placed_part.bounds.width)
    candidates = _component_label_candidates(placed_part, label_width)
    for candidate in _ranked_component_label_candidates(candidates, spec):
        if not _label_conflicts_with_pin_escapes(candidate, placed_part, spec):
            return candidate
    return candidates["above"]


def component_label_position(
    placed_part: PlacedPart, spec: PartSpec, value: str
) -> Point:
    """Return the SVG baseline point for one component label."""

    bounds = component_label_bounds(placed_part, spec, value)
    return Point(bounds.x, bounds.y + LABEL_ASCENT)


def _component_label_candidates(
    placed_part: PlacedPart, label_width: float
) -> dict[str, Bounds]:
    """Return candidate external component-label bounds."""

    bounds = placed_part.bounds
    center_y = bounds.y + (bounds.height - LABEL_HEIGHT) / 2.0
    return {
        "above": Bounds(
            x=bounds.x,
            y=bounds.y - LABEL_GAP - LABEL_HEIGHT,
            width=label_width,
            height=LABEL_HEIGHT,
        ),
        "below": Bounds(
            x=bounds.x,
            y=bounds.bottom + LABEL_GAP,
            width=label_width,
            height=LABEL_HEIGHT,
        ),
        "right": Bounds(
            x=bounds.right + LABEL_GAP,
            y=center_y,
            width=label_width,
            height=LABEL_HEIGHT,
        ),
        "left": Bounds(
            x=bounds.x - LABEL_GAP - label_width,
            y=center_y,
            width=label_width,
            height=LABEL_HEIGHT,
        ),
        "above_right": Bounds(
            x=bounds.right + LABEL_GAP,
            y=bounds.y - LABEL_GAP - LABEL_HEIGHT,
            width=label_width,
            height=LABEL_HEIGHT,
        ),
        "above_left": Bounds(
            x=bounds.x - LABEL_GAP - label_width,
            y=bounds.y - LABEL_GAP - LABEL_HEIGHT,
            width=label_width,
            height=LABEL_HEIGHT,
        ),
        "below_right": Bounds(
            x=bounds.right + LABEL_GAP,
            y=bounds.bottom + LABEL_GAP,
            width=label_width,
            height=LABEL_HEIGHT,
        ),
        "below_left": Bounds(
            x=bounds.x - LABEL_GAP - label_width,
            y=bounds.bottom + LABEL_GAP,
            width=label_width,
            height=LABEL_HEIGHT,
        ),
    }


def _ranked_component_label_candidates(
    candidates: Mapping[str, Bounds], spec: PartSpec
) -> tuple[Bounds, ...]:
    """Return label candidates ordered to avoid declared pin sides."""

    pin_sides = {pin.side for pin in spec.pins}
    names: list[str] = []
    if PinSide.TOP not in pin_sides:
        names.append("above")
    if PinSide.BOTTOM not in pin_sides:
        names.append("below")
    if PinSide.RIGHT not in pin_sides:
        names.append("right")
    if PinSide.LEFT not in pin_sides:
        names.append("left")
    names.extend(
        (
            "above_right",
            "below_right",
            "above_left",
            "below_left",
            "above",
            "below",
            "right",
            "left",
        )
    )
    result: list[Bounds] = []
    for name in names:
        candidate = candidates[name]
        if candidate not in result:
            result.append(candidate)
    return tuple(result)


def _label_conflicts_with_pin_escapes(
    label_bounds: Bounds, placed_part: PlacedPart, spec: PartSpec
) -> bool:
    """Return whether a label blocks any immediate pin escape segment."""

    blocked = label_bounds.expanded(ROUTING_CLEARANCE, ROUTING_CLEARANCE)
    for pin in spec.pins:
        anchor = pin_position(placed_part, spec, pin.name)
        escape = _pin_escape_point(placed_part, spec, pin.name, anchor)
        if _segment_crosses_bounds(WireSegment(anchor, escape), blocked):
            return True
    return False


def _initial_placement(
    circuit: Circuit, catalog: Mapping[str, PartSpec]
) -> tuple[PlacedPart, ...]:
    """Return a deterministic seed placement for force-directed refinement."""

    layers = _assign_layers(circuit, catalog)
    placed: list[PlacedPart] = []
    for layer in sorted(set(layers.values())):
        refs = sorted(ref for ref, ref_layer in layers.items() if ref_layer == layer)
        for row, ref in enumerate(refs):
            part = circuit.part_by_ref()[ref]
            spec = catalog[part.kind]
            placed.append(
                PlacedPart(
                    ref=ref,
                    kind=part.kind,
                    center=Point(
                        CANVAS_MARGIN + 70.0 + layer * LAYER_SPACING,
                        CANVAS_MARGIN + 70.0 + row * ROW_SPACING,
                    ),
                    size=_part_size(spec),
                )
            )
    return tuple(sorted(placed, key=lambda part: part.ref))


def _assign_layers(circuit: Circuit, catalog: Mapping[str, PartSpec]) -> dict[str, int]:
    """Assign deterministic left-to-right layout layers."""

    family_rank = {
        ComponentFamily.SOURCE: 0,
        ComponentFamily.POWER: 0,
        ComponentFamily.CONNECTOR: 1,
        ComponentFamily.PASSIVE: 2,
        ComponentFamily.SWITCH: 2,
        ComponentFamily.SEMICONDUCTOR: 3,
        ComponentFamily.LOGIC: 3,
        ComponentFamily.INTEGRATED: 3,
        ComponentFamily.ELECTROMECHANICAL: 4,
        ComponentFamily.LOAD: 4,
        ComponentFamily.MEASUREMENT: 5,
        ComponentFamily.CONTROLLED_SOURCE: 3,
    }
    layers: dict[str, int] = {}
    for part in circuit.parts:
        spec = catalog[part.kind]
        layers[part.ref] = family_rank[spec.family]
    return layers


def _force_directed_parts(
    circuit: Circuit,
    catalog: Mapping[str, PartSpec],
    placed_parts: tuple[PlacedPart, ...],
) -> tuple[PlacedPart, ...]:
    """Refine part positions with attractive nets and overlap repulsion."""

    part_by_ref = circuit.part_by_ref()
    sizes = {part.ref: part.size for part in placed_parts}
    positions = {part.ref: part.center for part in placed_parts}
    targets = dict(positions)
    links = _net_links(circuit)
    refs = tuple(sorted(positions))

    for iteration in range(FORCE_ITERATIONS):
        forces = {ref: Point(0.0, 0.0) for ref in refs}
        bounds = _placement_bounds_by_ref(circuit, catalog, positions, sizes)
        alpha = 1.0 - float(iteration) / float(FORCE_ITERATIONS)
        max_step = max(5.0, MAX_FORCE_STEP * alpha)

        for ref in refs:
            target = targets[ref]
            current = positions[ref]
            _add_force(
                forces,
                ref,
                (target.x - current.x) * 0.012,
                (target.y - current.y) * 0.003,
            )

        for ref_a, ref_b, weight in links:
            pos_a = positions[ref_a]
            pos_b = positions[ref_b]
            dx = pos_b.x - pos_a.x
            dy = pos_b.y - pos_a.y
            distance = max(hypot(dx, dy), 1.0)
            ideal = _ideal_link_length(
                catalog[part_by_ref[ref_a].kind],
                catalog[part_by_ref[ref_b].kind],
            )
            magnitude = (distance - ideal) * 0.012 * weight
            fx = dx / distance * magnitude
            fy = dy / distance * magnitude
            _add_force(forces, ref_a, fx, fy)
            _add_force(forces, ref_b, -fx, -fy)

        for ref_a, ref_b in combinations(refs, 2):
            correction = _overlap_correction(bounds[ref_a], bounds[ref_b])
            if correction is not None:
                _add_force(forces, ref_a, -correction.x * 0.55, -correction.y * 0.55)
                _add_force(forces, ref_b, correction.x * 0.55, correction.y * 0.55)
                continue
            pos_a = positions[ref_a]
            pos_b = positions[ref_b]
            dx = pos_b.x - pos_a.x
            dy = pos_b.y - pos_a.y
            distance = max(hypot(dx, dy), 1.0)
            close_distance = 210.0
            if distance < close_distance:
                magnitude = (close_distance - distance) * 0.018
                fx = dx / distance * magnitude
                fy = dy / distance * magnitude
                _add_force(forces, ref_a, -fx, -fy)
                _add_force(forces, ref_b, fx, fy)

        for ref in refs:
            force = forces[ref]
            step = min(max_step, hypot(force.x, force.y))
            if step == 0.0:
                continue
            scale = step / max(hypot(force.x, force.y), 1.0)
            positions[ref] = positions[ref].translate(force.x * scale, force.y * scale)

    return _placed_from_positions(part_by_ref, sizes, positions)


def _clear_overlaps(
    circuit: Circuit,
    catalog: Mapping[str, PartSpec],
    placed_parts: tuple[PlacedPart, ...],
) -> tuple[PlacedPart, ...]:
    """Run a deterministic hard-separation pass over placement bounds."""

    part_by_ref = circuit.part_by_ref()
    sizes = {part.ref: part.size for part in placed_parts}
    positions = {part.ref: part.center for part in placed_parts}
    refs = tuple(sorted(positions))

    for _ in range(120):
        changed = False
        bounds = _placement_bounds_by_ref(circuit, catalog, positions, sizes)
        for ref_a, ref_b in combinations(refs, 2):
            correction = _overlap_correction(bounds[ref_a], bounds[ref_b])
            if correction is None:
                continue
            positions[ref_a] = positions[ref_a].translate(
                -correction.x / 2.0,
                -correction.y / 2.0,
            )
            positions[ref_b] = positions[ref_b].translate(
                correction.x / 2.0,
                correction.y / 2.0,
            )
            changed = True
        if not changed:
            break

    return _placed_from_positions(part_by_ref, sizes, positions)


def _snap_parts(placed_parts: tuple[PlacedPart, ...]) -> tuple[PlacedPart, ...]:
    """Snap part centers to the renderer grid."""

    return tuple(
        PlacedPart(
            ref=part.ref,
            kind=part.kind,
            center=Point(
                GRID_SPACING * round(part.center.x / GRID_SPACING),
                GRID_SPACING * round(part.center.y / GRID_SPACING),
            ),
            size=part.size,
        )
        for part in placed_parts
    )


def _net_links(circuit: Circuit) -> tuple[tuple[str, str, float], ...]:
    """Return weighted part links induced by shared nets."""

    links: list[tuple[str, str, float]] = []
    for net in circuit.nets:
        refs = sorted(
            {connection.ref for connection in circuit.connections_for_net(net)}
        )
        if len(refs) < 2:
            continue
        weight = 1.0 / max(float(len(refs) - 1), 1.0)
        for ref_a, ref_b in combinations(refs, 2):
            links.append((ref_a, ref_b, weight))
    return tuple(links)


def _placement_bounds_by_ref(
    circuit: Circuit,
    catalog: Mapping[str, PartSpec],
    positions: Mapping[str, Point],
    sizes: Mapping[str, Size],
) -> dict[str, Bounds]:
    """Return placement bounds for mutable force-directed positions."""

    part_by_ref = circuit.part_by_ref()
    return {
        ref: placement_bounds(
            PlacedPart(
                ref=ref, kind=part_by_ref[ref].kind, center=position, size=sizes[ref]
            ),
            catalog[part_by_ref[ref].kind],
            part_by_ref[ref].value,
        )
        for ref, position in positions.items()
    }


def _placed_from_positions(
    part_by_ref: Mapping[str, PartInstance],
    sizes: Mapping[str, Size],
    positions: Mapping[str, Point],
) -> tuple[PlacedPart, ...]:
    """Build immutable placed parts from mutable position maps."""

    return tuple(
        PlacedPart(
            ref=ref,
            kind=part_by_ref[ref].kind,
            center=positions[ref],
            size=sizes[ref],
        )
        for ref in sorted(positions)
    )


def _ideal_link_length(spec_a: PartSpec, spec_b: PartSpec) -> float:
    """Return preferred center-to-center distance for connected parts."""

    width = _part_size(spec_a).width / 2.0 + _part_size(spec_b).width / 2.0
    return width + 130.0


def _overlap_correction(first: Bounds, second: Bounds) -> Point | None:
    """Return the smallest movement for separating ``second`` from ``first``."""

    if not first.overlaps(second):
        return None
    overlap_x = min(first.right, second.right) - max(first.x, second.x)
    overlap_y = min(first.bottom, second.bottom) - max(first.y, second.y)
    center_a = first.center
    center_b = second.center
    if overlap_x <= overlap_y:
        direction = 1.0 if center_b.x >= center_a.x else -1.0
        return Point(direction * (overlap_x + GRID_SPACING), 0.0)
    direction = 1.0 if center_b.y >= center_a.y else -1.0
    return Point(0.0, direction * (overlap_y + GRID_SPACING))


def _add_force(forces: dict[str, Point], ref: str, dx: float, dy: float) -> None:
    """Accumulate force for one reference designator."""

    force = forces[ref]
    forces[ref] = Point(force.x + dx, force.y + dy)


def _route_wires(
    circuit: Circuit,
    catalog: Mapping[str, PartSpec],
    placed_parts: Mapping[str, PlacedPart],
    *,
    route_pin_labels: bool,
) -> tuple[tuple[WirePath, ...], tuple[NetLabel, ...]]:
    """Build deterministic Manhattan paths through open routing channels."""

    paths: list[WirePath] = []
    net_labels: list[NetLabel] = []
    symbol_label_bounds = tuple(placed.bounds for placed in placed_parts.values())
    component_label_bounds_by_part = tuple(
        component_label_bounds(
            placed,
            catalog[circuit.part_by_ref()[ref].kind],
            circuit.part_by_ref()[ref].value,
        )
        for ref, placed in placed_parts.items()
    )
    symbol_bounds = tuple(
        bounds.expanded(ROUTING_CLEARANCE, ROUTING_CLEARANCE)
        for bounds in symbol_label_bounds
    )
    label_bounds = tuple(
        bounds.expanded(ROUTING_CLEARANCE, ROUTING_CLEARANCE)
        for bounds in component_label_bounds_by_part
    )
    pin_label_bounds: tuple[Bounds, ...] = ()
    pin_label_blocks: tuple[Bounds, ...] = ()
    if route_pin_labels:
        pin_label_blocks = tuple(
            _pin_label_bounds(
                pin_position(placed, spec, pin.name),
                pin.side,
                pin.name,
            )
            for ref, placed in placed_parts.items()
            for spec in (catalog[circuit.part_by_ref()[ref].kind],)
            for pin in spec.pins
        )
        pin_label_bounds = tuple(
            bounds.expanded(ROUTING_CLEARANCE, ROUTING_CLEARANCE)
            for bounds in pin_label_blocks
        )
    pin_approach_bounds = _pin_approach_bounds_by_pin(circuit, catalog, placed_parts)
    base_routing_bounds = symbol_bounds + label_bounds + pin_label_bounds
    labeled_stub_bounds_by_net = _labeled_stub_bounds_by_net(
        circuit,
        catalog,
        placed_parts,
    )
    routed_wire_bounds: tuple[Bounds, ...] = ()
    for net_index, net in enumerate(circuit.nets):
        connections = circuit.connections_for_net(net)
        anchors = _net_anchors(circuit, catalog, placed_parts, connections)
        if len(anchors) < 2:
            continue
        net_uses_labels = _uses_local_net_labels(net, connections)
        if net_uses_labels:
            segments, labels = _route_labeled_net(net, anchors)
            net_labels.extend(labels)
        else:
            current_pin_keys = {
                (connection.ref, connection.pin) for connection in connections
            }
            net_routing_bounds = (
                (
                    base_routing_bounds
                    + _pin_approach_routing_bounds(
                        pin_approach_bounds,
                        current_pin_keys,
                    )
                )
                + tuple(
                    bound
                    for label_net, bounds in labeled_stub_bounds_by_net.items()
                    if label_net != net
                    for bound in bounds
                )
                + routed_wire_bounds
            )
            segments = _route_net_tree(anchors, net_routing_bounds, net_index)
            if _route_has_blocked_segment(
                segments,
                _pin_stub_segments(anchors),
                net_routing_bounds,
            ):
                segments, labels = _route_labeled_net(net, anchors)
                net_labels.extend(labels)
                net_uses_labels = True
                labeled_stub_bounds_by_net[net] = tuple(
                    _segment_bounds(segment).expanded(
                        ROUTING_CLEARANCE,
                        ROUTING_CLEARANCE,
                    )
                    for segment in segments
                )
        deduped_segments = tuple(_dedupe_segments(segments))
        paths.append(WirePath(net=net, segments=deduped_segments))
        if not net_uses_labels:
            routed_wire_bounds += tuple(
                _segment_bounds(segment).expanded(ROUTING_CLEARANCE, ROUTING_CLEARANCE)
                for segment in deduped_segments
            )
    wires = tuple(paths)
    return wires, _place_net_labels(
        tuple(net_labels),
        symbol_label_bounds + component_label_bounds_by_part + pin_label_blocks,
        wires,
    )


def _labeled_stub_bounds_by_net(
    circuit: Circuit,
    catalog: Mapping[str, PartSpec],
    placed_parts: Mapping[str, PlacedPart],
) -> dict[str, tuple[Bounds, ...]]:
    """Return blocked bounds for local labeled-net stubs."""

    bounds_by_net: dict[str, tuple[Bounds, ...]] = {}
    for net in circuit.nets:
        connections = circuit.connections_for_net(net)
        if not _uses_local_net_labels(net, connections):
            continue
        anchors = _net_anchors(circuit, catalog, placed_parts, connections)
        bounds_by_net[net] = tuple(
            _segment_bounds(segment).expanded(ROUTING_CLEARANCE, ROUTING_CLEARANCE)
            for segment in _pin_stub_segments(anchors)
        )
    return bounds_by_net


def _pin_approach_bounds_by_pin(
    circuit: Circuit,
    catalog: Mapping[str, PartSpec],
    placed_parts: Mapping[str, PlacedPart],
) -> tuple[_PinApproachBound, ...]:
    """Return reserved pin-approach lanes keyed by component pin."""

    entries: list[_PinApproachBound] = []
    part_by_ref = circuit.part_by_ref()
    for ref, placed in placed_parts.items():
        spec = catalog[part_by_ref[ref].kind]
        if not _needs_pin_approach_keepout(spec):
            continue
        for pin in spec.pins:
            entries.append(
                _PinApproachBound(
                    ref=ref,
                    pin=pin.name,
                    side=pin.side,
                    bounds=_pin_approach_bounds_for_pin(placed, spec, pin.name),
                )
            )
    return tuple(entries)


def _route_has_blocked_segment(
    segments: list[WireSegment],
    allowed_stubs: list[WireSegment],
    bounds: tuple[Bounds, ...],
) -> bool:
    """Return whether a non-stub route segment crosses blocked geometry."""

    allowed_stub_keys = {_wire_segment_key(segment) for segment in allowed_stubs}
    return any(
        _wire_segment_key(segment) not in allowed_stub_keys
        and _segment_crosses_any_bounds(segment, bounds)
        for segment in segments
    )


def _wire_segment_key(
    segment: WireSegment,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return an order-independent key for one segment."""

    endpoints = (
        (segment.start.x, segment.start.y),
        (segment.end.x, segment.end.y),
    )
    first, second = endpoints
    if first <= second:
        return (first, second)
    return (second, first)


def _pin_approach_routing_bounds(
    entries: tuple[_PinApproachBound, ...],
    current_pin_keys: set[tuple[str, str]],
) -> tuple[Bounds, ...]:
    """Return pin-approach obstacles for one routed net.

    A net may leave its own pins along the normal approach-boundary escape. Other
    pins get expanded obstacles so unrelated routes cannot ride their approach
    boundaries and later look electrically connected.
    """

    bounds: list[Bounds] = []
    for entry in entries:
        if (entry.ref, entry.pin) in current_pin_keys:
            bounds.append(entry.bounds)
        else:
            bounds.append(_expanded_noncurrent_pin_approach(entry))
    return tuple(bounds)


def _expanded_noncurrent_pin_approach(entry: _PinApproachBound) -> Bounds:
    """Return a small obstacle expansion across a non-current pin lane."""

    if entry.side in (PinSide.LEFT, PinSide.RIGHT):
        return entry.bounds.expanded(PIN_APPROACH_BOUNDARY_CLEARANCE, 0.0)
    if entry.side in (PinSide.TOP, PinSide.BOTTOM):
        return entry.bounds.expanded(0.0, PIN_APPROACH_BOUNDARY_CLEARANCE)
    return entry.bounds


def _net_anchors(
    circuit: Circuit,
    catalog: Mapping[str, PartSpec],
    placed_parts: Mapping[str, PlacedPart],
    connections: tuple[Connection, ...],
) -> list[tuple[Connection, Point, Point, PinSide]]:
    """Return routed pin anchors for a net's connections."""

    anchors: list[tuple[Connection, Point, Point, PinSide]] = []
    for connection in connections:
        placed = placed_parts[connection.ref]
        spec = catalog[circuit.part_by_ref()[connection.ref].kind]
        pin_side = spec.pin(connection.pin).side
        anchor = pin_position(placed, spec, connection.pin)
        anchors.append(
            (
                connection,
                anchor,
                _pin_escape_point(placed, spec, connection.pin, anchor),
                pin_side,
            )
        )
    return anchors


def _uses_local_net_labels(net: str, connections: tuple[Connection, ...]) -> bool:
    """Return whether a high-fanout net should render as local net labels."""

    if net in {"0", "VCC"} and len(connections) >= 3:
        return True
    return len(connections) >= HIGH_FANOUT_NET_THRESHOLD


def _route_labeled_net(
    net: str,
    anchors: list[tuple[Connection, Point, Point, PinSide]],
) -> tuple[list[WireSegment], tuple[NetLabel, ...]]:
    """Route a high-fanout net as local labeled pin stubs."""

    label_text = _net_label_text(net)
    labels = tuple(
        NetLabel(
            net=net,
            text=label_text,
            side=pin_side,
            anchor=escape,
            position=_net_label_position(escape, pin_side, label_text),
        )
        for _, _, escape, pin_side in anchors
    )
    return _pin_stub_segments(anchors), labels


def _net_label_text(net: str) -> str:
    """Return renderer text for a shared net label."""

    if net == "0":
        return "GND"
    return net


def _net_label_position(anchor: Point, side: PinSide, label: str) -> Point:
    """Return the SVG baseline point for one local net label."""

    return _net_label_candidates(anchor, side, label)[0]


def _net_label_candidates(
    anchor: Point, side: PinSide, label: str
) -> tuple[Point, ...]:
    """Return candidate baseline positions for one local net label."""

    width = _text_width(label)
    if side is PinSide.LEFT:
        primary = Point(anchor.x - LABEL_GAP - width, anchor.y - LABEL_GAP)
        return _expand_net_label_candidates(
            primary,
            Point(anchor.x - LABEL_GAP - width, anchor.y + LABEL_GAP + LABEL_HEIGHT),
            Point(anchor.x + LABEL_GAP, anchor.y - LABEL_GAP),
            Point(anchor.x + LABEL_GAP, anchor.y + LABEL_GAP + LABEL_HEIGHT),
        )
    if side is PinSide.RIGHT:
        primary = Point(anchor.x + LABEL_GAP, anchor.y - LABEL_GAP)
        return _expand_net_label_candidates(
            primary,
            Point(anchor.x + LABEL_GAP, anchor.y + LABEL_GAP + LABEL_HEIGHT),
            Point(anchor.x - LABEL_GAP - width, anchor.y - LABEL_GAP),
            Point(anchor.x - LABEL_GAP - width, anchor.y + LABEL_GAP + LABEL_HEIGHT),
        )
    if side is PinSide.TOP:
        if label == "VCC":
            primary = Point(anchor.x - LABEL_GAP - width, anchor.y - LABEL_GAP)
            return _expand_net_label_candidates(
                primary,
                Point(anchor.x + LABEL_GAP, anchor.y - LABEL_GAP),
                Point(
                    anchor.x - LABEL_GAP - width, anchor.y + LABEL_GAP + LABEL_HEIGHT
                ),
                Point(anchor.x + LABEL_GAP, anchor.y + LABEL_GAP + LABEL_HEIGHT),
            )
        primary = Point(anchor.x + LABEL_GAP, anchor.y - LABEL_GAP)
        return _expand_net_label_candidates(
            primary,
            Point(anchor.x - LABEL_GAP - width, anchor.y - LABEL_GAP),
            Point(anchor.x + LABEL_GAP, anchor.y + LABEL_GAP + LABEL_HEIGHT),
            Point(anchor.x - LABEL_GAP - width, anchor.y + LABEL_GAP + LABEL_HEIGHT),
        )
    if label == "VCC":
        primary = Point(
            anchor.x - LABEL_GAP - width, anchor.y + LABEL_GAP + LABEL_HEIGHT
        )
        return _expand_net_label_candidates(
            primary,
            Point(anchor.x + LABEL_GAP, anchor.y + LABEL_GAP + LABEL_HEIGHT),
            Point(anchor.x - LABEL_GAP - width, anchor.y - LABEL_GAP),
            Point(anchor.x + LABEL_GAP, anchor.y - LABEL_GAP),
        )
    primary = Point(anchor.x + LABEL_GAP, anchor.y + LABEL_GAP + LABEL_HEIGHT)
    return _expand_net_label_candidates(
        primary,
        Point(anchor.x - LABEL_GAP - width, anchor.y + LABEL_GAP + LABEL_HEIGHT),
        Point(anchor.x + LABEL_GAP, anchor.y - LABEL_GAP),
        Point(anchor.x - LABEL_GAP - width, anchor.y - LABEL_GAP),
    )


def _expand_net_label_candidates(*points: Point) -> tuple[Point, ...]:
    """Return nearby label positions with deterministic farther fallbacks."""

    step = LABEL_HEIGHT + 2.0 * LABEL_GAP
    offsets = (0.0, step, -step, 2.0 * step, -2.0 * step, 3.0 * step, -3.0 * step)
    result: list[Point] = []
    for point in points:
        for dy in offsets:
            candidate = point.translate(0.0, dy)
            if candidate not in result:
                result.append(candidate)
    return tuple(result)


def _place_net_labels(
    labels: tuple[NetLabel, ...],
    routing_bounds: tuple[Bounds, ...],
    wires: tuple[WirePath, ...],
) -> tuple[NetLabel, ...]:
    """Return local net labels placed away from existing rendered geometry."""

    placed: list[NetLabel] = []
    occupied = [
        bound.expanded(ROUTING_CLEARANCE, ROUTING_CLEARANCE) for bound in routing_bounds
    ]
    wire_segments = tuple(segment for wire in wires for segment in wire.segments)
    for label in sorted(
        labels, key=lambda item: (item.anchor.y, item.anchor.x, item.net)
    ):
        candidates = [
            NetLabel(
                net=label.net,
                text=label.text,
                side=label.side,
                anchor=label.anchor,
                position=position,
            )
            for position in _net_label_candidates(label.anchor, label.side, label.text)
        ]
        chosen = min(
            candidates,
            key=lambda candidate: (
                _net_label_collision_score(candidate, occupied, wire_segments),
                _manhattan_distance(label.position, candidate.position),
                candidate.position.y,
                candidate.position.x,
            ),
        )
        placed.append(chosen)
        occupied.append(
            _net_label_bounds(chosen).expanded(ROUTING_CLEARANCE, ROUTING_CLEARANCE)
        )
    return tuple(placed)


def _net_label_collision_score(
    label: NetLabel,
    occupied: list[Bounds],
    wire_segments: tuple[WireSegment, ...],
) -> float:
    """Return a score for local net label collisions."""

    bounds = _net_label_bounds(label).expanded(
        NET_LABEL_CLEARANCE,
        NET_LABEL_CLEARANCE,
    )
    score = 0.0
    for blocked in occupied:
        if bounds.overlaps(blocked):
            score += 10_000.0 + _overlap_area(bounds, blocked)
    for segment in wire_segments:
        if _segment_crosses_bounds(segment, bounds):
            score += 50_000.0
    return score


def _overlap_area(first: Bounds, second: Bounds) -> float:
    """Return the positive overlap area between two bounds."""

    width = min(first.right, second.right) - max(first.x, second.x)
    height = min(first.bottom, second.bottom) - max(first.y, second.y)
    if width <= 0.0 or height <= 0.0:
        return 0.0
    return width * height


def _route_net_tree(
    anchors: list[tuple[Connection, Point, Point, PinSide]],
    routing_bounds: tuple[Bounds, ...],
    net_index: int,
) -> list[WireSegment]:
    """Route one net as a local tree over pin escape points."""

    segments = _pin_stub_segments(anchors)
    escape_points = tuple(escape for _, _, escape, _ in anchors)
    for first_index, second_index in _net_tree_edges(escape_points):
        start = escape_points[first_index]
        end = escape_points[second_index]
        path = _orthogonal_path_between(
            start,
            end,
            routing_bounds,
            channel_offset=_net_channel_offset(net_index),
        )
        if path is None:
            path = _fallback_tree_path(start, end, routing_bounds, net_index)
        segments.extend(_path_segments(path))
    return segments


def _pin_stub_segments(
    anchors: list[tuple[Connection, Point, Point, PinSide]],
) -> list[WireSegment]:
    """Return immediate pin-to-escape segments for one net."""

    return [
        WireSegment(anchor, escape)
        for _, anchor, escape, _ in anchors
        if anchor != escape
    ]


def _net_tree_edges(points: tuple[Point, ...]) -> tuple[tuple[int, int], ...]:
    """Return deterministic Manhattan MST edges over route points."""

    connected = {0}
    remaining = set(range(1, len(points)))
    edges: list[tuple[int, int]] = []
    while remaining:
        first, second = min(
            ((first, second) for first in connected for second in remaining),
            key=lambda edge: (
                _manhattan_distance(points[edge[0]], points[edge[1]]),
                min(edge),
                max(edge),
            ),
        )
        connected.add(second)
        remaining.remove(second)
        edges.append((first, second))
    return tuple(edges)


def _fallback_tree_path(
    start: Point,
    end: Point,
    routing_bounds: tuple[Bounds, ...],
    net_index: int,
) -> tuple[Point, ...]:
    """Return a deterministic fallback path for rare disconnected grids."""

    detour_xs = [
        start.x,
        end.x,
        min(bound.x for bound in routing_bounds)
        - 80.0
        - float(net_index % 4) * GRID_SPACING,
        max(bound.right for bound in routing_bounds)
        + 80.0
        + float(net_index % 4) * GRID_SPACING,
    ]
    detour_ys = [
        start.y,
        end.y,
        min(bound.y for bound in routing_bounds)
        - 80.0
        - float(net_index % 4) * GRID_SPACING,
        max(bound.bottom for bound in routing_bounds)
        + 80.0
        + float(net_index % 4) * GRID_SPACING,
    ]
    best: tuple[float, tuple[Point, ...]] | None = None
    paths = [
        (start, Point(detour_x, start.y), Point(detour_x, end.y), end)
        for detour_x in detour_xs
    ] + [
        (start, Point(start.x, detour_y), Point(end.x, detour_y), end)
        for detour_y in detour_ys
    ]
    for path in paths:
        if any(
            _segment_crosses_any_bounds(segment, routing_bounds)
            for segment in _path_segments(path)
        ):
            continue
        score = sum(
            _manhattan_distance(segment.start, segment.end)
            for segment in _path_segments(path)
        )
        if best is None or score < best[0]:
            best = (score, path)
    if best is not None:
        return _simplify_path(best[1])
    return _simplify_path(
        (
            start,
            Point(start.x, end.y),
            end,
        )
    )


def _manhattan_distance(first: Point, second: Point) -> float:
    """Return Manhattan distance between two points."""

    return abs(first.x - second.x) + abs(first.y - second.y)


def _net_channel_offset(net_index: int) -> float:
    """Return a small deterministic channel offset for one net."""

    return float((net_index % 5) - 2) * 4.0


def _pin_escape_point(
    placed: PlacedPart, spec: PartSpec, pin_name: str, anchor: Point
) -> Point:
    """Return the first routing point after leaving a part pin."""

    pin = spec.pin(pin_name)
    bounds = placed.bounds
    if pin.side is PinSide.LEFT:
        return Point(bounds.x - PIN_ESCAPE, anchor.y)
    if pin.side is PinSide.RIGHT:
        return Point(bounds.right + PIN_ESCAPE, anchor.y)
    if pin.side is PinSide.TOP:
        return Point(anchor.x, bounds.y - PIN_ESCAPE)
    if pin.side is PinSide.BOTTOM:
        return Point(anchor.x, bounds.bottom + PIN_ESCAPE)
    return anchor


def _pin_approach_bounds(placed: PlacedPart, spec: PartSpec) -> tuple[Bounds, ...]:
    """Return outside keepouts that reserve clean lanes into component pins."""

    if not _needs_pin_approach_keepout(spec):
        return ()
    bounds = placed.bounds
    keepouts: list[Bounds] = []
    anchors_by_side: dict[PinSide, list[Point]] = {
        side: [] for side in (PinSide.LEFT, PinSide.RIGHT, PinSide.TOP, PinSide.BOTTOM)
    }
    for pin in spec.pins:
        anchors_by_side[pin.side].append(pin_position(placed, spec, pin.name))
    for side, anchors in anchors_by_side.items():
        if not anchors:
            continue
        if side is PinSide.LEFT:
            min_y = min(anchor.y for anchor in anchors) - PIN_APPROACH_CLEARANCE
            max_y = max(anchor.y for anchor in anchors) + PIN_APPROACH_CLEARANCE
            keepouts.append(
                Bounds(
                    x=bounds.x - PIN_ESCAPE,
                    y=min_y,
                    width=PIN_ESCAPE,
                    height=max_y - min_y,
                )
            )
        elif side is PinSide.RIGHT:
            min_y = min(anchor.y for anchor in anchors) - PIN_APPROACH_CLEARANCE
            max_y = max(anchor.y for anchor in anchors) + PIN_APPROACH_CLEARANCE
            keepouts.append(
                Bounds(
                    x=bounds.right,
                    y=min_y,
                    width=PIN_ESCAPE,
                    height=max_y - min_y,
                )
            )
        elif side is PinSide.TOP:
            min_x = min(anchor.x for anchor in anchors) - PIN_APPROACH_CLEARANCE
            max_x = max(anchor.x for anchor in anchors) + PIN_APPROACH_CLEARANCE
            keepouts.append(
                Bounds(
                    x=min_x,
                    y=bounds.y - PIN_ESCAPE,
                    width=max_x - min_x,
                    height=PIN_ESCAPE,
                )
            )
        elif side is PinSide.BOTTOM:
            min_x = min(anchor.x for anchor in anchors) - PIN_APPROACH_CLEARANCE
            max_x = max(anchor.x for anchor in anchors) + PIN_APPROACH_CLEARANCE
            keepouts.append(
                Bounds(
                    x=min_x,
                    y=bounds.bottom,
                    width=max_x - min_x,
                    height=PIN_ESCAPE,
                )
            )
    return tuple(keepouts)


def _pin_approach_bounds_for_pin(
    placed: PlacedPart, spec: PartSpec, pin_name: str
) -> Bounds:
    """Return the reserved outside lane leading into one pin."""

    pin = spec.pin(pin_name)
    anchor = pin_position(placed, spec, pin_name)
    bounds = placed.bounds
    if pin.side is PinSide.LEFT:
        return Bounds(
            x=bounds.x - PIN_ESCAPE,
            y=anchor.y - PIN_APPROACH_CLEARANCE,
            width=PIN_ESCAPE,
            height=2.0 * PIN_APPROACH_CLEARANCE,
        )
    if pin.side is PinSide.RIGHT:
        return Bounds(
            x=bounds.right,
            y=anchor.y - PIN_APPROACH_CLEARANCE,
            width=PIN_ESCAPE,
            height=2.0 * PIN_APPROACH_CLEARANCE,
        )
    if pin.side is PinSide.TOP:
        return Bounds(
            x=anchor.x - PIN_APPROACH_CLEARANCE,
            y=bounds.y - PIN_ESCAPE,
            width=2.0 * PIN_APPROACH_CLEARANCE,
            height=PIN_ESCAPE,
        )
    if pin.side is PinSide.BOTTOM:
        return Bounds(
            x=anchor.x - PIN_APPROACH_CLEARANCE,
            y=bounds.bottom,
            width=2.0 * PIN_APPROACH_CLEARANCE,
            height=PIN_ESCAPE,
        )
    return Bounds(x=anchor.x, y=anchor.y, width=0.0, height=0.0)


def _needs_pin_approach_keepout(spec: PartSpec) -> bool:
    """Return whether a component has enough pin density to reserve face lanes."""

    return len(spec.pins) > 0


def _orthogonal_path_between(
    start: Point,
    end: Point,
    bounds: tuple[Bounds, ...],
    *,
    channel_offset: float = 0.0,
) -> tuple[Point, ...] | None:
    """Find a deterministic Manhattan path between two points."""

    for search_bounds in _routing_bound_scopes(start, end, bounds):
        path = _search_orthogonal_path(
            start,
            end,
            search_bounds,
            channel_offset=channel_offset,
        )
        if path is not None and _path_avoids_bounds(path, bounds):
            return path
    return None


def _routing_bound_scopes(
    start: Point, end: Point, bounds: tuple[Bounds, ...]
) -> tuple[tuple[Bounds, ...], ...]:
    """Return progressively wider obstacle scopes for one route search."""

    scopes: list[tuple[Bounds, ...]] = []
    for margin in (140.0, 280.0, 560.0):
        envelope = _route_envelope(start, end, margin)
        local_bounds = tuple(bound for bound in bounds if bound.overlaps(envelope))
        if local_bounds and local_bounds not in scopes:
            scopes.append(local_bounds)
    if bounds not in scopes:
        scopes.append(bounds)
    return tuple(scopes)


def _route_envelope(start: Point, end: Point, margin: float) -> Bounds:
    """Return the expanded search envelope between two route endpoints."""

    min_x = min(start.x, end.x) - margin
    min_y = min(start.y, end.y) - margin
    return Bounds(
        x=min_x,
        y=min_y,
        width=abs(end.x - start.x) + 2.0 * margin,
        height=abs(end.y - start.y) + 2.0 * margin,
    )


def _path_avoids_bounds(path: tuple[Point, ...], bounds: tuple[Bounds, ...]) -> bool:
    """Return whether a complete path avoids all blocked bounds."""

    return not any(
        _segment_crosses_any_bounds(segment, bounds) for segment in _path_segments(path)
    )


def _search_orthogonal_path(
    start: Point,
    end: Point,
    bounds: tuple[Bounds, ...],
    *,
    channel_offset: float,
) -> tuple[Point, ...] | None:
    """Search one orthogonal route against a fixed obstacle scope."""

    start_key = (start.x, start.y)
    end_key = (end.x, end.y)
    x_values = {start.x, end.x}
    y_values = {start.y, end.y}
    for bound in bounds:
        x_values.update(
            (
                bound.x - ROUTING_PADDING + channel_offset,
                bound.right + ROUTING_PADDING + channel_offset,
            )
        )
        y_values.update(
            (
                bound.y - ROUTING_PADDING + channel_offset,
                bound.bottom + ROUTING_PADDING + channel_offset,
            )
        )

    xs = tuple(sorted(x_values))
    ys = tuple(sorted(y_values))
    x_index = {value: index for index, value in enumerate(xs)}
    y_index = {value: index for index, value in enumerate(ys)}
    heap: list[tuple[float, float, tuple[float, float]]] = [
        (_manhattan_distance(start, end), 0.0, start_key)
    ]
    distances = {start_key: 0.0}
    previous: dict[tuple[float, float], tuple[float, float]] = {}

    while heap:
        _, cost, node = heappop(heap)
        if cost != distances[node]:
            continue
        if node == end_key:
            return _simplify_path(_reconstruct_path(node, previous))
        for neighbor in _grid_neighbors(node, xs, ys, x_index, y_index):
            segment = WireSegment(
                start=Point(node[0], node[1]),
                end=Point(neighbor[0], neighbor[1]),
            )
            if _segment_crosses_any_bounds(segment, bounds):
                continue
            next_cost = cost + abs(neighbor[0] - node[0]) + abs(neighbor[1] - node[1])
            if next_cost >= distances.get(neighbor, float("inf")):
                continue
            distances[neighbor] = next_cost
            previous[neighbor] = node
            heappush(
                heap,
                (
                    next_cost
                    + _manhattan_distance(Point(neighbor[0], neighbor[1]), end),
                    next_cost,
                    neighbor,
                ),
            )
    return None


def _grid_neighbors(
    node: tuple[float, float],
    xs: tuple[float, ...],
    ys: tuple[float, ...],
    x_index: Mapping[float, int],
    y_index: Mapping[float, int],
) -> tuple[tuple[float, float], ...]:
    """Return adjacent horizontal and vertical grid neighbors."""

    x, y = node
    neighbors: list[tuple[float, float]] = []
    current_x = x_index[x]
    current_y = y_index[y]
    if current_x > 0:
        neighbors.append((xs[current_x - 1], y))
    if current_x + 1 < len(xs):
        neighbors.append((xs[current_x + 1], y))
    if current_y > 0:
        neighbors.append((x, ys[current_y - 1]))
    if current_y + 1 < len(ys):
        neighbors.append((x, ys[current_y + 1]))
    return tuple(neighbors)


def _segment_crosses_any_bounds(
    segment: WireSegment, bounds: tuple[Bounds, ...]
) -> bool:
    """Return whether a segment crosses any blocked bounds."""

    return any(_segment_crosses_bounds(segment, bound) for bound in bounds)


def _segment_bounds(segment: WireSegment) -> Bounds:
    """Return axis-aligned bounds covering one wire segment."""

    min_x = min(segment.start.x, segment.end.x)
    min_y = min(segment.start.y, segment.end.y)
    return Bounds(
        x=min_x,
        y=min_y,
        width=abs(segment.end.x - segment.start.x),
        height=abs(segment.end.y - segment.start.y),
    )


def _reconstruct_path(
    node: tuple[float, float],
    previous: Mapping[tuple[float, float], tuple[float, float]],
) -> tuple[Point, ...]:
    """Reconstruct a path from predecessor links."""

    nodes = [node]
    while nodes[-1] in previous:
        nodes.append(previous[nodes[-1]])
    nodes.reverse()
    return tuple(Point(x, y) for x, y in nodes)


def _simplify_path(path: tuple[Point, ...]) -> tuple[Point, ...]:
    """Remove redundant collinear points from a route path."""

    simplified: list[Point] = []
    for point in path:
        if simplified and point == simplified[-1]:
            continue
        simplified.append(point)
        while len(simplified) >= 3 and _collinear(
            simplified[-3],
            simplified[-2],
            simplified[-1],
        ):
            simplified.pop(-2)
    return tuple(simplified)


def _collinear(first: Point, second: Point, third: Point) -> bool:
    """Return whether three path points lie on one axis."""

    return (first.x == second.x == third.x) or (first.y == second.y == third.y)


def _path_segments(path: tuple[Point, ...]) -> list[WireSegment]:
    """Convert a point path into wire segments."""

    return [
        WireSegment(start=start, end=end)
        for start, end in zip(path, path[1:])
        if start != end
    ]


def _segment_crosses_bounds(segment: WireSegment, bounds: Bounds) -> bool:
    """Return whether an axis-aligned segment crosses bounds."""

    if segment.start.x == segment.end.x:
        x = segment.start.x
        min_y = min(segment.start.y, segment.end.y)
        max_y = max(segment.start.y, segment.end.y)
        return (
            bounds.x + GEOMETRY_EPSILON < x < bounds.right - GEOMETRY_EPSILON
            and min_y < bounds.bottom - GEOMETRY_EPSILON
            and max_y > bounds.y + GEOMETRY_EPSILON
        )
    if segment.start.y == segment.end.y:
        y = segment.start.y
        min_x = min(segment.start.x, segment.end.x)
        max_x = max(segment.start.x, segment.end.x)
        return (
            bounds.y + GEOMETRY_EPSILON < y < bounds.bottom - GEOMETRY_EPSILON
            and min_x < bounds.right - GEOMETRY_EPSILON
            and max_x > bounds.x + GEOMETRY_EPSILON
        )
    return False


def _dedupe_segments(segments: list[WireSegment]) -> list[WireSegment]:
    """Return segments with deterministic duplicates removed."""

    seen: set[tuple[float, float, float, float]] = set()
    result: list[WireSegment] = []
    for segment in segments:
        key = (
            segment.start.x,
            segment.start.y,
            segment.end.x,
            segment.end.y,
        )
        reverse_key = (
            segment.end.x,
            segment.end.y,
            segment.start.x,
            segment.start.y,
        )
        if key in seen or reverse_key in seen:
            continue
        seen.add(key)
        result.append(segment)
    return result


def _normalize_layout(
    circuit: Circuit,
    catalog: Mapping[str, PartSpec],
    placed_parts: tuple[PlacedPart, ...],
    wires: tuple[WirePath, ...],
    net_labels: tuple[NetLabel, ...],
) -> tuple[tuple[PlacedPart, ...], tuple[WirePath, ...], tuple[NetLabel, ...], Size]:
    """Shift layout to a positive canvas and return final size."""

    min_x, min_y, max_x, max_y = _layout_extents(
        circuit,
        catalog,
        placed_parts,
        wires,
        net_labels,
    )
    dx = CANVAS_MARGIN - min_x
    dy = CANVAS_MARGIN - min_y
    shifted_parts = tuple(part.translate(dx, dy) for part in placed_parts)
    shifted_wires = tuple(_translate_wire_path(wire, dx, dy) for wire in wires)
    shifted_net_labels = tuple(label.translate(dx, dy) for label in net_labels)
    _, _, shifted_max_x, shifted_max_y = _layout_extents(
        circuit,
        catalog,
        shifted_parts,
        shifted_wires,
        shifted_net_labels,
    )
    return (
        shifted_parts,
        shifted_wires,
        shifted_net_labels,
        Size(width=shifted_max_x + CANVAS_MARGIN, height=shifted_max_y + CANVAS_MARGIN),
    )


def _layout_extents(
    circuit: Circuit,
    catalog: Mapping[str, PartSpec],
    placed_parts: tuple[PlacedPart, ...],
    wires: tuple[WirePath, ...],
    net_labels: tuple[NetLabel, ...],
) -> tuple[float, float, float, float]:
    """Return visual and wire extents for a layout."""

    part_by_ref = circuit.part_by_ref()
    boxes = [
        visual_bounds(
            part, catalog[part_by_ref[part.ref].kind], part_by_ref[part.ref].value
        )
        for part in placed_parts
    ]
    boxes.extend(_net_label_bounds(label) for label in net_labels)
    points = [
        point
        for wire in wires
        for segment in wire.segments
        for point in (segment.start, segment.end)
    ]
    points.extend(label.anchor for label in net_labels)
    if not boxes and not points:
        return 0.0, 0.0, 0.0, 0.0
    min_x = min([box.x for box in boxes] + [point.x for point in points])
    min_y = min([box.y for box in boxes] + [point.y for point in points])
    max_x = max([box.right for box in boxes] + [point.x for point in points])
    max_y = max([box.bottom for box in boxes] + [point.y for point in points])
    return min_x, min_y, max_x, max_y


def _translate_wire_path(wire: WirePath, dx: float, dy: float) -> WirePath:
    """Translate one wire path."""

    return WirePath(
        net=wire.net,
        segments=tuple(
            WireSegment(
                start=segment.start.translate(dx, dy),
                end=segment.end.translate(dx, dy),
            )
            for segment in wire.segments
        ),
    )


def _net_label_bounds(label: NetLabel) -> Bounds:
    """Return approximate rendered bounds for one local net label."""

    return Bounds(
        x=label.position.x,
        y=label.position.y - LABEL_ASCENT,
        width=_text_width(label.text),
        height=LABEL_HEIGHT,
    )


def _part_size(spec: PartSpec) -> Size:
    """Return deterministic symbol size for a part."""

    pin_count = max(len(spec.pins), 2)
    return Size(width=82.0, height=max(48.0, 16.0 * pin_count))


def _slot(index: int, count: int, length: float) -> float:
    """Return a pin slot coordinate along one side."""

    return length * float(index + 1) / float(count + 1)


def _pin_label_bounds(anchor: Point, side: PinSide, label: str) -> Bounds:
    """Return approximate rendered bounds for one pin label."""

    baseline = pin_label_position(anchor, side, label)
    return Bounds(
        x=baseline.x,
        y=baseline.y - LABEL_ASCENT,
        width=_text_width(label),
        height=LABEL_HEIGHT,
    )


def _text_width(text: str) -> float:
    """Return deterministic monospace text width estimate."""

    return max(1, len(text)) * LABEL_CHAR_WIDTH


def _union_bounds(bounds: list[Bounds]) -> Bounds:
    """Return the union of one or more bounds."""

    min_x = min(bound.x for bound in bounds)
    min_y = min(bound.y for bound in bounds)
    max_x = max(bound.right for bound in bounds)
    max_y = max(bound.bottom for bound in bounds)
    return Bounds(x=min_x, y=min_y, width=max_x - min_x, height=max_y - min_y)
