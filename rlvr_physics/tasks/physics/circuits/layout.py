"""Deterministic schematic layout planner for circuit renderers."""

from dataclasses import dataclass
from itertools import combinations
from math import hypot
from statistics import median
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
CANVAS_MARGIN = 48.0
LAYER_SPACING = 260.0
ROW_SPACING = 170.0
ROUTING_PADDING = 28.0
LABEL_CHAR_WIDTH = 7.2
LABEL_HEIGHT = 14.0
LABEL_ASCENT = 11.0
LABEL_GAP = 6.0
FORCE_ITERATIONS = 180
MAX_FORCE_STEP = 36.0


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
    """Rendered path for one net."""

    net: str
    segments: tuple[WireSegment, ...]


@dataclass(frozen=True)
class Layout:
    """Complete deterministic schematic layout."""

    parts: tuple[PlacedPart, ...]
    wires: tuple[WirePath, ...]
    size: Size

    def part_by_ref(self) -> Mapping[str, PlacedPart]:
        """Return a lookup from reference designator to placed part."""

        return {part.ref: part for part in self.parts}


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
        Schematic layout with placed components and routed wire paths.
    """

    initial_parts = _initial_placement(circuit, catalog)
    placed = _force_directed_parts(circuit, catalog, initial_parts)
    placed = _snap_parts(placed)
    placed = _clear_overlaps(circuit, catalog, placed)
    wires = _route_wires(circuit, catalog, {part.ref: part for part in placed})
    placed, wires, size = _normalize_layout(circuit, catalog, placed, wires)
    return Layout(
        parts=tuple(sorted(placed, key=lambda part: part.ref)), wires=wires, size=size
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
    label_text = f"{placed_part.ref} {value}".strip()
    label_width = max(_text_width(label_text), placed_part.bounds.width)
    boxes.append(
        Bounds(
            x=placed_part.bounds.x,
            y=placed_part.bounds.y - LABEL_GAP - LABEL_HEIGHT,
            width=label_width,
            height=LABEL_HEIGHT,
        )
    )
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

    return visual_bounds(placed_part, spec, value).expanded(
        ROUTING_PADDING, ROUTING_PADDING
    )


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
        return Point(anchor.x - _text_width(label) - LABEL_GAP, anchor.y + 4.0)
    if side is PinSide.RIGHT:
        return Point(anchor.x + LABEL_GAP, anchor.y + 4.0)
    if side is PinSide.TOP:
        return Point(anchor.x + LABEL_GAP, anchor.y - LABEL_GAP)
    return Point(anchor.x + LABEL_GAP, anchor.y + LABEL_GAP + LABEL_HEIGHT)


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
) -> tuple[WirePath, ...]:
    """Build deterministic Manhattan paths through open routing channels."""

    paths: list[WirePath] = []
    clearance_bounds = {
        ref: placement_bounds(
            placed,
            catalog[circuit.part_by_ref()[ref].kind],
            circuit.part_by_ref()[ref].value,
        )
        for ref, placed in placed_parts.items()
    }
    for net_index, net in enumerate(circuit.nets):
        connections = circuit.connections_for_net(net)
        anchors: list[tuple[Connection, Point]] = []
        for connection in connections:
            placed = placed_parts[connection.ref]
            spec = catalog[circuit.part_by_ref()[connection.ref].kind]
            anchors.append((connection, pin_position(placed, spec, connection.pin)))
        if len(anchors) < 2:
            continue
        spine_x = _choose_spine_x(anchors, clearance_bounds, net_index)
        spine_ys = [point.y for _, point in anchors]
        branch_routes: list[tuple[Point, Point, Point]] = []
        for connection, anchor in anchors:
            branch_y = _choose_branch_y(
                connection,
                anchor,
                spine_x,
                clearance_bounds,
            )
            elbow = Point(anchor.x, branch_y)
            spine = Point(spine_x, branch_y)
            spine_ys.append(branch_y)
            branch_routes.append((anchor, elbow, spine))

        min_y = min(spine_ys)
        max_y = max(spine_ys)
        segments: list[WireSegment] = []
        if min_y != max_y:
            segments.append(WireSegment(Point(spine_x, min_y), Point(spine_x, max_y)))
        for anchor, elbow, spine in branch_routes:
            if anchor != elbow:
                segments.append(WireSegment(anchor, elbow))
            if elbow != spine:
                segments.append(WireSegment(elbow, spine))
        paths.append(
            WirePath(
                net=net,
                segments=tuple(_dedupe_segments(segments)),
            )
        )
    return tuple(paths)


def _choose_branch_y(
    connection: Connection,
    anchor: Point,
    spine_x: float,
    bounds_by_ref: Mapping[str, Bounds],
) -> float:
    """Choose a horizontal branch channel that avoids unrelated parts."""

    candidate_values: set[float] = {anchor.y}
    for bounds in bounds_by_ref.values():
        candidate_values.add(bounds.y)
        candidate_values.add(bounds.y - ROUTING_PADDING)
        candidate_values.add(bounds.bottom)
        candidate_values.add(bounds.bottom + ROUTING_PADDING)
    edges = sorted(
        {bounds.y for bounds in bounds_by_ref.values()}
        | {bounds.bottom for bounds in bounds_by_ref.values()}
    )
    for top, bottom in zip(edges, edges[1:]):
        if bottom - top > 2.0 * ROUTING_PADDING:
            candidate_values.add((top + bottom) / 2.0)

    candidates = sorted(candidate_values)
    return min(
        candidates,
        key=lambda candidate: (
            _branch_score(connection, anchor, spine_x, candidate, bounds_by_ref),
            abs(candidate - anchor.y),
            candidate,
        ),
    )


def _branch_score(
    connection: Connection,
    anchor: Point,
    spine_x: float,
    branch_y: float,
    bounds_by_ref: Mapping[str, Bounds],
) -> float:
    """Score one routed branch candidate."""

    score = abs(branch_y - anchor.y) + abs(spine_x - anchor.x)
    vertical = WireSegment(anchor, Point(anchor.x, branch_y))
    horizontal = WireSegment(Point(anchor.x, branch_y), Point(spine_x, branch_y))
    for ref, bounds in bounds_by_ref.items():
        if ref == connection.ref:
            continue
        if _segment_crosses_bounds(vertical, bounds):
            score += 100_000.0
        if _segment_crosses_bounds(horizontal, bounds):
            score += 100_000.0
    return score


def _choose_spine_x(
    anchors: list[tuple[Connection, Point]],
    bounds_by_ref: Mapping[str, Bounds],
    net_index: int,
) -> float:
    """Choose a vertical routing spine that avoids placement bounds."""

    points = [point for _, point in anchors]
    median_x = float(median(point.x for point in points))
    min_x = min(
        [point.x for point in points] + [bounds.x for bounds in bounds_by_ref.values()]
    )
    max_x = max(
        [point.x for point in points]
        + [bounds.right for bounds in bounds_by_ref.values()]
    )
    candidate_values: set[float] = {
        median_x,
        min_x - 80.0 - float(net_index % 4) * GRID_SPACING,
        max_x + 80.0 + float(net_index % 4) * GRID_SPACING,
    }
    for bounds in bounds_by_ref.values():
        candidate_values.add(bounds.x)
        candidate_values.add(bounds.x - ROUTING_PADDING)
        candidate_values.add(bounds.right)
        candidate_values.add(bounds.right + ROUTING_PADDING)
    edges = sorted(
        {bounds.x for bounds in bounds_by_ref.values()}
        | {bounds.right for bounds in bounds_by_ref.values()}
    )
    for left, right in zip(edges, edges[1:]):
        if right - left > 2.0 * ROUTING_PADDING:
            candidate_values.add((left + right) / 2.0)

    candidates = sorted(candidate_values)
    return min(
        candidates,
        key=lambda candidate: (
            _spine_score(candidate, anchors, bounds_by_ref),
            abs(candidate - median_x),
            candidate,
        ),
    )


def _spine_score(
    spine_x: float,
    anchors: list[tuple[Connection, Point]],
    bounds_by_ref: Mapping[str, Bounds],
) -> float:
    """Score one routing spine candidate."""

    points = [point for _, point in anchors]
    min_y = min(point.y for point in points)
    max_y = max(point.y for point in points)
    score = sum(abs(point.x - spine_x) for point in points)
    vertical = WireSegment(Point(spine_x, min_y), Point(spine_x, max_y))
    for ref, bounds in bounds_by_ref.items():
        if _segment_crosses_bounds(vertical, bounds):
            score += 100_000.0
    for connection, anchor in anchors:
        branch = WireSegment(anchor, Point(spine_x, anchor.y))
        for ref, bounds in bounds_by_ref.items():
            if ref == connection.ref:
                continue
            if _segment_crosses_bounds(branch, bounds):
                score += 100_000.0
    return score


def _segment_crosses_bounds(segment: WireSegment, bounds: Bounds) -> bool:
    """Return whether an axis-aligned segment crosses bounds."""

    if segment.start.x == segment.end.x:
        x = segment.start.x
        min_y = min(segment.start.y, segment.end.y)
        max_y = max(segment.start.y, segment.end.y)
        return (
            bounds.x < x < bounds.right and min_y < bounds.bottom and max_y > bounds.y
        )
    if segment.start.y == segment.end.y:
        y = segment.start.y
        min_x = min(segment.start.x, segment.end.x)
        max_x = max(segment.start.x, segment.end.x)
        return (
            bounds.y < y < bounds.bottom and min_x < bounds.right and max_x > bounds.x
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
) -> tuple[tuple[PlacedPart, ...], tuple[WirePath, ...], Size]:
    """Shift layout to a positive canvas and return final size."""

    min_x, min_y, max_x, max_y = _layout_extents(circuit, catalog, placed_parts, wires)
    dx = CANVAS_MARGIN - min_x
    dy = CANVAS_MARGIN - min_y
    shifted_parts = tuple(part.translate(dx, dy) for part in placed_parts)
    shifted_wires = tuple(_translate_wire_path(wire, dx, dy) for wire in wires)
    _, _, shifted_max_x, shifted_max_y = _layout_extents(
        circuit,
        catalog,
        shifted_parts,
        shifted_wires,
    )
    return (
        shifted_parts,
        shifted_wires,
        Size(width=shifted_max_x + CANVAS_MARGIN, height=shifted_max_y + CANVAS_MARGIN),
    )


def _layout_extents(
    circuit: Circuit,
    catalog: Mapping[str, PartSpec],
    placed_parts: tuple[PlacedPart, ...],
    wires: tuple[WirePath, ...],
) -> tuple[float, float, float, float]:
    """Return visual and wire extents for a layout."""

    part_by_ref = circuit.part_by_ref()
    boxes = [
        visual_bounds(
            part, catalog[part_by_ref[part.ref].kind], part_by_ref[part.ref].value
        )
        for part in placed_parts
    ]
    points = [
        point
        for wire in wires
        for segment in wire.segments
        for point in (segment.start, segment.end)
    ]
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


def _part_size(spec: PartSpec) -> Size:
    """Return deterministic symbol size for a part."""

    pin_count = max(len(spec.pins), 2)
    return Size(width=92.0, height=max(56.0, 18.0 * pin_count))


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
