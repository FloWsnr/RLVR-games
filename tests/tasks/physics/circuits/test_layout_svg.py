"""Tests for schematic layout and SVG drawing."""

from functools import cache
from hashlib import sha256
from importlib.resources import files
from math import inf, nan
from pathlib import Path
from random import Random
from typing import Mapping

import pytest

import rlvr_physics.tasks.physics.circuits.svg as svg_module
from rlvr_physics.core.rendering import PNG_MIME_TYPE, validate_png_image_data
from rlvr_physics.tasks.physics.circuits import (
    Bounds,
    Circuit,
    CircuitBuilder,
    CircuitRenderStyle,
    DEFAULT_RENDER_STYLE,
    GeneratorConfig,
    Layout,
    NetLabel,
    PartInstance,
    PartSpec,
    PinSide,
    PlacedPart,
    Point,
    Size,
    WirePath,
    default_catalog,
    default_motif_weights,
    default_motifs,
    draw_png,
    draw_svg,
    generate_circuit,
    plan_layout,
    to_png,
    WireSegment,
)
from rlvr_physics.tasks.physics.circuits.layout import (
    _RoutingObstacleIndex,
    _segment_crosses_bounds as _layout_segment_crosses_bounds,
    _net_label_bounds,
    _plan_layout,
    _pin_approach_bounds,
    _pin_escape_point,
    _pin_label_bounds,
    component_label_bounds,
    pin_position,
    placement_bounds,
)
from rlvr_physics.tasks.physics.circuits.symbol_assets import (
    SymbolAsset,
    asset_component_label_bounds,
    asset_for_part,
    asset_render_bounds_for_part,
    asset_terminals_for_part,
    draw_asset_part,
    symbol_fragments_for_scale,
)
from rlvr_physics.tasks.physics.circuits.svg import (
    _asset_pin_position,
    _junction_points,
    _rendered_wire_segments,
)
from tests.tasks.physics.circuits.test_model import divider_circuit


GENERATED_IMAGE_ROOT = Path(__file__).with_name("images")
GENERATED_MOTIF_IMAGE_DIR = GENERATED_IMAGE_ROOT / "motif"
GENERATED_CIRCUIT_IMAGE_DIR = GENERATED_IMAGE_ROOT / "circuit"
GENERATED_CASES = (
    (0, 3),
    (1, 3),
    (7, 4),
    (2, 4),
    (4, 5),
    (6, 5),
    (9, 3),
    (36, 4),
    (40, 5),
)
GENERATED_LAYOUT_FINGERPRINTS = {
    (0, 3): "7113aab8c49450c872b2955ea6cc9c91bfc10cdb1215febe15eee58e7d5b34dc",
    (1, 3): "d640ebf1c21a92635f64f1629647d86b7989a3a77d35a08ee2e0a923a084f827",
    (7, 4): "ffb238ae8c13f725beecfbe8f11317357fa9e07b2e9b15337df02d928ecc80de",
    (2, 4): "d307e3eb03c7dc09e2f367716170d4688ca5e338bf10378afe00efed26e896e0",
    (4, 5): "773ab232acfeb4e526af828cf9485ba35064c32f5d3c288a634ee4f9de3d2e2c",
    (6, 5): "e9941783afa9f26f9cc5ec244e1cbc1748fc395f54d67eb6f3e3dd725829de05",
    (9, 3): "7a500804c7e882186241c2a3090b425c082594048db86096cda2310670471195",
    (36, 4): "7f19fabe15433c47cd91fbb96470f797faf0dfbce7eae88cf0d82f16aced1f3f",
    (40, 5): "4af835a10566d023a91dcd8d2cbf75d7118412eb30c070be96aa1685396be9f7",
}
GENERATED_ASSET_LAYOUT_FINGERPRINTS = {
    (0, 3): "d6db662f89125548364831804d5889ec104894418a2ca191b56b87d5c5b493e6",
    (1, 3): "059e61d903b86f19e17c922b81fb448a5a65247ae8a3378afaf157782c8e70cd",
    (7, 4): "fa5b8734c2b7ff19b5ee228cb0606805af2fb7cd91c1961993071adfde61c582",
    (2, 4): "696ea4d6489f2baa6b958b02129158823257c611eb9622c254f51deaae05be25",
    (4, 5): "e7c850dc1f31a52f4d002d8d9c3c326f7ed82893b61b2b909e56cdb1b8a6f2df",
    (6, 5): "5fb81902bd80c757216a5feac842a0f8a2efb0704d149dc8e5bd5a2bd8d099a5",
    (9, 3): "e9f84e1d33cc3364715097296598ad10cc60c05f58d470bd570a10d7e41b45f0",
    (36, 4): "46a677c385753709d235790c0146bce6c20d41d3c63e627470c7677dbf4f0efe",
    (40, 5): "6223a28a15871313bbfcef9f6612cf102b735d4b238ed03dad2486c40072400b",
}
GEOMETRY_EPSILON = 1.0e-6
WIRE_CLEARANCE = 1.0
DYNAMIC_PACKAGE_PIN_LEAD = 14.0
DYNAMIC_PACKAGE_KINDS = (
    "controlled_switch",
    "counter_4bit",
    "instrumentation_amplifier",
    "timer_555",
)


def test_layout_places_parts_without_overlap() -> None:
    layout = plan_layout(divider_circuit(), default_catalog())
    bounds = [part.bounds for part in layout.parts]

    for index, current in enumerate(bounds):
        for other in bounds[index + 1 :]:
            assert not current.overlaps(other)


def test_routing_obstacle_index_matches_naive_bounds_scan() -> None:
    bounds = (
        Bounds(10.0, 10.0, 40.0, 30.0),
        Bounds(-20.0, -10.0, 15.0, 50.0),
        Bounds(70.0, 20.0, 0.0, 30.0),
        Bounds(80.0, -30.0, 25.0, 25.0),
    )
    segments = (
        WireSegment(Point(20.0, 0.0), Point(20.0, 60.0)),
        WireSegment(Point(10.0, 0.0), Point(10.0, 60.0)),
        WireSegment(Point(30.0, 40.0), Point(30.0, 60.0)),
        WireSegment(Point(-10.0, -20.0), Point(-10.0, 20.0)),
        WireSegment(Point(0.0, 20.0), Point(60.0, 20.0)),
        WireSegment(Point(0.0, 10.0), Point(60.0, 10.0)),
        WireSegment(Point(70.0, 25.0), Point(95.0, 25.0)),
        WireSegment(Point(60.0, -20.0), Point(120.0, -20.0)),
    )
    index = _RoutingObstacleIndex(bounds)

    for segment in segments:
        assert index.segment_crosses_any(segment) is any(
            _layout_segment_crosses_bounds(segment, bound) for bound in bounds
        )


def test_layout_places_rendered_bounds_without_overlap() -> None:
    catalog = default_catalog()

    for seed, element_count in GENERATED_CASES:
        circuit, layout = _planned_generated_case(seed, element_count)
        part_by_ref = circuit.part_by_ref()
        bounds = [
            placement_bounds(
                part,
                catalog[part_by_ref[part.ref].kind],
                part_by_ref[part.ref].value,
            )
            for part in layout.parts
        ]

        for index, current in enumerate(bounds):
            for other in bounds[index + 1 :]:
                assert not current.overlaps(other), (seed, element_count)


def test_layout_routes_wires_to_declared_pin_positions() -> None:
    catalog = default_catalog()

    for seed, element_count in GENERATED_CASES:
        circuit, _ = _planned_generated_case(seed, element_count)
        layout = _plan_layout(circuit, catalog, route_pin_labels=True)
        _assert_wire_endpoints_hit_pins(
            (seed, element_count),
            circuit,
            layout,
            catalog,
        )


def test_layout_routes_only_orthogonal_wire_segments() -> None:
    for seed, element_count in GENERATED_CASES:
        _, layout = _planned_generated_case(seed, element_count)

        for wire in layout.wires:
            for segment in wire.segments:
                assert _segment_is_orthogonal(segment), (
                    seed,
                    element_count,
                    wire.net,
                    segment,
                )


def test_layout_connects_every_default_motif_net_geometry() -> None:
    catalog = default_catalog()

    for index, (name, motif) in enumerate(default_motifs().items()):
        circuit, layout = _planned_motif_case(index, name, motif.element_count + 1)
        _assert_wire_paths_connect_declared_pins(name, circuit, layout, catalog)


def test_layout_routes_wires_around_unrelated_rendered_bounds() -> None:
    catalog = default_catalog()

    for seed, element_count in GENERATED_CASES:
        circuit, layout = _planned_generated_case(seed, element_count)
        part_by_ref = circuit.part_by_ref()
        placed_by_ref = layout.part_by_ref()
        boxes_by_ref = {
            ref: _default_rendered_boxes(
                placed,
                catalog[part_by_ref[ref].kind],
                part_by_ref[ref].value,
            )
            for ref, placed in placed_by_ref.items()
        }

        for wire in layout.wires:
            connected_refs = {
                connection.ref for connection in circuit.connections_for_net(wire.net)
            }
            for segment in wire.segments:
                for ref, boxes in boxes_by_ref.items():
                    if ref in connected_refs:
                        continue
                    for bounds in boxes:
                        assert not _segment_crosses_bounds(
                            segment,
                            bounds.expanded(WIRE_CLEARANCE, WIRE_CLEARANCE),
                        ), (
                            seed,
                            element_count,
                            wire.net,
                            ref,
                        )


def test_layout_routes_wires_around_all_symbol_bounds() -> None:
    for seed, element_count in GENERATED_CASES:
        _, layout = _planned_generated_case(seed, element_count)

        for wire in layout.wires:
            for segment in wire.segments:
                assert _segment_is_orthogonal(segment), (
                    seed,
                    element_count,
                    wire.net,
                    segment,
                )
                for part in layout.parts:
                    assert not _segment_crosses_bounds(segment, part.bounds), (
                        seed,
                        element_count,
                        wire.net,
                        part.ref,
                    )


def test_layout_routes_wires_around_all_component_labels() -> None:
    catalog = default_catalog()

    for seed, element_count in GENERATED_CASES:
        circuit, layout = _planned_generated_case(seed, element_count)
        part_by_ref = circuit.part_by_ref()
        label_bounds = [
            component_label_bounds(
                part,
                catalog[part_by_ref[part.ref].kind],
                part_by_ref[part.ref].value,
            )
            for part in layout.parts
        ]

        for wire in layout.wires:
            for segment in wire.segments:
                for bounds in label_bounds:
                    assert not _segment_crosses_bounds(
                        segment,
                        bounds.expanded(WIRE_CLEARANCE, WIRE_CLEARANCE),
                    ), (
                        seed,
                        element_count,
                        wire.net,
                        bounds,
                    )


def test_asset_aware_layout_routes_wires_around_drawn_component_labels() -> None:
    catalog = default_catalog()

    seed, element_count = GENERATED_CASES[0]
    circuit, _ = _planned_generated_case(seed, element_count)
    layout = _plan_layout(
        circuit,
        catalog,
        route_pin_labels=False,
        pin_position_resolver=_asset_pin_position,
        component_label_bounds_resolver=asset_component_label_bounds,
    )
    part_by_ref = circuit.part_by_ref()
    label_bounds = [
        asset_component_label_bounds(
            part,
            catalog[part_by_ref[part.ref].kind],
            part_by_ref[part.ref],
        )
        for part in layout.parts
    ]

    for wire in layout.wires:
        for segment in wire.segments:
            for bounds in label_bounds:
                assert not _segment_crosses_bounds(
                    segment,
                    bounds.expanded(WIRE_CLEARANCE, WIRE_CLEARANCE),
                ), (
                    seed,
                    element_count,
                    wire.net,
                    bounds,
                )


def test_draw_svg_planning_uses_drawn_component_label_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = default_catalog()
    calls: list[str] = []
    original_resolver = svg_module.asset_component_label_bounds

    def recording_resolver(
        part: PlacedPart,
        spec: PartSpec,
        instance: PartInstance,
    ) -> Bounds:
        calls.append(part.ref)
        return original_resolver(part, spec, instance)

    monkeypatch.setattr(
        svg_module,
        "asset_component_label_bounds",
        recording_resolver,
    )

    draw_svg(divider_circuit(), catalog)

    assert {"GND1", "R1", "R2", "V1"}.issubset(calls)


def test_layout_routes_shared_spines_around_component_labels() -> None:
    catalog = default_catalog()
    seed = 21
    element_count = 4
    circuit, layout = _planned_generated_case(seed, element_count)
    part_by_ref = circuit.part_by_ref()
    label_bounds = [
        component_label_bounds(
            part,
            catalog[part_by_ref[part.ref].kind],
            part_by_ref[part.ref].value,
        )
        for part in layout.parts
    ]

    for wire in layout.wires:
        for segment in wire.segments:
            for bounds in label_bounds:
                assert not _segment_crosses_bounds(
                    segment,
                    bounds.expanded(WIRE_CLEARANCE, WIRE_CLEARANCE),
                ), (
                    seed,
                    element_count,
                    wire.net,
                    bounds,
                )


def test_layout_routes_pin_label_mode_around_pin_labels() -> None:
    catalog = default_catalog()
    pin_label_cases = ((0, 3),)

    for seed, element_count in pin_label_cases:
        circuit, _ = _planned_generated_case(seed, element_count)
        layout = _plan_layout(circuit, catalog, route_pin_labels=True)
        part_by_ref = circuit.part_by_ref()
        pin_label_bounds = [
            _pin_label_bounds(
                pin_position(part, spec, pin.name),
                pin.side,
                pin.name,
            )
            for part in layout.parts
            for spec in (catalog[part_by_ref[part.ref].kind],)
            for pin in spec.pins
        ]

        for wire in layout.wires:
            for segment in wire.segments:
                for bounds in pin_label_bounds:
                    assert not _segment_crosses_bounds(
                        segment,
                        bounds.expanded(WIRE_CLEARANCE, WIRE_CLEARANCE),
                    ), (
                        seed,
                        element_count,
                        wire.net,
                        bounds,
                    )


def test_layout_reserves_component_pin_approach_lanes() -> None:
    catalog = default_catalog()

    for seed, element_count in GENERATED_CASES:
        circuit, layout = _planned_generated_case(seed, element_count)
        _assert_pin_approach_lanes_clear(
            (seed, element_count),
            circuit,
            layout,
            catalog,
        )
    for index, (name, motif) in enumerate(default_motifs().items()):
        circuit, layout = _planned_motif_case(index, name, motif.element_count + 1)
        _assert_pin_approach_lanes_clear(name, circuit, layout, catalog)


def test_layout_uses_local_labels_for_shared_global_nets() -> None:
    circuit, layout = _planned_generated_case(2, 5)
    labeled_nets = {label.net for label in layout.net_labels}

    assert "0" in labeled_nets
    assert all(label.text == "GND" for label in layout.net_labels if label.net == "0")
    assert len([label for label in layout.net_labels if label.net == "0"]) == len(
        circuit.connections_for_net("0")
    )


def test_layout_places_local_net_labels_clear_of_rendered_geometry() -> None:
    catalog = default_catalog()

    for seed, element_count in (*GENERATED_CASES, (12, 3), (21, 4), (33, 5)):
        circuit, layout = _planned_generated_case(seed, element_count)
        _assert_net_labels_are_clear((seed, element_count), circuit, layout, catalog)
    for index, (name, motif) in enumerate(default_motifs().items()):
        circuit, layout = _planned_motif_case(index, name, motif.element_count + 1)
        _assert_net_labels_are_clear(name, circuit, layout, catalog)


def test_layout_routes_signals_around_labeled_net_stubs() -> None:
    for seed, element_count in (*GENERATED_CASES, (12, 3), (21, 4)):
        _, layout = _planned_generated_case(seed, element_count)
        _assert_cross_net_labeled_stubs_are_clear((seed, element_count), layout)
    for index, (name, motif) in enumerate(default_motifs().items()):
        _, layout = _planned_motif_case(index, name, motif.element_count + 1)
        _assert_cross_net_labeled_stubs_are_clear(name, layout)


def test_layout_avoids_collinear_cross_net_wire_overlaps() -> None:
    for seed, element_count in (*GENERATED_CASES, (11, 3)):
        _, layout = _planned_generated_case(seed, element_count)
        _assert_no_collinear_cross_net_wire_overlaps((seed, element_count), layout)
    for index, (name, motif) in enumerate(default_motifs().items()):
        _, layout = _planned_motif_case(index, name, motif.element_count + 1)
        _assert_no_collinear_cross_net_wire_overlaps(name, layout)


def test_layout_routes_every_default_motif_cleanly() -> None:
    catalog = default_catalog()

    for index, (name, motif) in enumerate(default_motifs().items()):
        circuit, layout = _planned_motif_case(index, name, motif.element_count + 1)
        part_by_ref = circuit.part_by_ref()
        placed_by_ref = layout.part_by_ref()
        boxes_by_ref = {
            ref: _default_rendered_boxes(
                placed,
                catalog[part_by_ref[ref].kind],
                part_by_ref[ref].value,
            )
            for ref, placed in placed_by_ref.items()
        }
        label_bounds = [
            component_label_bounds(
                part,
                catalog[part_by_ref[part.ref].kind],
                part_by_ref[part.ref].value,
            )
            for part in layout.parts
        ]

        _assert_wire_endpoints_hit_pins(name, circuit, layout, catalog)
        for wire in layout.wires:
            connected_refs = {
                connection.ref for connection in circuit.connections_for_net(wire.net)
            }
            for segment in wire.segments:
                for part in layout.parts:
                    assert not _segment_crosses_bounds(segment, part.bounds), (
                        name,
                        wire.net,
                        part.ref,
                    )
                for ref, boxes in boxes_by_ref.items():
                    if ref in connected_refs:
                        continue
                    for bounds in boxes:
                        assert not _segment_crosses_bounds(
                            segment,
                            bounds.expanded(WIRE_CLEARANCE, WIRE_CLEARANCE),
                        ), (
                            name,
                            wire.net,
                            ref,
                        )
                for bounds in label_bounds:
                    assert not _segment_crosses_bounds(
                        segment,
                        bounds.expanded(WIRE_CLEARANCE, WIRE_CLEARANCE),
                    ), (
                        name,
                        wire.net,
                        bounds,
                    )
                for bounds in (_net_label_bounds(label) for label in layout.net_labels):
                    assert not _segment_crosses_bounds(
                        segment,
                        bounds.expanded(WIRE_CLEARANCE, WIRE_CLEARANCE),
                    ), (
                        name,
                        wire.net,
                        bounds,
                    )


def test_generated_layout_fingerprints_preserve_representative_cases() -> None:
    for case, expected in GENERATED_LAYOUT_FINGERPRINTS.items():
        _, layout = _planned_generated_case(*case)

        assert _layout_fingerprint(layout) == expected


def test_generated_asset_layout_fingerprints_preserve_representative_cases() -> None:
    catalog = default_catalog()

    for case, expected in GENERATED_ASSET_LAYOUT_FINGERPRINTS.items():
        circuit, _ = _planned_generated_case(*case)
        layout = _plan_layout(
            circuit,
            catalog,
            route_pin_labels=False,
            pin_position_resolver=_asset_pin_position,
            component_label_bounds_resolver=asset_component_label_bounds,
        )

        assert _layout_fingerprint(layout) == expected


def test_svg_draws_circuit_symbols_and_labels() -> None:
    circuit = divider_circuit()
    catalog = default_catalog()
    svg = draw_svg(circuit, catalog)

    assert DEFAULT_RENDER_STYLE == CircuitRenderStyle(
        wire_stroke_width=2.7,
        symbol_stroke_width=2.5,
        symbol_solid_stroke_width=2.0,
        pin_stroke_width=2.5,
        terminal_dot_radius=3.6,
        junction_dot_radius=4.4,
    )
    assert svg.startswith("<svg ")
    assert 'id="R1"' in svg
    assert 'class="circuit-symbol"' in svg
    assert 'class="symbol-asset"' in svg
    assert 'data-symbol="resistor"' in svg
    assert 'data-symbol="variable_resistor"' not in svg
    assert "vector-effect" not in svg
    assert ".wire{stroke:#1f2937;stroke-width:2.7" in svg
    assert ".symbol{stroke:#111827;stroke-width:2.5" in svg
    assert ".symbol-solid{stroke:#111827;stroke-width:2.0" in svg
    assert ".pin{stroke:#6b7280;stroke-width:2.5" in svg
    assert ".label{font:11px monospace" in svg
    assert "<clipPath" not in svg
    assert "<use " not in svg
    assert ">1</text>" not in svg
    assert "R2 2k" in svg
    assert 'class="background"' in svg
    assert 'fill="#ffffff"' in svg
    assert 'class="wire"' in svg
    assert 'class="junction"' in svg
    assert '<circle class="junction"' in svg
    assert 'r="3.6"/>' in svg
    assert 'r="1.1"/>' not in svg
    assert svg.endswith("</svg>\n")


def test_svg_accepts_dynamic_render_style() -> None:
    circuit = divider_circuit()
    catalog = default_catalog()
    style = CircuitRenderStyle(
        wire_stroke_width=4.0,
        symbol_stroke_width=3.0,
        symbol_solid_stroke_width=2.0,
        pin_stroke_width=3.5,
        terminal_dot_radius=6.0,
        junction_dot_radius=7.0,
    )

    svg = draw_svg(circuit, catalog, style=style)

    assert ".wire{stroke:#1f2937;stroke-width:4.0" in svg
    assert ".symbol{stroke:#111827;stroke-width:3.0" in svg
    assert ".symbol-solid{stroke:#111827;stroke-width:2.0" in svg
    assert ".pin{stroke:#6b7280;stroke-width:3.5" in svg
    assert 'r="6.0"/>' in svg


def test_render_style_rejects_non_positive_dimensions() -> None:
    with pytest.raises(
        ValueError, match="wire_stroke_width must be finite and positive"
    ):
        CircuitRenderStyle(
            wire_stroke_width=0.0,
            symbol_stroke_width=2.5,
            symbol_solid_stroke_width=2.0,
            pin_stroke_width=2.5,
            terminal_dot_radius=3.6,
            junction_dot_radius=4.4,
        )
    with pytest.raises(
        ValueError,
        match="terminal_dot_radius must be finite and positive",
    ):
        CircuitRenderStyle(
            wire_stroke_width=2.7,
            symbol_stroke_width=2.5,
            symbol_solid_stroke_width=2.0,
            pin_stroke_width=2.5,
            terminal_dot_radius=nan,
            junction_dot_radius=4.4,
        )
    with pytest.raises(
        ValueError,
        match="junction_dot_radius must be finite and positive",
    ):
        CircuitRenderStyle(
            wire_stroke_width=2.7,
            symbol_stroke_width=2.5,
            symbol_solid_stroke_width=2.0,
            pin_stroke_width=2.5,
            terminal_dot_radius=3.6,
            junction_dot_radius=inf,
        )

    with pytest.raises(ValueError, match="stroke_scale must be finite and positive"):
        DEFAULT_RENDER_STYLE.scaled(stroke_scale=0.0, dot_scale=1.0)
    with pytest.raises(ValueError, match="stroke_scale must be finite and positive"):
        DEFAULT_RENDER_STYLE.scaled(stroke_scale=nan, dot_scale=1.0)
    with pytest.raises(ValueError, match="dot_scale must be finite and positive"):
        DEFAULT_RENDER_STYLE.scaled(stroke_scale=1.0, dot_scale=0.0)
    with pytest.raises(ValueError, match="dot_scale must be finite and positive"):
        DEFAULT_RENDER_STYLE.scaled(stroke_scale=1.0, dot_scale=inf)


def test_logic_symbols_use_distinct_assets() -> None:
    catalog = default_catalog()
    builder = CircuitBuilder("logic-assets", catalog)
    builder.add_part("GND1", "ground", "0", {}, {})
    builder.connect("GND1", "0", "0")
    builder.add_part("U1", "and_gate", "AND", {}, {})
    builder.connect("U1", "in1", "A")
    builder.connect("U1", "in2", "B")
    builder.connect("U1", "out", "C")
    builder.connect("U1", "vcc", "VCC")
    builder.connect("U1", "gnd", "0")
    builder.add_part("U2", "or_gate", "OR", {}, {})
    builder.connect("U2", "in1", "A")
    builder.connect("U2", "in2", "B")
    builder.connect("U2", "out", "D")
    builder.connect("U2", "vcc", "VCC")
    builder.connect("U2", "gnd", "0")
    builder.add_part("U3", "not_gate", "NOT", {}, {})
    builder.connect("U3", "in1", "D")
    builder.connect("U3", "out", "E")
    builder.connect("U3", "vcc", "VCC")
    builder.connect("U3", "gnd", "0")
    builder.add_part("U4", "nand_gate", "NAND", {}, {})
    builder.connect("U4", "in1", "A")
    builder.connect("U4", "in2", "B")
    builder.connect("U4", "out", "F")
    builder.connect("U4", "vcc", "VCC")
    builder.connect("U4", "gnd", "0")
    builder.add_part("U5", "xor_gate", "XOR", {}, {})
    builder.connect("U5", "in1", "A")
    builder.connect("U5", "in2", "B")
    builder.connect("U5", "out", "G")
    builder.connect("U5", "vcc", "VCC")
    builder.connect("U5", "gnd", "0")

    svg = draw_svg(builder.freeze(), catalog)

    assert 'data-symbol="and_gate"' in svg
    assert 'data-symbol="or_gate"' in svg
    assert 'data-symbol="not_gate"' in svg
    assert 'data-symbol="nand_gate"' in svg
    assert 'data-symbol="xor_gate"' in svg


def test_controlled_current_source_uses_current_symbol() -> None:
    catalog = default_catalog()
    builder = CircuitBuilder("controlled-current-assets", catalog)
    builder.add_part("GND1", "ground", "0", {}, {})
    builder.connect("GND1", "0", "0")
    builder.add_part("G1", "vccs", "gm=1m", {"gain": 1e-3}, {})
    builder.connect("G1", "cp", "VCC")
    builder.connect("G1", "cn", "0")
    builder.connect("G1", "p", "OUT")
    builder.connect("G1", "n", "0")

    svg = draw_svg(builder.freeze(), catalog)

    assert 'data-symbol="controlled_current_source"' in svg


def test_switch_state_selects_open_or_closed_symbol() -> None:
    catalog = default_catalog()
    spec = catalog["ideal_switch"]
    part = PlacedPart(
        ref="S1",
        kind="ideal_switch",
        center=Point(100.0, 100.0),
        size=Size(82.0, 48.0),
    )

    open_fragments = "\n".join(
        draw_asset_part(
            part,
            spec,
            PartInstance(
                "S1",
                "ideal_switch",
                "jumper",
                {"state_resistance_ohm": 1e12},
                {"state": "open"},
            ),
        )
    )
    closed_fragments = "\n".join(
        draw_asset_part(
            part,
            spec,
            PartInstance(
                "S1",
                "ideal_switch",
                "jumper",
                {"state_resistance_ohm": 0.05},
                {"state": "closed"},
            ),
        )
    )
    closed_by_resistance_fragments = "\n".join(
        draw_asset_part(
            part,
            spec,
            PartInstance(
                "S1",
                "ideal_switch",
                "jumper",
                {"state_resistance_ohm": 0.05},
                {},
            ),
        )
    )
    closed_with_stale_metadata_fragments = "\n".join(
        draw_asset_part(
            part,
            spec,
            PartInstance(
                "S1",
                "ideal_switch",
                "jumper",
                {"state_resistance_ohm": 0.05},
                {"state": "open"},
            ),
        )
    )

    assert 'data-symbol="spst_switch"' in open_fragments
    assert 'data-symbol="spst_switch_closed"' not in open_fragments
    assert 'data-symbol="spst_switch_closed"' in closed_fragments
    assert 'data-symbol="spst_switch_closed"' in closed_by_resistance_fragments
    assert 'data-symbol="spst_switch_closed"' in closed_with_stale_metadata_fragments


def test_svg_pin_anchors_are_declared_in_symbol_assets() -> None:
    catalog = default_catalog()
    led_asset = asset_for_part("led", catalog["led"])
    switch_asset = asset_for_part("ideal_switch", catalog["ideal_switch"])

    assert led_asset is not None
    assert switch_asset is not None
    led_anode = led_asset.anchor("a")
    led_cathode = led_asset.anchor("k")
    switch_pin_1 = switch_asset.anchor("1")
    switch_pin_2 = switch_asset.anchor("2")

    assert led_anode is not None
    assert led_cathode is not None
    assert switch_pin_1 is not None
    assert switch_pin_2 is not None
    assert abs(led_anode.x - 6.90927) <= 1.0e-5
    assert abs(led_anode.y - 49.50004) <= 1.0e-5
    assert abs(led_cathode.x - 6.90927) <= 1.0e-5
    assert abs(led_cathode.y - 0.5) <= 1.0e-5
    assert abs(switch_pin_1.x - 14.78612) <= 1.0e-5
    assert abs(switch_pin_1.y - 49.50186) <= 1.0e-5
    assert abs(switch_pin_2.x - 14.78612) <= 1.0e-5
    assert abs(switch_pin_2.y - 0.5) <= 1.0e-5
    crystal_asset = asset_for_part("crystal", catalog["crystal"])

    assert crystal_asset is not None
    _assert_asset_anchor(crystal_asset, "1", 9.01058, 49.478763)
    _assert_asset_anchor(crystal_asset, "2", 9.01058, 0.5)


def test_crystal_asset_rotation_places_rendered_pins_left_and_right() -> None:
    catalog = default_catalog()
    spec = catalog["crystal"]
    part = PlacedPart(
        ref="XTAL1",
        kind="crystal",
        center=Point(100.0, 100.0),
        size=Size(82.0, 48.0),
    )
    instance = PartInstance("XTAL1", "crystal", "XTAL", {}, {})

    terminals = asset_terminals_for_part(part, spec, instance)

    assert terminals["1"].x < part.center.x < terminals["2"].x
    assert abs(terminals["1"].y - part.center.y) <= GEOMETRY_EPSILON
    assert abs(terminals["2"].y - part.center.y) <= GEOMETRY_EPSILON


def test_default_part_assets_declare_every_catalog_pin() -> None:
    catalog = default_catalog()

    for kind, spec in catalog.items():
        instances = [
            PartInstance(f"{spec.ref_prefix}1", kind, "value", {}, {}),
        ]
        if kind == "ideal_switch":
            instances.append(
                PartInstance(
                    "S1",
                    kind,
                    "closed",
                    {"state_resistance_ohm": 0.05},
                    {"state": "closed"},
                )
            )
        for instance in instances:
            part = PlacedPart(
                ref=instance.ref,
                kind=kind,
                center=Point(100.0, 100.0),
                size=Size(82.0, max(48.0, 16.0 * len(spec.pins))),
            )
            terminals = asset_terminals_for_part(part, spec, instance)

            assert set(terminals) == {pin.name for pin in spec.pins}, (
                kind,
                instance.metadata,
            )


def test_asset_pin_resolver_fails_when_anchor_is_missing() -> None:
    catalog = default_catalog()
    spec = catalog["led"]

    with pytest.raises(ValueError, match="does not declare pin"):
        _asset_pin_position(
            PlacedPart(
                ref="D1",
                kind="led",
                center=Point(100.0, 100.0),
                size=Size(82.0, 48.0),
            ),
            spec,
            PartInstance("D1", "led", "LED", {}, {}),
            "missing",
        )


def test_svg_routes_to_asset_pin_anchors_without_corrective_leads() -> None:
    catalog = default_catalog()

    for seed, element_count in GENERATED_CASES:
        circuit, _ = _planned_generated_case(seed, element_count)
        svg = draw_svg(circuit, catalog)

        _assert_no_corrective_symbol_leads(svg, circuit)

    for index, (name, motif) in enumerate(default_motifs().items()):
        circuit, _ = _planned_motif_case(index, name, motif.element_count + 1)
        svg = draw_svg(circuit, catalog)

        _assert_no_corrective_symbol_leads(svg, circuit)


def test_svg_supplied_layout_keeps_terminal_dots_on_layout_pins() -> None:
    catalog = default_catalog()
    builder = CircuitBuilder("led-indicator", catalog)
    builder.add_part("V1", "voltage_source_dc", "5V", {"voltage_v": 5.0}, {})
    builder.connect("V1", "p", "VCC")
    builder.connect("V1", "n", "0")
    builder.add_part("GND1", "ground", "0", {}, {})
    builder.connect("GND1", "0", "0")
    builder.add_part("R1", "resistor", "1k", {"resistance_ohm": 1000.0}, {})
    builder.add_part("D1", "led", "LED", {}, {})
    builder.connect("R1", "1", "VCC")
    builder.connect("R1", "2", "N1")
    builder.connect("D1", "a", "N1")
    builder.connect("D1", "k", "0")
    circuit = builder.freeze()
    layout = plan_layout(circuit, catalog)
    placed_led = layout.part_by_ref()["D1"]
    led_instance = circuit.part_by_ref()["D1"]
    led_spec = catalog[led_instance.kind]
    layout_anchor = pin_position(placed_led, led_spec, "a")
    asset_anchor = asset_terminals_for_part(
        placed_led,
        led_spec,
        led_instance,
    )["a"]

    svg = draw_svg(circuit, catalog, layout)

    assert _point_key(layout_anchor) != _point_key(asset_anchor)
    assert (
        f'<circle class="junction" cx="{layout_anchor.x:.1f}" '
        f'cy="{layout_anchor.y:.1f}" r="3.6"/>'
    ) in svg


def test_asset_aware_layout_routes_wire_endpoints_to_asset_pins() -> None:
    catalog = default_catalog()

    for index, (name, motif) in enumerate(default_motifs().items()):
        circuit, _ = _planned_motif_case(index, name, motif.element_count + 1)
        layout = _plan_layout(
            circuit,
            catalog,
            route_pin_labels=False,
            pin_position_resolver=_asset_pin_position,
        )

        _assert_wire_endpoints_hit_asset_pins(name, circuit, layout, catalog)


def test_pull_resistor_assets_match_catalog_pin_sides() -> None:
    catalog = default_catalog()

    for kind, expected_rail_side in (
        ("pullup_resistor", PinSide.TOP),
        ("pulldown_resistor", PinSide.BOTTOM),
    ):
        spec = catalog[kind]
        part = PlacedPart(
            ref=f"{spec.ref_prefix}1",
            kind=kind,
            center=Point(100.0, 100.0),
            size=Size(82.0, 48.0),
        )
        terminals = asset_terminals_for_part(
            part,
            spec,
            PartInstance(f"{spec.ref_prefix}1", kind, "10k", {}, {}),
        )

        assert terminals["net"].x == part.bounds.x
        assert terminals["net"].y == part.center.y
        assert terminals["rail"].y == (
            part.bounds.y if expected_rail_side is PinSide.TOP else part.bounds.bottom
        )


def test_svg_component_labels_anchor_to_drawn_asset_bounds() -> None:
    catalog = default_catalog()
    spec = catalog["resistor"]
    part = PlacedPart(
        ref="R1",
        kind="resistor",
        center=Point(100.0, 100.0),
        size=Size(82.0, 48.0),
    )
    instance = PartInstance("R1", "resistor", "1000", {}, {})

    generic_label = component_label_bounds(part, spec, instance.value)
    asset_bounds = asset_render_bounds_for_part(part, spec, instance)
    asset_label = asset_component_label_bounds(part, spec, instance)
    fragments = "\n".join(draw_asset_part(part, spec, instance))

    assert asset_bounds.y > part.bounds.y
    assert asset_label.y > generic_label.y + 10.0
    assert abs(asset_bounds.y - asset_label.bottom - 5.0) <= 1.0e-6
    assert f'y="{asset_label.y + 10.0:.1f}"' in fragments
    assert f'y="{generic_label.y + 10.0:.1f}"' not in fragments


def test_symbol_mask_leaves_asset_pin_boundary_visible() -> None:
    catalog = default_catalog()
    spec = catalog["resistor"]
    part = PlacedPart(
        ref="R1",
        kind="resistor",
        center=Point(100.0, 100.0),
        size=Size(82.0, 48.0),
    )

    fragments = "\n".join(
        draw_asset_part(part, spec, PartInstance("R1", "resistor", "1k", {}, {}))
    )
    mask_bounds = asset_render_bounds_for_part(
        part,
        spec,
        PartInstance("R1", "resistor", "1k", {}, {}),
    )

    assert (
        f'<rect class="symbol-mask" x="{mask_bounds.x + 1.5:.1f}" '
        f'y="{mask_bounds.y + 1.5:.1f}"'
    ) in fragments
    assert f'x="{part.bounds.x - 2.0:.1f}"' not in fragments


def test_transistor_symbol_mask_does_not_hide_base_pin_lead() -> None:
    catalog = default_catalog()
    kind = "bjt_npn"
    spec = catalog[kind]
    part = PlacedPart(
        ref="Q1",
        kind=kind,
        center=Point(100.0, 100.0),
        size=Size(82.0, 48.0),
    )
    instance = PartInstance("Q1", kind, "BJT", {}, {})
    terminals = asset_terminals_for_part(part, spec, instance)
    mask_bounds = asset_render_bounds_for_part(part, spec, instance)
    mask_left = mask_bounds.x + 1.5

    fragments = "\n".join(draw_asset_part(part, spec, instance))

    assert part.bounds.x + 1.5 < terminals["b"].x < mask_left
    assert f'x="{mask_left:.1f}"' in fragments
    assert f'x="{part.bounds.x + 1.5:.1f}"' not in fragments


def test_transistor_asset_anchors_match_visible_terminal_leads() -> None:
    catalog = default_catalog()
    npn_asset = asset_for_part("bjt_npn", catalog["bjt_npn"])

    assert npn_asset is not None
    _assert_asset_anchor(npn_asset, "b", 0.5, 24.992567)
    _assert_asset_anchor(npn_asset, "c", 27.474328, 0.5)
    _assert_asset_anchor(npn_asset, "e", 27.474328, 49.485129)


def test_net_label_is_exported_with_public_layout_type() -> None:
    label = NetLabel(
        net="VCC",
        text="VCC",
        side=PinSide.LEFT,
        anchor=Point(10.0, 20.0),
        position=Point(0.0, 20.0),
    )

    layout = Layout(parts=(), wires=(), size=Size(10.0, 10.0), net_labels=(label,))

    assert layout.net_labels == (label,)


def test_controlled_source_assets_draw_visible_terminal_leads() -> None:
    asset_root = files("rlvr_physics.tasks.physics.circuits.assets")

    for filename in ("controlled_source.svg", "controlled_current_source.svg"):
        text = asset_root.joinpath(filename).read_text(encoding="utf-8")
        assert "M 0,16.7 H 11.8" in text
        assert "M 0,33.3 H 11.8" in text
        assert "M 52.2,16.7 H 64" in text
        assert "M 52.2,33.3 H 64" in text


def test_generic_ic_asset_draws_pins_on_layout_slots() -> None:
    text = (
        files("rlvr_physics.tasks.physics.circuits.assets")
        .joinpath("generic_ic.svg")
        .read_text(encoding="utf-8")
    )

    assert "M 0,23.3 H 15" in text
    assert "M 0,46.7 H 15" in text


@pytest.mark.parametrize(
    "kind",
    DYNAMIC_PACKAGE_KINDS,
)
def test_dynamic_package_symbols_draw_visible_lead_for_every_catalog_pin(
    kind: str,
) -> None:
    catalog = default_catalog()
    spec = catalog[kind]
    part = PlacedPart(
        ref=f"{spec.ref_prefix}1",
        kind=kind,
        center=Point(100.0, 100.0),
        size=Size(82.0, max(48.0, 16.0 * len(spec.pins))),
    )
    instance = PartInstance(f"{spec.ref_prefix}1", kind, "value", {}, {})

    fragments = "\n".join(draw_asset_part(part, spec, instance))

    assert asset_render_bounds_for_part(part, spec, instance) == part.bounds
    assert 'class="symbol-asset"' not in fragments
    for pin in spec.pins:
        anchor = pin_position(part, spec, pin.name)
        lead = _dynamic_package_inner_point(anchor, pin.side)
        assert (
            f'<line class="symbol-pin" x1="{anchor.x:.1f}" '
            f'y1="{anchor.y:.1f}" x2="{lead.x:.1f}" y2="{lead.y:.1f}"/>'
        ) in fragments, (
            kind,
            pin.name,
        )


def test_dynamic_package_symbols_do_not_expose_static_assets() -> None:
    catalog = default_catalog()

    for kind in DYNAMIC_PACKAGE_KINDS:
        assert asset_for_part(kind, catalog[kind]) is None


def test_op_amp_asset_anchors_match_visible_triangle_terminal_leads() -> None:
    catalog = default_catalog()
    asset = asset_for_part("op_amp", catalog["op_amp"])
    text = (
        files("rlvr_physics.tasks.physics.circuits.assets")
        .joinpath("op_amp.svg")
        .read_text(encoding="utf-8")
    )

    assert asset is not None
    assert "M 0,23 H 18 M 0,47 H 18 M 70,35 H 82 M 41,0 V 21 M 41,49 V 70" in text
    _assert_asset_anchor(asset, "noninv", 0.0, 23.0)
    _assert_asset_anchor(asset, "inv", 0.0, 47.0)
    _assert_asset_anchor(asset, "out", 82.0, 35.0)
    _assert_asset_anchor(asset, "vpos", 41.0, 0.0)
    _assert_asset_anchor(asset, "vneg", 41.0, 70.0)


def test_logic_and_op_amp_assets_draw_visible_supply_leads() -> None:
    asset_root = files("rlvr_physics.tasks.physics.circuits.assets")
    expected_leads = {
        "logic.svg": ("M 35,0 V 10", "M 35,60 V 70"),
        "nand_gate.svg": ("M 35,0 V 10", "M 35,60 V 70"),
        "or_gate.svg": ("M 35,0 V 12", "M 35,58 V 70"),
        "xor_gate.svg": ("M 35,0 V 12", "M 35,58 V 70"),
        "not_gate.svg": ("M 35,0 V 23", "M 35,47 V 70"),
        "op_amp.svg": ("M 41,0 V 21", "M 41,49 V 70"),
    }

    for filename, leads in expected_leads.items():
        text = asset_root.joinpath(filename).read_text(encoding="utf-8")
        for lead in leads:
            assert lead in text


def test_multi_pin_asset_terminals_are_declared_in_asset_bounds() -> None:
    catalog = default_catalog()
    multi_pin_kinds = (
        "ammeter",
        "battery",
        "and_gate",
        "or_gate",
        "op_amp",
        "comparator",
        "generic_ic",
        "instrumentation_amplifier",
        "inductor_looped",
        "polarized_capacitor",
        "power_rail",
        "pushbutton_switch",
        "test_point",
        "timer_555",
        "transformer",
        "connector_2",
        "counter_4bit",
        "variable_resistor",
        "vcvs",
        "vccs",
        "voltmeter",
        "relay",
    )

    for kind in multi_pin_kinds:
        spec = catalog[kind]
        part = PlacedPart(
            ref=f"{spec.ref_prefix}1",
            kind=kind,
            center=Point(100.0, 100.0),
            size=Size(82.0, max(48.0, 16.0 * len(spec.pins))),
        )
        asset = asset_for_part(kind, spec)
        instance = PartInstance(f"{spec.ref_prefix}1", kind, "value", {}, {})
        if kind in DYNAMIC_PACKAGE_KINDS:
            assert asset is None
        else:
            assert asset is not None
        terminals = asset_terminals_for_part(part, spec, instance)
        rendered_bounds = asset_render_bounds_for_part(part, spec, instance)

        assert set(terminals) == {pin.name for pin in spec.pins}
        for pin_name, terminal in terminals.items():
            assert (
                rendered_bounds.x - GEOMETRY_EPSILON
                <= terminal.x
                <= (rendered_bounds.right + GEOMETRY_EPSILON)
            ), (kind, pin_name)
            assert (
                rendered_bounds.y - GEOMETRY_EPSILON
                <= terminal.y
                <= (rendered_bounds.bottom + GEOMETRY_EPSILON)
            ), (kind, pin_name)


def test_polarized_capacitor_positive_pin_renders_on_left() -> None:
    catalog = default_catalog()
    spec = catalog["polarized_capacitor"]
    part = PlacedPart(
        ref="C1",
        kind="polarized_capacitor",
        center=Point(100.0, 100.0),
        size=Size(82.0, 48.0),
    )
    instance = PartInstance("C1", "polarized_capacitor", "10u", {}, {})

    terminals = asset_terminals_for_part(part, spec, instance)

    assert terminals["p"].x < terminals["n"].x


def test_test_point_uses_dot_scaled_layout_size() -> None:
    catalog = default_catalog()
    builder = CircuitBuilder("test-point", catalog)
    builder.add_part("GND1", "ground", "0", {}, {})
    builder.add_part("TP1", "test_point", "", {}, {})
    builder.connect("GND1", "0", "0")
    builder.connect("TP1", "net", "0")
    circuit = builder.freeze()

    layout = plan_layout(circuit, catalog)
    test_point = layout.part_by_ref()["TP1"]
    rendered = draw_svg(circuit, catalog)

    assert test_point.size == Size(width=14.0, height=14.0)
    assert 'data-symbol="junction_dot"' in rendered


def test_svg_merges_overlapping_wire_segments_before_drawing() -> None:
    rendered = _rendered_wire_segments(
        (
            WireSegment(Point(0.0, 10.0), Point(20.0, 10.0)),
            WireSegment(Point(10.0, 10.0), Point(30.0, 10.0)),
            WireSegment(Point(40.0, 0.0), Point(40.0, 20.0)),
            WireSegment(Point(40.0, 10.0), Point(40.0, 30.0)),
        )
    )

    assert rendered == (
        WireSegment(Point(0.0, 10.0), Point(30.0, 10.0)),
        WireSegment(Point(40.0, 0.0), Point(40.0, 30.0)),
    )


def test_svg_does_not_merge_adjacent_cross_net_segments() -> None:
    circuit = divider_circuit()
    base_layout = plan_layout(circuit, default_catalog())
    layout = Layout(
        parts=base_layout.parts,
        wires=(
            WirePath("0", (WireSegment(Point(10.0, 10.0), Point(20.0, 10.0)),)),
            WirePath("VCC", (WireSegment(Point(20.0, 10.0), Point(30.0, 10.0)),)),
        ),
        size=Size(200.0, 200.0),
    )

    svg = draw_svg(circuit, default_catalog(), layout)

    assert 'x1="10.0" y1="10.0" x2="20.0" y2="10.0"' in svg
    assert 'x1="20.0" y1="10.0" x2="30.0" y2="10.0"' in svg
    assert 'x1="10.0" y1="10.0" x2="30.0" y2="10.0"' not in svg
    assert '<circle class="junction" cx="20.0" cy="10.0"' not in svg


def test_svg_marks_same_net_interior_crossings_as_junctions() -> None:
    crossing_segments = (
        WireSegment(Point(10.0, 0.0), Point(10.0, 20.0)),
        WireSegment(Point(0.0, 10.0), Point(20.0, 10.0)),
        WireSegment(Point(40.0, 0.0), Point(40.0, 20.0)),
        WireSegment(Point(40.0, 20.0), Point(60.0, 20.0)),
    )
    junctions = _junction_points(crossing_segments)
    base_layout = plan_layout(divider_circuit(), default_catalog())
    layout = Layout(
        parts=base_layout.parts,
        wires=(WirePath("VCC", crossing_segments),),
        size=Size(100.0, 100.0),
    )

    svg = draw_svg(divider_circuit(), default_catalog(), layout)

    assert Point(10.0, 10.0) in junctions
    assert Point(40.0, 20.0) not in junctions
    assert '<circle class="junction" cx="10.0" cy="10.0" r="4.4"/>' in svg
    assert 'r="1.6"/>' not in svg


def test_svg_asset_stroke_normalization_doubles_group_inherited_lines() -> None:
    catalog = default_catalog()
    asset = asset_for_part("lamp", catalog["lamp"])

    assert asset is not None

    fragments = "\n".join(symbol_fragments_for_scale(asset, 2.0))
    custom_fragments = "\n".join(
        symbol_fragments_for_scale(asset, 2.0, stroke_width=3.0)
    )

    assert 'stroke-width="1.25"' in fragments
    assert 'stroke-width="1.4"' not in fragments
    assert 'stroke-width="1.5"' in custom_fragments
    assert 'stroke-width="1.25"' not in custom_fragments


def test_svg_can_draw_pin_labels_when_requested() -> None:
    circuit = divider_circuit()
    catalog = default_catalog()
    svg = draw_svg(circuit, catalog, show_pin_labels=True)

    assert ">1</text>" in svg


def test_svg_uses_supplied_layout_when_pin_labels_are_requested() -> None:
    circuit = divider_circuit()
    catalog = default_catalog()
    layout = plan_layout(circuit, catalog)
    shifted_layout = Layout(
        parts=tuple(part.translate(1000.0, 1000.0) for part in layout.parts),
        wires=tuple(
            WirePath(
                net=wire.net,
                segments=tuple(
                    WireSegment(
                        start=segment.start.translate(1000.0, 1000.0),
                        end=segment.end.translate(1000.0, 1000.0),
                    )
                    for segment in wire.segments
                ),
            )
            for wire in layout.wires
        ),
        size=Size(
            width=layout.size.width + 1000.0,
            height=layout.size.height + 1000.0,
        ),
    )

    svg = draw_svg(circuit, catalog, shifted_layout, show_pin_labels=True)

    first_part = shifted_layout.parts[0]
    first_instance = circuit.part_by_ref()[first_part.ref]
    first_spec = catalog[first_instance.kind]
    first_asset_bounds = asset_render_bounds_for_part(
        first_part,
        first_spec,
        first_instance,
    )
    assert f'width="{shifted_layout.size.width:.0f}"' in svg
    assert f'x="{first_asset_bounds.x + 1.5:.1f}"' in svg


def test_symbol_drawer_covers_default_part_catalog() -> None:
    catalog = default_catalog()

    for kind, spec in catalog.items():
        fragments = draw_asset_part(
            PlacedPart(
                ref=f"{spec.ref_prefix}1",
                kind=kind,
                center=Point(100.0, 100.0),
                size=Size(110.0, 90.0),
            ),
            spec,
            PartInstance(f"{spec.ref_prefix}1", kind, "value", {}, {}),
        )

        assert any('class="circuit-symbol"' in fragment for fragment in fragments)


def test_vendored_svg_assets_cover_common_symbols() -> None:
    catalog = default_catalog()

    for kind, spec in catalog.items():
        if kind in DYNAMIC_PACKAGE_KINDS:
            assert asset_for_part(kind, spec) is None
            continue
        assert asset_for_part(kind, spec) is not None


@pytest.mark.parametrize(
    ("kind", "asset_key"),
    (
        ("battery", "battery"),
        ("inductor_looped", "inductor_looped"),
        ("polarized_capacitor", "polarized_capacitor"),
        ("power_rail", "power_rail"),
        ("pushbutton_switch", "pushbutton_switch"),
        ("test_point", "junction_dot"),
        ("variable_resistor", "variable_resistor"),
    ),
)
def test_asset_backed_symbols_are_reachable_for_motifs(
    kind: str, asset_key: str
) -> None:
    catalog = default_catalog()

    asset = asset_for_part(kind, catalog[kind])

    assert asset is not None
    assert asset.key == asset_key


def test_exported_symbol_assets_are_used_directly() -> None:
    asset_root = files("rlvr_physics.tasks.physics.circuits.assets")

    for asset_file in asset_root.iterdir():
        if not asset_file.name.endswith(".svg"):
            continue
        text = asset_file.read_text(encoding="utf-8")
        assert "<svg" in text
        assert "<g" in text


def test_relay_asset_draws_visible_terminal_leads() -> None:
    relay = files("rlvr_physics.tasks.physics.circuits.assets").joinpath("relay.svg")
    text = relay.read_text(encoding="utf-8")

    assert "M 0,24 H 18" in text
    assert "M 0,46 H 12" in text
    assert "M 44,24 H 70" in text
    assert "M 44,46 H 70" in text
    assert "V 70" in text


def test_svg_rasterizes_to_png() -> None:
    circuit = divider_circuit()
    catalog = default_catalog()
    png = to_png(draw_svg(circuit, catalog))

    validate_png_image_data(png, PNG_MIME_TYPE)


def test_png_drawer_plans_layout_before_rasterizing() -> None:
    png = draw_png(divider_circuit(), default_catalog())

    validate_png_image_data(png, PNG_MIME_TYPE)


def test_generated_circuit_png_artifacts_are_written() -> None:
    catalog = default_catalog()
    _prepare_generated_image_dir(GENERATED_CIRCUIT_IMAGE_DIR)
    image_paths: list[Path] = []
    expected_paths = {
        GENERATED_CIRCUIT_IMAGE_DIR
        / f"circuit_seed_{seed:03d}_motifs_{element_count:02d}.png"
        for seed, element_count in GENERATED_CASES
    }

    for seed, element_count in GENERATED_CASES:
        generated = generate_circuit(
            GeneratorConfig(
                seed=seed,
                motif_count_min=element_count,
                motif_count_max=element_count,
                motif_weights=default_motif_weights(),
            ),
            catalog,
        )
        png = draw_png(generated.circuit, catalog)
        image_path = (
            GENERATED_CIRCUIT_IMAGE_DIR
            / f"circuit_seed_{seed:03d}_motifs_{element_count:02d}.png"
        )

        validate_png_image_data(png, PNG_MIME_TYPE)
        image_path.write_bytes(png)
        assert image_path.stat().st_size > 0
        image_paths.append(image_path)

    assert len(image_paths) == len(GENERATED_CASES)
    assert set(image_paths) == expected_paths
    assert set(GENERATED_CIRCUIT_IMAGE_DIR.glob("*.png")) == expected_paths


def test_default_motif_png_artifacts_are_written() -> None:
    catalog = default_catalog()
    _prepare_generated_image_dir(GENERATED_MOTIF_IMAGE_DIR)
    motifs = tuple(default_motifs().items())
    image_paths: list[Path] = []
    expected_paths = {
        GENERATED_MOTIF_IMAGE_DIR / f"motif_{index:02d}_{name}.png"
        for index, (name, _) in enumerate(motifs, start=1)
    }

    for index, (name, motif) in enumerate(motifs, start=1):
        circuit = _motif_rendering_circuit(name)
        png = draw_png(circuit, catalog)
        image_path = GENERATED_MOTIF_IMAGE_DIR / f"motif_{index:02d}_{name}.png"

        validate_png_image_data(png, PNG_MIME_TYPE)
        image_path.write_bytes(png)
        assert image_path.stat().st_size > 0
        image_paths.append(image_path)

    assert len(image_paths) == len(motifs)
    assert set(image_paths) == expected_paths
    assert set(GENERATED_MOTIF_IMAGE_DIR.glob("*.png")) == expected_paths


def _prepare_generated_image_dir(path: Path) -> None:
    """Create one generated image directory and remove stale PNG artifacts.

    Parameters
    ----------
    path:
        Directory that receives deterministic PNG renders for one artifact group.
    """

    path.mkdir(parents=True, exist_ok=True)
    for image_path in path.glob("*.png"):
        image_path.unlink()


def _segment_crosses_bounds(segment: WireSegment, bounds: Bounds) -> bool:
    """Return whether an axis-aligned segment crosses bounds."""

    assert _segment_is_orthogonal(segment), segment
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


def _segment_is_orthogonal(segment: WireSegment) -> bool:
    """Return whether a wire segment lies on one axis."""

    return segment.start.x == segment.end.x or segment.start.y == segment.end.y


def _assert_wire_endpoints_hit_pins(
    case_id: object,
    circuit: Circuit,
    layout: Layout,
    catalog: Mapping[str, PartSpec],
) -> None:
    """Assert routed wire endpoints land on declared circuit pins."""

    part_by_ref = circuit.part_by_ref()
    placed_by_ref = layout.part_by_ref()
    wire_by_net = {wire.net: wire for wire in layout.wires}

    for connection in circuit.connections:
        if len(circuit.connections_for_net(connection.net)) < 2:
            continue
        part = part_by_ref[connection.ref]
        spec = catalog[part.kind]
        anchor = pin_position(
            placed_by_ref[connection.ref],
            spec,
            connection.pin,
        )
        endpoints = {
            _point_key(point)
            for segment in wire_by_net[connection.net].segments
            for point in (segment.start, segment.end)
        }

        assert _point_key(anchor) in endpoints, (
            case_id,
            connection.ref,
            connection.pin,
            connection.net,
        )


def _assert_wire_endpoints_hit_asset_pins(
    case_id: object,
    circuit: Circuit,
    layout: Layout,
    catalog: Mapping[str, PartSpec],
) -> None:
    """Assert routed wire endpoints land on rendered SVG asset terminals."""

    part_by_ref = circuit.part_by_ref()
    placed_by_ref = layout.part_by_ref()
    wire_by_net = {wire.net: wire for wire in layout.wires}

    for connection in circuit.connections:
        if len(circuit.connections_for_net(connection.net)) < 2:
            continue
        part = part_by_ref[connection.ref]
        spec = catalog[part.kind]
        terminal = asset_terminals_for_part(
            placed_by_ref[connection.ref],
            spec,
            part,
        )[connection.pin]
        endpoints = {
            _point_key(point)
            for segment in wire_by_net[connection.net].segments
            for point in (segment.start, segment.end)
        }

        assert _point_key(terminal) in endpoints, (
            case_id,
            connection.ref,
            connection.pin,
            connection.net,
        )


def _assert_wire_paths_connect_declared_pins(
    case_id: object,
    circuit: Circuit,
    layout: Layout,
    catalog: Mapping[str, PartSpec],
) -> None:
    """Assert every routed net is geometrically connected."""

    part_by_ref = circuit.part_by_ref()
    placed_by_ref = layout.part_by_ref()
    wire_by_net = {wire.net: wire for wire in layout.wires}

    for net in circuit.nets:
        connections = circuit.connections_for_net(net)
        if len(connections) < 2:
            continue
        wire = wire_by_net[net]
        graph = _wire_connectivity_graph(wire.segments)
        pin_points = [
            _point_key(
                pin_position(
                    placed_by_ref[connection.ref],
                    catalog[part_by_ref[connection.ref].kind],
                    connection.pin,
                )
            )
            for connection in connections
        ]
        label_points = [
            _point_key(label.anchor) for label in layout.net_labels if label.net == net
        ]

        if label_points:
            for pin_point in pin_points:
                assert pin_point in graph, (case_id, net, pin_point)
            for label_point in label_points:
                assert label_point in graph, (case_id, net, label_point)
            label_roots = {
                _find_root(graph, label_point) for label_point in label_points
            }
            for pin_point in pin_points:
                assert _find_root(graph, pin_point) in label_roots, (
                    case_id,
                    net,
                    pin_points,
                    label_points,
                )
            continue

        for pin_point in pin_points:
            assert pin_point in graph, (case_id, net, pin_point)
        root = _find_root(graph, pin_points[0])
        for pin_point in pin_points[1:]:
            assert _find_root(graph, pin_point) == root, (case_id, net, pin_points)


def _assert_net_labels_are_clear(
    case_id: object,
    circuit: Circuit,
    layout: Layout,
    catalog: Mapping[str, PartSpec],
) -> None:
    """Assert local net labels do not overlap rendered symbols, labels, or wires."""

    part_by_ref = circuit.part_by_ref()
    rendered_boxes = [
        bounds
        for part in layout.parts
        for bounds in _default_rendered_boxes(
            part,
            catalog[part_by_ref[part.ref].kind],
            part_by_ref[part.ref].value,
        )
    ]
    net_label_bounds = [_net_label_bounds(label) for label in layout.net_labels]

    for index, current in enumerate(net_label_bounds):
        for other in net_label_bounds[index + 1 :]:
            assert not current.expanded(WIRE_CLEARANCE, WIRE_CLEARANCE).overlaps(
                other.expanded(WIRE_CLEARANCE, WIRE_CLEARANCE)
            ), (case_id, current, other)
        for box in rendered_boxes:
            assert not current.expanded(WIRE_CLEARANCE, WIRE_CLEARANCE).overlaps(
                box.expanded(WIRE_CLEARANCE, WIRE_CLEARANCE)
            ), (case_id, current, box)
        for wire in layout.wires:
            for segment in wire.segments:
                assert not _segment_crosses_bounds(
                    segment,
                    current.expanded(WIRE_CLEARANCE, WIRE_CLEARANCE),
                ), (case_id, wire.net, current, segment)


def _assert_cross_net_labeled_stubs_are_clear(case_id: object, layout: Layout) -> None:
    """Assert routed signal wires avoid local labeled-net stubs."""

    labeled_nets = {label.net for label in layout.net_labels}
    labeled_stub_bounds = [
        (wire.net, _segment_bounds(segment).expanded(WIRE_CLEARANCE, WIRE_CLEARANCE))
        for wire in layout.wires
        if wire.net in labeled_nets
        for segment in wire.segments
    ]
    for wire in layout.wires:
        if wire.net in labeled_nets:
            continue
        for segment in wire.segments:
            for label_net, bounds in labeled_stub_bounds:
                assert not _segment_crosses_bounds(segment, bounds), (
                    case_id,
                    wire.net,
                    label_net,
                    bounds,
                    segment,
                )


def _assert_no_collinear_cross_net_wire_overlaps(
    case_id: object, layout: Layout
) -> None:
    """Assert different nets do not share a visible collinear wire span."""

    for first_index, first_wire in enumerate(layout.wires):
        for first_segment in first_wire.segments:
            for second_wire in layout.wires[first_index + 1 :]:
                for second_segment in second_wire.segments:
                    assert not _segments_overlap_collinearly(
                        first_segment,
                        second_segment,
                    ), (
                        case_id,
                        first_wire.net,
                        second_wire.net,
                        first_segment,
                        second_segment,
                    )


def _assert_pin_approach_lanes_clear(
    case_id: object,
    circuit: Circuit,
    layout: Layout,
    catalog: Mapping[str, PartSpec],
) -> None:
    """Assert only component pin stubs use the reserved pin-approach lanes."""

    part_by_ref = circuit.part_by_ref()
    placed_by_ref = layout.part_by_ref()
    approach_bounds = {
        ref: _pin_approach_bounds(placed, catalog[part_by_ref[ref].kind])
        for ref, placed in placed_by_ref.items()
    }
    pin_stub_keys = _pin_stub_segment_keys(circuit, layout, catalog)

    for wire in layout.wires:
        for segment in wire.segments:
            if _segment_key(segment) in pin_stub_keys:
                continue
            for ref, bounds_list in approach_bounds.items():
                for bounds in bounds_list:
                    assert not _segment_crosses_bounds(segment, bounds), (
                        case_id,
                        wire.net,
                        ref,
                        bounds,
                    )


def _pin_stub_segment_keys(
    circuit: Circuit,
    layout: Layout,
    catalog: Mapping[str, PartSpec],
) -> set[tuple[tuple[float, float], tuple[float, float]]]:
    """Return the expected immediate pin-stub segments."""

    part_by_ref = circuit.part_by_ref()
    placed_by_ref = layout.part_by_ref()
    keys: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    for connection in circuit.connections:
        part = part_by_ref[connection.ref]
        spec = catalog[part.kind]
        placed = placed_by_ref[connection.ref]
        anchor = pin_position(placed, spec, connection.pin)
        escape = _pin_escape_point(placed, spec, connection.pin, anchor)
        keys.add(_segment_key(WireSegment(anchor, escape)))
    return keys


def _wire_connectivity_graph(
    segments: tuple[WireSegment, ...],
) -> dict[tuple[float, float], tuple[float, float]]:
    """Return disjoint-set parent links for one wire path."""

    points_by_segment: list[set[tuple[float, float]]] = [
        {_point_key(segment.start), _point_key(segment.end)} for segment in segments
    ]
    for first_index, first in enumerate(segments):
        for second_index in range(first_index + 1, len(segments)):
            second = segments[second_index]
            intersections = _segment_intersections(first, second)
            if not intersections:
                continue
            points_by_segment[first_index].update(intersections)
            points_by_segment[second_index].update(intersections)

    graph: dict[tuple[float, float], tuple[float, float]] = {}
    for segment, segment_points in zip(segments, points_by_segment):
        ordered = sorted(
            segment_points,
            key=lambda point: (
                point[0] if segment.start.x != segment.end.x else point[1]
            ),
        )
        for point in ordered:
            graph.setdefault(point, point)
        for start, end in zip(ordered, ordered[1:]):
            _union_roots(graph, start, end)
    return graph


def _segment_intersections(
    first: WireSegment, second: WireSegment
) -> set[tuple[float, float]]:
    """Return intersection points between two axis-aligned segments."""

    first_vertical = first.start.x == first.end.x
    second_vertical = second.start.x == second.end.x
    if first_vertical and second_vertical:
        if first.start.x != second.start.x:
            return set()
        top = max(min(first.start.y, first.end.y), min(second.start.y, second.end.y))
        bottom = min(max(first.start.y, first.end.y), max(second.start.y, second.end.y))
        if top > bottom:
            return set()
        return {
            _point_key(Point(first.start.x, top)),
            _point_key(Point(first.start.x, bottom)),
        }
    if not first_vertical and not second_vertical:
        if first.start.y != second.start.y:
            return set()
        left = max(min(first.start.x, first.end.x), min(second.start.x, second.end.x))
        right = min(max(first.start.x, first.end.x), max(second.start.x, second.end.x))
        if left > right:
            return set()
        return {
            _point_key(Point(left, first.start.y)),
            _point_key(Point(right, first.start.y)),
        }

    vertical = first if first_vertical else second
    horizontal = second if first_vertical else first
    x = vertical.start.x
    y = horizontal.start.y
    if min(horizontal.start.x, horizontal.end.x) <= x <= max(
        horizontal.start.x, horizontal.end.x
    ) and min(vertical.start.y, vertical.end.y) <= y <= max(
        vertical.start.y, vertical.end.y
    ):
        return {_point_key(Point(x, y))}
    return set()


def _find_root(
    graph: dict[tuple[float, float], tuple[float, float]],
    point: tuple[float, float],
) -> tuple[float, float]:
    """Return the disjoint-set root for one point."""

    while graph[point] != point:
        graph[point] = graph[graph[point]]
        point = graph[point]
    return point


def _union_roots(
    graph: dict[tuple[float, float], tuple[float, float]],
    first: tuple[float, float],
    second: tuple[float, float],
) -> None:
    """Join two disjoint-set roots."""

    first_root = _find_root(graph, first)
    second_root = _find_root(graph, second)
    if first_root != second_root:
        graph[second_root] = first_root


def _layout_fingerprint(layout: Layout) -> str:
    """Return a stable digest for layout geometry regression checks."""

    data = (
        (round(layout.size.width, 6), round(layout.size.height, 6)),
        tuple(
            (
                part.ref,
                part.kind,
                round(part.center.x, 6),
                round(part.center.y, 6),
                round(part.size.width, 6),
                round(part.size.height, 6),
            )
            for part in layout.parts
        ),
        tuple(
            (
                wire.net,
                tuple(
                    (
                        round(segment.start.x, 6),
                        round(segment.start.y, 6),
                        round(segment.end.x, 6),
                        round(segment.end.y, 6),
                    )
                    for segment in wire.segments
                ),
            )
            for wire in layout.wires
        ),
        tuple(
            (
                label.net,
                label.text,
                label.side.value,
                round(label.anchor.x, 6),
                round(label.anchor.y, 6),
                round(label.position.x, 6),
                round(label.position.y, 6),
            )
            for label in layout.net_labels
        ),
    )
    return sha256(repr(data).encode()).hexdigest()


@cache
def _planned_generated_case(seed: int, element_count: int) -> tuple[Circuit, Layout]:
    """Return the generated circuit and layout for one regression case."""

    catalog = default_catalog()
    generated = generate_circuit(
        GeneratorConfig(
            seed=seed,
            motif_count_min=element_count,
            motif_count_max=element_count,
            motif_weights=default_motif_weights(),
        ),
        catalog,
    )
    return generated.circuit, plan_layout(generated.circuit, catalog)


@cache
def _planned_motif_case(
    index: int, motif_name: str, element_count: int
) -> tuple[Circuit, Layout]:
    """Return the generated circuit and layout for one motif case."""

    catalog = default_catalog()
    circuit = _motif_rendering_circuit(motif_name)
    return circuit, plan_layout(circuit, catalog)


def _motif_rendering_circuit(motif_name: str) -> Circuit:
    """Build one default motif circuit for image artifact inspection."""

    catalog = default_catalog()
    motif = default_motifs()[motif_name]
    ctx = _MotifRenderingContext(CircuitBuilder(f"motif-{motif_name}", catalog))
    motif.build(ctx, {})
    nets = ctx.builder._nets
    if "VCC" in nets:
        source = ctx.add_part("V", "voltage_source_dc", "5V", {"voltage_v": 5.0}, {})
        ctx.builder.connect(source, "p", "VCC")
        ctx.builder.connect(source, "n", "0")
    if "0" in nets:
        ground = ctx.add_part("GND", "ground", "0", {}, {})
        ctx.builder.connect(ground, "0", "0")
    return ctx.builder.freeze()


class _MotifRenderingContext:
    """Motif context used by image artifact tests."""

    def __init__(self, builder: CircuitBuilder) -> None:
        """Initialize a motif rendering context."""

        self.builder = builder
        self.rng = Random(123)
        self.counters: dict[str, int] = {}
        self.node_counter = 0
        self.motif_counters: dict[str, int] = {}
        self.negative_supply_nets: set[str] = set()

    def add_part(
        self,
        prefix: str,
        kind: str,
        value: str,
        parameters: Mapping[str, object],
        metadata: Mapping[str, object],
    ) -> str:
        """Add a numbered part to the rendered motif circuit."""

        number = self.counters.get(prefix, 0) + 1
        self.counters[prefix] = number
        ref = f"{prefix}{number}"
        self.builder.add_part(ref, kind, value, parameters, metadata)
        return ref

    def node(self) -> str:
        """Return a fresh node name."""

        self.node_counter += 1
        return f"N{self.node_counter}"

    def motif_instance_id(self, motif_name: str) -> str:
        """Return a fresh deterministic motif instance id."""

        number = self.motif_counters.get(motif_name, 0) + 1
        self.motif_counters[motif_name] = number
        return f"{motif_name}#{number}"

    def add_negative_supply(
        self, net: str, motif_name: str, instance_id: str
    ) -> tuple[str, ...]:
        """Add one negative supply source per generated net."""

        if net in self.negative_supply_nets:
            return ()
        self.negative_supply_nets.add(net)
        negative = self.add_part(
            "VEE",
            "voltage_source_dc",
            "-5V",
            {"voltage_v": -5.0},
            {
                "role": "negative_supply",
                "motif": motif_name,
                "motif_instance": instance_id,
            },
        )
        self.builder.connect(negative, "p", net)
        self.builder.connect(negative, "n", "0")
        return (negative,)


def _default_rendered_boxes(
    part: PlacedPart,
    spec: PartSpec,
    value: str,
) -> tuple[Bounds, Bounds]:
    """Return default-rendered symbol and label boxes."""

    return (part.bounds, component_label_bounds(part, spec, value))


def _part_lead_prefix(svg: str, ref: str) -> str:
    """Return the circuit-symbol fragment before embedded asset artwork."""

    group_start = svg.index(f'<g id="{ref}" class="circuit-symbol">')
    next_part_start = svg.find('\n<g id="', group_start + 1)
    group_end = next_part_start if next_part_start >= 0 else svg.index("\n</svg>")
    asset_start = svg.find('<g class="symbol-asset"', group_start, group_end)
    if asset_start < 0:
        asset_start = group_end
    return svg[group_start:asset_start]


def _assert_asset_anchor(asset: SymbolAsset, pin_name: str, x: float, y: float) -> None:
    """Assert one loaded SVG asset pin anchor has the expected source point."""

    anchor = asset.anchor(pin_name)

    assert anchor is not None, pin_name
    assert abs(anchor.x - x) <= 1.0e-5
    assert abs(anchor.y - y) <= 1.0e-5


def _dynamic_package_inner_point(anchor: Point, side: PinSide) -> Point:
    """Return the expected inner endpoint for one dynamic package pin."""

    if side is PinSide.LEFT:
        return anchor.translate(DYNAMIC_PACKAGE_PIN_LEAD, 0.0)
    if side is PinSide.RIGHT:
        return anchor.translate(-DYNAMIC_PACKAGE_PIN_LEAD, 0.0)
    if side is PinSide.TOP:
        return anchor.translate(0.0, DYNAMIC_PACKAGE_PIN_LEAD)
    return anchor.translate(0.0, -DYNAMIC_PACKAGE_PIN_LEAD)


def _assert_no_corrective_symbol_leads(svg: str, circuit: Circuit) -> None:
    """Assert default SVG rendering does not add layout-to-asset lead lines."""

    for part in circuit.parts:
        assert '<line class="symbol"' not in _part_lead_prefix(svg, part.ref), part.ref


def _point_key(point: Point) -> tuple[float, float]:
    """Return a stable comparable coordinate key."""

    return (round(point.x, 6), round(point.y, 6))


def _segment_key(
    segment: WireSegment,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return a direction-independent wire-segment key."""

    start = _point_key(segment.start)
    end = _point_key(segment.end)
    return (start, end) if start <= end else (end, start)


def _segment_bounds(segment: WireSegment) -> Bounds:
    """Return axis-aligned bounds covering one wire segment."""

    return Bounds(
        x=min(segment.start.x, segment.end.x),
        y=min(segment.start.y, segment.end.y),
        width=abs(segment.end.x - segment.start.x),
        height=abs(segment.end.y - segment.start.y),
    )


def _segments_overlap_collinearly(
    first: WireSegment,
    second: WireSegment,
) -> bool:
    """Return whether two axis-aligned segments share a positive-length span."""

    if first.start.x == first.end.x and second.start.x == second.end.x:
        if first.start.x != second.start.x:
            return False
        return (
            _interval_overlap_length(
                min(first.start.y, first.end.y),
                max(first.start.y, first.end.y),
                min(second.start.y, second.end.y),
                max(second.start.y, second.end.y),
            )
            > GEOMETRY_EPSILON
        )
    if first.start.y == first.end.y and second.start.y == second.end.y:
        if first.start.y != second.start.y:
            return False
        return (
            _interval_overlap_length(
                min(first.start.x, first.end.x),
                max(first.start.x, first.end.x),
                min(second.start.x, second.end.x),
                max(second.start.x, second.end.x),
            )
            > GEOMETRY_EPSILON
        )
    return False


def _interval_overlap_length(
    first_start: float,
    first_end: float,
    second_start: float,
    second_end: float,
) -> float:
    """Return positive overlap length for two one-dimensional intervals."""

    return min(first_end, second_end) - max(first_start, second_start)
