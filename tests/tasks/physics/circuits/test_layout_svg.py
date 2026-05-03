"""Tests for schematic layout and SVG drawing."""

from importlib.resources import files
from pathlib import Path

from rlvr_physics.core.rendering import PNG_MIME_TYPE, validate_png_image_data
from rlvr_physics.tasks.physics.circuits import (
    Bounds,
    GeneratorConfig,
    PlacedPart,
    Point,
    Size,
    default_catalog,
    default_motif_weights,
    draw_png,
    draw_svg,
    generate_circuit,
    plan_layout,
    to_png,
    WireSegment,
)
from rlvr_physics.tasks.physics.circuits.layout import placement_bounds, visual_bounds
from rlvr_physics.tasks.physics.circuits.symbol_assets import (
    asset_for_part,
    draw_asset_part,
)
from tests.tasks.physics.circuits.test_model import divider_circuit


GENERATED_IMAGE_DIR = Path(__file__).with_name("images")
GENERATED_CASES = (
    (0, 6),
    (1, 8),
    (2, 10),
    (3, 12),
    (4, 16),
    (5, 20),
    (9, 15),
)


def test_layout_places_parts_without_overlap() -> None:
    layout = plan_layout(divider_circuit(), default_catalog())
    bounds = [part.bounds for part in layout.parts]

    for index, current in enumerate(bounds):
        for other in bounds[index + 1 :]:
            assert not current.overlaps(other)


def test_layout_places_rendered_bounds_without_overlap() -> None:
    catalog = default_catalog()

    for seed, element_count in GENERATED_CASES:
        generated = generate_circuit(
            GeneratorConfig(
                seed=seed,
                element_count=element_count,
                motif_weights=default_motif_weights(),
            ),
            catalog,
        )
        layout = plan_layout(generated.circuit, catalog)
        part_by_ref = generated.circuit.part_by_ref()
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


def test_layout_routes_wires_around_unrelated_rendered_bounds() -> None:
    catalog = default_catalog()

    for seed, element_count in GENERATED_CASES:
        generated = generate_circuit(
            GeneratorConfig(
                seed=seed,
                element_count=element_count,
                motif_weights=default_motif_weights(),
            ),
            catalog,
        )
        circuit = generated.circuit
        layout = plan_layout(circuit, catalog)
        part_by_ref = circuit.part_by_ref()
        placed_by_ref = layout.part_by_ref()
        bounds_by_ref = {
            ref: visual_bounds(
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
                for ref, bounds in bounds_by_ref.items():
                    if ref in connected_refs:
                        continue
                    assert not _segment_crosses_bounds(segment, bounds), (
                        seed,
                        element_count,
                        wire.net,
                        ref,
                    )


def test_svg_draws_circuit_symbols_and_labels() -> None:
    circuit = divider_circuit()
    catalog = default_catalog()
    svg = draw_svg(circuit, catalog)

    assert svg.startswith("<svg ")
    assert 'id="R1"' in svg
    assert 'class="circuit-symbol"' in svg
    assert 'class="symbol-asset"' in svg
    assert 'data-symbol="resistor"' in svg
    assert 'data-symbol="variable_resistor"' not in svg
    assert "vector-effect" not in svg
    assert "stroke-width:0.7608695652173912" in svg
    assert "<clipPath" not in svg
    assert "<use " not in svg
    assert ">1</text>" not in svg
    assert "R2 2k" in svg
    assert 'class="background"' in svg
    assert 'fill="#ffffff"' in svg
    assert 'class="wire"' in svg
    assert svg.endswith("</svg>\n")


def test_svg_can_draw_pin_labels_when_requested() -> None:
    circuit = divider_circuit()
    catalog = default_catalog()
    svg = draw_svg(circuit, catalog, show_pin_labels=True)

    assert ">1</text>" in svg


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
            "value",
        )

        assert any('class="circuit-symbol"' in fragment for fragment in fragments)


def test_vendored_svg_assets_cover_common_symbols() -> None:
    catalog = default_catalog()

    for kind, spec in catalog.items():
        assert asset_for_part(kind, spec) is not None


def test_exported_symbol_assets_are_used_directly() -> None:
    asset_root = files("rlvr_physics.tasks.physics.circuits.assets")

    for asset_file in asset_root.iterdir():
        if not asset_file.name.endswith(".svg"):
            continue
        text = asset_file.read_text(encoding="utf-8")
        assert "<svg" in text
        assert "<g" in text


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
    GENERATED_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    for seed, element_count in GENERATED_CASES:
        generated = generate_circuit(
            GeneratorConfig(
                seed=seed,
                element_count=element_count,
                motif_weights=default_motif_weights(),
            ),
            catalog,
        )
        png = draw_png(generated.circuit, catalog)
        image_path = (
            GENERATED_IMAGE_DIR
            / f"generated_seed_{seed:03d}_count_{element_count:02d}.png"
        )

        validate_png_image_data(png, PNG_MIME_TYPE)
        image_path.write_bytes(png)
        assert image_path.stat().st_size > 0


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
