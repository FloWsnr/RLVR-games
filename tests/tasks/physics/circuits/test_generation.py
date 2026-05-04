"""Tests for procedural circuit generation."""

from collections import Counter

from rlvr_physics.tasks.physics.circuits import (
    AnalysisSupport,
    GeneratorConfig,
    check_circuit,
    default_catalog,
    default_motif_weights,
    default_motifs,
    export_spice,
    generate_circuit,
    operating_point_analysis,
)


def test_generate_circuit_is_deterministic_and_hits_count() -> None:
    config = GeneratorConfig(
        seed=123,
        element_count=10,
        motif_weights=default_motif_weights(),
    )

    generated = generate_circuit(config, default_catalog())
    repeated = generate_circuit(config, default_catalog())

    assert generated.circuit.content_hash() == repeated.circuit.content_hash()
    assert generated.motif_names == repeated.motif_names
    assert sum(part.kind != "ground" for part in generated.circuit.parts) == 10


def test_generated_circuit_plain_data_does_not_expose_seed() -> None:
    config = GeneratorConfig(
        seed=123,
        element_count=10,
        motif_weights=default_motif_weights(),
    )

    generated = generate_circuit(config, default_catalog())
    plain_data = generated.circuit.to_plain_data()
    metadata = plain_data["metadata"]

    assert generated.seed == 123
    assert plain_data["name"] == "generated-circuit"
    assert isinstance(metadata, dict)
    assert "seed" not in metadata


def test_generate_circuit_passes_erc_without_errors() -> None:
    config = GeneratorConfig(
        seed=456,
        element_count=12,
        motif_weights=default_motif_weights(),
    )
    generated = generate_circuit(config, default_catalog())

    report = check_circuit(
        generated.circuit,
        default_catalog(),
        AnalysisSupport.SPICE_EXPORT,
    )

    assert report.issues == ()


def test_generate_circuit_can_emit_every_default_motif() -> None:
    catalog = default_catalog()

    for name, motif in default_motifs().items():
        generated = generate_circuit(
            GeneratorConfig(
                seed=123,
                element_count=motif.element_count + 1,
                motif_weights={name: 1.0},
            ),
            catalog,
        )

        assert generated.motif_names == (name,)


def test_single_default_motif_generations_are_spice_exportable() -> None:
    catalog = default_catalog()

    for name, motif in default_motifs().items():
        generated = generate_circuit(
            GeneratorConfig(
                seed=123,
                element_count=motif.element_count + 1,
                motif_weights={name: 1.0},
            ),
            catalog,
        )
        report = check_circuit(
            generated.circuit,
            catalog,
            AnalysisSupport.SPICE_EXPORT,
        )

        assert report.issues == (), (name, report.issues)
        assert export_spice(
            generated.circuit,
            catalog,
            operating_point_analysis(),
        ).text.endswith(".op\n.end\n")


def test_default_generation_uses_every_part_kind_multiple_times() -> None:
    catalog = default_catalog()
    weights = default_motif_weights()
    part_counts: Counter[str] = Counter()

    for seed in range(10):
        generated = generate_circuit(
            GeneratorConfig(
                seed=seed,
                element_count=80,
                motif_weights=weights,
            ),
            catalog,
        )
        part_counts.update(part.kind for part in generated.circuit.parts)

    sparse_kinds = {
        kind: part_counts[kind] for kind in catalog if part_counts[kind] < 2
    }
    assert sparse_kinds == {}


def test_generate_circuit_seed_sweep_passes_erc_without_errors() -> None:
    catalog = default_catalog()
    weights = default_motif_weights()

    for seed in range(200):
        for element_count in range(2, 25):
            generated = generate_circuit(
                GeneratorConfig(
                    seed=seed,
                    element_count=element_count,
                    motif_weights=weights,
                ),
                catalog,
            )
            report = check_circuit(
                generated.circuit,
                catalog,
                AnalysisSupport.SPICE_EXPORT,
            )

            assert sum(part.kind != "ground" for part in generated.circuit.parts) == (
                element_count
            )
            assert report.issues == (), (seed, element_count, report.issues)
