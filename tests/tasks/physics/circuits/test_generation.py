"""Tests for procedural circuit generation."""

from rlvr_physics.tasks.physics.circuits import (
    AnalysisSupport,
    GeneratorConfig,
    check_circuit,
    default_catalog,
    default_motif_weights,
    generate_circuit,
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

    assert report.is_valid


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
            assert report.is_valid, (seed, element_count, report.errors)
