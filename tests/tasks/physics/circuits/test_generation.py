"""Tests for procedural circuit generation."""

from pathlib import Path
import shutil
import subprocess

import pytest

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

    assert report.errors == ()


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

        assert report.errors == (), (name, report.errors)
        assert export_spice(
            generated.circuit,
            catalog,
            operating_point_analysis(),
        ).text.endswith(".op\n.end\n")


def test_default_generated_spice_refs_are_unique() -> None:
    catalog = default_catalog()
    weights = default_motif_weights()

    for seed in range(50):
        for element_count in range(2, 25):
            generated = generate_circuit(
                GeneratorConfig(
                    seed=seed,
                    element_count=element_count,
                    motif_weights=weights,
                ),
                catalog,
            )
            netlist = export_spice(
                generated.circuit,
                catalog,
                operating_point_analysis(),
            )
            refs = _spice_element_refs(netlist.text)

            assert len(refs) == len(set(refs)), (seed, element_count, refs)


def test_controlled_source_motifs_run_in_ngspice(tmp_path: Path) -> None:
    ngspice = shutil.which("ngspice")
    if ngspice is None:
        pytest.skip("ngspice executable is not available")
    catalog = default_catalog()

    for motif_name in (
        "vccs_voltage_to_current_driver",
        "vcvs_ideal_voltage_gain_block",
    ):
        motif = default_motifs()[motif_name]
        generated = generate_circuit(
            GeneratorConfig(
                seed=123,
                element_count=motif.element_count + 1,
                motif_weights={motif_name: 1.0},
            ),
            catalog,
        )
        netlist = export_spice(
            generated.circuit,
            catalog,
            operating_point_analysis(),
        )
        netlist_path = tmp_path / f"{motif_name}.cir"
        netlist_path.write_text(netlist.text, encoding="utf-8")
        completed = subprocess.run(
            (ngspice, "-b", str(netlist_path)),
            check=False,
            capture_output=True,
            text=True,
            timeout=20.0,
        )

        assert completed.returncode == 0, completed.stdout + completed.stderr


def test_default_generation_uses_catalog_motifs_before_fallbacks() -> None:
    catalog = default_catalog()
    generated = generate_circuit(
        GeneratorConfig(
            seed=123,
            element_count=12,
            motif_weights=default_motif_weights(),
        ),
        catalog,
    )

    assert generated.motif_names
    assert any(name in default_motifs() for name in generated.motif_names)


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
            assert report.errors == (), (seed, element_count, report.errors)


def _spice_element_refs(netlist: str) -> tuple[str, ...]:
    """Return top-level SPICE element references from a netlist."""

    refs: list[str] = []
    in_subcircuit = False
    for line in netlist.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("*", ".")):
            if stripped.lower().startswith(".subckt"):
                in_subcircuit = True
            elif stripped.lower().startswith(".ends"):
                in_subcircuit = False
            continue
        if not in_subcircuit:
            refs.append(stripped.split()[0])
    return tuple(refs)
