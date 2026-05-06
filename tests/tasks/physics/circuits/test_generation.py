"""Tests for procedural circuit generation."""

from pathlib import Path
from random import Random
import shutil
import subprocess
from typing import Any, Mapping

import pytest

from rlvr_physics.tasks.physics.circuits import (
    AnalysisSupport,
    Circuit,
    CircuitBuilder,
    GeneratedCircuit,
    GeneratorConfig,
    InstantiatedMotif,
    MotifPortRole,
    check_circuit,
    default_catalog,
    default_motif_weights,
    default_motifs,
    export_spice,
    generate_circuit,
    operating_point_analysis,
)

REJECTED_GENERATION_WARNING_CODES = {
    "excessive_drive",
    "insufficient_drive",
    "pin_conflict",
    "single_pin_net",
}


def test_generate_circuit_is_deterministic_and_hits_motif_count() -> None:
    config = GeneratorConfig(
        seed=123,
        motif_count_min=3,
        motif_count_max=5,
        motif_weights=default_motif_weights(),
    )

    generated = generate_circuit(config, default_catalog())
    repeated = generate_circuit(config, default_catalog())

    assert generated.circuit.content_hash() == repeated.circuit.content_hash()
    assert generated.motif_names == repeated.motif_names
    assert generated.motif_instances == repeated.motif_instances
    assert 3 <= len(generated.motif_names) <= 5
    assert len(generated.motif_instances) == len(generated.motif_names)


def test_generated_circuit_plain_data_does_not_expose_seed() -> None:
    config = GeneratorConfig(
        seed=123,
        motif_count_min=3,
        motif_count_max=5,
        motif_weights=default_motif_weights(),
    )

    generated = generate_circuit(config, default_catalog())
    plain_data = generated.circuit.to_plain_data()
    metadata = plain_data["metadata"]

    assert generated.seed == 123
    assert plain_data["name"] == "generated-circuit"
    assert isinstance(metadata, dict)
    assert metadata["target_motif_count"] == len(generated.motif_names)
    assert "seed" not in metadata


def test_generate_circuit_passes_structural_checks() -> None:
    config = GeneratorConfig(
        seed=456,
        motif_count_min=3,
        motif_count_max=5,
        motif_weights=default_motif_weights(),
    )
    generated = generate_circuit(config, default_catalog())

    report = check_circuit(
        generated.circuit,
        default_catalog(),
        AnalysisSupport.SPICE_EXPORT,
    )

    assert report.errors == ()
    assert not any(
        issue.code in REJECTED_GENERATION_WARNING_CODES for issue in report.warnings
    )


def test_generate_circuit_rejects_part_count_api_shape() -> None:
    old_shape: dict[str, Any] = {
        "seed": 123,
        "element_count": 10,
        "motif_weights": default_motif_weights(),
    }

    with pytest.raises(TypeError):
        GeneratorConfig(**old_shape)


def test_generate_circuit_rejects_too_few_motifs() -> None:
    with pytest.raises(ValueError, match="motif_count_min"):
        generate_circuit(
            GeneratorConfig(
                seed=123,
                motif_count_min=2,
                motif_count_max=2,
                motif_weights=default_motif_weights(),
            ),
            default_catalog(),
        )


def test_generate_circuit_rejects_too_many_motifs() -> None:
    with pytest.raises(ValueError, match="motif_count_max"):
        generate_circuit(
            GeneratorConfig(
                seed=123,
                motif_count_min=3,
                motif_count_max=6,
                motif_weights=default_motif_weights(),
            ),
            default_catalog(),
        )


def test_generate_circuit_rejects_incomplete_weight_role_sets() -> None:
    with pytest.raises(ValueError, match="supply source motif"):
        generate_circuit(
            GeneratorConfig(
                seed=123,
                motif_count_min=3,
                motif_count_max=3,
                motif_weights={"inverting_op_amp_amplifier": 1.0},
            ),
            default_catalog(),
        )


def test_default_generated_spice_refs_are_unique() -> None:
    catalog = default_catalog()
    weights = default_motif_weights()

    for seed in range(50):
        generated = generate_circuit(
            GeneratorConfig(
                seed=seed,
                motif_count_min=3,
                motif_count_max=5,
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

        assert len(refs) == len(set(refs)), (seed, generated.motif_names, refs)


def test_controlled_source_motifs_run_in_ngspice(tmp_path: Path) -> None:
    ngspice = shutil.which("ngspice")
    if ngspice is None:
        pytest.skip("ngspice executable is not available")
    catalog = default_catalog()

    for motif_name in (
        "vccs_voltage_to_current_driver",
        "vcvs_ideal_voltage_gain_block",
    ):
        circuit = _built_motif_circuit(motif_name)
        netlist = export_spice(
            circuit,
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


def test_default_generation_uses_only_catalog_motifs() -> None:
    generated = generate_circuit(
        GeneratorConfig(
            seed=123,
            motif_count_min=3,
            motif_count_max=5,
            motif_weights=default_motif_weights(),
        ),
        default_catalog(),
    )

    assert generated.motif_names
    assert len(generated.motif_names) == len(generated.motif_instances)
    assert set(generated.motif_names) <= set(default_motifs())
    assert "load_resistor" not in generated.motif_names


def test_default_weighted_motifs_are_reachable() -> None:
    catalog = default_catalog()
    weights = default_motif_weights()
    weighted_motifs = set(weights)
    seen_motifs: set[str] = set()

    for seed in range(1000):
        generated = generate_circuit(
            GeneratorConfig(
                seed=seed,
                motif_count_min=3,
                motif_count_max=5,
                motif_weights=weights,
            ),
            catalog,
        )
        seen_motifs.update(generated.motif_names)
        if seen_motifs == weighted_motifs:
            break

    assert seen_motifs == weighted_motifs


def test_generated_parts_are_owned_by_motif_instances() -> None:
    generated = generate_circuit(
        GeneratorConfig(
            seed=15,
            motif_count_min=5,
            motif_count_max=5,
            motif_weights=default_motif_weights(),
        ),
        default_catalog(),
    )
    owners: dict[str, str] = {}

    for instance in generated.motif_instances:
        for ref in instance.part_refs:
            assert ref not in owners, (ref, generated.motif_names)
            owners[ref] = instance.instance_id

    assert set(owners) == {part.ref for part in generated.circuit.parts}
    for part in generated.circuit.parts:
        assert part.metadata["motif_instance"] == owners[part.ref]


def test_generated_circuit_has_cross_motif_signal_net() -> None:
    generated = generate_circuit(
        GeneratorConfig(
            seed=123,
            motif_count_min=3,
            motif_count_max=5,
            motif_weights=default_motif_weights(),
        ),
        default_catalog(),
    )

    assert _has_cross_motif_nonrail_net(generated)


def test_generated_path_motifs_consume_previous_signal() -> None:
    generated = generate_circuit(
        GeneratorConfig(
            seed=9,
            motif_count_min=5,
            motif_count_max=5,
            motif_weights=default_motif_weights(),
        ),
        default_catalog(),
    )
    instances = generated.motif_instances
    previous_path_instance = instances[1]
    consumed_count = 0

    for current in instances[2:]:
        current_nets = _nonrail_port_nets_for_roles(
            current,
            {MotifPortRole.SINK},
        )
        if not current_nets:
            continue
        previous_nets = _nonrail_port_nets_for_roles(
            previous_path_instance,
            {MotifPortRole.SOURCE, MotifPortRole.PROBE},
        )

        assert previous_nets & current_nets, generated.motif_names
        previous_path_instance = current
        consumed_count += 1

    assert consumed_count >= 1, generated.motif_names


def test_dual_rail_motifs_share_one_negative_supply_source() -> None:
    weights = {name: 0.0 for name in default_motif_weights()} | {
        "battery_powered_led_indicator": 1.0,
        "voltage_divider_with_voltmeter": 1.0,
        "inverting_op_amp_amplifier": 1.0,
        "non_inverting_op_amp_amplifier": 1.0,
    }

    generated = generate_circuit(
        GeneratorConfig(
            seed=4,
            motif_count_min=4,
            motif_count_max=4,
            motif_weights=weights,
        ),
        default_catalog(),
    )
    report = check_circuit(
        generated.circuit,
        default_catalog(),
        AnalysisSupport.SPICE_EXPORT,
    )
    negative_sources = [
        part
        for part in generated.circuit.parts
        if part.kind == "voltage_source_dc"
        and part.metadata.get("role") == "negative_supply"
    ]

    assert set(generated.motif_names) == {
        "battery_powered_led_indicator",
        "voltage_divider_with_voltmeter",
        "inverting_op_amp_amplifier",
        "non_inverting_op_amp_amplifier",
    }
    assert len(negative_sources) == 1
    assert report.errors == ()
    assert not any(
        issue.code in REJECTED_GENERATION_WARNING_CODES for issue in report.warnings
    )


def test_generate_circuit_seed_sweep_passes_structural_checks() -> None:
    catalog = default_catalog()
    weights = default_motif_weights()

    for seed in range(200):
        generated = generate_circuit(
            GeneratorConfig(
                seed=seed,
                motif_count_min=3,
                motif_count_max=5,
                motif_weights=weights,
            ),
            catalog,
        )
        report = check_circuit(
            generated.circuit,
            catalog,
            AnalysisSupport.SPICE_EXPORT,
        )

        assert 3 <= len(generated.motif_names) <= 5
        assert report.errors == (), (seed, generated.motif_names, report.errors)
        assert not any(
            issue.code in REJECTED_GENERATION_WARNING_CODES for issue in report.warnings
        ), (seed, generated.motif_names, report.warnings)
        assert _has_cross_motif_nonrail_net(generated)
        assert _has_ordered_signal_path(generated)


class _MotifTestContext:
    """Minimal motif context for standalone motif checks."""

    def __init__(self) -> None:
        """Initialize an empty motif test context."""

        self.builder = CircuitBuilder("motif", default_catalog())
        self.rng = Random(123)
        self.counters: dict[str, int] = {}
        self.non_ground_count = 0
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
        """Add a numbered part to the test circuit."""

        number = self.counters.get(prefix, 0) + 1
        self.counters[prefix] = number
        ref = f"{prefix}{number}"
        self.builder.add_part(ref, kind, value, parameters, metadata)
        if kind != "ground":
            self.non_ground_count += 1
        return ref

    def node(self) -> str:
        """Return a fresh generated node name."""

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


def _built_motif_circuit(motif_name: str) -> Circuit:
    """Return a circuit built from one default motif."""

    motif = default_motifs()[motif_name]
    ctx = _MotifTestContext()
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


def _has_cross_motif_nonrail_net(generated: GeneratedCircuit) -> bool:
    """Return whether a generated circuit has a non-rail cross-motif net."""

    circuit = generated.circuit
    instance_by_part = {
        part_ref: instance.instance_id
        for instance in generated.motif_instances
        for part_ref in instance.part_refs
    }
    for net in circuit.nets:
        if net in {"0", "VCC", "VEE"}:
            continue
        instance_ids = {
            instance_by_part[connection.ref]
            for connection in circuit.connections_for_net(net)
            if connection.ref in instance_by_part
        }
        if len(instance_ids) > 1:
            return True
    return False


def _nonrail_port_nets_for_roles(
    instance: InstantiatedMotif, roles: set[MotifPortRole]
) -> set[str]:
    """Return non-rail nets for selected port roles on a motif instance."""

    motif = default_motifs()[instance.motif_name]
    return {
        instance.port_nets[port.name]
        for port in motif.ports
        if port.role in roles
        and instance.port_nets[port.name] not in {"0", "VCC", "VEE"}
    }


def _has_ordered_signal_path(generated: GeneratedCircuit) -> bool:
    """Return whether generated path motifs consume and expose signal in order."""

    instances = generated.motif_instances
    previous_path_instance = instances[1]
    consumed_count = 0

    for current in instances[2:]:
        current_nets = _nonrail_port_nets_for_roles(
            current,
            {MotifPortRole.SINK},
        )
        if not current_nets:
            continue
        previous_nets = _nonrail_port_nets_for_roles(
            previous_path_instance,
            {MotifPortRole.SOURCE, MotifPortRole.PROBE},
        )
        if not previous_nets & current_nets:
            return False
        previous_path_instance = current
        consumed_count += 1
    final_nets = _nonrail_port_nets_for_roles(
        previous_path_instance,
        {MotifPortRole.SOURCE, MotifPortRole.PROBE},
    )
    return consumed_count > 0 and bool(final_nets)


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
