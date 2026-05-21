"""Tests for procedural circuit generation."""

from random import Random
import re
from typing import Any, Mapping

import pytest

from rlvr_physics.tasks.physics.circuits import (
    CheckIssue,
    CheckReport,
    CircuitBuilder,
    CircuitSupplyPort,
    GeneratedCircuit,
    GeneratorConfig,
    InstantiatedMotif,
    MotifPortRole,
    check_circuit,
    default_catalog,
    default_motif_weights,
    default_motifs,
    generate_circuit,
)

REJECTED_GENERATION_WARNING_CODES = {
    "empty_net",
    "excessive_drive",
    "insufficient_drive",
    "pin_conflict",
    "single_pin_net",
}
SAFE_GENERATED_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|0")


def test_generate_circuit_is_deterministic_and_hits_motif_count() -> None:
    catalog = default_catalog()
    weights = default_motif_weights()
    config = GeneratorConfig(
        seed=123,
        motif_count_min=3,
        motif_count_max=5,
        motif_weights=weights,
    )

    generated = generate_circuit(config, catalog)
    repeated = generate_circuit(config, catalog)

    assert generated.circuit.content_hash() == repeated.circuit.content_hash()
    assert generated.motif_names == repeated.motif_names
    assert generated.motif_instances == repeated.motif_instances
    assert (
        config.motif_count_min <= len(generated.motif_names) <= config.motif_count_max
    )
    assert len(generated.motif_instances) == len(generated.motif_names)
    assert set(generated.motif_names) <= set(default_motifs())
    assert "load_resistor" not in generated.motif_names


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
    assert plain_data["name"] == "generated_circuit"
    assert isinstance(metadata, dict)
    assert metadata["target_motif_count"] == len(generated.motif_names)
    assert "seed" not in metadata


def test_generated_circuit_uses_safe_canonical_identifiers() -> None:
    generated = generate_circuit(
        GeneratorConfig(
            seed=123,
            motif_count_min=3,
            motif_count_max=5,
            motif_weights=default_motif_weights(),
        ),
        default_catalog(),
    )

    assert SAFE_GENERATED_IDENTIFIER_PATTERN.fullmatch(generated.circuit.name)
    assert all(
        SAFE_GENERATED_IDENTIFIER_PATTERN.fullmatch(part.ref)
        for part in generated.circuit.parts
    )
    assert all(
        SAFE_GENERATED_IDENTIFIER_PATTERN.fullmatch(net)
        for net in generated.circuit.nets
    )
    assert all(
        SAFE_GENERATED_IDENTIFIER_PATTERN.fullmatch(instance.instance_id)
        for instance in generated.motif_instances
    )


def test_generated_circuit_declares_supply_port_without_applied_voltage() -> None:
    generated = generate_circuit(
        GeneratorConfig(
            seed=123,
            motif_count_min=3,
            motif_count_max=3,
            motif_weights=default_motif_weights(),
        ),
        default_catalog(),
    )
    applied_supplies = [
        part
        for part in generated.circuit.parts
        if part.kind == "voltage_source_dc"
        and part.metadata.get("role") in {"main_supply", "negative_supply"}
    ]
    metadata = generated.circuit.metadata

    assert generated.motif_names[0] == "supply_port"
    assert generated.supply_ports == (CircuitSupplyPort("VCC", "VCC", "0"),)
    assert metadata["supply_ports"] == (
        {"name": "VCC", "positive_net": "VCC", "reference_net": "0"},
    )
    assert applied_supplies == []


def test_generate_circuit_passes_structural_checks() -> None:
    config = GeneratorConfig(
        seed=456,
        motif_count_min=3,
        motif_count_max=5,
        motif_weights=default_motif_weights(),
    )
    generated = generate_circuit(config, default_catalog())

    report = check_circuit(generated.circuit, default_catalog())

    assert report.errors == ()
    assert _unexpected_generation_warnings(generated, report) == ()


def test_generate_circuit_rejects_part_count_api_shape() -> None:
    old_shape: dict[str, Any] = {
        "seed": 123,
        "element_count": 10,
        "motif_weights": default_motif_weights(),
    }

    with pytest.raises(TypeError):
        GeneratorConfig(**old_shape)


@pytest.mark.parametrize("motif_count", (0, 2))
def test_generate_circuit_rejects_too_few_motifs(motif_count: int) -> None:
    with pytest.raises(ValueError, match="motif_count_min"):
        generate_circuit(
            GeneratorConfig(
                seed=123,
                motif_count_min=motif_count,
                motif_count_max=motif_count,
                motif_weights=default_motif_weights(),
            ),
            default_catalog(),
        )


def test_generate_circuit_rejects_inverted_motif_range() -> None:
    with pytest.raises(ValueError, match="motif_count_max"):
        generate_circuit(
            GeneratorConfig(
                seed=123,
                motif_count_min=5,
                motif_count_max=4,
                motif_weights=default_motif_weights(),
            ),
            default_catalog(),
        )


def test_generate_circuit_accepts_motif_counts_above_five() -> None:
    generated = generate_circuit(
        GeneratorConfig(
            seed=0,
            motif_count_min=8,
            motif_count_max=8,
            motif_weights=default_motif_weights(),
        ),
        default_catalog(),
    )

    assert len(generated.motif_names) == 8


def test_generate_circuit_rejects_incomplete_weight_role_sets() -> None:
    with pytest.raises(ValueError, match="signal source motif"):
        generate_circuit(
            GeneratorConfig(
                seed=123,
                motif_count_min=3,
                motif_count_max=3,
                motif_weights={"inverting_op_amp_amplifier": 1.0},
            ),
            default_catalog(),
        )


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
        seen_motifs.update(name for name in generated.motif_names if name in weights)
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


def test_dual_rail_motifs_declare_one_negative_supply_port() -> None:
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
    report = check_circuit(generated.circuit, default_catalog())
    negative_ports = [
        port for port in generated.supply_ports if port.positive_net == "VEE"
    ]
    applied_sources = [
        part
        for part in generated.circuit.parts
        if part.kind == "voltage_source_dc"
        and part.metadata.get("role") in {"main_supply", "negative_supply"}
    ]

    assert set(generated.motif_names) == {
        "supply_port",
        "voltage_divider_with_voltmeter",
        "inverting_op_amp_amplifier",
        "non_inverting_op_amp_amplifier",
    }
    assert negative_ports == [CircuitSupplyPort("VEE", "VEE", "0")]
    assert applied_sources == []
    assert report.errors == ()
    assert _unexpected_generation_warnings(generated, report) == ()


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
        report = check_circuit(generated.circuit, catalog)

        assert 3 <= len(generated.motif_names) <= 5
        assert report.errors == (), (seed, generated.motif_names, report.errors)
        assert _unexpected_generation_warnings(generated, report) == (), (
            seed,
            generated.motif_names,
            report.warnings,
        )
        assert all(
            generated.circuit.connections_for_net(port.positive_net)
            for port in generated.supply_ports
        )
        assert _has_cross_motif_nonrail_net(generated)
        assert _has_ordered_signal_path(generated)


def _unexpected_generation_warnings(
    generated: GeneratedCircuit, report: CheckReport
) -> tuple[CheckIssue, ...]:
    """Return structural warnings that are not explained by external supplies."""

    supply_nets = {port.positive_net for port in generated.supply_ports}
    return tuple(
        issue
        for issue in report.warnings
        if issue.code in REJECTED_GENERATION_WARNING_CODES
        and not (issue.code == "insufficient_drive" and set(issue.nets) <= supply_nets)
    )


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
        return f"{motif_name}_{number}"

    def add_negative_supply(
        self, net: str, motif_name: str, instance_id: str
    ) -> tuple[str, ...]:
        """Declare one negative supply port per generated net."""

        if net in self.negative_supply_nets:
            return ()
        self.negative_supply_nets.add(net)
        self.builder.add_net(net)
        self.builder.add_net("0")
        return ()


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
