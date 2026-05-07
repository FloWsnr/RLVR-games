"""Procedural circuit generation from composable motifs."""

from dataclasses import dataclass
from math import isfinite
from random import Random
import re
from typing import Mapping

from rlvr_physics.tasks.physics.circuits.erc import check_circuit
from rlvr_physics.tasks.physics.circuits.motifs import (
    CircuitMotif,
    InstantiatedMotif,
    MotifPort,
    MotifPortRole,
    MotifSignalKind,
    default_motifs,
)
from rlvr_physics.tasks.physics.circuits.model import (
    AnalysisSupport,
    Circuit,
    CircuitBuilder,
    PartSpec,
    is_ground_net,
)
from rlvr_physics.tasks.physics.circuits.spice_export import operating_point_analysis
from rlvr_physics.tasks.physics.circuits.spice_sim import SpiceSimulationSpec

_MAX_GENERATION_ATTEMPTS = 500
_REJECTED_WARNING_CODES = {
    "excessive_drive",
    "insufficient_drive",
    "pin_conflict",
    "single_pin_net",
}
_GENERATED_CIRCUIT_NAME = "generated_circuit"


class CircuitGenerationError(ValueError):
    """Raised when motif composition cannot produce a valid circuit."""


@dataclass(frozen=True)
class GeneratorConfig:
    """Configuration for deterministic motif-only circuit generation.

    Parameters
    ----------
    seed:
        Deterministic generator seed.
    supply_voltage_v:
        Main generated VCC supply voltage in volts.
    motif_count_min:
        Minimum number of motif instances to compose.
    motif_count_max:
        Maximum number of motif instances to compose.
    motif_weights:
        Relative motif weights by motif name.
    """

    seed: int
    supply_voltage_v: float
    motif_count_min: int
    motif_count_max: int
    motif_weights: Mapping[str, float]


@dataclass(frozen=True)
class GeneratedCircuit:
    """Generated circuit plus provenance metadata."""

    circuit: Circuit
    motif_names: tuple[str, ...]
    motif_instances: tuple[InstantiatedMotif, ...]
    simulation_spec: SpiceSimulationSpec
    seed: int


@dataclass(frozen=True)
class _LiveSignal:
    """One generated net available for binding downstream motif inputs."""

    net: str
    signal: MotifSignalKind


def generate_circuit(
    config: GeneratorConfig, catalog: Mapping[str, PartSpec]
) -> GeneratedCircuit:
    """Generate a deterministic motif-composed circuit.

    Parameters
    ----------
    config:
        Generation configuration.
    catalog:
        Component catalog.

    Returns
    -------
    GeneratedCircuit
        Generated canonical circuit and motif provenance.
    """

    _validate_config(config)
    motif_catalog = default_motifs()
    _validate_weighted_motif_roles(config, motif_catalog)
    rng = Random(config.seed)
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        motif_count = rng.randint(config.motif_count_min, config.motif_count_max)
        generated = _try_generate_candidate(
            config, catalog, motif_catalog, rng, motif_count
        )
        if generated is not None:
            return generated
    raise CircuitGenerationError(
        "could not compose a connected motif-only circuit "
        f"after {_MAX_GENERATION_ATTEMPTS} attempts"
    )


class _GenerationContext:
    """Mutable generator context."""

    def __init__(
        self, builder: CircuitBuilder, rng: Random, supply_voltage_v: float
    ) -> None:
        """Initialize context state."""

        self.builder = builder
        self.rng = rng
        self.supply_voltage_v = supply_voltage_v
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
        """Add a new numbered part and return its reference."""

        prefix = _safe_identifier(prefix, "P")
        number = self.counters.get(prefix, 0) + 1
        self.counters[prefix] = number
        ref = f"{prefix}{number}"
        self.builder.add_part(ref, kind, value, parameters, metadata)
        if kind != "ground":
            self.non_ground_count += 1
        return ref

    def node(self) -> str:
        """Return a fresh node name."""

        self.node_counter += 1
        return f"N{self.node_counter}"

    def motif_instance_id(self, motif_name: str) -> str:
        """Return a fresh deterministic motif instance id."""

        number = self.motif_counters.get(motif_name, 0) + 1
        self.motif_counters[motif_name] = number
        return f"{_safe_identifier(motif_name, 'motif')}_{number}"

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


def _validate_config(config: GeneratorConfig) -> None:
    """Validate generator configuration."""

    if config.motif_count_min < 3:
        raise ValueError("motif_count_min must be at least 3")
    if config.motif_count_max > 5:
        raise ValueError("motif_count_max must be at most 5")
    if config.motif_count_max < config.motif_count_min:
        raise ValueError("motif_count_max must be greater than or equal to min")
    if not isfinite(float(config.supply_voltage_v)) or config.supply_voltage_v <= 0.0:
        raise ValueError("supply_voltage_v must be positive and finite")


def _safe_identifier(value: str, fallback: str) -> str:
    """Return a generated identifier using letters, digits, and underscores."""

    identifier = re.sub(r"\W+", "_", value).strip("_")
    if not identifier:
        identifier = fallback
    if identifier[0].isdigit():
        identifier = f"{fallback}_{identifier}"
    return identifier


def _validate_weighted_motif_roles(
    config: GeneratorConfig, motif_catalog: Mapping[str, CircuitMotif]
) -> None:
    """Validate that enabled motifs can cover required composition roles."""

    unknown_names = set(config.motif_weights) - set(motif_catalog)
    if unknown_names:
        raise ValueError(f"unknown motif weights: {tuple(sorted(unknown_names))}")
    if not any(weight > 0.0 for weight in config.motif_weights.values()):
        raise ValueError("motif_weights must enable at least one motif")
    if not _signal_source_motifs(motif_catalog, config.motif_weights):
        raise ValueError("motif_weights must enable at least one signal source motif")
    if not any(
        _required_sink_ports(motif) and _non_power_source_ports(motif)
        for motif in motif_catalog.values()
        if config.motif_weights.get(motif.name, 0.0) > 0.0
    ):
        raise ValueError(
            "motif_weights must enable at least one signal path motif "
            "with a sink and source/probe endpoint"
        )


def _try_generate_candidate(
    config: GeneratorConfig,
    catalog: Mapping[str, PartSpec],
    motif_catalog: Mapping[str, CircuitMotif],
    rng: Random,
    motif_count: int,
) -> GeneratedCircuit | None:
    """Try to compose one candidate circuit."""

    ctx = _GenerationContext(
        CircuitBuilder(_GENERATED_CIRCUIT_NAME, catalog), rng, config.supply_voltage_v
    )
    used_names: set[str] = set()
    instances: list[InstantiatedMotif] = []
    live_signals: list[_LiveSignal] = []

    supply = motif_catalog["dc_supply_source"]
    supply_instance = supply.build(ctx, {"source_vcc": "VCC"})
    instances.append(supply_instance)
    used_names.add(supply.name)
    live_power = _LiveSignal("VCC", MotifSignalKind.POWER)
    live_signals.append(live_power)

    source = _choose_weighted(
        rng,
        _signal_source_motifs(motif_catalog, config.motif_weights),
        config.motif_weights,
        used_names,
    )
    if source is None:
        return None
    source_output = _choose_port(rng, _non_power_source_ports(source))
    if source_output is None:
        return None
    source_bindings = _base_bindings(source)
    if not _bind_required_sinks(source, live_signals, source_bindings, rng):
        return None
    source_net = "SIG1"
    source_bindings[source_output.name] = source_net
    source_instance = source.build(ctx, source_bindings)
    instances.append(source_instance)
    used_names.add(source.name)
    live_signals.append(_LiveSignal(source_net, source_output.signal))
    path_signal = _LiveSignal(source_net, source_output.signal)

    while len(instances) < motif_count:
        final_step = len(instances) == motif_count - 1
        path_candidates = _compatible_path_motifs(
            motif_catalog,
            config.motif_weights,
            path_signal,
            require_output=True,
        )
        candidates = path_candidates
        if not final_step:
            candidates = _unique_motifs(
                (
                    *path_candidates,
                    *_accessory_motifs(motif_catalog, config.motif_weights),
                )
            )
        motif = _choose_weighted(rng, candidates, config.motif_weights, used_names)
        if motif is None:
            return None
        bindings = _base_bindings(motif)
        required_signal = path_signal if _required_sink_ports(motif) else None
        if not _bind_required_sinks(
            motif, live_signals, bindings, rng, required_signal=required_signal
        ):
            return None
        output = None
        if not final_step and _required_sink_ports(motif):
            output = _choose_port(rng, _non_power_source_ports(motif))
            if output is None:
                return None
        if output is not None:
            output_net = f"SIG{len(instances)}"
            bindings[output.name] = output_net
            path_signal = _LiveSignal(output_net, output.signal)
        instance = motif.build(ctx, bindings)
        instances.append(instance)
        used_names.add(motif.name)
        if output is not None:
            live_signals.append(path_signal)

    circuit = ctx.builder.freeze()
    if not _candidate_is_valid(circuit, catalog, motif_catalog, instances):
        return None
    motif_names = tuple(instance.motif_name for instance in instances)
    circuit = _with_generation_metadata(circuit, motif_names, instances, motif_count)
    simulation_spec = _default_simulation_spec()
    return GeneratedCircuit(
        circuit=circuit,
        motif_names=motif_names,
        motif_instances=tuple(instances),
        simulation_spec=simulation_spec,
        seed=config.seed,
    )


def _with_generation_metadata(
    circuit: Circuit,
    motif_names: tuple[str, ...],
    instances: list[InstantiatedMotif],
    motif_count: int,
) -> Circuit:
    """Return a circuit with procedural generation metadata attached."""

    return Circuit(
        name=circuit.name,
        parts=circuit.parts,
        nets=circuit.nets,
        connections=circuit.connections,
        metadata={
            "source": "procedural",
            "target_motif_count": motif_count,
            "motifs": motif_names,
            "motif_instances": tuple(
                {
                    "motif": instance.motif_name,
                    "instance_id": instance.instance_id,
                    "ports": dict(instance.port_nets),
                    "parts": instance.part_refs,
                }
                for instance in instances
            ),
        },
    )


def _default_simulation_spec() -> SpiceSimulationSpec:
    """Return the default operating-point simulation spec for generation."""

    return SpiceSimulationSpec(analysis=operating_point_analysis())


def _choose_weighted(
    rng: Random,
    candidates: tuple[CircuitMotif, ...],
    weights: Mapping[str, float],
    used_names: set[str],
) -> CircuitMotif | None:
    """Choose one candidate, preferring motifs not already used."""

    preferred = tuple(motif for motif in candidates if motif.name not in used_names)
    pool = preferred if preferred else candidates
    viable = tuple((motif, weights.get(motif.name, 0.0)) for motif in pool)
    viable = tuple((motif, weight) for motif, weight in viable if weight > 0.0)
    if not viable:
        return None
    total = sum(weight for _, weight in viable)
    pick = rng.random() * total
    running = 0.0
    for motif, weight in viable:
        running += weight
        if pick <= running:
            return motif
    return viable[-1][0]


def _choose_port(rng: Random, ports: tuple[MotifPort, ...]) -> MotifPort | None:
    """Choose one port deterministically."""

    if not ports:
        return None
    return ports[rng.randrange(len(ports))]


def _signal_source_motifs(
    motif_catalog: Mapping[str, CircuitMotif], weights: Mapping[str, float]
) -> tuple[CircuitMotif, ...]:
    """Return weighted motifs that can produce a non-power signal."""

    return tuple(
        motif
        for motif in motif_catalog.values()
        if weights.get(motif.name, 0.0) > 0.0
        and _non_power_source_ports(motif)
        and _all_required_sinks_are_supply_only(motif)
    )


def _compatible_path_motifs(
    motif_catalog: Mapping[str, CircuitMotif],
    weights: Mapping[str, float],
    path_signal: _LiveSignal,
    *,
    require_output: bool,
) -> tuple[CircuitMotif, ...]:
    """Return motifs that can consume the current path signal."""

    return tuple(
        motif
        for motif in motif_catalog.values()
        if weights.get(motif.name, 0.0) > 0.0
        and any(
            _signals_are_compatible(path_signal.signal, port.signal)
            for port in _required_sink_ports(motif)
        )
        and (not require_output or _non_power_source_ports(motif))
    )


def _accessory_motifs(
    motif_catalog: Mapping[str, CircuitMotif],
    weights: Mapping[str, float],
) -> tuple[CircuitMotif, ...]:
    """Return self-contained rail accessories that can fill extra motif slots."""

    return tuple(
        motif
        for motif in motif_catalog.values()
        if weights.get(motif.name, 0.0) > 0.0
        and not _required_sink_ports(motif)
        and not _non_power_source_ports(motif)
    )


def _unique_motifs(motifs: tuple[CircuitMotif, ...]) -> tuple[CircuitMotif, ...]:
    """Return motifs in first-seen order without duplicates."""

    seen: set[str] = set()
    result: list[CircuitMotif] = []
    for motif in motifs:
        if motif.name in seen:
            continue
        seen.add(motif.name)
        result.append(motif)
    return tuple(result)


def _base_bindings(motif: CircuitMotif) -> dict[str, str]:
    """Return default supply and ground bindings for a motif."""

    bindings: dict[str, str] = {}
    for port in motif.ports:
        if port.role is MotifPortRole.GROUND:
            bindings[port.name] = "0"
        elif port.role is MotifPortRole.SUPPLY:
            bindings[port.name] = "VEE" if port.net == "VEE" else "VCC"
    return bindings


def _bind_required_sinks(
    motif: CircuitMotif,
    live_signals: list[_LiveSignal],
    bindings: dict[str, str],
    rng: Random,
    *,
    required_signal: _LiveSignal | None = None,
) -> bool:
    """Bind every required sink port to a compatible live signal."""

    sink_ports = _required_sink_ports(motif)
    if required_signal is not None:
        required_ports = tuple(
            port
            for port in sink_ports
            if _signals_are_compatible(required_signal.signal, port.signal)
        )
        if not required_ports:
            return False
        required_port = required_ports[rng.randrange(len(required_ports))]
        bindings[required_port.name] = required_signal.net

    for port in sink_ports:
        if port.name in bindings:
            continue
        compatible = tuple(
            signal
            for signal in live_signals
            if _signals_are_compatible(signal.signal, port.signal)
        )
        if not compatible:
            return False
        bindings[port.name] = compatible[rng.randrange(len(compatible))].net
    return True


def _required_sink_ports(motif: CircuitMotif) -> tuple[MotifPort, ...]:
    """Return required non-supply sink ports."""

    return tuple(
        port
        for port in motif.ports
        if port.required
        and port.role is MotifPortRole.SINK
        and port.signal
        not in {
            MotifSignalKind.POWER,
            MotifSignalKind.REFERENCE,
        }
    )


def _source_ports(
    motif: CircuitMotif, signal: MotifSignalKind
) -> tuple[MotifPort, ...]:
    """Return source ports for one signal kind."""

    return tuple(
        port
        for port in motif.ports
        if port.role is MotifPortRole.SOURCE and port.signal is signal
    )


def _non_power_source_ports(motif: CircuitMotif) -> tuple[MotifPort, ...]:
    """Return source/probe ports that can extend a signal path."""

    return tuple(
        port
        for port in motif.ports
        if port.role in {MotifPortRole.SOURCE, MotifPortRole.PROBE}
        and port.signal
        not in {
            MotifSignalKind.POWER,
            MotifSignalKind.REFERENCE,
        }
    )


def _all_required_sinks_are_supply_only(motif: CircuitMotif) -> bool:
    """Return whether a motif can start a generated path."""

    return not _required_sink_ports(motif)


def _signals_are_compatible(source: MotifSignalKind, sink: MotifSignalKind) -> bool:
    """Return whether a source signal can drive a sink signal."""

    if source is sink:
        return True
    if source is MotifSignalKind.POWER:
        return sink in {MotifSignalKind.ANALOG, MotifSignalKind.DIGITAL}
    if source is MotifSignalKind.DIGITAL:
        return sink is MotifSignalKind.ANALOG
    return False


def _candidate_is_valid(
    circuit: Circuit,
    catalog: Mapping[str, PartSpec],
    motif_catalog: Mapping[str, CircuitMotif],
    instances: list[InstantiatedMotif],
) -> bool:
    """Return whether a generated circuit satisfies structural constraints."""

    report = check_circuit(circuit, catalog, AnalysisSupport.SPICE_EXPORT)
    if report.errors:
        return False
    if any(issue.code in _REJECTED_WARNING_CODES for issue in report.warnings):
        return False
    return _has_cross_motif_nonrail_net(
        circuit, instances
    ) and _has_ordered_signal_path(tuple(instances), motif_catalog)


def _has_cross_motif_nonrail_net(
    circuit: Circuit, instances: list[InstantiatedMotif]
) -> bool:
    """Return whether at least one non-rail net crosses motif instances."""

    instance_by_part = {
        part_ref: instance.instance_id
        for instance in instances
        for part_ref in instance.part_refs
    }
    for net in circuit.nets:
        if is_ground_net(net) or net in {"VCC", "VEE"}:
            continue
        instance_ids = {
            instance_by_part[connection.ref]
            for connection in circuit.connections_for_net(net)
            if connection.ref in instance_by_part
        }
        if len(instance_ids) > 1:
            return True
    return False


def _has_ordered_signal_path(
    instances: tuple[InstantiatedMotif, ...],
    motif_catalog: Mapping[str, CircuitMotif],
) -> bool:
    """Return whether path motifs consume and expose a signal in order."""

    if len(instances) < 3:
        return False
    previous_path = instances[1]
    consumed_count = 0
    for current in instances[2:]:
        current_sinks = _nonrail_port_nets_for_roles(
            current,
            motif_catalog,
            {MotifPortRole.SINK},
        )
        if not current_sinks:
            continue
        previous_outputs = _nonrail_port_nets_for_roles(
            previous_path,
            motif_catalog,
            {MotifPortRole.SOURCE, MotifPortRole.PROBE},
        )
        if not previous_outputs & current_sinks:
            return False
        previous_path = current
        consumed_count += 1
    final_outputs = _nonrail_port_nets_for_roles(
        previous_path,
        motif_catalog,
        {MotifPortRole.SOURCE, MotifPortRole.PROBE},
    )
    return consumed_count > 0 and bool(final_outputs)


def _nonrail_port_nets_for_roles(
    instance: InstantiatedMotif,
    motif_catalog: Mapping[str, CircuitMotif],
    roles: set[MotifPortRole],
) -> set[str]:
    """Return non-rail port nets for selected roles on one instance."""

    motif = motif_catalog[instance.motif_name]
    return {
        instance.port_nets[port.name]
        for port in motif.ports
        if port.role in roles
        and not is_ground_net(instance.port_nets[port.name])
        and instance.port_nets[port.name] not in {"VCC", "VEE"}
    }
