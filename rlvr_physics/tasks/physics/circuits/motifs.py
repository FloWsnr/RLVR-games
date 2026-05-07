"""Reusable procedural circuit motif catalog."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from random import Random
import re
from types import MappingProxyType
from typing import Callable, Mapping, Protocol

from rlvr_physics.tasks.physics.circuits.parts import default_part_catalog
from rlvr_physics.tasks.physics.circuits.model import CircuitBuilder
from rlvr_physics.tasks.physics.circuits.model import ComponentFamily, PinKind


class MotifContext(Protocol):
    """Builder-facing context required by procedural motifs.

    Attributes
    ----------
    builder:
        Circuit builder that receives new parts and connections.
    rng:
        Deterministic random number generator owned by the generation run.
    supply_voltage_v:
        Main generated supply voltage in volts.
    """

    builder: CircuitBuilder
    rng: Random
    supply_voltage_v: float

    def add_part(
        self,
        prefix: str,
        kind: str,
        value: str,
        parameters: Mapping[str, object],
        metadata: Mapping[str, object],
    ) -> str:
        """Add a generated part.

        Parameters
        ----------
        prefix:
            Reference designator prefix.
        kind:
            Component kind from the part catalog.
        value:
            Display value.
        parameters:
            Structured component parameters.
        metadata:
            Auxiliary component metadata.

        Returns
        -------
        str
            Reference designator for the new part.
        """
        ...

    def node(self) -> str:
        """Return a fresh generated node name.

        Returns
        -------
        str
            Deterministic node name unique within the generated circuit.
        """
        ...

    def motif_instance_id(self, motif_name: str) -> str:
        """Return a fresh deterministic motif instance id.

        Parameters
        ----------
        motif_name:
            Stable motif name.

        Returns
        -------
        str
            Unique motif instance id within the generated circuit.
        """
        ...

    def add_negative_supply(
        self, net: str, motif_name: str, instance_id: str
    ) -> tuple[str, ...]:
        """Add or reuse a negative supply source for ``net``.

        Parameters
        ----------
        net:
            Generated negative supply net.
        motif_name:
            Motif requesting the supply.
        instance_id:
            Motif instance requesting the supply.

        Returns
        -------
        tuple[str, ...]
            Newly added part references, or an empty tuple when the supply
            already exists.
        """
        ...


class MotifPortRole(Enum):
    """Role a motif port plays during procedural composition."""

    SOURCE = "source"
    SINK = "sink"
    SUPPLY = "supply"
    GROUND = "ground"
    PROBE = "probe"


class MotifSignalKind(Enum):
    """Signal class used to decide whether two motif ports can be bound."""

    ANALOG = "analog"
    DIGITAL = "digital"
    POWER = "power"
    REFERENCE = "reference"


@dataclass(frozen=True)
class MotifPort:
    """One declared boundary port on a reusable motif.

    Parameters
    ----------
    name:
        Stable port name within the motif.
    net:
        Local motif net represented by the port.
    role:
        Composition role for the port.
    signal:
        Signal kind expected or produced by the port.
    required:
        Whether generation must bind this port to an external net.
    """

    name: str
    net: str
    role: MotifPortRole
    signal: MotifSignalKind
    required: bool


@dataclass(frozen=True)
class InstantiatedMotif:
    """Result of building one motif instance into a circuit.

    Parameters
    ----------
    motif_name:
        Name of the motif catalog entry.
    instance_id:
        Deterministic identifier for this motif instance within the circuit.
    port_nets:
        Mapping from motif port name to actual generated circuit net.
    part_refs:
        Generated part references owned by this motif instance.
    """

    motif_name: str
    instance_id: str
    port_nets: Mapping[str, str]
    part_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        """Freeze mutable mapping inputs after construction."""

        object.__setattr__(self, "port_nets", MappingProxyType(dict(self.port_nets)))


MotifBuilder = Callable[[MotifContext, Mapping[str, str]], InstantiatedMotif]
_EMPTY_PARAMETERS: Mapping[str, object] = MappingProxyType({})


@dataclass(frozen=True)
class CircuitMotif:
    """Catalog entry for one reusable procedural circuit motif.

    Parameters
    ----------
    name:
        Stable motif identifier used in generation config and provenance.
    element_count:
        Number of non-ground parts added by the motif.
    default_weight:
        Default relative selection weight.
    ports:
        Boundary contract used by procedural composition.
    build:
        Motif builder function.
    """

    name: str
    element_count: int
    default_weight: float
    ports: tuple[MotifPort, ...]
    build: MotifBuilder


@dataclass(frozen=True)
class _MotifPart:
    """Declarative part entry for one netlist-style motif."""

    ref: str
    kind: str
    value: str
    parameters: Mapping[str, object]


@dataclass(frozen=True)
class _MotifSpec:
    """Declarative netlist-style motif specification."""

    name: str
    default_weight: float
    parts: tuple[_MotifPart, ...]
    connections: tuple[tuple[str, str, str], ...]


def default_motifs() -> Mapping[str, CircuitMotif]:
    """Return the built-in procedural motif catalog.

    Returns
    -------
    Mapping[str, CircuitMotif]
        Immutable mapping from motif name to motif definition.
    """

    return _DEFAULT_MOTIFS


def default_motif_weights() -> Mapping[str, float]:
    """Return default relative motif weights.

    Returns
    -------
    Mapping[str, float]
        Weight mapping keyed by motif name.
    """

    return {
        name: motif.default_weight
        for name, motif in _DEFAULT_MOTIFS.items()
        if motif.default_weight > 0.0
    }


def choose_motif(
    rng: Random,
    motif_catalog: Mapping[str, CircuitMotif],
    weights: Mapping[str, float],
) -> CircuitMotif | None:
    """Choose one weighted motif.

    Parameters
    ----------
    rng:
        Deterministic random number generator.
    motif_catalog:
        Available motif definitions.
    weights:
        Relative motif weights keyed by motif name.
    Returns
    -------
    CircuitMotif | None
        Chosen motif, or ``None`` when no weighted motif is enabled.
    """

    viable = [
        (motif, weight)
        for name, motif in motif_catalog.items()
        for weight in (weights.get(name, 0.0),)
        if weight > 0.0
    ]
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


def _part(
    ref: str, kind: str, value: str, parameters: Mapping[str, object]
) -> _MotifPart:
    """Return one declarative motif part."""

    return _MotifPart(ref=ref, kind=kind, value=value, parameters=parameters)


def _plain_part(ref: str, kind: str, value: str) -> _MotifPart:
    """Return one declarative motif part without parameters."""

    return _part(ref, kind, value, _EMPTY_PARAMETERS)


def _res(ref: str, value: str, resistance_ohm: float) -> _MotifPart:
    """Return one resistor motif part."""

    return _part(ref, "resistor", value, {"resistance_ohm": resistance_ohm})


def _cap(ref: str, value: str, capacitance_f: float) -> _MotifPart:
    """Return one capacitor motif part."""

    return _part(ref, "capacitor", value, {"capacitance_f": capacitance_f})


def _ind(ref: str, value: str, inductance_h: float) -> _MotifPart:
    """Return one inductor motif part."""

    return _part(ref, "inductor", value, {"inductance_h": inductance_h})


def _dc_source(ref: str, value: str, voltage_v: float) -> _MotifPart:
    """Return one DC voltage source motif part."""

    return _part(ref, "voltage_source_dc", value, {"voltage_v": voltage_v})


def _ac_source(ref: str, value: str) -> _MotifPart:
    """Return one AC voltage source motif part."""

    return _part(ref, "voltage_source_ac", value, {"voltage_spec": value})


def _sw(ref: str, state: str, resistance_ohm: float) -> _MotifPart:
    """Return one ideal switch motif part."""

    return _part(ref, "ideal_switch", state, {"state_resistance_ohm": resistance_ohm})


def _controlled_sw(ref: str) -> _MotifPart:
    """Return one controlled switch motif part."""

    return _plain_part(ref, "controlled_switch", "ctrl")


def _battery(ref: str, value: str, voltage_v: float) -> _MotifPart:
    """Return one battery motif part."""

    return _part(ref, "battery", value, {"voltage_v": voltage_v})


def _ammeter(ref: str) -> _MotifPart:
    """Return one ammeter motif part."""

    return _part(ref, "ammeter", "0", {"voltage_v": 0.0})


def _voltmeter(ref: str) -> _MotifPart:
    """Return one voltmeter motif part."""

    return _part(ref, "voltmeter", "1T", {"resistance_ohm": 1e12})


def _pullup(ref: str, value: str, resistance_ohm: float) -> _MotifPart:
    """Return one pull-up resistor motif part."""

    return _part(ref, "pullup_resistor", value, {"resistance_ohm": resistance_ohm})


def _pulldown(ref: str, value: str, resistance_ohm: float) -> _MotifPart:
    """Return one pull-down resistor motif part."""

    return _part(ref, "pulldown_resistor", value, {"resistance_ohm": resistance_ohm})


def _pushbutton(ref: str, resistance_ohm: float) -> _MotifPart:
    """Return one pushbutton switch motif part."""

    return _part(
        ref,
        "pushbutton_switch",
        "closed" if resistance_ohm < 1.0 else "open",
        {"state_resistance_ohm": resistance_ohm},
    )


def _ideal_switch(ref: str, resistance_ohm: float) -> _MotifPart:
    """Return one ideal switch motif part."""

    return _part(
        ref,
        "ideal_switch",
        "closed" if resistance_ohm < 1.0 else "open",
        {"state_resistance_ohm": resistance_ohm},
    )


def _variable_res(ref: str, value: str, resistance_ohm: float) -> _MotifPart:
    """Return one variable resistor motif part."""

    return _part(ref, "variable_resistor", value, {"resistance_ohm": resistance_ohm})


def _configured_supply_motif() -> CircuitMotif:
    """Return the generator-owned configurable DC supply motif."""

    name = "dc_supply_source"

    def build(ctx: MotifContext, port_bindings: Mapping[str, str]) -> InstantiatedMotif:
        """Build the configured supply source into ``ctx``."""

        unknown_ports = set(port_bindings) - {"ground_n_0", "source_vcc"}
        if unknown_ports:
            raise ValueError(f"unknown port for {name}: {tuple(sorted(unknown_ports))}")
        ground_net = port_bindings.get("ground_n_0", "0")
        if not ground_net:
            raise ValueError(f"empty binding for {name}.ground_n_0")
        vcc_net = port_bindings.get("source_vcc", "VCC")
        if not vcc_net:
            raise ValueError(f"empty binding for {name}.source_vcc")
        supply_voltage_v = float(ctx.supply_voltage_v)
        if not isfinite(supply_voltage_v):
            raise ValueError("supply_voltage_v must be finite")
        instance_id = ctx.motif_instance_id(name)
        ref = ctx.add_part(
            "V",
            "voltage_source_dc",
            _voltage_display(supply_voltage_v),
            {"voltage_v": supply_voltage_v},
            {
                "role": "main_supply",
                "motif": name,
                "motif_instance": instance_id,
                "source_ref": "VMAIN",
            },
        )
        ctx.builder.connect(ref, "p", vcc_net)
        ctx.builder.connect(ref, "n", ground_net)
        return InstantiatedMotif(
            motif_name=name,
            instance_id=instance_id,
            port_nets={"ground_n_0": ground_net, "source_vcc": vcc_net},
            part_refs=(ref,),
        )

    return CircuitMotif(
        name=name,
        element_count=1,
        default_weight=0.0,
        ports=(
            MotifPort(
                name="ground_n_0",
                net="0",
                role=MotifPortRole.GROUND,
                signal=MotifSignalKind.REFERENCE,
                required=True,
            ),
            MotifPort(
                name="source_vcc",
                net="VCC",
                role=MotifPortRole.SOURCE,
                signal=MotifSignalKind.POWER,
                required=False,
            ),
        ),
        build=build,
    )


def _voltage_display(value: float) -> str:
    """Return compact display text for a voltage value."""

    return f"{value:.12g}V"


def _build_default_motifs() -> Mapping[str, CircuitMotif]:
    """Build the immutable built-in motif catalog."""

    motifs = {"dc_supply_source": _configured_supply_motif()}
    for spec in _DEFAULT_MOTIF_SPECS:
        motifs[spec.name] = CircuitMotif(
            spec.name,
            _element_count(spec),
            spec.default_weight,
            _motif_ports(spec),
            _build_netlist_motif(spec),
        )
    return MappingProxyType(motifs)


def _build_netlist_motif(spec: _MotifSpec) -> MotifBuilder:
    """Return a builder for one declarative motif specification."""

    def build(ctx: MotifContext, port_bindings: Mapping[str, str]) -> InstantiatedMotif:
        """Build one declarative motif into ``ctx``."""

        net_bindings = _net_bindings_for_ports(spec, port_bindings)
        ref_map: dict[str, str] = {}
        net_map: dict[str, str] = {}
        part_refs: list[str] = []
        instance_id = ctx.motif_instance_id(spec.name)
        if _uses_net(spec, "VEE"):
            negative_rail = _local_net(ctx, "VEE", net_map, net_bindings)
            part_refs.extend(
                ctx.add_negative_supply(negative_rail, spec.name, instance_id)
            )
        for part in spec.parts:
            generated_ref = ctx.add_part(
                _reference_prefix(part.ref),
                part.kind,
                part.value,
                part.parameters,
                {
                    "motif": spec.name,
                    "motif_instance": instance_id,
                    "source_ref": part.ref,
                },
            )
            ref_map[part.ref] = generated_ref
            part_refs.append(generated_ref)
        for ref, pin, net in spec.connections:
            ctx.builder.connect(
                ref_map[ref], pin, _local_net(ctx, net, net_map, net_bindings)
            )
        return InstantiatedMotif(
            motif_name=spec.name,
            instance_id=instance_id,
            port_nets={
                port.name: _local_net(ctx, port.net, net_map, net_bindings)
                for port in _motif_ports(spec)
            },
            part_refs=tuple(part_refs),
        )

    return build


def _element_count(spec: _MotifSpec) -> int:
    """Return the non-ground part count added by ``spec``."""

    return sum(1 for part in spec.parts if part.kind != "ground") + (
        1 if _uses_net(spec, "VEE") else 0
    )


def _motif_ports(spec: _MotifSpec) -> tuple[MotifPort, ...]:
    """Return the inferred boundary contract for one motif."""

    catalog = default_part_catalog()
    parts = {part.ref: part for part in spec.parts}
    connected_by_net: dict[str, list[tuple[str, str]]] = {}
    for ref, pin, net in spec.connections:
        connected_by_net.setdefault(net, []).append((ref, pin))

    source_nets: set[str] = set()
    sink_nets: set[str] = set()
    probe_nets: set[str] = set()
    digital_nets: set[str] = set()
    for net, connections in connected_by_net.items():
        for ref, pin_name in connections:
            part = parts.get(ref)
            if part is None:
                continue
            spec_for_part = catalog[part.kind]
            pin = spec_for_part.pin(pin_name)
            if spec_for_part.family is ComponentFamily.LOGIC:
                digital_nets.add(net)
            if (
                spec_for_part.family is ComponentFamily.LOGIC
                and pin.kind is PinKind.OUTPUT
            ):
                source_nets.add(net)
            if pin.kind is PinKind.POWER_OUT and _looks_like_power_output_net(net):
                source_nets.add(net)
            if spec_for_part.kind in {"test_point", "voltmeter", "ammeter"}:
                probe_nets.add(net)

    for net in connected_by_net:
        normalized = _normalize_net(net)
        if normalized in {"0", "VCC", "VEE"}:
            continue
        if (
            _looks_like_sink_net(net)
            and net not in source_nets
            and not _has_local_bias_or_switch(connected_by_net[net], parts)
        ):
            sink_nets.add(net)
        if _looks_like_source_net(net):
            source_nets.add(net)
        if _looks_like_power_output_net(net):
            source_nets.add(net)
        if _looks_like_probe_net(net):
            probe_nets.add(net)
        if _looks_like_digital_net(net):
            digital_nets.add(net)

    ports: dict[tuple[MotifPortRole, str], MotifPort] = {}
    for net in connected_by_net:
        normalized = _normalize_net(net)
        if normalized == "0":
            _add_port(
                ports,
                net,
                MotifPortRole.GROUND,
                MotifSignalKind.REFERENCE,
                required=True,
            )
        elif normalized in {"VCC", "VEE"}:
            _add_port(
                ports,
                net,
                MotifPortRole.SUPPLY,
                MotifSignalKind.POWER,
                required=True,
            )

    if _uses_net(spec, "VEE"):
        _add_port(
            ports,
            "0",
            MotifPortRole.GROUND,
            MotifSignalKind.REFERENCE,
            required=True,
        )

    for net in sorted(source_nets, key=_net_sort_key):
        _add_port(
            ports,
            net,
            MotifPortRole.SOURCE,
            _signal_kind_for_net(net, digital_nets, source_nets),
            required=False,
        )
    for net in sorted(sink_nets - source_nets, key=_net_sort_key):
        _add_port(
            ports,
            net,
            MotifPortRole.SINK,
            _signal_kind_for_net(net, digital_nets, source_nets),
            required=True,
        )
    for net in sorted(probe_nets - source_nets, key=_net_sort_key):
        _add_port(
            ports,
            net,
            MotifPortRole.PROBE,
            _signal_kind_for_net(net, digital_nets, source_nets),
            required=False,
        )
    return tuple(ports[key] for key in sorted(ports, key=lambda item: item[0].value))


def _add_port(
    ports: dict[tuple[MotifPortRole, str], MotifPort],
    net: str,
    role: MotifPortRole,
    signal: MotifSignalKind,
    *,
    required: bool,
) -> None:
    """Add one unique motif port."""

    name = f"{role.value}_{_safe_port_suffix(net)}"
    ports.setdefault(
        (role, net),
        MotifPort(
            name=name,
            net=net,
            role=role,
            signal=signal,
            required=required,
        ),
    )


def _safe_port_suffix(net: str) -> str:
    """Return a stable identifier suffix for a local net."""

    suffix = re.sub(r"[^a-z0-9]+", "_", net.lower()).strip("_")
    if not suffix:
        return "net"
    if suffix[0].isdigit():
        return f"n_{suffix}"
    return suffix


def _safe_generated_identifier(value: str, fallback: str) -> str:
    """Return a generated identifier using letters, digits, and underscores."""

    identifier = re.sub(r"\W+", "_", value).strip("_")
    if not identifier:
        identifier = fallback
    if identifier[0].isdigit():
        identifier = f"{fallback}_{identifier}"
    return identifier


def _signal_kind_for_net(
    net: str, digital_nets: set[str], source_nets: set[str]
) -> MotifSignalKind:
    """Infer a composition signal kind for one local motif net."""

    normalized = _normalize_net(net)
    if normalized == "0":
        return MotifSignalKind.REFERENCE
    if normalized in {"VCC", "VEE"}:
        return MotifSignalKind.POWER
    if net in source_nets and _looks_like_power_source_net(net):
        return MotifSignalKind.POWER
    if net in digital_nets or _looks_like_digital_net(net):
        return MotifSignalKind.DIGITAL
    return MotifSignalKind.ANALOG


def _looks_like_sink_net(net: str) -> bool:
    """Return whether a net name usually represents an external input."""

    upper = net.upper()
    if upper in {"IN", "INA", "INB", "INP", "INN", "EXT_IN", "CTRL", "PWM"}:
        return True
    if upper in {"TRIG_IN", "CLK", "SAMPLE", "RESET_N", "S_N", "R_N"}:
        return True
    return upper in {"A", "B"}


def _looks_like_source_net(net: str) -> bool:
    """Return whether a net name usually represents an observable output."""

    upper = net.upper()
    if upper in {
        "OUT",
        "VOUT",
        "OUTP",
        "OUTN",
        "Y",
        "SUM",
        "CARRY",
        "SENSE",
        "SIG",
        "VMID",
        "VSW",
        "NLOAD",
    }:
        return True
    return bool(re.fullmatch(r"Q[A-D0-9]?", upper))


def _looks_like_probe_net(net: str) -> bool:
    """Return whether a net name usually represents an observable probe."""

    return net.upper() in {"OUT", "VOUT", "SENSE", "VMID", "SIG"}


def _looks_like_digital_net(net: str) -> bool:
    """Return whether a net name usually carries a digital signal."""

    upper = net.upper()
    if upper in {"A", "B", "Y", "SUM", "CARRY", "CLK", "RESET_N", "S_N", "R_N"}:
        return True
    if upper in {"CTRL", "PWM", "SAMPLE", "SIG"}:
        return True
    return bool(re.fullmatch(r"Q[A-D0-9]?", upper))


def _looks_like_power_source_net(net: str) -> bool:
    """Return whether a source-like net is better treated as power."""

    upper = net.upper()
    return upper in {"VCC", "VDD", "VEE", "VIN", "VBAT", "VRAW", "VRECT", "VSW"}


def _looks_like_power_output_net(net: str) -> bool:
    """Return whether a powered net is a motif output rather than excitation."""

    return net.upper() in {"VBAT", "VRAW", "VRECT"}


def _has_local_bias_or_switch(
    connections: list[tuple[str, str]],
    parts: Mapping[str, _MotifPart],
) -> bool:
    """Return whether a net is locally driven by an input-bias subcircuit."""

    biased_kinds = {
        "ideal_switch",
        "pullup_resistor",
        "pulldown_resistor",
        "pushbutton_switch",
    }
    return any(
        (part := parts.get(ref)) is not None and part.kind in biased_kinds
        for ref, _ in connections
    )


def _net_sort_key(net: str) -> tuple[int, str]:
    """Sort common boundary names before internal-looking names."""

    upper = net.upper()
    priority = 0 if _looks_like_sink_net(upper) or _looks_like_source_net(upper) else 1
    return priority, upper


def _uses_net(spec: _MotifSpec, net: str) -> bool:
    """Return whether ``spec`` connects any pin to ``net``."""

    return any(connection_net == net for _, _, connection_net in spec.connections)


def _reference_prefix(ref: str) -> str:
    """Return the generated reference prefix for one source reference."""

    prefix = ref.rstrip("0123456789")
    if prefix:
        return prefix
    return ref


def _normalize_net(net: str) -> str:
    """Normalize common supply aliases used by source netlists."""

    supply_aliases = {
        "VDD": "VCC",
        "VEXC": "VCC",
        "VBIAS": "VCC",
    }
    return supply_aliases.get(net, net)


def _net_bindings_for_ports(
    spec: _MotifSpec, port_bindings: Mapping[str, str]
) -> Mapping[str, str]:
    """Return a local-net binding map from external port bindings."""

    ports = {port.name: port for port in _motif_ports(spec)}
    bindings: dict[str, str] = {}
    for port_name, external_net in port_bindings.items():
        if port_name not in ports:
            raise ValueError(f"unknown port for {spec.name}: {port_name}")
        if not external_net:
            raise ValueError(f"empty binding for {spec.name}.{port_name}")
        port = ports[port_name]
        for local_net in {port.net, _normalize_net(port.net)}:
            previous = bindings.get(local_net)
            if previous is not None and previous != external_net:
                raise ValueError(
                    f"conflicting bindings for {spec.name} net {port.net}: "
                    f"{previous} and {external_net}"
                )
            bindings[local_net] = external_net
    return MappingProxyType(bindings)


def _local_net(
    ctx: MotifContext,
    net: str,
    net_map: dict[str, str],
    net_bindings: Mapping[str, str],
) -> str:
    """Return a build-local net name, preserving only global rails."""

    normalized = _normalize_net(net)
    if net in net_bindings:
        return net_bindings[net]
    if normalized in net_bindings:
        return net_bindings[normalized]
    if normalized in {"0", "VCC"}:
        return normalized
    if normalized not in net_map:
        net_prefix = _safe_generated_identifier(normalized, "N")
        net_map[normalized] = f"{net_prefix}_{ctx.node()}"
    return net_map[normalized]


_DEFAULT_MOTIF_SPECS = (
    _MotifSpec(
        "bridge_rectifier",
        0.8,
        (
            _plain_part("T1", "transformer", "1:1"),
            _ac_source("VAC1", "AC 6"),
            _plain_part("D1", "diode", "D"),
            _plain_part("D2", "diode", "D"),
            _plain_part("D3", "diode", "D"),
            _plain_part("D4", "diode", "D"),
            _res("RLOAD", "1k", 1000.0),
        ),
        (
            ("VAC1", "p", "ACPRI"),
            ("VAC1", "n", "0"),
            ("T1", "p1", "ACPRI"),
            ("T1", "p2", "0"),
            ("T1", "s1", "AC1"),
            ("T1", "s2", "AC2"),
            ("D1", "a", "AC1"),
            ("D1", "k", "VRAW"),
            ("D2", "a", "AC2"),
            ("D2", "k", "VRAW"),
            ("D3", "a", "0"),
            ("D3", "k", "AC1"),
            ("D4", "a", "0"),
            ("D4", "k", "AC2"),
            ("RLOAD", "1", "VRAW"),
            ("RLOAD", "2", "0"),
        ),
    ),
    _MotifSpec(
        "crc_power_filter",
        1.0,
        (
            _dc_source("VRECT", "12V", 12.0),
            _cap("C1", "100u", 1e-4),
            _res("R1", "10", 10.0),
            _cap("C2", "100u", 1e-4),
            _res("RLOAD", "1k", 1000.0),
        ),
        (
            ("VRECT", "p", "VRAW"),
            ("VRECT", "n", "0"),
            ("C1", "1", "VRAW"),
            ("C1", "2", "0"),
            ("R1", "1", "VRAW"),
            ("R1", "2", "OUT"),
            ("C2", "1", "OUT"),
            ("C2", "2", "0"),
            ("RLOAD", "1", "OUT"),
            ("RLOAD", "2", "0"),
        ),
    ),
    _MotifSpec(
        "zener_shunt_regulator",
        1.0,
        (
            _dc_source("VIN", "12V", 12.0),
            _res("R1", "470", 470.0),
            _plain_part("DZ1", "zener", "5V1"),
            _cap("C1", "10u", 1e-5),
            _res("RLOAD", "1k", 1000.0),
        ),
        (
            ("VIN", "p", "VIN"),
            ("VIN", "n", "0"),
            ("R1", "1", "VIN"),
            ("R1", "2", "OUT"),
            ("DZ1", "k", "OUT"),
            ("DZ1", "a", "0"),
            ("C1", "1", "OUT"),
            ("C1", "2", "0"),
            ("RLOAD", "1", "OUT"),
            ("RLOAD", "2", "0"),
        ),
    ),
    _MotifSpec(
        "rc_low_pass_filter",
        1.0,
        (
            _res("R1", "1k", 1000.0),
            _cap("C1", "1u", 1e-6),
            _res("RLOAD", "10k", 10000.0),
        ),
        (
            ("R1", "1", "IN"),
            ("R1", "2", "OUT"),
            ("C1", "1", "OUT"),
            ("C1", "2", "0"),
            ("RLOAD", "1", "OUT"),
            ("RLOAD", "2", "0"),
        ),
    ),
    _MotifSpec(
        "rc_high_pass_filter",
        1.0,
        (
            _cap("C1", "1u", 1e-6),
            _res("R1", "10k", 10000.0),
            _res("RLOAD", "10k", 10000.0),
        ),
        (
            ("C1", "1", "IN"),
            ("C1", "2", "OUT"),
            ("R1", "1", "OUT"),
            ("R1", "2", "0"),
            ("RLOAD", "1", "OUT"),
            ("RLOAD", "2", "0"),
        ),
    ),
    _MotifSpec(
        "passive_rlc_band_pass_filter",
        0.9,
        (_cap("C1", "100n", 1e-7), _ind("L1", "1m", 1e-3), _res("R1", "1k", 1000.0)),
        (
            ("C1", "1", "IN"),
            ("C1", "2", "N1"),
            ("L1", "1", "N1"),
            ("L1", "2", "OUT"),
            ("R1", "1", "OUT"),
            ("R1", "2", "0"),
        ),
    ),
    _MotifSpec(
        "twin_t_notch_filter",
        0.7,
        (
            _res("R1", "10k", 10000.0),
            _res("R2", "10k", 10000.0),
            _cap("C3", "10n", 1e-8),
            _cap("C1", "10n", 1e-8),
            _cap("C2", "10n", 1e-8),
            _res("R3", "5k", 5000.0),
        ),
        (
            ("R1", "1", "IN"),
            ("R1", "2", "NR"),
            ("R2", "1", "NR"),
            ("R2", "2", "OUT"),
            ("C3", "1", "NR"),
            ("C3", "2", "0"),
            ("C1", "1", "IN"),
            ("C1", "2", "NC"),
            ("C2", "1", "NC"),
            ("C2", "2", "OUT"),
            ("R3", "1", "NC"),
            ("R3", "2", "0"),
        ),
    ),
    _MotifSpec(
        "fixed_bias_bjt_common_emitter_amplifier",
        0.8,
        (
            _res("RB", "100k", 100000.0),
            _cap("CIN", "1u", 1e-6),
            _plain_part("Q1", "bjt_npn", "NPN"),
            _res("RC", "4.7k", 4700.0),
            _cap("COUT", "1u", 1e-6),
            _res("RLOAD", "10k", 10000.0),
        ),
        (
            ("RB", "1", "VCC"),
            ("RB", "2", "NB"),
            ("CIN", "1", "IN"),
            ("CIN", "2", "NB"),
            ("Q1", "b", "NB"),
            ("Q1", "c", "NC"),
            ("Q1", "e", "0"),
            ("RC", "1", "VCC"),
            ("RC", "2", "NC"),
            ("COUT", "1", "NC"),
            ("COUT", "2", "OUT"),
            ("RLOAD", "1", "OUT"),
            ("RLOAD", "2", "0"),
        ),
    ),
    _MotifSpec(
        "voltage_divider_biased_common_emitter_amplifier",
        0.7,
        (
            _res("R1", "100k", 100000.0),
            _res("R2", "22k", 22000.0),
            _cap("CIN", "1u", 1e-6),
            _plain_part("Q1", "bjt_npn", "NPN"),
            _res("RC", "4.7k", 4700.0),
            _res("RE", "1k", 1000.0),
            _cap("CE", "47u", 4.7e-5),
            _cap("COUT", "1u", 1e-6),
            _res("RLOAD", "10k", 10000.0),
        ),
        (
            ("R1", "1", "VCC"),
            ("R1", "2", "NB"),
            ("R2", "1", "NB"),
            ("R2", "2", "0"),
            ("CIN", "1", "IN"),
            ("CIN", "2", "NB"),
            ("Q1", "b", "NB"),
            ("Q1", "c", "NC"),
            ("Q1", "e", "NE"),
            ("RC", "1", "VCC"),
            ("RC", "2", "NC"),
            ("RE", "1", "NE"),
            ("RE", "2", "0"),
            ("CE", "1", "NE"),
            ("CE", "2", "0"),
            ("COUT", "1", "NC"),
            ("COUT", "2", "OUT"),
            ("RLOAD", "1", "OUT"),
            ("RLOAD", "2", "0"),
        ),
    ),
    _MotifSpec(
        "common_collector_emitter_follower",
        0.8,
        (
            _res("R1", "100k", 100000.0),
            _res("R2", "22k", 22000.0),
            _cap("CIN", "1u", 1e-6),
            _plain_part("Q1", "bjt_npn", "NPN"),
            _res("RE", "1k", 1000.0),
            _cap("COUT", "1u", 1e-6),
            _res("RLOAD", "10k", 10000.0),
        ),
        (
            ("R1", "1", "VCC"),
            ("R1", "2", "NB"),
            ("R2", "1", "NB"),
            ("R2", "2", "0"),
            ("CIN", "1", "IN"),
            ("CIN", "2", "NB"),
            ("Q1", "b", "NB"),
            ("Q1", "c", "VCC"),
            ("Q1", "e", "NE"),
            ("RE", "1", "NE"),
            ("RE", "2", "0"),
            ("COUT", "1", "NE"),
            ("COUT", "2", "OUT"),
            ("RLOAD", "1", "OUT"),
            ("RLOAD", "2", "0"),
        ),
    ),
    _MotifSpec(
        "bjt_differential_pair",
        0.7,
        (
            _plain_part("Q1", "bjt_npn", "NPN"),
            _plain_part("Q2", "bjt_npn", "NPN"),
            _res("RC1", "4.7k", 4700.0),
            _res("RC2", "4.7k", 4700.0),
            _res("REE", "2.2k", 2200.0),
        ),
        (
            ("Q1", "b", "INP"),
            ("Q2", "b", "INN"),
            ("Q1", "c", "OUTN"),
            ("Q2", "c", "OUTP"),
            ("Q1", "e", "NTAIL"),
            ("Q2", "e", "NTAIL"),
            ("RC1", "1", "VCC"),
            ("RC1", "2", "OUTN"),
            ("RC2", "1", "VCC"),
            ("RC2", "2", "OUTP"),
            ("REE", "1", "NTAIL"),
            ("REE", "2", "VEE"),
        ),
    ),
    _MotifSpec(
        "common_source_fet_amplifier",
        0.8,
        (
            _res("RG1", "1M", 1000000.0),
            _res("RG2", "220k", 220000.0),
            _cap("CIN", "1u", 1e-6),
            _plain_part("M1", "mosfet_n", "NMOS"),
            _res("RD", "4.7k", 4700.0),
            _res("RS", "1k", 1000.0),
            _cap("COUT", "1u", 1e-6),
            _res("RLOAD", "10k", 10000.0),
        ),
        (
            ("RG1", "1", "VDD"),
            ("RG1", "2", "NG"),
            ("RG2", "1", "NG"),
            ("RG2", "2", "0"),
            ("CIN", "1", "IN"),
            ("CIN", "2", "NG"),
            ("M1", "g", "NG"),
            ("M1", "d", "ND"),
            ("M1", "s", "NS"),
            ("RD", "1", "VDD"),
            ("RD", "2", "ND"),
            ("RS", "1", "NS"),
            ("RS", "2", "0"),
            ("COUT", "1", "ND"),
            ("COUT", "2", "OUT"),
            ("RLOAD", "1", "OUT"),
            ("RLOAD", "2", "0"),
        ),
    ),
    _MotifSpec(
        "tuned_lc_band_pass_amplifier",
        0.7,
        (
            _res("RB1", "100k", 100000.0),
            _res("RB2", "22k", 22000.0),
            _cap("CIN", "1u", 1e-6),
            _plain_part("Q1", "bjt_npn", "NPN"),
            _ind("L1", "1m", 1e-3),
            _cap("CTANK", "100n", 1e-7),
            _cap("COUT", "1u", 1e-6),
            _res("RLOAD", "10k", 10000.0),
        ),
        (
            ("RB1", "1", "VCC"),
            ("RB1", "2", "NB"),
            ("RB2", "1", "NB"),
            ("RB2", "2", "0"),
            ("CIN", "1", "IN"),
            ("CIN", "2", "NB"),
            ("Q1", "b", "NB"),
            ("Q1", "e", "0"),
            ("Q1", "c", "NC"),
            ("L1", "1", "VCC"),
            ("L1", "2", "NC"),
            ("CTANK", "1", "VCC"),
            ("CTANK", "2", "NC"),
            ("COUT", "1", "NC"),
            ("COUT", "2", "OUT"),
            ("RLOAD", "1", "OUT"),
            ("RLOAD", "2", "0"),
        ),
    ),
    _MotifSpec(
        "inverting_op_amp_amplifier",
        0.8,
        (
            _plain_part("U1", "op_amp", "ideal"),
            _res("RIN", "10k", 10000.0),
            _res("RF", "100k", 100000.0),
        ),
        (
            ("U1", "vpos", "VCC"),
            ("U1", "vneg", "VEE"),
            ("U1", "noninv", "0"),
            ("RIN", "1", "IN"),
            ("RIN", "2", "NSUM"),
            ("RF", "1", "OUT"),
            ("RF", "2", "NSUM"),
            ("U1", "inv", "NSUM"),
            ("U1", "out", "OUT"),
        ),
    ),
    _MotifSpec(
        "non_inverting_op_amp_amplifier",
        0.8,
        (
            _plain_part("U1", "op_amp", "ideal"),
            _res("RG", "10k", 10000.0),
            _res("RF", "100k", 100000.0),
        ),
        (
            ("U1", "vpos", "VCC"),
            ("U1", "vneg", "VEE"),
            ("U1", "noninv", "IN"),
            ("U1", "inv", "NFB"),
            ("RG", "1", "NFB"),
            ("RG", "2", "0"),
            ("RF", "1", "OUT"),
            ("RF", "2", "NFB"),
            ("U1", "out", "OUT"),
        ),
    ),
    _MotifSpec(
        "differential_op_amp_subtractor",
        0.7,
        (
            _res("R1", "10k", 10000.0),
            _res("R2", "10k", 10000.0),
            _res("R3", "10k", 10000.0),
            _res("R4", "10k", 10000.0),
            _plain_part("U1", "op_amp", "ideal"),
        ),
        (
            ("R1", "1", "INA"),
            ("R1", "2", "NMINUS"),
            ("R2", "1", "OUT"),
            ("R2", "2", "NMINUS"),
            ("R3", "1", "INB"),
            ("R3", "2", "NPLUS"),
            ("R4", "1", "NPLUS"),
            ("R4", "2", "0"),
            ("U1", "inv", "NMINUS"),
            ("U1", "noninv", "NPLUS"),
            ("U1", "out", "OUT"),
            ("U1", "vpos", "VCC"),
            ("U1", "vneg", "VEE"),
        ),
    ),
    _MotifSpec(
        "voltage_comparator",
        0.8,
        (
            _res("R1", "10k", 10000.0),
            _res("R2", "10k", 10000.0),
            _plain_part("U1", "comparator", "cmp"),
            _res("RPU", "10k", 10000.0),
        ),
        (
            ("R1", "1", "VCC"),
            ("R1", "2", "VREF"),
            ("R2", "1", "VREF"),
            ("R2", "2", "0"),
            ("U1", "noninv", "IN"),
            ("U1", "inv", "VREF"),
            ("U1", "out", "OUT"),
            ("U1", "vpos", "VCC"),
            ("U1", "vneg", "0"),
            ("RPU", "1", "VCC"),
            ("RPU", "2", "OUT"),
        ),
    ),
    _MotifSpec(
        "schmitt_trigger_comparator",
        0.7,
        (
            _res("RA", "10k", 10000.0),
            _res("RB", "10k", 10000.0),
            _plain_part("U1", "comparator", "cmp"),
            _res("RREF", "10k", 10000.0),
            _res("RFB", "100k", 100000.0),
            _res("RPU", "10k", 10000.0),
        ),
        (
            ("RA", "1", "VCC"),
            ("RA", "2", "VREF"),
            ("RB", "1", "VREF"),
            ("RB", "2", "0"),
            ("U1", "inv", "IN"),
            ("U1", "noninv", "NTH"),
            ("RREF", "1", "VREF"),
            ("RREF", "2", "NTH"),
            ("RFB", "1", "OUT"),
            ("RFB", "2", "NTH"),
            ("U1", "out", "OUT"),
            ("U1", "vpos", "VCC"),
            ("U1", "vneg", "0"),
            ("RPU", "1", "VCC"),
            ("RPU", "2", "OUT"),
        ),
    ),
    _MotifSpec(
        "rc_phase_shift_oscillator",
        0.6,
        (
            _plain_part("U1", "op_amp", "ideal"),
            _res("RF", "300k", 300000.0),
            _res("RIN", "10k", 10000.0),
            _cap("C1", "10n", 1e-8),
            _res("R1", "10k", 10000.0),
            _cap("C2", "10n", 1e-8),
            _res("R2", "10k", 10000.0),
            _cap("C3", "10n", 1e-8),
            _res("R3", "10k", 10000.0),
        ),
        (
            ("U1", "vpos", "VCC"),
            ("U1", "vneg", "VEE"),
            ("U1", "noninv", "0"),
            ("U1", "inv", "NSUM"),
            ("RF", "1", "OUT"),
            ("RF", "2", "NSUM"),
            ("RIN", "1", "N3"),
            ("RIN", "2", "NSUM"),
            ("C1", "1", "OUT"),
            ("C1", "2", "N1"),
            ("R1", "1", "N1"),
            ("R1", "2", "0"),
            ("C2", "1", "N1"),
            ("C2", "2", "N2"),
            ("R2", "1", "N2"),
            ("R2", "2", "0"),
            ("C3", "1", "N2"),
            ("C3", "2", "N3"),
            ("R3", "1", "N3"),
            ("R3", "2", "0"),
            ("U1", "out", "OUT"),
        ),
    ),
    _MotifSpec(
        "wien_bridge_oscillator",
        0.6,
        (
            _plain_part("U1", "op_amp", "ideal"),
            _cap("C1", "10n", 1e-8),
            _res("R1", "10k", 10000.0),
            _res("R2", "10k", 10000.0),
            _cap("C2", "10n", 1e-8),
            _res("RF", "20k", 20000.0),
            _res("RG", "10k", 10000.0),
            _plain_part("D1", "diode", "D"),
            _plain_part("D2", "diode", "D"),
        ),
        (
            ("U1", "vpos", "VCC"),
            ("U1", "vneg", "VEE"),
            ("U1", "noninv", "NWIEN"),
            ("U1", "inv", "NFB"),
            ("U1", "out", "OUT"),
            ("C1", "1", "OUT"),
            ("C1", "2", "N1"),
            ("R1", "1", "N1"),
            ("R1", "2", "NWIEN"),
            ("R2", "1", "NWIEN"),
            ("R2", "2", "0"),
            ("C2", "1", "NWIEN"),
            ("C2", "2", "0"),
            ("RF", "1", "OUT"),
            ("RF", "2", "NFB"),
            ("RG", "1", "NFB"),
            ("RG", "2", "0"),
            ("D1", "a", "NFB"),
            ("D1", "k", "OUT"),
            ("D2", "a", "OUT"),
            ("D2", "k", "NFB"),
        ),
    ),
    _MotifSpec(
        "colpitts_lc_oscillator",
        0.6,
        (
            _res("RB1", "100k", 100000.0),
            _res("RB2", "22k", 22000.0),
            _plain_part("Q1", "bjt_npn", "NPN"),
            _ind("RFC", "10m", 1e-2),
            _ind("L1", "1m", 1e-3),
            _cap("C1", "100n", 1e-7),
            _cap("C2", "100n", 1e-7),
            _res("RE", "1k", 1000.0),
            _cap("COUT", "1u", 1e-6),
        ),
        (
            ("RB1", "1", "VCC"),
            ("RB1", "2", "NB"),
            ("RB2", "1", "NB"),
            ("RB2", "2", "0"),
            ("Q1", "b", "NB"),
            ("Q1", "c", "NTANK"),
            ("Q1", "e", "NFB"),
            ("RFC", "1", "VCC"),
            ("RFC", "2", "NTANK"),
            ("L1", "1", "NTANK"),
            ("L1", "2", "0"),
            ("C1", "1", "NTANK"),
            ("C1", "2", "NFB"),
            ("C2", "1", "NFB"),
            ("C2", "2", "0"),
            ("RE", "1", "NFB"),
            ("RE", "2", "0"),
            ("COUT", "1", "NTANK"),
            ("COUT", "2", "OUT"),
        ),
    ),
    _MotifSpec(
        "pierce_crystal_oscillator",
        0.7,
        (
            _plain_part("U1", "not_gate", "INV"),
            _res("RFB", "1M", 1000000.0),
            _plain_part("XTAL1", "crystal", "XTAL"),
            _cap("C1", "22p", 2.2e-11),
            _cap("C2", "22p", 2.2e-11),
        ),
        (
            ("U1", "vcc", "VCC"),
            ("U1", "gnd", "0"),
            ("U1", "in1", "NXIN"),
            ("U1", "out", "NXOUT"),
            ("RFB", "1", "NXOUT"),
            ("RFB", "2", "NXIN"),
            ("XTAL1", "1", "NXIN"),
            ("XTAL1", "2", "NXOUT"),
            ("C1", "1", "NXIN"),
            ("C1", "2", "0"),
            ("C2", "1", "NXOUT"),
            ("C2", "2", "0"),
        ),
    ),
    _MotifSpec(
        "photodiode_transimpedance_amplifier",
        0.7,
        (
            _plain_part("PD1", "photodiode", "PD"),
            _part("IPD", "current_source_dc", "10u", {"current_a": 1e-5}),
            _plain_part("U1", "op_amp", "ideal"),
            _res("RF", "1M", 1000000.0),
            _cap("CF", "10p", 1e-11),
        ),
        (
            ("PD1", "k", "VBIAS"),
            ("PD1", "a", "NSUM"),
            ("IPD", "p", "VBIAS"),
            ("IPD", "n", "NSUM"),
            ("U1", "inv", "NSUM"),
            ("U1", "noninv", "0"),
            ("RF", "1", "OUT"),
            ("RF", "2", "NSUM"),
            ("CF", "1", "OUT"),
            ("CF", "2", "NSUM"),
            ("U1", "out", "OUT"),
            ("U1", "vpos", "VCC"),
            ("U1", "vneg", "VEE"),
        ),
    ),
    _MotifSpec(
        "wheatstone_bridge_instrumentation_amplifier",
        0.7,
        (
            _res("R1", "1k", 1000.0),
            _res("R2", "1k", 1000.0),
            _res("R3", "1k", 1000.0),
            _res("R4", "1k", 1000.0),
            _plain_part("UINA", "instrumentation_amplifier", "INA"),
            _res("RG", "1k", 1000.0),
        ),
        (
            ("R1", "1", "VEXC"),
            ("R1", "2", "NP"),
            ("R2", "1", "NP"),
            ("R2", "2", "0"),
            ("R3", "1", "VEXC"),
            ("R3", "2", "NN"),
            ("R4", "1", "NN"),
            ("R4", "2", "0"),
            ("UINA", "inp", "NP"),
            ("UINA", "inn", "NN"),
            ("UINA", "ref", "0"),
            ("UINA", "out", "OUT"),
            ("UINA", "vpos", "VCC"),
            ("UINA", "vneg", "0"),
            ("RG", "1", "NRG1"),
            ("RG", "2", "NRG2"),
            ("UINA", "rg1", "NRG1"),
            ("UINA", "rg2", "NRG2"),
        ),
    ),
    _MotifSpec(
        "timer_555_astable_oscillator",
        0.7,
        (
            _plain_part("U555", "timer_555", "555"),
            _cap("CCTRL", "10n", 1e-8),
            _res("RA", "10k", 10000.0),
            _res("RB", "100k", 100000.0),
            _cap("CT", "100n", 1e-7),
        ),
        (
            ("U555", "gnd", "0"),
            ("U555", "vcc", "VCC"),
            ("U555", "reset", "VCC"),
            ("U555", "ctrl", "NCTRL"),
            ("CCTRL", "1", "NCTRL"),
            ("CCTRL", "2", "0"),
            ("RA", "1", "VCC"),
            ("RA", "2", "NDIS"),
            ("RB", "1", "NDIS"),
            ("RB", "2", "NTIME"),
            ("U555", "disch", "NDIS"),
            ("U555", "thresh", "NTIME"),
            ("U555", "trig", "NTIME"),
            ("CT", "1", "NTIME"),
            ("CT", "2", "0"),
            ("U555", "out", "OUT"),
        ),
    ),
    _MotifSpec(
        "timer_555_monostable_one_shot",
        0.7,
        (
            _plain_part("U555", "timer_555", "555"),
            _cap("CCTRL", "10n", 1e-8),
            _res("R1", "100k", 100000.0),
            _cap("C1", "1u", 1e-6),
        ),
        (
            ("U555", "gnd", "0"),
            ("U555", "vcc", "VCC"),
            ("U555", "reset", "VCC"),
            ("U555", "ctrl", "NCTRL"),
            ("CCTRL", "1", "NCTRL"),
            ("CCTRL", "2", "0"),
            ("U555", "trig", "TRIG_IN"),
            ("U555", "out", "OUT"),
            ("R1", "1", "VCC"),
            ("R1", "2", "NTIME"),
            ("C1", "1", "NTIME"),
            ("C1", "2", "0"),
            ("U555", "thresh", "NTIME"),
            ("U555", "disch", "NTIME"),
        ),
    ),
    _MotifSpec(
        "npn_low_side_relay_driver",
        0.8,
        (
            _res("RBASE", "1k", 1000.0),
            _res("RPD", "100k", 100000.0),
            _plain_part("Q1", "bjt_npn", "NPN"),
            _plain_part("K1", "relay", "relay"),
            _plain_part("DFLY", "diode", "D"),
            _part("LA", "lamp", "lamp", {"resistance_ohm": 100.0}),
        ),
        (
            ("RBASE", "1", "CTRL"),
            ("RBASE", "2", "NB"),
            ("RPD", "1", "NB"),
            ("RPD", "2", "0"),
            ("Q1", "b", "NB"),
            ("Q1", "e", "0"),
            ("Q1", "c", "NLOAD"),
            ("K1", "coil_p", "VCC"),
            ("K1", "coil_n", "NLOAD"),
            ("K1", "com", "NCONTACT"),
            ("K1", "no", "VCC"),
            ("K1", "nc", "0"),
            ("DFLY", "a", "NLOAD"),
            ("DFLY", "k", "VCC"),
            ("LA", "1", "NCONTACT"),
            ("LA", "2", "0"),
        ),
    ),
    _MotifSpec(
        "nmos_low_side_pwm_driver",
        0.8,
        (
            _res("RG", "100", 100.0),
            _res("RPD", "100k", 100000.0),
            _plain_part("M1", "mosfet_n", "NMOS"),
            _part("MOT", "motor", "motor", {"resistance_ohm": 25.0}),
            _plain_part("DFLY", "diode", "D"),
        ),
        (
            ("RG", "1", "PWM"),
            ("RG", "2", "NG"),
            ("RPD", "1", "NG"),
            ("RPD", "2", "0"),
            ("M1", "g", "NG"),
            ("M1", "s", "0"),
            ("M1", "d", "NLOAD"),
            ("MOT", "1", "VCC"),
            ("MOT", "2", "NLOAD"),
            ("DFLY", "a", "NLOAD"),
            ("DFLY", "k", "VCC"),
        ),
    ),
    _MotifSpec(
        "asynchronous_buck_converter",
        0.7,
        (
            _dc_source("VIN", "12V", 12.0),
            _controlled_sw("S1"),
            _plain_part("D1", "diode", "D"),
            _ind("L1", "100u", 1e-4),
            _cap("COUT", "100u", 1e-4),
            _res("RLOAD", "10", 10.0),
        ),
        (
            ("VIN", "p", "VIN"),
            ("VIN", "n", "0"),
            ("S1", "in", "VIN"),
            ("S1", "out", "NSW"),
            ("S1", "ctrl", "PWM"),
            ("D1", "a", "0"),
            ("D1", "k", "NSW"),
            ("L1", "1", "NSW"),
            ("L1", "2", "OUT"),
            ("COUT", "1", "OUT"),
            ("COUT", "2", "0"),
            ("RLOAD", "1", "OUT"),
            ("RLOAD", "2", "0"),
        ),
    ),
    _MotifSpec(
        "asynchronous_boost_converter",
        0.7,
        (
            _dc_source("VIN", "5V", 5.0),
            _ind("L1", "100u", 1e-4),
            _plain_part("M1", "mosfet_n", "NMOS"),
            _plain_part("D1", "diode", "D"),
            _cap("COUT", "100u", 1e-4),
            _res("RLOAD", "100", 100.0),
        ),
        (
            ("VIN", "p", "VIN"),
            ("VIN", "n", "0"),
            ("L1", "1", "VIN"),
            ("L1", "2", "NSW"),
            ("M1", "d", "NSW"),
            ("M1", "s", "0"),
            ("M1", "g", "PWM"),
            ("D1", "a", "NSW"),
            ("D1", "k", "OUT"),
            ("COUT", "1", "OUT"),
            ("COUT", "2", "0"),
            ("RLOAD", "1", "OUT"),
            ("RLOAD", "2", "0"),
        ),
    ),
    _MotifSpec(
        "diode_capacitor_voltage_doubler",
        0.8,
        (
            _ac_source("VAC1", "AC 3"),
            _cap("C1", "10u", 1e-5),
            _plain_part("D1", "diode", "D"),
            _plain_part("D2", "diode", "D"),
            _cap("C2", "10u", 1e-5),
            _res("RLOAD", "10k", 10000.0),
        ),
        (
            ("VAC1", "p", "AC_IN"),
            ("VAC1", "n", "0"),
            ("C1", "1", "AC_IN"),
            ("C1", "2", "N1"),
            ("D1", "a", "0"),
            ("D1", "k", "N1"),
            ("D2", "a", "N1"),
            ("D2", "k", "OUT"),
            ("C2", "1", "OUT"),
            ("C2", "2", "0"),
            ("RLOAD", "1", "OUT"),
            ("RLOAD", "2", "0"),
        ),
    ),
    _MotifSpec(
        "peak_detector_envelope_detector",
        0.9,
        (
            _ac_source("VSIG", "AC 1"),
            _plain_part("D1", "diode", "D"),
            _cap("C1", "1u", 1e-6),
            _res("RDIS", "100k", 100000.0),
            _res("RLOAD", "100k", 100000.0),
        ),
        (
            ("VSIG", "p", "IN"),
            ("VSIG", "n", "0"),
            ("D1", "a", "IN"),
            ("D1", "k", "OUT"),
            ("C1", "1", "OUT"),
            ("C1", "2", "0"),
            ("RDIS", "1", "OUT"),
            ("RDIS", "2", "0"),
            ("RLOAD", "1", "OUT"),
            ("RLOAD", "2", "0"),
        ),
    ),
    _MotifSpec(
        "precision_half_wave_rectifier",
        0.8,
        (
            _plain_part("U1", "op_amp", "ideal"),
            _plain_part("D1", "diode", "D"),
            _res("RLOAD", "10k", 10000.0),
        ),
        (
            ("U1", "noninv", "IN"),
            ("U1", "inv", "OUT"),
            ("U1", "out", "NDRV"),
            ("D1", "a", "NDRV"),
            ("D1", "k", "OUT"),
            ("RLOAD", "1", "OUT"),
            ("RLOAD", "2", "0"),
            ("U1", "vpos", "VCC"),
            ("U1", "vneg", "VEE"),
        ),
    ),
    _MotifSpec(
        "sample_and_hold",
        0.8,
        (
            _controlled_sw("S1"),
            _cap("C1", "1u", 1e-6),
            _plain_part("U1", "op_amp", "ideal"),
        ),
        (
            ("S1", "in", "IN"),
            ("S1", "out", "NHOLD"),
            ("S1", "ctrl", "SAMPLE"),
            ("C1", "1", "NHOLD"),
            ("C1", "2", "0"),
            ("U1", "noninv", "NHOLD"),
            ("U1", "inv", "OUT"),
            ("U1", "out", "OUT"),
            ("U1", "vpos", "VCC"),
            ("U1", "vneg", "VEE"),
        ),
    ),
    _MotifSpec(
        "two_input_nand_gate",
        1.0,
        (_plain_part("U1", "nand_gate", "NAND"),),
        (
            ("U1", "vcc", "VCC"),
            ("U1", "gnd", "0"),
            ("U1", "in1", "A"),
            ("U1", "in2", "B"),
            ("U1", "out", "Y"),
        ),
    ),
    _MotifSpec(
        "cross_coupled_nand_sr_latch",
        0.8,
        (
            _plain_part("U1", "nand_gate", "NAND"),
            _plain_part("U2", "nand_gate", "NAND"),
        ),
        (
            ("U1", "vcc", "VCC"),
            ("U1", "gnd", "0"),
            ("U2", "vcc", "VCC"),
            ("U2", "gnd", "0"),
            ("U1", "in1", "S_N"),
            ("U1", "in2", "Q_BAR"),
            ("U1", "out", "Q"),
            ("U2", "in1", "R_N"),
            ("U2", "in2", "Q"),
            ("U2", "out", "Q_BAR"),
        ),
    ),
    _MotifSpec(
        "half_adder",
        0.8,
        (
            _plain_part("XOR1", "xor_gate", "XOR"),
            _plain_part("AND1", "and_gate", "AND"),
            _plain_part("TPSUM", "test_point", "SUM"),
            _plain_part("TPCARRY", "test_point", "CARRY"),
        ),
        (
            ("XOR1", "vcc", "VCC"),
            ("XOR1", "gnd", "0"),
            ("XOR1", "in1", "A"),
            ("XOR1", "in2", "B"),
            ("XOR1", "out", "SUM"),
            ("AND1", "vcc", "VCC"),
            ("AND1", "gnd", "0"),
            ("AND1", "in1", "A"),
            ("AND1", "in2", "B"),
            ("AND1", "out", "CARRY"),
            ("TPSUM", "net", "SUM"),
            ("TPCARRY", "net", "CARRY"),
        ),
    ),
    _MotifSpec(
        "four_bit_synchronous_binary_counter",
        0.5,
        (
            _plain_part("U1", "counter_4bit", "74HC161"),
            _plain_part("TPQ0", "test_point", "Q0"),
            _plain_part("TPQ1", "test_point", "Q1"),
            _plain_part("TPQ2", "test_point", "Q2"),
            _plain_part("TPQ3", "test_point", "Q3"),
            _plain_part("TPCARRY", "test_point", "CARRY"),
        ),
        (
            ("U1", "vcc", "VCC"),
            ("U1", "gnd", "0"),
            ("U1", "clk", "CLK"),
            ("U1", "clr_n", "RESET_N"),
            ("U1", "load_n", "VCC"),
            ("U1", "enp", "VCC"),
            ("U1", "ent", "VCC"),
            ("U1", "a", "0"),
            ("U1", "b", "0"),
            ("U1", "c", "0"),
            ("U1", "d", "0"),
            ("U1", "qa", "Q0"),
            ("U1", "qb", "Q1"),
            ("U1", "qc", "Q2"),
            ("U1", "qd", "Q3"),
            ("U1", "rco", "CARRY"),
            ("TPQ0", "net", "Q0"),
            ("TPQ1", "net", "Q1"),
            ("TPQ2", "net", "Q2"),
            ("TPQ3", "net", "Q3"),
            ("TPCARRY", "net", "CARRY"),
        ),
    ),
    _MotifSpec(
        "inline_load_current_meter",
        0.7,
        (_battery("B1", "9V", 9.0), _ammeter("AM1"), _res("RLOAD", "1k", 1000.0)),
        (
            ("B1", "p", "VBAT"),
            ("B1", "n", "0"),
            ("AM1", "p", "VBAT"),
            ("AM1", "n", "NLOAD"),
            ("RLOAD", "1", "NLOAD"),
            ("RLOAD", "2", "0"),
        ),
    ),
    _MotifSpec(
        "battery_powered_led_indicator",
        0.8,
        (
            _battery("B1", "9V", 9.0),
            _res("R1", "1k", 1000.0),
            _plain_part("LED1", "led", "LED"),
        ),
        (
            ("B1", "p", "VBAT"),
            ("B1", "n", "0"),
            ("R1", "1", "VBAT"),
            ("R1", "2", "NLED"),
            ("LED1", "a", "NLED"),
            ("LED1", "k", "0"),
        ),
    ),
    _MotifSpec(
        "two_pin_external_input_connector_rc_filter",
        0.7,
        (
            _plain_part("J1", "connector_2", "J2"),
            _res("R1", "1k", 1000.0),
            _cap("C1", "100n", 1e-7),
        ),
        (
            ("J1", "1", "EXT_IN"),
            ("J1", "2", "0"),
            ("R1", "1", "EXT_IN"),
            ("R1", "2", "OUT"),
            ("C1", "1", "OUT"),
            ("C1", "2", "0"),
        ),
    ),
    _MotifSpec(
        "generic_ic_powered_logic_block",
        0.7,
        (
            _plain_part("U1", "generic_ic", "IC"),
            _cap("CDEC", "100n", 1e-7),
            _pullup("RPU", "10k", 10000.0),
            _pulldown("RPD", "10k", 10000.0),
            _res("RLED", "1k", 1000.0),
            _plain_part("LED1", "led", "LED"),
        ),
        (
            ("U1", "vcc", "VCC"),
            ("U1", "gnd", "0"),
            ("CDEC", "1", "VCC"),
            ("CDEC", "2", "0"),
            ("RPU", "net", "IN1"),
            ("RPU", "rail", "VCC"),
            ("RPD", "net", "IN2"),
            ("RPD", "rail", "0"),
            ("U1", "in1", "IN1"),
            ("U1", "in2", "IN2"),
            ("U1", "out1", "OUT"),
            ("RLED", "1", "OUT"),
            ("RLED", "2", "LED_A"),
            ("LED1", "a", "LED_A"),
            ("LED1", "k", "0"),
        ),
    ),
    _MotifSpec(
        "explicit_ground_reference_node",
        0.5,
        (
            _plain_part("GND1", "ground", "0"),
            _cap("CDEC", "100n", 1e-7),
            _res("RLOAD", "1k", 1000.0),
        ),
        (
            ("GND1", "0", "0"),
            ("CDEC", "1", "VCC"),
            ("CDEC", "2", "0"),
            ("RLOAD", "1", "VCC"),
            ("RLOAD", "2", "0"),
        ),
    ),
    _MotifSpec(
        "ideal_switch_power_disconnect",
        0.7,
        (
            _battery("B1", "9V", 9.0),
            _ideal_switch("SW1", 0.1),
            _cap("COUT", "100u", 1e-4),
            _res("RLOAD", "1k", 1000.0),
        ),
        (
            ("B1", "p", "VBAT"),
            ("B1", "n", "0"),
            ("SW1", "1", "VBAT"),
            ("SW1", "2", "VSW"),
            ("COUT", "1", "VSW"),
            ("COUT", "2", "0"),
            ("RLOAD", "1", "VSW"),
            ("RLOAD", "2", "0"),
        ),
    ),
    _MotifSpec(
        "looped_inductor_parallel_resonant_tank",
        0.6,
        (
            _cap("CIN", "10p", 1e-11),
            _part("LLOOP1", "inductor_looped", "1u", {"inductance_h": 1e-6}),
            _cap("CTANK", "100p", 1e-10),
            _res("RLOSS", "100k", 100000.0),
        ),
        (
            ("CIN", "1", "IN"),
            ("CIN", "2", "OUT"),
            ("LLOOP1", "1", "OUT"),
            ("LLOOP1", "2", "0"),
            ("CTANK", "1", "OUT"),
            ("CTANK", "2", "0"),
            ("RLOSS", "1", "OUT"),
            ("RLOSS", "2", "0"),
        ),
    ),
    _MotifSpec(
        "n_jfet_source_follower_buffer",
        0.7,
        (
            _cap("CIN", "1u", 1e-6),
            _res("RG", "1M", 1000000.0),
            _plain_part("J1", "jfet_n", "NJFET"),
            _res("RS", "2.2k", 2200.0),
            _cap("COUT", "1u", 1e-6),
            _res("RLOAD", "10k", 10000.0),
        ),
        (
            ("CIN", "1", "IN"),
            ("CIN", "2", "NG"),
            ("RG", "1", "NG"),
            ("RG", "2", "0"),
            ("J1", "g", "NG"),
            ("J1", "d", "VDD"),
            ("J1", "s", "NS"),
            ("RS", "1", "NS"),
            ("RS", "2", "0"),
            ("COUT", "1", "NS"),
            ("COUT", "2", "OUT"),
            ("RLOAD", "1", "OUT"),
            ("RLOAD", "2", "0"),
        ),
    ),
    _MotifSpec(
        "p_jfet_high_side_current_source",
        0.6,
        (
            _res("RSET", "1k", 1000.0),
            _plain_part("J1", "jfet_p", "PJFET"),
            _res("RLOAD", "1k", 1000.0),
        ),
        (
            ("RSET", "1", "VCC"),
            ("RSET", "2", "NS"),
            ("J1", "s", "NS"),
            ("J1", "g", "VCC"),
            ("J1", "d", "OUT"),
            ("RLOAD", "1", "OUT"),
            ("RLOAD", "2", "0"),
        ),
    ),
    _MotifSpec(
        "led_power_indicator",
        0.8,
        (_res("RLED", "1k", 1000.0), _plain_part("LED1", "led", "LED")),
        (
            ("RLED", "1", "VCC"),
            ("RLED", "2", "NLED"),
            ("LED1", "a", "NLED"),
            ("LED1", "k", "0"),
        ),
    ),
    _MotifSpec(
        "pmos_high_side_load_switch",
        0.7,
        (
            _plain_part("MP1", "mosfet_p", "PMOS"),
            _plain_part("QN1", "bjt_npn", "NPN"),
            _res("RGPU", "100k", 100000.0),
            _res("RB", "10k", 10000.0),
            _cap("COUT", "10u", 1e-5),
            _res("RLOAD", "100", 100.0),
        ),
        (
            ("MP1", "s", "VCC"),
            ("MP1", "d", "VOUT"),
            ("MP1", "g", "NGATE"),
            ("RGPU", "1", "VCC"),
            ("RGPU", "2", "NGATE"),
            ("RB", "1", "CTRL"),
            ("RB", "2", "NB"),
            ("QN1", "b", "NB"),
            ("QN1", "e", "0"),
            ("QN1", "c", "NGATE"),
            ("COUT", "1", "VOUT"),
            ("COUT", "2", "0"),
            ("RLOAD", "1", "VOUT"),
            ("RLOAD", "2", "0"),
        ),
    ),
    _MotifSpec(
        "two_input_or_gate_with_led_output",
        0.7,
        (
            _plain_part("U1", "or_gate", "OR"),
            _pulldown("RPD_A", "10k", 10000.0),
            _pulldown("RPD_B", "10k", 10000.0),
            _pushbutton("SWA", 0.1),
            _pushbutton("SWB", 0.1),
            _res("RLED", "1k", 1000.0),
            _plain_part("LED1", "led", "LED"),
        ),
        (
            ("U1", "vcc", "VCC"),
            ("U1", "gnd", "0"),
            ("RPD_A", "net", "A"),
            ("RPD_A", "rail", "0"),
            ("RPD_B", "net", "B"),
            ("RPD_B", "rail", "0"),
            ("SWA", "1", "VCC"),
            ("SWA", "2", "A"),
            ("SWB", "1", "VCC"),
            ("SWB", "2", "B"),
            ("U1", "in1", "A"),
            ("U1", "in2", "B"),
            ("U1", "out", "Y"),
            ("RLED", "1", "Y"),
            ("RLED", "2", "LED_A"),
            ("LED1", "a", "LED_A"),
            ("LED1", "k", "0"),
        ),
    ),
    _MotifSpec(
        "bulk_polarized_supply_capacitor",
        0.7,
        (
            _part("CPOL1", "polarized_capacitor", "100u", {"capacitance_f": 1e-4}),
            _cap("CFAST", "100n", 1e-7),
            _res("RLOAD", "1k", 1000.0),
        ),
        (
            ("CPOL1", "p", "VCC"),
            ("CPOL1", "n", "0"),
            ("CFAST", "1", "VCC"),
            ("CFAST", "2", "0"),
            ("RLOAD", "1", "VCC"),
            ("RLOAD", "2", "0"),
        ),
    ),
    _MotifSpec(
        "visible_power_rail_distribution",
        0.6,
        (
            _plain_part("PWR1", "power_rail", "VCC"),
            _cap("CDEC", "100n", 1e-7),
            _res("RLOAD", "1k", 1000.0),
            _res("RLED", "1k", 1000.0),
            _plain_part("LED1", "led", "LED"),
        ),
        (
            ("PWR1", "net", "VCC"),
            ("CDEC", "1", "VCC"),
            ("CDEC", "2", "0"),
            ("RLOAD", "1", "VCC"),
            ("RLOAD", "2", "0"),
            ("RLED", "1", "VCC"),
            ("RLED", "2", "LED_A"),
            ("LED1", "a", "LED_A"),
            ("LED1", "k", "0"),
        ),
    ),
    _MotifSpec(
        "default_low_digital_input",
        0.7,
        (
            _pulldown("RPD1", "10k", 10000.0),
            _pushbutton("SW1", 1e12),
            _res("RLED", "1k", 1000.0),
            _plain_part("LED1", "led", "LED"),
        ),
        (
            ("RPD1", "net", "SIG"),
            ("RPD1", "rail", "0"),
            ("SW1", "1", "VCC"),
            ("SW1", "2", "SIG"),
            ("RLED", "1", "SIG"),
            ("RLED", "2", "LED_A"),
            ("LED1", "a", "LED_A"),
            ("LED1", "k", "0"),
        ),
    ),
    _MotifSpec(
        "active_low_reset_pullup",
        0.7,
        (
            _pullup("RPU1", "10k", 10000.0),
            _pushbutton("SW1", 1e12),
            _cap("C1", "100n", 1e-7),
        ),
        (
            ("RPU1", "net", "RESET_N"),
            ("RPU1", "rail", "VCC"),
            ("SW1", "1", "RESET_N"),
            ("SW1", "2", "0"),
            ("C1", "1", "RESET_N"),
            ("C1", "2", "0"),
        ),
    ),
    _MotifSpec(
        "rc_debounced_pushbutton_input",
        0.7,
        (
            _pushbutton("SW1", 0.1),
            _pulldown("RPD", "10k", 10000.0),
            _res("RFILT", "1k", 1000.0),
            _cap("CFILT", "100n", 1e-7),
            _res("RREF_TOP", "10k", 10000.0),
            _res("RREF_BOT", "10k", 10000.0),
            _plain_part("CMP1", "comparator", "CMP"),
        ),
        (
            ("SW1", "1", "VCC"),
            ("SW1", "2", "NBUTTON"),
            ("RPD", "net", "NBUTTON"),
            ("RPD", "rail", "0"),
            ("RFILT", "1", "NBUTTON"),
            ("RFILT", "2", "NFILT"),
            ("CFILT", "1", "NFILT"),
            ("CFILT", "2", "0"),
            ("RREF_TOP", "1", "VCC"),
            ("RREF_TOP", "2", "VREF"),
            ("RREF_BOT", "1", "VREF"),
            ("RREF_BOT", "2", "0"),
            ("CMP1", "noninv", "NFILT"),
            ("CMP1", "inv", "VREF"),
            ("CMP1", "out", "OUT"),
            ("CMP1", "vpos", "VCC"),
            ("CMP1", "vneg", "0"),
        ),
    ),
    _MotifSpec(
        "test_point_on_filtered_signal",
        0.7,
        (
            _res("R1", "1k", 1000.0),
            _cap("C1", "100n", 1e-7),
            _plain_part("TP1", "test_point", "TP"),
        ),
        (
            ("R1", "1", "IN"),
            ("R1", "2", "SENSE"),
            ("C1", "1", "SENSE"),
            ("C1", "2", "0"),
            ("TP1", "net", "SENSE"),
        ),
    ),
    _MotifSpec(
        "variable_cutoff_rc_low_pass",
        0.7,
        (
            _variable_res("RV1", "10k", 10000.0),
            _cap("C1", "100n", 1e-7),
            _res("RLOAD", "10k", 10000.0),
        ),
        (
            ("RV1", "1", "IN"),
            ("RV1", "2", "OUT"),
            ("C1", "1", "OUT"),
            ("C1", "2", "0"),
            ("RLOAD", "1", "OUT"),
            ("RLOAD", "2", "0"),
        ),
    ),
    _MotifSpec(
        "vccs_voltage_to_current_driver",
        0.6,
        (_part("G1", "vccs", "1m", {"gain": 0.001}), _res("RLOAD", "1k", 1000.0)),
        (
            ("G1", "cp", "VCC"),
            ("G1", "cn", "0"),
            ("G1", "p", "NLOAD"),
            ("G1", "n", "0"),
            ("RLOAD", "1", "VCC"),
            ("RLOAD", "2", "NLOAD"),
        ),
    ),
    _MotifSpec(
        "vcvs_ideal_voltage_gain_block",
        0.6,
        (_part("E1", "vcvs", "10", {"gain": 10.0}), _res("RLOAD", "10k", 10000.0)),
        (
            ("E1", "cp", "VCC"),
            ("E1", "cn", "0"),
            ("E1", "p", "OUT"),
            ("E1", "n", "0"),
            ("RLOAD", "1", "OUT"),
            ("RLOAD", "2", "0"),
        ),
    ),
    _MotifSpec(
        "voltage_divider_with_voltmeter",
        0.7,
        (_voltmeter("VM1"), _res("R1", "10k", 10000.0), _res("R2", "10k", 10000.0)),
        (
            ("R1", "1", "VCC"),
            ("R1", "2", "VMID"),
            ("R2", "1", "VMID"),
            ("R2", "2", "0"),
            ("VM1", "p", "VMID"),
            ("VM1", "n", "0"),
        ),
    ),
)

_DEFAULT_MOTIFS = _build_default_motifs()
