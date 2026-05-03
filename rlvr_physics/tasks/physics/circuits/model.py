"""Canonical circuit data structures and explicit builder API."""

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from rlvr_physics.core.payloads import freeze_mapping, stable_hash, to_plain_data


class PinKind(Enum):
    """Electrical role used by ERC and renderers."""

    INPUT = "input"
    OUTPUT = "output"
    BIDIRECTIONAL = "bidirectional"
    TRISTATE = "tristate"
    PASSIVE = "passive"
    UNSPECIFIED = "unspecified"
    POWER_IN = "power_in"
    POWER_OUT = "power_out"
    OPEN_COLLECTOR = "open_collector"
    OPEN_EMITTER = "open_emitter"
    PULLUP = "pullup"
    PULLDOWN = "pulldown"
    NO_CONNECT = "no_connect"
    FREE = "free"


class ComponentFamily(Enum):
    """High-level component family for layout, ERC, and generation."""

    PASSIVE = "passive"
    SOURCE = "source"
    CONTROLLED_SOURCE = "controlled_source"
    SEMICONDUCTOR = "semiconductor"
    SWITCH = "switch"
    ELECTROMECHANICAL = "electromechanical"
    LOGIC = "logic"
    MEASUREMENT = "measurement"
    CONNECTOR = "connector"
    POWER = "power"
    LOAD = "load"
    INTEGRATED = "integrated"


class AnalysisSupport(Enum):
    """Backend analysis capability advertised by a part specification."""

    SPICE_EXPORT = "spice_export"
    LINEAR_DC = "linear_dc"
    TRANSIENT_EXPORT = "transient_export"


class PinSide(Enum):
    """Preferred visual side for a part pin."""

    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"


@dataclass(frozen=True)
class PinSpec:
    """Static definition of one component pin.

    Parameters
    ----------
    name:
        Stable pin name used in builder connections.
    kind:
        Electrical role used by ERC.
    side:
        Preferred side for schematic layout and SVG drawing.
    """

    name: str
    kind: PinKind
    side: PinSide


@dataclass(frozen=True)
class SpiceSpec:
    """SPICE emission metadata for a part kind.

    Parameters
    ----------
    prefix:
        SPICE element prefix, such as ``R`` or ``V``. Use ``X`` for subcircuits.
    pin_order:
        Pin names in SPICE node order.
    value_parameter:
        Parameter key used for primitive value emission.
    default_value:
        Value emitted when an instance does not provide ``value_parameter``.
    model_name:
        Optional model or subcircuit name emitted after node names.
    model_definition:
        Optional local ``.model`` or ``.subckt`` text.
    """

    prefix: str
    pin_order: tuple[str, ...]
    value_parameter: str | None
    default_value: str
    model_name: str | None
    model_definition: str | None


@dataclass(frozen=True)
class PartSpec:
    """Catalog definition for one reusable component kind.

    Parameters
    ----------
    kind:
        Stable component kind identifier.
    display_name:
        Human-readable component name.
    ref_prefix:
        Conventional reference designator prefix.
    family:
        Component family for layout, ERC, and generation.
    pins:
        Pin definitions accepted by this component kind.
    icon:
        Internal SVG icon identifier.
    spice:
        SPICE export metadata, or ``None`` for drawing-only parts.
    generation_tags:
        Tags used by the procedural generator.
    analysis_support:
        Supported backend analysis capabilities.
    """

    kind: str
    display_name: str
    ref_prefix: str
    family: ComponentFamily
    pins: tuple[PinSpec, ...]
    icon: str
    spice: SpiceSpec | None
    generation_tags: tuple[str, ...]
    analysis_support: tuple[AnalysisSupport, ...]

    @property
    def pin_names(self) -> tuple[str, ...]:
        """Return the valid pin names for this component kind.

        Returns
        -------
        tuple[str, ...]
            Pin names in catalog order.
        """

        return tuple(pin.name for pin in self.pins)

    def pin(self, name: str) -> PinSpec:
        """Return a named pin specification.

        Parameters
        ----------
        name:
            Pin name to look up.

        Returns
        -------
        PinSpec
            Matching pin specification.

        Raises
        ------
        KeyError
            Raised when ``name`` is not defined for this part kind.
        """

        for pin in self.pins:
            if pin.name == name:
                return pin
        raise KeyError(name)


@dataclass(frozen=True)
class PartInstance:
    """One component instance in a canonical circuit.

    Parameters
    ----------
    ref:
        Stable reference designator, such as ``R1``.
    kind:
        Component kind from the catalog.
    value:
        Display/SPICE value string.
    parameters:
        Frozen structured parameters used by solvers, exporters, and tasks.
    metadata:
        Frozen auxiliary metadata.
    """

    ref: str
    kind: str
    value: str
    parameters: Mapping[str, object]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        """Freeze mutable mapping inputs after construction."""

        object.__setattr__(self, "parameters", _freeze_mapping(self.parameters))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True)
class Connection:
    """Connection between one part pin and one circuit net.

    Parameters
    ----------
    ref:
        Part reference designator.
    pin:
        Pin name on the part.
    net:
        Net name.
    """

    ref: str
    pin: str
    net: str


@dataclass(frozen=True)
class Circuit:
    """Immutable canonical circuit topology.

    Parameters
    ----------
    name:
        Stable circuit name.
    parts:
        Component instances in deterministic order.
    nets:
        Net names in deterministic order.
    connections:
        Pin-to-net connections in deterministic order.
    metadata:
        Frozen circuit metadata.
    """

    name: str
    parts: tuple[PartInstance, ...]
    nets: tuple[str, ...]
    connections: tuple[Connection, ...]
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        """Freeze mutable metadata after construction."""

        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def part_by_ref(self) -> Mapping[str, PartInstance]:
        """Return a reference-designator lookup for parts.

        Returns
        -------
        Mapping[str, PartInstance]
            Mapping from part reference to part instance.
        """

        return MappingProxyType({part.ref: part for part in self.parts})

    def connections_for_net(self, net: str) -> tuple[Connection, ...]:
        """Return all connections attached to a net.

        Parameters
        ----------
        net:
            Net name to inspect.

        Returns
        -------
        tuple[Connection, ...]
            Connections attached to ``net``.
        """

        return tuple(
            connection for connection in self.connections if connection.net == net
        )

    def net_for_pin(self, ref: str, pin: str) -> str | None:
        """Return the net attached to a part pin.

        Parameters
        ----------
        ref:
            Part reference designator.
        pin:
            Pin name on the part.

        Returns
        -------
        str | None
            Net name when the pin is connected; otherwise ``None``.
        """

        for connection in self.connections:
            if connection.ref == ref and connection.pin == pin:
                return connection.net
        return None

    def to_plain_data(self) -> dict[str, object]:
        """Return a deterministic plain-data representation.

        Returns
        -------
        dict[str, object]
            JSON-oriented circuit data.
        """

        return {
            "name": self.name,
            "parts": [
                {
                    "ref": part.ref,
                    "kind": part.kind,
                    "value": part.value,
                    "parameters": to_plain_data(part.parameters),
                    "metadata": to_plain_data(part.metadata),
                }
                for part in self.parts
            ],
            "nets": list(self.nets),
            "connections": [
                {"ref": item.ref, "pin": item.pin, "net": item.net}
                for item in self.connections
            ],
            "metadata": to_plain_data(self.metadata),
        }

    def content_hash(self) -> str:
        """Return a stable SHA-256 hash for the circuit content.

        Returns
        -------
        str
            Stable content hash.
        """

        return stable_hash(self.to_plain_data())


class CircuitTopologyError(ValueError):
    """Raised when a circuit cannot be constructed as valid topology."""


class CircuitBuilder:
    """Explicit builder for immutable canonical circuits.

    Parameters
    ----------
    name:
        Stable circuit name.
    part_specs:
        Catalog mapping used to validate part kinds and pins.
    """

    def __init__(self, name: str, part_specs: Mapping[str, PartSpec]) -> None:
        """Initialize an empty builder.

        Parameters
        ----------
        name:
            Stable circuit name.
        part_specs:
            Catalog mapping used to validate part kinds and pins.
        """

        self._name = name
        self._part_specs = part_specs
        self._parts: dict[str, PartInstance] = {}
        self._nets: set[str] = set()
        self._connections: dict[tuple[str, str], Connection] = {}
        self._metadata: dict[str, object] = {}

    def add_part(
        self,
        ref: str,
        kind: str,
        value: str,
        parameters: Mapping[str, object],
        metadata: Mapping[str, object],
    ) -> PartInstance:
        """Add one part instance.

        Parameters
        ----------
        ref:
            Stable reference designator.
        kind:
            Component kind from the catalog.
        value:
            Display/SPICE value string.
        parameters:
            Structured instance parameters.
        metadata:
            Auxiliary metadata.

        Returns
        -------
        PartInstance
            Added part instance.
        """

        if ref in self._parts:
            raise CircuitTopologyError(f"duplicate part reference: {ref}")
        if kind not in self._part_specs:
            raise CircuitTopologyError(f"unknown part kind for {ref}: {kind}")
        part = PartInstance(
            ref=ref,
            kind=kind,
            value=value,
            parameters=parameters,
            metadata=metadata,
        )
        self._parts[ref] = part
        return part

    def add_net(self, net: str) -> None:
        """Add a named net.

        Parameters
        ----------
        net:
            Net name.
        """

        if not net:
            raise CircuitTopologyError("net name cannot be empty")
        self._nets.add(net)

    def connect(self, ref: str, pin: str, net: str) -> None:
        """Connect one part pin to one net.

        Parameters
        ----------
        ref:
            Part reference designator.
        pin:
            Pin name on the part.
        net:
            Net name.
        """

        if ref not in self._parts:
            raise CircuitTopologyError(f"unknown part reference: {ref}")
        spec = self._part_specs[self._parts[ref].kind]
        if pin not in spec.pin_names:
            raise CircuitTopologyError(f"unknown pin {pin!r} on {ref}")
        key = (ref, pin)
        if key in self._connections:
            raise CircuitTopologyError(f"pin already connected: {ref}.{pin}")
        self.add_net(net)
        self._connections[key] = Connection(ref=ref, pin=pin, net=net)

    def set_metadata(self, metadata: Mapping[str, object]) -> None:
        """Replace circuit metadata.

        Parameters
        ----------
        metadata:
            Metadata to store on the frozen circuit.
        """

        self._metadata = dict(metadata)

    def freeze(self) -> Circuit:
        """Freeze the builder into an immutable circuit.

        Returns
        -------
        Circuit
            Immutable canonical circuit.
        """

        connected_nets = {connection.net for connection in self._connections.values()}
        nets = tuple(sorted(self._nets | connected_nets, key=_ground_aware_sort_key))
        return Circuit(
            name=self._name,
            parts=tuple(self._parts[ref] for ref in sorted(self._parts)),
            nets=nets,
            connections=tuple(
                self._connections[key] for key in sorted(self._connections)
            ),
            metadata=self._metadata,
        )


def is_ground_net(net: str) -> bool:
    """Return whether a net is an electrical reference node.

    Parameters
    ----------
    net:
        Net name to inspect.

    Returns
    -------
    bool
        Whether ``net`` is a ground/reference alias.
    """

    return net.upper() in {"0", "GND", "GROUND", "AGND", "DGND"}


def _ground_aware_sort_key(net: str) -> tuple[int, str]:
    """Sort ground aliases before other nets."""

    return (0 if is_ground_net(net) else 1, net)


def _freeze_mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    """Return a recursively immutable mapping."""

    return freeze_mapping(values)
