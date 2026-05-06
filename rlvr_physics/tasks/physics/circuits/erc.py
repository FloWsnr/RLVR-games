"""Electrical rule checking for canonical circuits."""

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Mapping

from rlvr_physics.tasks.physics.circuits.model import (
    AnalysisSupport,
    Circuit,
    Connection,
    PartSpec,
    PinKind,
    is_ground_net,
)


class IssueSeverity(Enum):
    """Severity of one circuit validation issue."""

    ERROR = "error"
    WARNING = "warning"


class DriveStrength(IntEnum):
    """Ordered electrical drive strength used by ERC."""

    NO_CONNECT = 0
    NONE = 1
    PASSIVE = 2
    PULL = 3
    ONE_SIDE = 4
    TRISTATE = 5
    PUSH_PULL = 6
    POWER = 7


@dataclass(frozen=True)
class CheckIssue:
    """One structured ERC issue.

    Parameters
    ----------
    severity:
        Error or warning severity.
    code:
        Stable issue code.
    message:
        Public-safe issue message.
    refs:
        Affected part references.
    nets:
        Affected net names.
    pins:
        Affected ``ref.pin`` names.
    """

    severity: IssueSeverity
    code: str
    message: str
    refs: tuple[str, ...]
    nets: tuple[str, ...]
    pins: tuple[str, ...]


@dataclass(frozen=True)
class CheckReport:
    """Complete ERC report for a circuit.

    Parameters
    ----------
    issues:
        Structured ERC issues.
    """

    issues: tuple[CheckIssue, ...]

    @property
    def errors(self) -> tuple[CheckIssue, ...]:
        """Return all error issues.

        Returns
        -------
        tuple[CheckIssue, ...]
            Issues with ``ERROR`` severity.
        """

        return tuple(
            issue for issue in self.issues if issue.severity is IssueSeverity.ERROR
        )

    @property
    def warnings(self) -> tuple[CheckIssue, ...]:
        """Return all warning issues.

        Returns
        -------
        tuple[CheckIssue, ...]
            Issues with ``WARNING`` severity.
        """

        return tuple(
            issue for issue in self.issues if issue.severity is IssueSeverity.WARNING
        )

    @property
    def is_valid(self) -> bool:
        """Return whether the report contains no errors.

        Returns
        -------
        bool
            ``True`` when no errors were found.
        """

        return not self.errors


@dataclass(frozen=True)
class PinElectricalInfo:
    """ERC drive and receive constraints for one pin kind."""

    drive: DriveStrength
    min_receive: DriveStrength
    max_receive: DriveStrength


PIN_INFO: Mapping[PinKind, PinElectricalInfo] = {
    PinKind.INPUT: PinElectricalInfo(
        DriveStrength.NONE, DriveStrength.PASSIVE, DriveStrength.POWER
    ),
    PinKind.OUTPUT: PinElectricalInfo(
        DriveStrength.PUSH_PULL, DriveStrength.NONE, DriveStrength.PASSIVE
    ),
    PinKind.BIDIRECTIONAL: PinElectricalInfo(
        DriveStrength.TRISTATE, DriveStrength.NONE, DriveStrength.POWER
    ),
    PinKind.TRISTATE: PinElectricalInfo(
        DriveStrength.TRISTATE, DriveStrength.NONE, DriveStrength.TRISTATE
    ),
    PinKind.PASSIVE: PinElectricalInfo(
        DriveStrength.PASSIVE, DriveStrength.NONE, DriveStrength.POWER
    ),
    PinKind.UNSPECIFIED: PinElectricalInfo(
        DriveStrength.NONE, DriveStrength.NONE, DriveStrength.POWER
    ),
    PinKind.POWER_IN: PinElectricalInfo(
        DriveStrength.NONE, DriveStrength.POWER, DriveStrength.POWER
    ),
    PinKind.POWER_OUT: PinElectricalInfo(
        DriveStrength.POWER, DriveStrength.NONE, DriveStrength.PASSIVE
    ),
    PinKind.OPEN_COLLECTOR: PinElectricalInfo(
        DriveStrength.ONE_SIDE, DriveStrength.NONE, DriveStrength.TRISTATE
    ),
    PinKind.OPEN_EMITTER: PinElectricalInfo(
        DriveStrength.ONE_SIDE, DriveStrength.NONE, DriveStrength.TRISTATE
    ),
    PinKind.PULLUP: PinElectricalInfo(
        DriveStrength.PULL, DriveStrength.NONE, DriveStrength.POWER
    ),
    PinKind.PULLDOWN: PinElectricalInfo(
        DriveStrength.PULL, DriveStrength.NONE, DriveStrength.POWER
    ),
    PinKind.NO_CONNECT: PinElectricalInfo(
        DriveStrength.NO_CONNECT,
        DriveStrength.NO_CONNECT,
        DriveStrength.NO_CONNECT,
    ),
    PinKind.FREE: PinElectricalInfo(
        DriveStrength.NONE, DriveStrength.NO_CONNECT, DriveStrength.POWER
    ),
}


def check_circuit(
    circuit: Circuit,
    catalog: Mapping[str, PartSpec],
    analysis_support: AnalysisSupport,
) -> CheckReport:
    """Run electronic rule checks on a circuit.

    Parameters
    ----------
    circuit:
        Circuit to check.
    catalog:
        Component catalog for part and pin metadata.
    analysis_support:
        Analysis mode expected by the caller.

    Returns
    -------
    CheckReport
        Structured ERC report.
    """

    checker = _CircuitChecker(circuit, catalog, analysis_support)
    return checker.run()


class _CircuitChecker:
    """Internal stateful ERC implementation."""

    def __init__(
        self,
        circuit: Circuit,
        catalog: Mapping[str, PartSpec],
        analysis_support: AnalysisSupport,
    ) -> None:
        """Initialize checker state."""

        self.circuit = circuit
        self.catalog = catalog
        self.analysis_support = analysis_support
        self.issues: list[CheckIssue] = []
        self.parts = circuit.part_by_ref()

    def run(self) -> CheckReport:
        """Run all ERC checks."""

        self._check_parts()
        self._check_connections()
        self._check_nets()
        self._check_reference_node()
        self._check_analysis_support()
        return CheckReport(tuple(self.issues))

    def _check_parts(self) -> None:
        """Check duplicate references and unknown component kinds."""

        refs: set[str] = set()
        for part in self.circuit.parts:
            if part.ref in refs:
                self._add(
                    IssueSeverity.ERROR,
                    "duplicate_ref",
                    f"duplicate part reference: {part.ref}",
                    refs=(part.ref,),
                )
            refs.add(part.ref)
            if part.kind not in self.catalog:
                self._add(
                    IssueSeverity.ERROR,
                    "unknown_part_kind",
                    f"unknown part kind on {part.ref}: {part.kind}",
                    refs=(part.ref,),
                )

    def _check_connections(self) -> None:
        """Check each pin connection."""

        seen_pins: set[tuple[str, str]] = set()
        for connection in self.circuit.connections:
            part = self.parts.get(connection.ref)
            if part is None:
                self._add(
                    IssueSeverity.ERROR,
                    "unknown_connection_ref",
                    f"connection references unknown part: {connection.ref}",
                    refs=(connection.ref,),
                    nets=(connection.net,),
                )
                continue
            spec = self.catalog.get(part.kind)
            if spec is None:
                continue
            if connection.pin not in spec.pin_names:
                self._add(
                    IssueSeverity.ERROR,
                    "unknown_pin",
                    f"unknown pin {connection.pin!r} on {connection.ref}",
                    refs=(connection.ref,),
                    pins=(f"{connection.ref}.{connection.pin}",),
                    nets=(connection.net,),
                )
                continue
            pin_key = (connection.ref, connection.pin)
            if pin_key in seen_pins:
                self._add(
                    IssueSeverity.ERROR,
                    "pin_connected_twice",
                    f"pin connected more than once: {connection.ref}.{connection.pin}",
                    refs=(connection.ref,),
                    pins=(f"{connection.ref}.{connection.pin}",),
                )
            seen_pins.add(pin_key)
            pin = spec.pin(connection.pin)
            if pin.kind is PinKind.NO_CONNECT:
                self._add(
                    IssueSeverity.ERROR,
                    "no_connect_pin_connected",
                    f"no-connect pin is connected: {connection.ref}.{connection.pin}",
                    refs=(connection.ref,),
                    pins=(f"{connection.ref}.{connection.pin}",),
                    nets=(connection.net,),
                )

        connected_pins = seen_pins
        for part in self.circuit.parts:
            spec = self.catalog.get(part.kind)
            if spec is None:
                continue
            for pin in spec.pins:
                if pin.kind is PinKind.FREE:
                    continue
                if (part.ref, pin.name) not in connected_pins:
                    severity = (
                        IssueSeverity.WARNING
                        if pin.kind in {PinKind.NO_CONNECT, PinKind.UNSPECIFIED}
                        else IssueSeverity.ERROR
                    )
                    self._add(
                        severity,
                        "unconnected_pin",
                        f"unconnected pin: {part.ref}.{pin.name}",
                        refs=(part.ref,),
                        pins=(f"{part.ref}.{pin.name}",),
                    )

    def _check_nets(self) -> None:
        """Check net fanout, pin conflicts, and drive strength."""

        connected_by_net = {
            net: self.circuit.connections_for_net(net) for net in self.circuit.nets
        }
        for net, connections in connected_by_net.items():
            if len(connections) == 0:
                self._add(
                    IssueSeverity.WARNING,
                    "empty_net",
                    f"no pins attached to net {net}",
                    nets=(net,),
                )
                continue
            if len(connections) == 1:
                self._add(
                    IssueSeverity.WARNING,
                    "single_pin_net",
                    f"only one pin attached to net {net}",
                    nets=(net,),
                    pins=(f"{connections[0].ref}.{connections[0].pin}",),
                )

            pin_kinds = self._pin_kinds(connections)
            self._check_pin_conflicts(net, pin_kinds)
            self._check_net_drive(net, pin_kinds)

    def _check_reference_node(self) -> None:
        """Check that the circuit has an electrical reference node."""

        if not any(is_ground_net(net) for net in self.circuit.nets):
            self._add(
                IssueSeverity.ERROR,
                "missing_reference_node",
                "circuit has no ground/reference net",
            )

    def _check_analysis_support(self) -> None:
        """Check whether parts support the requested analysis mode."""

        for part in self.circuit.parts:
            spec = self.catalog.get(part.kind)
            if spec is None:
                continue
            if self.analysis_support not in spec.analysis_support:
                self._add(
                    IssueSeverity.WARNING,
                    "unsupported_analysis",
                    (
                        f"{part.ref} ({part.kind}) does not support "
                        f"{self.analysis_support.value}"
                    ),
                    refs=(part.ref,),
                )

    def _pin_kinds(
        self, connections: tuple[Connection, ...]
    ) -> tuple[tuple[str, str, PinKind], ...]:
        """Return pin kinds for a set of connections."""

        result: list[tuple[str, str, PinKind]] = []
        for item in connections:
            ref = item.ref
            pin_name = item.pin
            part = self.parts.get(ref)
            if part is None:
                continue
            spec = self.catalog.get(part.kind)
            if spec is None or pin_name not in spec.pin_names:
                continue
            result.append((ref, pin_name, spec.pin(pin_name).kind))
        return tuple(result)

    def _check_pin_conflicts(
        self, net: str, pin_kinds: tuple[tuple[str, str, PinKind], ...]
    ) -> None:
        """Check pairwise pin contention on a net."""

        for idx, (ref_a, pin_a, kind_a) in enumerate(pin_kinds):
            for ref_b, pin_b, kind_b in pin_kinds[idx + 1 :]:
                severity, detail = _pin_conflict(kind_a, kind_b)
                if severity is None:
                    continue
                self._add(
                    severity,
                    "pin_conflict",
                    (
                        f"pin conflict on net {net}: {ref_a}.{pin_a} "
                        f"with {ref_b}.{pin_b}{detail}"
                    ),
                    refs=tuple(sorted((ref_a, ref_b))),
                    pins=(f"{ref_a}.{pin_a}", f"{ref_b}.{pin_b}"),
                    nets=(net,),
                )

    def _check_net_drive(
        self, net: str, pin_kinds: tuple[tuple[str, str, PinKind], ...]
    ) -> None:
        """Check whether connected pins receive enough drive."""

        if is_ground_net(net):
            return
        pin_drives = tuple(
            (ref, pin_name, PIN_INFO[kind].drive) for ref, pin_name, kind in pin_kinds
        )
        net_drive = max(
            (drive for _, _, drive in pin_drives), default=DriveStrength.NONE
        )
        for ref, pin_name, kind in pin_kinds:
            info = PIN_INFO[kind]
            if info.min_receive > net_drive:
                self._add(
                    IssueSeverity.WARNING,
                    "insufficient_drive",
                    f"insufficient drive on net {net} for {ref}.{pin_name}",
                    refs=(ref,),
                    pins=(f"{ref}.{pin_name}",),
                    nets=(net,),
                )
            other_drive = max(
                (
                    drive
                    for other_ref, other_pin, drive in pin_drives
                    if (other_ref, other_pin) != (ref, pin_name)
                ),
                default=DriveStrength.NONE,
            )
            if other_drive > info.max_receive:
                self._add(
                    IssueSeverity.WARNING,
                    "excessive_drive",
                    f"excessive drive on net {net} for {ref}.{pin_name}",
                    refs=(ref,),
                    pins=(f"{ref}.{pin_name}",),
                    nets=(net,),
                )

    def _add(
        self,
        severity: IssueSeverity,
        code: str,
        message: str,
        refs: tuple[str, ...] = (),
        nets: tuple[str, ...] = (),
        pins: tuple[str, ...] = (),
    ) -> None:
        """Append an ERC issue."""

        self.issues.append(
            CheckIssue(
                severity=severity,
                code=code,
                message=message,
                refs=tuple(sorted(refs)),
                nets=tuple(sorted(nets)),
                pins=pins,
            )
        )


def _pin_conflict(first: PinKind, second: PinKind) -> tuple[IssueSeverity | None, str]:
    """Return the conflict status for two pin kinds."""

    pair = frozenset((first, second))
    if first is PinKind.NO_CONNECT or second is PinKind.NO_CONNECT:
        return IssueSeverity.ERROR, ""
    if first is PinKind.OUTPUT and second is PinKind.OUTPUT:
        return IssueSeverity.ERROR, ""
    if pair == frozenset((PinKind.POWER_OUT, PinKind.POWER_OUT)):
        return IssueSeverity.ERROR, ""
    if PinKind.POWER_OUT in pair and (
        PinKind.OUTPUT in pair or PinKind.TRISTATE in pair
    ):
        return IssueSeverity.ERROR, ""
    if pair == frozenset((PinKind.PULLUP, PinKind.PULLDOWN)):
        return IssueSeverity.ERROR, " (pull-up connected to pull-down)"
    if first is PinKind.PULLUP and second is PinKind.PULLUP:
        return IssueSeverity.WARNING, " (multiple pull-ups)"
    if first is PinKind.PULLDOWN and second is PinKind.PULLDOWN:
        return IssueSeverity.WARNING, " (multiple pull-downs)"
    if PinKind.UNSPECIFIED in pair:
        return IssueSeverity.WARNING, " (unspecified pin)"
    if PinKind.OPEN_COLLECTOR in pair and PinKind.OUTPUT in pair:
        return IssueSeverity.ERROR, ""
    if PinKind.OPEN_EMITTER in pair and PinKind.OUTPUT in pair:
        return IssueSeverity.ERROR, ""
    return None, ""
