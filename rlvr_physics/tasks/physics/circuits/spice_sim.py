"""ngspice-backed simulation for canonical circuits."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from types import MappingProxyType
from typing import Mapping

from rlvr_physics.tasks.physics.circuits.model import (
    Circuit,
    PartSpec,
)
from rlvr_physics.tasks.physics.circuits.spice_export import (
    SpiceAnalysis,
    SpiceAnalysisKind,
    SpiceNetlist,
    export_spice,
)


@dataclass(frozen=True)
class SpiceSimulationSpec:
    """Simulation request used by circuit tasks.

    Parameters
    ----------
    analysis:
        SPICE analysis to execute.
    voltage_sources:
        External operating voltage sources to overlay for this simulation.
    """

    analysis: SpiceAnalysis
    voltage_sources: tuple["SpiceVoltageSource", ...]


@dataclass(frozen=True)
class SpiceVoltageSource:
    """External operating voltage applied during simulation.

    Parameters
    ----------
    name:
        Stable source name used to derive an emitted SPICE reference.
    positive_net:
        Canonical circuit net connected to the source positive terminal.
    negative_net:
        Canonical circuit net connected to the source negative terminal.
    voltage_v:
        Source voltage in volts from positive terminal to negative terminal.
    """

    name: str
    positive_net: str
    negative_net: str
    voltage_v: float


@dataclass(frozen=True)
class SpiceSimulatorConfig:
    """Runtime configuration for the ngspice command-line simulator.

    Parameters
    ----------
    ngspice_command:
        Executable path or command name.
    timeout_s:
        Maximum wall-clock seconds for one simulation.
    """

    ngspice_command: str
    timeout_s: float


@dataclass(frozen=True)
class SpiceSimulationIssue:
    """One structured simulator issue.

    Parameters
    ----------
    code:
        Stable issue code.
    message:
        Public-safe issue message.
    """

    code: str
    message: str


@dataclass(frozen=True)
class SpiceSimulationResult:
    """Structured result from an ngspice simulation.

    Parameters
    ----------
    ok:
        Whether ngspice ran and all node voltages returned finite values.
    analysis:
        Analysis that was executed.
    values:
        Node voltages keyed by canonical circuit net name.
    issues:
        Structured simulator issues.
    stdout:
        Raw ngspice stdout, for debug use.
    stderr:
        Raw ngspice stderr, for debug use.
    netlist:
        Exported netlist and canonical-to-SPICE name maps used for execution.
    """

    ok: bool
    analysis: SpiceAnalysis
    values: Mapping[str, float]
    issues: tuple[SpiceSimulationIssue, ...]
    stdout: str
    stderr: str
    netlist: SpiceNetlist


@dataclass(frozen=True)
class _PlannedNodeVoltage:
    """Internal node-voltage measurement with its ngspice expression."""

    net: str
    expression: str


def default_spice_simulator_config() -> SpiceSimulatorConfig:
    """Return a default ngspice command configuration.

    Returns
    -------
    SpiceSimulatorConfig
        Configuration using ``ngspice`` from ``PATH`` when available.
    """

    command = shutil.which("ngspice") or "ngspice"
    return SpiceSimulatorConfig(ngspice_command=command, timeout_s=20.0)


def simulate_spice(
    circuit: Circuit,
    catalog: Mapping[str, PartSpec],
    spec: SpiceSimulationSpec,
    config: SpiceSimulatorConfig,
) -> SpiceSimulationResult:
    """Run ngspice for one canonical circuit and simulation spec.

    Parameters
    ----------
    circuit:
        Canonical circuit to simulate.
    catalog:
        Component catalog.
    spec:
        Simulation request.
    config:
        Runtime ngspice configuration.

    Returns
    -------
    SpiceSimulationResult
        Structured simulation result.
    """

    _validate_config(config)
    if spec.analysis.kind is not SpiceAnalysisKind.OPERATING_POINT:
        raise ValueError("spice simulation v1 supports operating-point analysis only")
    exported_netlist = export_spice(circuit, catalog, spec.analysis)
    netlist = _netlist_with_operating_conditions(exported_netlist, spec.voltage_sources)
    planned_nodes = _planned_node_voltages(netlist)
    with TemporaryDirectory(prefix="rlvr-spice-") as temp_name:
        temp_dir = Path(temp_name)
        node_voltage_path = temp_dir / "node_voltages.dat"
        netlist_path = temp_dir / "circuit.cir"
        netlist_path.write_text(
            _simulation_netlist_text(netlist.text, node_voltage_path, planned_nodes),
            encoding="utf-8",
        )
        try:
            completed = subprocess.run(
                (config.ngspice_command, "-b", str(netlist_path)),
                check=False,
                capture_output=True,
                text=True,
                timeout=config.timeout_s,
            )
        except FileNotFoundError:
            return _failed_result(
                spec.analysis,
                netlist,
                "ngspice_not_found",
                f"ngspice executable not found: {config.ngspice_command}",
            )
        except subprocess.TimeoutExpired as exc:
            return SpiceSimulationResult(
                ok=False,
                analysis=spec.analysis,
                values=MappingProxyType({}),
                issues=(
                    SpiceSimulationIssue(
                        code="ngspice_timeout",
                        message=(
                            f"ngspice timed out after {config.timeout_s:.12g} seconds"
                        ),
                    ),
                ),
                stdout=_timeout_output_text(exc.stdout),
                stderr=_timeout_output_text(exc.stderr),
                netlist=netlist,
            )
        if completed.returncode != 0:
            return SpiceSimulationResult(
                ok=False,
                analysis=spec.analysis,
                values=MappingProxyType({}),
                issues=(
                    SpiceSimulationIssue(
                        code="ngspice_failed",
                        message=f"ngspice exited with status {completed.returncode}",
                    ),
                ),
                stdout=completed.stdout,
                stderr=completed.stderr,
                netlist=netlist,
            )
        try:
            values = _read_node_voltages(node_voltage_path, netlist, planned_nodes)
        except ValueError as exc:
            return SpiceSimulationResult(
                ok=False,
                analysis=spec.analysis,
                values=MappingProxyType({}),
                issues=(
                    SpiceSimulationIssue(
                        code="node_voltage_parse_failed",
                        message=str(exc),
                    ),
                ),
                stdout=completed.stdout,
                stderr=completed.stderr,
                netlist=netlist,
            )
    return SpiceSimulationResult(
        ok=True,
        analysis=spec.analysis,
        values=values,
        issues=(),
        stdout=completed.stdout,
        stderr=completed.stderr,
        netlist=netlist,
    )


def _failed_result(
    analysis: SpiceAnalysis,
    netlist: SpiceNetlist,
    code: str,
    message: str,
) -> SpiceSimulationResult:
    """Return a failed result without simulator output."""

    return SpiceSimulationResult(
        ok=False,
        analysis=analysis,
        values=MappingProxyType({}),
        issues=(SpiceSimulationIssue(code=code, message=message),),
        stdout="",
        stderr="",
        netlist=netlist,
    )


def _timeout_output_text(value: str | bytes | None) -> str:
    """Return timeout output as text."""

    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _validate_config(config: SpiceSimulatorConfig) -> None:
    """Validate simulator configuration."""

    if not config.ngspice_command.strip():
        raise ValueError("ngspice_command cannot be empty")
    if not isfinite(float(config.timeout_s)) or config.timeout_s <= 0.0:
        raise ValueError("timeout_s must be positive and finite")


def _netlist_with_operating_conditions(
    netlist: SpiceNetlist,
    voltage_sources: tuple[SpiceVoltageSource, ...],
) -> SpiceNetlist:
    """Return netlist text with external operating sources overlaid."""

    source_lines = _external_voltage_source_lines(netlist, voltage_sources)
    if not source_lines:
        return netlist
    return SpiceNetlist(
        text=_insert_before_analysis_card(netlist.text, source_lines),
        analysis=netlist.analysis,
        node_names=netlist.node_names,
        element_refs=netlist.element_refs,
    )


def _external_voltage_source_lines(
    netlist: SpiceNetlist,
    voltage_sources: tuple[SpiceVoltageSource, ...],
) -> tuple[str, ...]:
    """Return SPICE source lines for external operating voltages."""

    used_names = {ref.upper() for ref in netlist.element_refs.values()}
    source_names: set[str] = set()
    lines: list[str] = []
    for source in voltage_sources:
        _validate_voltage_source(source, netlist)
        source_name = source.name.upper()
        if source_name in source_names:
            raise ValueError(f"duplicate voltage source name: {source.name}")
        source_names.add(source_name)
        ref = _unique_source_ref(source.name, used_names)
        positive_node = netlist.node_names[source.positive_net]
        negative_node = netlist.node_names[source.negative_net]
        lines.append(
            f"{ref} {positive_node} {negative_node} DC {_fmt_float(source.voltage_v)}"
        )
    return tuple(lines)


def _validate_voltage_source(source: SpiceVoltageSource, netlist: SpiceNetlist) -> None:
    """Validate one external voltage source against a netlist."""

    if not source.name.strip():
        raise ValueError("voltage source name cannot be empty")
    if source.positive_net not in netlist.node_names:
        raise ValueError(f"unknown voltage source positive net: {source.positive_net}")
    if source.negative_net not in netlist.node_names:
        raise ValueError(f"unknown voltage source negative net: {source.negative_net}")
    if source.positive_net == source.negative_net:
        raise ValueError(
            f"voltage source terminals must use different nets: {source.name}"
        )
    if (
        netlist.node_names[source.positive_net]
        == netlist.node_names[source.negative_net]
    ):
        raise ValueError(
            f"voltage source terminals map to the same SPICE node: {source.name}"
        )
    if not isfinite(float(source.voltage_v)):
        raise ValueError(f"voltage source voltage must be finite: {source.name}")


def _unique_source_ref(name: str, used_names: set[str]) -> str:
    """Return a unique SPICE voltage-source reference."""

    base = f"VOP_{_safe_identifier(name)}"
    candidate = base
    suffix = 2
    while candidate.upper() in used_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_names.add(candidate.upper())
    return candidate


def _safe_identifier(value: str) -> str:
    """Return a SPICE-safe identifier suffix."""

    identifier = "".join(
        char if char.isalnum() or char == "_" else "_" for char in value
    )
    identifier = identifier.strip("_")
    if not identifier:
        return "SOURCE"
    if identifier[0].isdigit():
        return f"SOURCE_{identifier}"
    return identifier


def _fmt_float(value: float) -> str:
    """Return compact floating point text."""

    return f"{value:.12g}"


def _insert_before_analysis_card(text: str, source_lines: tuple[str, ...]) -> str:
    """Insert source lines before the exported analysis card."""

    lines = text.rstrip().splitlines()
    if len(lines) < 2 or lines[-1].lower() != ".end":
        raise ValueError(
            "exported SPICE netlist must end with an analysis card and .end"
        )
    return "\n".join((*lines[:-2], *source_lines, lines[-2], lines[-1])) + "\n"


def _planned_node_voltages(netlist: SpiceNetlist) -> tuple[_PlannedNodeVoltage, ...]:
    """Return all non-ground node-voltage measurements for a netlist."""

    return tuple(
        _PlannedNodeVoltage(net=net, expression=f"v({node})")
        for net, node in netlist.node_names.items()
        if node != "0"
    )


def _simulation_netlist_text(
    exported_text: str,
    node_voltage_path: Path,
    nodes: tuple[_PlannedNodeVoltage, ...],
) -> str:
    """Insert an ngspice control block into exported netlist text."""

    lines = exported_text.rstrip().splitlines()
    if len(lines) < 2 or lines[-1].lower() != ".end":
        raise ValueError(
            "exported SPICE netlist must end with an analysis card and .end"
        )
    control = [
        ".control",
        "set wr_singlescale",
        "set wr_vecnames",
        "op",
    ]
    if nodes:
        expressions = " ".join(node.expression for node in nodes)
        control.append(f"wrdata {node_voltage_path.as_posix()} {expressions}")
    control.append(".endc")
    return "\n".join((*lines[:-1], *control, lines[-1])) + "\n"


def _read_node_voltages(
    node_voltage_path: Path,
    netlist: SpiceNetlist,
    nodes: tuple[_PlannedNodeVoltage, ...],
) -> Mapping[str, float]:
    """Read one-row ngspice ``wrdata`` output."""

    values = {net: 0.0 for net, node in netlist.node_names.items() if node == "0"}
    if not nodes:
        return MappingProxyType(values)
    if not node_voltage_path.exists():
        raise ValueError("ngspice did not write node-voltage output")
    lines = [
        line.strip()
        for line in node_voltage_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) < 2:
        raise ValueError("node-voltage output does not contain a data row")
    numeric_values: list[float] = []
    for token in " ".join(lines[1:]).split():
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(f"non-numeric node-voltage output token: {token}") from exc
        if not isfinite(value):
            raise ValueError(f"non-finite node-voltage output value: {token}")
        numeric_values.append(value)
    expected_count = len(nodes) + 1
    if len(numeric_values) != expected_count:
        raise ValueError(
            f"expected {expected_count} node-voltage output columns, "
            f"found {len(numeric_values)}"
        )
    node_values = numeric_values[-len(nodes) :]
    values.update({node.net: value for node, value in zip(nodes, node_values)})
    return MappingProxyType(values)
