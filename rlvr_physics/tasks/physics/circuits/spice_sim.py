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
    """

    analysis: SpiceAnalysis


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
    netlist = export_spice(circuit, catalog, spec.analysis)
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
    if not lines or lines[-1].lower() != ".end":
        raise ValueError("exported SPICE netlist must end with .end")
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
