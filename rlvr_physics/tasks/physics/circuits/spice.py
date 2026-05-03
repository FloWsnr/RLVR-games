"""Dependency-free SPICE netlist export for canonical circuits."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite
import re
from typing import Mapping

from rlvr_physics.tasks.physics.circuits.model import (
    Circuit,
    CircuitTopologyError,
    ComponentFamily,
    PartInstance,
    PartSpec,
    SpiceSpec,
    is_ground_net,
)


class SpiceAnalysisKind(Enum):
    """Supported exported SPICE analysis cards."""

    OPERATING_POINT = "op"
    DC_SWEEP = "dc"
    TRANSIENT = "tran"


@dataclass(frozen=True)
class SpiceAnalysis:
    """SPICE analysis card request.

    Parameters
    ----------
    kind:
        Analysis card kind.
    source_ref:
        Source reference for DC sweeps.
    start:
        Start value for sweeps.
    stop:
        Stop value for sweeps.
    step:
        Step value for sweeps or transient time step.
    stop_time:
        Stop time for transient analysis.
    """

    kind: SpiceAnalysisKind
    source_ref: str | None
    start: float | None
    stop: float | None
    step: float | None
    stop_time: float | None


@dataclass(frozen=True)
class SpiceNetlist:
    """Exported SPICE netlist.

    Parameters
    ----------
    text:
        Full SPICE netlist text.
    analysis:
        Analysis card used in the export.
    """

    text: str
    analysis: SpiceAnalysis


def operating_point_analysis() -> SpiceAnalysis:
    """Return an operating-point analysis request.

    Returns
    -------
    SpiceAnalysis
        Operating-point request.
    """

    return SpiceAnalysis(
        kind=SpiceAnalysisKind.OPERATING_POINT,
        source_ref=None,
        start=None,
        stop=None,
        step=None,
        stop_time=None,
    )


def dc_sweep_analysis(
    source_ref: str, start: float, stop: float, step: float
) -> SpiceAnalysis:
    """Return a DC sweep analysis request.

    Parameters
    ----------
    source_ref:
        Source reference to sweep.
    start:
        Sweep start value.
    stop:
        Sweep stop value.
    step:
        Sweep step value.

    Returns
    -------
    SpiceAnalysis
        DC sweep request.
    """

    source_ref, start, stop, step = _validate_dc_sweep_values(
        source_ref, start, stop, step
    )
    return SpiceAnalysis(
        kind=SpiceAnalysisKind.DC_SWEEP,
        source_ref=source_ref,
        start=start,
        stop=stop,
        step=step,
        stop_time=None,
    )


def transient_analysis(step: float, stop_time: float) -> SpiceAnalysis:
    """Return a transient export request.

    Parameters
    ----------
    step:
        Transient output step.
    stop_time:
        Transient stop time.

    Returns
    -------
    SpiceAnalysis
        Transient request.
    """

    step, stop_time = _validate_transient_values(step, stop_time)
    return SpiceAnalysis(
        kind=SpiceAnalysisKind.TRANSIENT,
        source_ref=None,
        start=None,
        stop=None,
        step=step,
        stop_time=stop_time,
    )


def export_spice(
    circuit: Circuit,
    catalog: Mapping[str, PartSpec],
    analysis: SpiceAnalysis,
) -> SpiceNetlist:
    """Export a circuit to deterministic SPICE text.

    Parameters
    ----------
    circuit:
        Circuit to export.
    catalog:
        Component catalog.
    analysis:
        Analysis card to append.

    Returns
    -------
    SpiceNetlist
        Full netlist text and analysis metadata.
    """

    lines = [f"* RLVR-physics circuit: {circuit.name}"]
    model_lines: dict[str, str] = {}
    for part in circuit.parts:
        spec = catalog[part.kind]
        if spec.kind == "ground":
            continue
        if spec.spice is None:
            raise CircuitTopologyError(
                f"cannot export {part.ref} ({part.kind}) to SPICE"
            )
        lines.append(_part_line(circuit, part, spec.spice))
        if spec.spice.model_definition is not None:
            model_lines[spec.spice.model_definition] = spec.spice.model_definition

    lines.extend(sorted(model_lines))
    lines.append(_analysis_card(circuit, catalog, analysis))
    lines.append(".end")
    return SpiceNetlist(text="\n".join(lines) + "\n", analysis=analysis)


def _part_line(circuit: Circuit, part: PartInstance, spice: SpiceSpec) -> str:
    """Return one SPICE element line."""

    nodes = [_node_for_pin(circuit, part.ref, pin) for pin in spice.pin_order]
    ref = _spice_ref(part.ref, spice.prefix)
    pieces = [ref, *nodes]
    if spice.model_name is not None:
        pieces.append(spice.model_name)
    value = _spice_value(part, spice)
    if value:
        pieces.append(value)
    return " ".join(pieces)


def _node_for_pin(circuit: Circuit, ref: str, pin: str) -> str:
    """Return SPICE node name for a connected pin."""

    net = circuit.net_for_pin(ref, pin)
    if net is None:
        raise CircuitTopologyError(f"cannot export unconnected pin: {ref}.{pin}")
    if is_ground_net(net):
        return "0"
    return _legalize(net)


def _spice_ref(ref: str, prefix: str) -> str:
    """Return a legal SPICE reference designator."""

    legal = _legalize(ref)
    if legal.upper().startswith(prefix.upper()):
        return legal
    return f"{prefix}{legal}"


def _spice_value(part: PartInstance, spice: SpiceSpec) -> str:
    """Return an emitted SPICE value string."""

    if spice.value_parameter is None:
        return spice.default_value
    if spice.value_parameter in part.parameters:
        value = part.parameters[spice.value_parameter]
        if spice.prefix in {"V", "I"} and isinstance(value, int | float):
            return f"DC {_fmt_float(float(value))}"
        return _fmt_parameter(value)
    if part.value:
        return part.value
    return spice.default_value


def _analysis_card(
    circuit: Circuit,
    catalog: Mapping[str, PartSpec],
    analysis: SpiceAnalysis,
) -> str:
    """Return the SPICE analysis card."""

    if analysis.kind is SpiceAnalysisKind.OPERATING_POINT:
        return ".op"
    if analysis.kind is SpiceAnalysisKind.DC_SWEEP:
        source_ref, start, stop, step = _validate_dc_sweep_values(
            analysis.source_ref,
            analysis.start,
            analysis.stop,
            analysis.step,
        )
        spice_ref = _dc_sweep_spice_ref(circuit, catalog, source_ref)
        return (
            f".dc {spice_ref} {_fmt_float(start)} {_fmt_float(stop)} {_fmt_float(step)}"
        )
    step, stop_time = _validate_transient_values(analysis.step, analysis.stop_time)
    return f".tran {_fmt_float(step)} {_fmt_float(stop_time)}"


def _dc_sweep_spice_ref(
    circuit: Circuit, catalog: Mapping[str, PartSpec], source_ref: str
) -> str:
    """Return the emitted SPICE ref for a swept independent source."""

    part = circuit.part_by_ref().get(source_ref)
    if part is None:
        raise CircuitTopologyError(f"dc sweep source not found: {source_ref}")
    spec = catalog[part.kind]
    if spec.spice is None:
        raise CircuitTopologyError(
            f"dc sweep source has no SPICE representation: {source_ref}"
        )
    if spec.family is not ComponentFamily.SOURCE or spec.spice.prefix.upper() not in {
        "I",
        "V",
    }:
        raise ValueError(
            f"dc sweep source must be an independent voltage or current source: "
            f"{source_ref}"
        )
    return _spice_ref(part.ref, spec.spice.prefix)


def _validate_dc_sweep_values(
    source_ref: str | None,
    start: float | None,
    stop: float | None,
    step: float | None,
) -> tuple[str, float, float, float]:
    """Validate and normalize DC sweep values."""

    if source_ref is None or not source_ref.strip():
        raise ValueError("dc sweep requires a non-empty source_ref")
    start_value = _require_finite("dc sweep start", start)
    stop_value = _require_finite("dc sweep stop", stop)
    step_value = _require_positive("dc sweep step", step)
    if stop_value < start_value:
        raise ValueError("dc sweep stop must be greater than or equal to start")
    return source_ref.strip(), start_value, stop_value, step_value


def _validate_transient_values(
    step: float | None, stop_time: float | None
) -> tuple[float, float]:
    """Validate and normalize transient analysis values."""

    step_value = _require_positive("transient step", step)
    stop_value = _require_positive("transient stop_time", stop_time)
    if stop_value < step_value:
        raise ValueError("transient stop_time must be greater than or equal to step")
    return step_value, stop_value


def _require_finite(name: str, value: float | None) -> float:
    """Return a finite float or raise ``ValueError``."""

    if value is None:
        raise ValueError(f"{name} is required")
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _require_positive(name: str, value: float | None) -> float:
    """Return a positive finite float or raise ``ValueError``."""

    numeric = _require_finite(name, value)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be positive")
    return numeric


def _fmt_parameter(value: object) -> str:
    """Return a compact SPICE parameter value."""

    if isinstance(value, float):
        return _fmt_float(value)
    if isinstance(value, int):
        return str(value)
    return str(value)


def _fmt_float(value: float) -> str:
    """Return compact floating point text."""

    return f"{value:.12g}"


def _legalize(name: str) -> str:
    """Return a SPICE-safe identifier."""

    return re.sub(r"\W", "_", name)
