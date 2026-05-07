"""Dependency-free SPICE netlist export for canonical circuits."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite
import re
from types import MappingProxyType
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

_CAPACITOR_LIKE_KINDS = {"capacitor", "polarized_capacitor"}
_CAPACITOR_LEAKAGE_OHM = "1e9"


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
    node_names:
        Mapping from canonical net name to emitted SPICE node name.
    element_refs:
        Mapping from canonical part reference to emitted SPICE element reference.
    """

    text: str
    analysis: SpiceAnalysis
    node_names: Mapping[str, str]
    element_refs: Mapping[str, str]


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
    node_names = _node_names(circuit)
    element_refs = _element_refs(circuit, catalog)
    used_refs = {ref.upper() for ref in element_refs.values()}
    model_lines: dict[str, str] = {}
    for part in circuit.parts:
        spec = catalog[part.kind]
        if spec.kind == "ground":
            continue
        if spec.spice is None:
            raise CircuitTopologyError(
                f"cannot export {part.ref} ({part.kind}) to SPICE"
            )
        element_ref = element_refs[part.ref]
        lines.append(_part_line(circuit, part, spec.spice, node_names, element_ref))
        lines.extend(
            _extra_part_lines(
                circuit,
                part,
                spec,
                node_names,
                element_ref,
                used_refs,
            )
        )
        if spec.spice.model_definition is not None:
            model_lines[spec.spice.model_definition] = spec.spice.model_definition

    lines.extend(sorted(model_lines))
    lines.append(_analysis_card(circuit, catalog, analysis, element_refs))
    lines.append(".end")
    return SpiceNetlist(
        text="\n".join(lines) + "\n",
        analysis=analysis,
        node_names=MappingProxyType(node_names),
        element_refs=MappingProxyType(element_refs),
    )


def _part_line(
    circuit: Circuit,
    part: PartInstance,
    spice: SpiceSpec,
    node_names: Mapping[str, str],
    element_ref: str,
) -> str:
    """Return one SPICE element line."""

    nodes = [
        _node_for_pin(circuit, part.ref, pin, node_names) for pin in spice.pin_order
    ]
    pieces = [element_ref, *nodes]
    if spice.model_name is not None:
        pieces.append(spice.model_name)
    value = _spice_value(part, spice)
    if value:
        pieces.append(value)
    return " ".join(pieces)


def _extra_part_lines(
    circuit: Circuit,
    part: PartInstance,
    spec: PartSpec,
    node_names: Mapping[str, str],
    element_ref: str,
    used_refs: set[str],
) -> tuple[str, ...]:
    """Return auxiliary SPICE lines needed for a part's executable model."""

    if spec.kind not in _CAPACITOR_LIKE_KINDS or spec.spice is None:
        return ()
    first_node = _node_for_pin(circuit, part.ref, spec.spice.pin_order[0], node_names)
    second_node = _node_for_pin(circuit, part.ref, spec.spice.pin_order[1], node_names)
    leakage_ref = _unique_extra_ref(f"R{element_ref}_LEAK", used_refs)
    return (f"{leakage_ref} {first_node} {second_node} {_CAPACITOR_LEAKAGE_OHM}",)


def _unique_extra_ref(candidate: str, used_refs: set[str]) -> str:
    """Return a unique legal auxiliary SPICE reference."""

    legal_candidate = _legalize(candidate)
    if not legal_candidate.upper().startswith("R"):
        legal_candidate = f"R{legal_candidate}"
    ref = legal_candidate
    suffix = 2
    while ref.upper() in used_refs:
        ref = f"{legal_candidate}_{suffix}"
        suffix += 1
    used_refs.add(ref.upper())
    return ref


def _node_for_pin(
    circuit: Circuit, ref: str, pin: str, node_names: Mapping[str, str]
) -> str:
    """Return SPICE node name for a connected pin."""

    net = circuit.net_for_pin(ref, pin)
    if net is None:
        raise CircuitTopologyError(f"cannot export unconnected pin: {ref}.{pin}")
    return node_names[net]


def _node_names(circuit: Circuit) -> dict[str, str]:
    """Return emitted SPICE node names keyed by canonical circuit net."""

    names = {net: "0" for net in circuit.nets if is_ground_net(net)}
    non_ground_nets = tuple(net for net in circuit.nets if not is_ground_net(net))
    names.update(_unique_legal_names(non_ground_nets, reserved=("0",)))
    return names


def _element_refs(
    circuit: Circuit, catalog: Mapping[str, PartSpec]
) -> Mapping[str, str]:
    """Return unique emitted SPICE references keyed by canonical part reference."""

    candidates: list[tuple[str, str]] = []
    for part in circuit.parts:
        spec = catalog[part.kind]
        if spec.kind == "ground" or spec.spice is None:
            continue
        candidates.append((part.ref, _spice_ref(part.ref, spec.spice.prefix)))
    return MappingProxyType(_unique_candidate_names(tuple(candidates)))


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
    element_refs: Mapping[str, str],
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
        spice_ref = _dc_sweep_spice_ref(circuit, catalog, source_ref, element_refs)
        return (
            f".dc {spice_ref} {_fmt_float(start)} {_fmt_float(stop)} {_fmt_float(step)}"
        )
    step, stop_time = _validate_transient_values(analysis.step, analysis.stop_time)
    return f".tran {_fmt_float(step)} {_fmt_float(stop_time)}"


def _dc_sweep_spice_ref(
    circuit: Circuit,
    catalog: Mapping[str, PartSpec],
    source_ref: str,
    element_refs: Mapping[str, str],
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
    return element_refs[part.ref]


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

    legal = re.sub(r"\W", "_", name)
    if legal:
        return legal
    return "_"


def _unique_legal_names(
    names: tuple[str, ...], *, reserved: tuple[str, ...] = ()
) -> dict[str, str]:
    """Return unique legal names keyed by canonical names."""

    return _unique_candidate_names(
        tuple((name, _legalize(name)) for name in names), reserved
    )


def _unique_candidate_names(
    candidates: tuple[tuple[str, str], ...], reserved: tuple[str, ...] = ()
) -> dict[str, str]:
    """Return unique emitted names from precomputed legal candidates."""

    used = {name.upper() for name in reserved}
    result: dict[str, str] = {}
    for canonical, base in candidates:
        candidate = base
        suffix = 2
        while candidate.upper() in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate.upper())
        result[canonical] = candidate
    return result
