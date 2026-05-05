"""Reusable procedural circuit motif catalog."""

from dataclasses import dataclass
from random import Random
from types import MappingProxyType
from typing import Callable, Mapping, Protocol

from rlvr_physics.tasks.physics.circuits.model import CircuitBuilder


class MotifContext(Protocol):
    """Builder-facing context required by procedural motifs.

    Attributes
    ----------
    builder:
        Circuit builder that receives new parts and connections.
    rng:
        Deterministic random number generator owned by the generation run.
    """

    builder: CircuitBuilder
    rng: Random

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


MotifBuilder = Callable[[MotifContext, int], bool]
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
    build:
        Motif builder function.
    """

    name: str
    element_count: int
    default_weight: float
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
    remaining: int,
) -> CircuitMotif | None:
    """Choose a motif that fits within an element budget.

    Parameters
    ----------
    rng:
        Deterministic random number generator.
    motif_catalog:
        Available motif definitions.
    weights:
        Relative motif weights keyed by motif name.
    remaining:
        Remaining non-ground element budget.

    Returns
    -------
    CircuitMotif | None
        Chosen motif, or ``None`` when no weighted motif fits.
    """

    viable = [
        (motif, weight)
        for name, motif in motif_catalog.items()
        for weight in (weights.get(name, 0.0),)
        if weight > 0.0 and motif.element_count <= remaining
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


def add_load_resistor(ctx: MotifContext) -> None:
    """Add one fallback load resistor.

    Parameters
    ----------
    ctx:
        Mutable generation context.
    """

    resistor = ctx.add_part(
        "R",
        "resistor",
        "1k",
        {"resistance_ohm": 1000.0},
        {},
    )
    ctx.builder.connect(resistor, "1", "VCC")
    ctx.builder.connect(resistor, "2", "0")


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


def _build_default_motifs() -> Mapping[str, CircuitMotif]:
    """Build the immutable built-in motif catalog."""

    motifs = {
        spec.name: CircuitMotif(
            spec.name,
            _element_count(spec),
            spec.default_weight,
            _build_netlist_motif(spec),
        )
        for spec in _DEFAULT_MOTIF_SPECS
    }
    return MappingProxyType(motifs)


def _build_netlist_motif(spec: _MotifSpec) -> MotifBuilder:
    """Return a builder for one declarative motif specification."""

    def build(ctx: MotifContext, remaining: int) -> bool:
        """Build one declarative motif into ``ctx``."""

        if remaining < _element_count(spec):
            return False
        ref_map: dict[str, str] = {}
        net_map: dict[str, str] = {}
        if _uses_net(spec, "VEE"):
            negative_rail = _local_net(ctx, "VEE", net_map)
            negative = ctx.add_part(
                "VEE",
                "voltage_source_dc",
                "-5V",
                {"voltage_v": -5.0},
                {"role": "negative_supply", "motif": spec.name},
            )
            ctx.builder.connect(negative, "p", negative_rail)
            ctx.builder.connect(negative, "n", "0")
        for part in spec.parts:
            ref_map[part.ref] = ctx.add_part(
                _reference_prefix(part.ref),
                part.kind,
                part.value,
                part.parameters,
                {"motif": spec.name, "source_ref": part.ref},
            )
        for ref, pin, net in spec.connections:
            ctx.builder.connect(ref_map[ref], pin, _local_net(ctx, net, net_map))
        return True

    return build


def _element_count(spec: _MotifSpec) -> int:
    """Return the non-ground part count added by ``spec``."""

    return len(spec.parts) + (1 if _uses_net(spec, "VEE") else 0)


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


def _local_net(ctx: MotifContext, net: str, net_map: dict[str, str]) -> str:
    """Return a build-local net name, preserving only global rails."""

    normalized = _normalize_net(net)
    if normalized in {"0", "VCC"}:
        return normalized
    if normalized not in net_map:
        net_map[normalized] = f"{normalized}_{ctx.node()}"
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
        "series_pass_transistor_regulator",
        0.8,
        (
            _dc_source("VIN", "12V", 12.0),
            _res("RZ", "1k", 1000.0),
            _plain_part("DZ1", "zener", "5V1"),
            _plain_part("Q1", "bjt_npn", "NPN"),
            _cap("COUT", "10u", 1e-5),
            _res("RLOAD", "1k", 1000.0),
        ),
        (
            ("VIN", "p", "VIN"),
            ("VIN", "n", "0"),
            ("RZ", "1", "VIN"),
            ("RZ", "2", "NZ"),
            ("DZ1", "k", "NZ"),
            ("DZ1", "a", "0"),
            ("Q1", "c", "VIN"),
            ("Q1", "b", "NZ"),
            ("Q1", "e", "OUT"),
            ("COUT", "1", "OUT"),
            ("COUT", "2", "0"),
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
        "class_ab_push_pull_power_amplifier",
        0.6,
        (
            _cap("CIN", "1u", 1e-6),
            _res("RBIAS1", "10k", 10000.0),
            _plain_part("D1", "diode", "D"),
            _plain_part("D2", "diode", "D"),
            _res("RBIAS2", "10k", 10000.0),
            _plain_part("QN", "bjt_npn", "NPN"),
            _plain_part("QP", "bjt_pnp", "PNP"),
            _res("RLOAD", "8", 8.0),
        ),
        (
            ("CIN", "1", "IN"),
            ("CIN", "2", "NDRV"),
            ("RBIAS1", "1", "VCC"),
            ("RBIAS1", "2", "NBH"),
            ("D1", "a", "NBH"),
            ("D1", "k", "NDRV"),
            ("D2", "a", "NDRV"),
            ("D2", "k", "NBL"),
            ("RBIAS2", "1", "NBL"),
            ("RBIAS2", "2", "VEE"),
            ("QN", "b", "NBH"),
            ("QN", "c", "VCC"),
            ("QN", "e", "OUT"),
            ("QP", "b", "NBL"),
            ("QP", "c", "VEE"),
            ("QP", "e", "OUT"),
            ("RLOAD", "1", "OUT"),
            ("RLOAD", "2", "0"),
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
            _part("LOAD", "lamp", "lamp", {"resistance_ohm": 100.0}),
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
            ("LOAD", "1", "NCONTACT"),
            ("LOAD", "2", "0"),
        ),
    ),
    _MotifSpec(
        "nmos_low_side_pwm_driver",
        0.8,
        (
            _res("RG", "100", 100.0),
            _res("RPD", "100k", 100000.0),
            _plain_part("M1", "mosfet_n", "NMOS"),
            _part("LOAD", "motor", "motor", {"resistance_ohm": 25.0}),
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
            ("LOAD", "1", "VCC"),
            ("LOAD", "2", "NLOAD"),
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
        ),
    ),
    _MotifSpec(
        "four_bit_synchronous_binary_counter",
        0.5,
        (_plain_part("U1", "counter_4bit", "74HC161"),),
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
        ),
    ),
)

_DEFAULT_MOTIFS = _build_default_motifs()
