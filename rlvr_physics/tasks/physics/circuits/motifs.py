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
        for name, weight in weights.items()
        if weight > 0.0
        for motif in (motif_catalog.get(name),)
        if motif is not None and motif.element_count <= remaining
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

    resistor = _resistor(ctx)
    ctx.builder.connect(resistor, "1", "VCC")
    ctx.builder.connect(resistor, "2", "0")


def _build_default_motifs() -> Mapping[str, CircuitMotif]:
    """Build the immutable built-in motif catalog."""

    motifs = {
        "divider": CircuitMotif("divider", 2, 1.0, _add_divider),
        "rc_lowpass": CircuitMotif("rc_lowpass", 2, 1.0, _add_rc_lowpass),
        "led_indicator": CircuitMotif("led_indicator", 2, 1.0, _add_led_indicator),
        "pullup_switch": CircuitMotif("pullup_switch", 2, 1.0, _add_pullup_switch),
        "pulldown_switch": CircuitMotif(
            "pulldown_switch", 2, 0.8, _add_pulldown_switch
        ),
        "bridge": CircuitMotif("bridge", 4, 0.6, _add_bridge),
        "transistor_switch": CircuitMotif(
            "transistor_switch", 3, 0.6, _add_transistor_switch
        ),
        "jfet_bias": CircuitMotif("jfet_bias", 3, 0.4, _add_jfet_bias),
        "controlled_gain": CircuitMotif(
            "controlled_gain", 2, 0.4, _add_controlled_gain
        ),
        "opamp_buffer": CircuitMotif("opamp_buffer", 3, 0.4, _add_opamp_buffer),
        "comparator_threshold": CircuitMotif(
            "comparator_threshold", 5, 0.3, _add_comparator_threshold
        ),
    }
    return MappingProxyType(motifs)


def _add_divider(ctx: MotifContext, remaining: int) -> bool:
    """Add a voltage divider motif."""

    if remaining < 2:
        return False
    mid = ctx.node()
    r1 = _resistor(ctx)
    r2 = _resistor(ctx)
    ctx.builder.connect(r1, "1", "VCC")
    ctx.builder.connect(r1, "2", mid)
    ctx.builder.connect(r2, "1", mid)
    ctx.builder.connect(r2, "2", "0")
    return True


def _add_rc_lowpass(ctx: MotifContext, remaining: int) -> bool:
    """Add an RC low-pass motif."""

    if remaining < 2:
        return False
    mid = ctx.node()
    r1 = _resistor(ctx)
    c1 = ctx.add_part("C", "capacitor", "1u", {"capacitance_f": 1e-6}, {})
    ctx.builder.connect(r1, "1", "VCC")
    ctx.builder.connect(r1, "2", mid)
    ctx.builder.connect(c1, "1", mid)
    ctx.builder.connect(c1, "2", "0")
    return True


def _add_led_indicator(ctx: MotifContext, remaining: int) -> bool:
    """Add a resistor and LED indicator motif."""

    if remaining < 2:
        return False
    node = ctx.node()
    resistor = _resistor(ctx)
    led = ctx.add_part("D", "led", "LED", {}, {})
    ctx.builder.connect(resistor, "1", "VCC")
    ctx.builder.connect(resistor, "2", node)
    ctx.builder.connect(led, "a", node)
    ctx.builder.connect(led, "k", "0")
    return True


def _add_pullup_switch(ctx: MotifContext, remaining: int) -> bool:
    """Add a pull-up resistor and switch motif."""

    if remaining < 2:
        return False
    node = ctx.node()
    resistor = ctx.add_part(
        "RPU",
        "pullup_resistor",
        "10k",
        {"resistance_ohm": 10000.0},
        {},
    )
    switch = ctx.add_part(
        "S",
        "ideal_switch",
        "open",
        {"state_resistance_ohm": 1e12},
        {"state": "open"},
    )
    ctx.builder.connect(resistor, "net", node)
    ctx.builder.connect(resistor, "rail", "VCC")
    ctx.builder.connect(switch, "1", node)
    ctx.builder.connect(switch, "2", "0")
    return True


def _add_pulldown_switch(ctx: MotifContext, remaining: int) -> bool:
    """Add a pull-down resistor and switch motif."""

    if remaining < 2:
        return False
    node = ctx.node()
    resistor = ctx.add_part(
        "RPD",
        "pulldown_resistor",
        "10k",
        {"resistance_ohm": 10000.0},
        {},
    )
    switch = ctx.add_part(
        "S",
        "ideal_switch",
        "open",
        {"state_resistance_ohm": 1e12},
        {"state": "open"},
    )
    ctx.builder.connect(resistor, "net", node)
    ctx.builder.connect(resistor, "rail", "0")
    ctx.builder.connect(switch, "1", "VCC")
    ctx.builder.connect(switch, "2", node)
    return True


def _add_bridge(ctx: MotifContext, remaining: int) -> bool:
    """Add a four-resistor bridge motif."""

    if remaining < 4:
        return False
    left = ctx.node()
    right = ctx.node()
    for net_a, net_b in (("VCC", left), (left, "0"), ("VCC", right), (right, "0")):
        resistor = _resistor(ctx)
        ctx.builder.connect(resistor, "1", net_a)
        ctx.builder.connect(resistor, "2", net_b)
    return True


def _add_transistor_switch(ctx: MotifContext, remaining: int) -> bool:
    """Add a simple NPN low-side transistor switch motif."""

    if remaining < 3:
        return False
    collector = ctx.node()
    base = ctx.node()
    load = _resistor(ctx)
    bias = _resistor(ctx)
    transistor = ctx.add_part("Q", "bjt_npn", "NPN", {}, {})
    ctx.builder.connect(load, "1", "VCC")
    ctx.builder.connect(load, "2", collector)
    ctx.builder.connect(bias, "1", "VCC")
    ctx.builder.connect(bias, "2", base)
    ctx.builder.connect(transistor, "c", collector)
    ctx.builder.connect(transistor, "b", base)
    ctx.builder.connect(transistor, "e", "0")
    return True


def _add_jfet_bias(ctx: MotifContext, remaining: int) -> bool:
    """Add a JFET with a drain load and gate bias."""

    if remaining < 3:
        return False
    drain = ctx.node()
    gate = ctx.node()
    load = _resistor(ctx)
    jfet = ctx.add_part("J", "jfet_n", "NJFET", {}, {})
    ctx.builder.connect(load, "1", "VCC")
    ctx.builder.connect(load, "2", drain)
    ctx.builder.connect(jfet, "d", drain)
    ctx.builder.connect(jfet, "g", gate)
    ctx.builder.connect(jfet, "s", "0")
    bias = ctx.add_part(
        "RPD",
        "pulldown_resistor",
        "1M",
        {"resistance_ohm": 1_000_000.0},
        {},
    )
    ctx.builder.connect(bias, "net", gate)
    ctx.builder.connect(bias, "rail", "0")
    return True


def _add_controlled_gain(ctx: MotifContext, remaining: int) -> bool:
    """Add a voltage-controlled source with a load."""

    if remaining < 2:
        return False
    out = ctx.node()
    source = ctx.add_part("E", "vcvs", "gain=2", {"gain": 2.0}, {})
    load = _resistor(ctx)
    ctx.builder.connect(source, "p", out)
    ctx.builder.connect(source, "n", "0")
    ctx.builder.connect(source, "cp", "VCC")
    ctx.builder.connect(source, "cn", "0")
    ctx.builder.connect(load, "1", out)
    ctx.builder.connect(load, "2", "0")
    return True


def _add_opamp_buffer(ctx: MotifContext, remaining: int) -> bool:
    """Add a minimal op-amp buffer motif."""

    if remaining < 3:
        return False
    input_net = ctx.node()
    output_net = ctx.node()
    source = _resistor(ctx)
    load = _resistor(ctx)
    opamp = ctx.add_part("U", "op_amp", "ideal", {}, {})
    ctx.builder.connect(source, "1", "VCC")
    ctx.builder.connect(source, "2", input_net)
    ctx.builder.connect(load, "1", output_net)
    ctx.builder.connect(load, "2", "0")
    ctx.builder.connect(opamp, "noninv", input_net)
    ctx.builder.connect(opamp, "inv", output_net)
    ctx.builder.connect(opamp, "vpos", "VCC")
    ctx.builder.connect(opamp, "vneg", "0")
    ctx.builder.connect(opamp, "out", output_net)
    return True


def _add_comparator_threshold(ctx: MotifContext, remaining: int) -> bool:
    """Add a comparator threshold motif."""

    if remaining < 5:
        return False
    sense = ctx.node()
    threshold = ctx.node()
    output = ctx.node()
    sense_resistor = _resistor(ctx)
    threshold_top = _resistor(ctx)
    threshold_bottom = _resistor(ctx)
    comparator = ctx.add_part("U", "comparator", "cmp", {}, {})
    ctx.builder.connect(sense_resistor, "1", "VCC")
    ctx.builder.connect(sense_resistor, "2", sense)
    ctx.builder.connect(threshold_top, "1", "VCC")
    ctx.builder.connect(threshold_top, "2", threshold)
    ctx.builder.connect(threshold_bottom, "1", threshold)
    ctx.builder.connect(threshold_bottom, "2", "0")
    ctx.builder.connect(comparator, "noninv", sense)
    ctx.builder.connect(comparator, "inv", threshold)
    ctx.builder.connect(comparator, "vpos", "VCC")
    ctx.builder.connect(comparator, "vneg", "0")
    ctx.builder.connect(comparator, "out", output)
    load = _resistor(ctx)
    ctx.builder.connect(load, "1", output)
    ctx.builder.connect(load, "2", "0")
    return True


def _resistor(ctx: MotifContext) -> str:
    """Add a generated resistor."""

    value = float(ctx.rng.choice((220, 470, 1000, 2200, 4700, 10000)))
    return ctx.add_part(
        "R",
        "resistor",
        f"{value:g}",
        {"resistance_ohm": value},
        {},
    )


_DEFAULT_MOTIFS = _build_default_motifs()
