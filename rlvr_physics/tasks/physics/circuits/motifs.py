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

    resistor = _resistor(ctx)
    ctx.builder.connect(resistor, "1", "VCC")
    ctx.builder.connect(resistor, "2", "0")


def _build_default_motifs() -> Mapping[str, CircuitMotif]:
    """Build the immutable built-in motif catalog."""

    motifs = {
        "divider": CircuitMotif("divider", 2, 1.0, _add_divider),
        "rc_lowpass": CircuitMotif("rc_lowpass", 2, 1.0, _add_rc_lowpass),
        "lc_tank": CircuitMotif("lc_tank", 3, 0.8, _add_lc_tank),
        "pi_filter": CircuitMotif("pi_filter", 4, 0.7, _add_pi_filter),
        "led_indicator": CircuitMotif("led_indicator", 2, 1.0, _add_led_indicator),
        "diode_clamp_reference": CircuitMotif(
            "diode_clamp_reference", 4, 0.6, _add_diode_clamp_reference
        ),
        "rectifier_filter": CircuitMotif(
            "rectifier_filter", 5, 0.5, _add_rectifier_filter
        ),
        "current_bias_monitor": CircuitMotif(
            "current_bias_monitor", 4, 0.5, _add_current_bias_monitor
        ),
        "current_led_clamp": CircuitMotif(
            "current_led_clamp", 4, 0.5, _add_current_led_clamp
        ),
        "pullup_switch": CircuitMotif("pullup_switch", 2, 1.0, _add_pullup_switch),
        "pulldown_switch": CircuitMotif(
            "pulldown_switch", 2, 0.8, _add_pulldown_switch
        ),
        "bridge": CircuitMotif("bridge", 5, 0.6, _add_bridge),
        "transistor_switch": CircuitMotif(
            "transistor_switch", 3, 0.6, _add_transistor_switch
        ),
        "pnp_high_side_switch": CircuitMotif(
            "pnp_high_side_switch", 4, 0.5, _add_pnp_high_side_switch
        ),
        "complementary_bjt_pair": CircuitMotif(
            "complementary_bjt_pair", 4, 0.4, _add_complementary_bjt_pair
        ),
        "nmos_low_side_driver": CircuitMotif(
            "nmos_low_side_driver", 4, 0.5, _add_nmos_low_side_driver
        ),
        "pmos_high_side_driver": CircuitMotif(
            "pmos_high_side_driver", 4, 0.5, _add_pmos_high_side_driver
        ),
        "cmos_pair": CircuitMotif("cmos_pair", 5, 0.4, _add_cmos_pair),
        "jfet_bias": CircuitMotif("jfet_bias", 3, 0.4, _add_jfet_bias),
        "p_jfet_bias": CircuitMotif("p_jfet_bias", 3, 0.4, _add_p_jfet_bias),
        "dual_jfet_stage": CircuitMotif(
            "dual_jfet_stage", 6, 0.3, _add_dual_jfet_stage
        ),
        "controlled_gain": CircuitMotif(
            "controlled_gain", 2, 0.4, _add_controlled_gain
        ),
        "controlled_current_load": CircuitMotif(
            "controlled_current_load", 3, 0.4, _add_controlled_current_load
        ),
        "dual_controlled_source": CircuitMotif(
            "dual_controlled_source", 4, 0.3, _add_dual_controlled_source
        ),
        "opamp_buffer": CircuitMotif("opamp_buffer", 3, 0.4, _add_opamp_buffer),
        "active_filter_stage": CircuitMotif(
            "active_filter_stage", 3, 0.4, _add_active_filter_stage
        ),
        "comparator_threshold": CircuitMotif(
            "comparator_threshold", 5, 0.3, _add_comparator_threshold
        ),
        "threshold_led_monitor": CircuitMotif(
            "threshold_led_monitor", 6, 0.3, _add_threshold_led_monitor
        ),
        "relay_lamp_driver": CircuitMotif(
            "relay_lamp_driver", 4, 0.4, _add_relay_lamp_driver
        ),
        "relay_motor_driver": CircuitMotif(
            "relay_motor_driver", 4, 0.4, _add_relay_motor_driver
        ),
        "motor_current_monitor": CircuitMotif(
            "motor_current_monitor", 4, 0.4, _add_motor_current_monitor
        ),
        "lamp_indicator_load": CircuitMotif(
            "lamp_indicator_load", 3, 0.5, _add_lamp_indicator_load
        ),
        "transformer_symbol_load": CircuitMotif(
            "transformer_symbol_load", 3, 0.3, _add_transformer_symbol_load
        ),
        "isolated_transformer_symbol_probe": CircuitMotif(
            "isolated_transformer_symbol_probe",
            4,
            0.3,
            _add_isolated_transformer_symbol_probe,
        ),
        "connector_probe": CircuitMotif(
            "connector_probe", 3, 0.4, _add_connector_probe
        ),
        "connector_loopback": CircuitMotif(
            "connector_loopback", 5, 0.4, _add_connector_loopback
        ),
        "generic_ic_interface": CircuitMotif(
            "generic_ic_interface", 5, 0.4, _add_generic_ic_interface
        ),
        "generic_ic_probe": CircuitMotif(
            "generic_ic_probe", 4, 0.4, _add_generic_ic_probe
        ),
        "logic_and_indicator": CircuitMotif(
            "logic_and_indicator", 5, 0.5, _add_logic_and_indicator
        ),
        "logic_or_lamp": CircuitMotif("logic_or_lamp", 4, 0.5, _add_logic_or_lamp),
        "inverter_chain": CircuitMotif("inverter_chain", 5, 0.5, _add_inverter_chain),
        "mixed_logic_chain": CircuitMotif(
            "mixed_logic_chain", 8, 0.3, _add_mixed_logic_chain
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

    if remaining < 5:
        return False
    left = ctx.node()
    right = ctx.node()
    for net_a, net_b in (("VCC", left), (left, "0"), ("VCC", right), (right, "0")):
        resistor = _resistor(ctx)
        ctx.builder.connect(resistor, "1", net_a)
        ctx.builder.connect(resistor, "2", net_b)
    meter = ctx.add_part("VM", "voltmeter", "VM", {"resistance_ohm": 1e12}, {})
    ctx.builder.connect(meter, "p", left)
    ctx.builder.connect(meter, "n", right)
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


def _add_lc_tank(ctx: MotifContext, remaining: int) -> bool:
    """Add an LC resonant branch with damping."""

    if remaining < 3:
        return False
    tank = ctx.node()
    feed = _resistor(ctx)
    inductor = ctx.add_part("L", "inductor", "1m", {"inductance_h": 1e-3}, {})
    capacitor = ctx.add_part("C", "capacitor", "100n", {"capacitance_f": 1e-7}, {})
    ctx.builder.connect(feed, "1", "VCC")
    ctx.builder.connect(feed, "2", tank)
    ctx.builder.connect(inductor, "1", tank)
    ctx.builder.connect(inductor, "2", "0")
    ctx.builder.connect(capacitor, "1", tank)
    ctx.builder.connect(capacitor, "2", "0")
    return True


def _add_pi_filter(ctx: MotifContext, remaining: int) -> bool:
    """Add a capacitive-input pi filter motif."""

    if remaining < 4:
        return False
    out = ctx.node()
    input_cap = ctx.add_part("C", "capacitor", "10u", {"capacitance_f": 1e-5}, {})
    inductor = ctx.add_part("L", "inductor", "10m", {"inductance_h": 1e-2}, {})
    output_cap = ctx.add_part("C", "capacitor", "10u", {"capacitance_f": 1e-5}, {})
    load = _resistor(ctx)
    ctx.builder.connect(input_cap, "1", "VCC")
    ctx.builder.connect(input_cap, "2", "0")
    ctx.builder.connect(inductor, "1", "VCC")
    ctx.builder.connect(inductor, "2", out)
    ctx.builder.connect(output_cap, "1", out)
    ctx.builder.connect(output_cap, "2", "0")
    ctx.builder.connect(load, "1", out)
    ctx.builder.connect(load, "2", "0")
    return True


def _add_diode_clamp_reference(ctx: MotifContext, remaining: int) -> bool:
    """Add a diode and zener clamp with a voltage probe."""

    if remaining < 4:
        return False
    node = ctx.node()
    feed = _resistor(ctx)
    diode = ctx.add_part("D", "diode", "D", {}, {})
    zener = ctx.add_part("D", "zener", "5V1", {}, {})
    meter = ctx.add_part("VM", "voltmeter", "VM", {"resistance_ohm": 1e12}, {})
    ctx.builder.connect(feed, "1", "VCC")
    ctx.builder.connect(feed, "2", node)
    ctx.builder.connect(diode, "a", node)
    ctx.builder.connect(diode, "k", "VCC")
    ctx.builder.connect(zener, "k", node)
    ctx.builder.connect(zener, "a", "0")
    ctx.builder.connect(meter, "p", node)
    ctx.builder.connect(meter, "n", "0")
    return True


def _add_rectifier_filter(ctx: MotifContext, remaining: int) -> bool:
    """Add a local source, rectifier, filter capacitor, and probe."""

    if remaining < 5:
        return False
    input_net = ctx.node()
    rectified = ctx.node()
    source = _voltage_source(ctx)
    diode = ctx.add_part("D", "diode", "D", {}, {})
    capacitor = ctx.add_part("C", "capacitor", "47u", {"capacitance_f": 4.7e-5}, {})
    load = _resistor(ctx)
    meter = ctx.add_part("VM", "voltmeter", "VM", {"resistance_ohm": 1e12}, {})
    ctx.builder.connect(source, "p", input_net)
    ctx.builder.connect(source, "n", "VCC")
    ctx.builder.connect(diode, "a", input_net)
    ctx.builder.connect(diode, "k", rectified)
    ctx.builder.connect(capacitor, "1", rectified)
    ctx.builder.connect(capacitor, "2", "0")
    ctx.builder.connect(load, "1", rectified)
    ctx.builder.connect(load, "2", "0")
    ctx.builder.connect(meter, "p", rectified)
    ctx.builder.connect(meter, "n", "0")
    return True


def _add_current_bias_monitor(ctx: MotifContext, remaining: int) -> bool:
    """Add a current source bias branch with current and voltage probes."""

    if remaining < 4:
        return False
    bias = ctx.node()
    sensed = ctx.node()
    source = _current_source(ctx)
    meter = ctx.add_part("AM", "ammeter", "AM", {"voltage_v": 0.0}, {})
    load = _resistor(ctx)
    voltmeter = ctx.add_part("VM", "voltmeter", "VM", {"resistance_ohm": 1e12}, {})
    ctx.builder.connect(source, "p", "VCC")
    ctx.builder.connect(source, "n", bias)
    ctx.builder.connect(meter, "p", bias)
    ctx.builder.connect(meter, "n", sensed)
    ctx.builder.connect(load, "1", sensed)
    ctx.builder.connect(load, "2", "0")
    ctx.builder.connect(voltmeter, "p", sensed)
    ctx.builder.connect(voltmeter, "n", "0")
    return True


def _add_current_led_clamp(ctx: MotifContext, remaining: int) -> bool:
    """Add a current-driven LED with zener protection."""

    if remaining < 4:
        return False
    node = ctx.node()
    source = _current_source(ctx)
    led = ctx.add_part("D", "led", "LED", {}, {})
    zener = ctx.add_part("D", "zener", "5V1", {}, {})
    shunt = _resistor(ctx)
    ctx.builder.connect(source, "p", "VCC")
    ctx.builder.connect(source, "n", node)
    ctx.builder.connect(led, "a", node)
    ctx.builder.connect(led, "k", "0")
    ctx.builder.connect(zener, "k", node)
    ctx.builder.connect(zener, "a", "0")
    ctx.builder.connect(shunt, "1", node)
    ctx.builder.connect(shunt, "2", "0")
    return True


def _add_pnp_high_side_switch(ctx: MotifContext, remaining: int) -> bool:
    """Add a PNP high-side switch motif."""

    if remaining < 4:
        return False
    output = ctx.node()
    base = ctx.node()
    transistor = ctx.add_part("Q", "bjt_pnp", "PNP", {}, {})
    load = _resistor(ctx)
    pullup = ctx.add_part(
        "RPU",
        "pullup_resistor",
        "100k",
        {"resistance_ohm": 100_000.0},
        {},
    )
    switch = ctx.add_part(
        "S",
        "ideal_switch",
        "open",
        {"state_resistance_ohm": 1e12},
        {"state": "open"},
    )
    ctx.builder.connect(transistor, "e", "VCC")
    ctx.builder.connect(transistor, "c", output)
    ctx.builder.connect(transistor, "b", base)
    ctx.builder.connect(load, "1", output)
    ctx.builder.connect(load, "2", "0")
    ctx.builder.connect(pullup, "net", base)
    ctx.builder.connect(pullup, "rail", "VCC")
    ctx.builder.connect(switch, "1", base)
    ctx.builder.connect(switch, "2", "0")
    return True


def _add_complementary_bjt_pair(ctx: MotifContext, remaining: int) -> bool:
    """Add complementary NPN and PNP transistor pair."""

    if remaining < 4:
        return False
    output = ctx.node()
    base = ctx.node()
    npn = ctx.add_part("Q", "bjt_npn", "NPN", {}, {})
    pnp = ctx.add_part("Q", "bjt_pnp", "PNP", {}, {})
    bias = _resistor(ctx)
    load = _resistor(ctx)
    ctx.builder.connect(pnp, "e", "VCC")
    ctx.builder.connect(pnp, "c", output)
    ctx.builder.connect(pnp, "b", base)
    ctx.builder.connect(npn, "c", output)
    ctx.builder.connect(npn, "b", base)
    ctx.builder.connect(npn, "e", "0")
    ctx.builder.connect(bias, "1", "VCC")
    ctx.builder.connect(bias, "2", base)
    ctx.builder.connect(load, "1", output)
    ctx.builder.connect(load, "2", "0")
    return True


def _add_nmos_low_side_driver(ctx: MotifContext, remaining: int) -> bool:
    """Add an NMOS low-side driver with a switched gate."""

    if remaining < 4:
        return False
    output = ctx.node()
    gate = ctx.node()
    transistor = ctx.add_part("Q", "mosfet_n", "NMOS", {}, {})
    load = _resistor(ctx)
    pulldown = ctx.add_part(
        "RPD",
        "pulldown_resistor",
        "100k",
        {"resistance_ohm": 100_000.0},
        {},
    )
    switch = ctx.add_part(
        "S",
        "ideal_switch",
        "open",
        {"state_resistance_ohm": 1e12},
        {"state": "open"},
    )
    ctx.builder.connect(load, "1", "VCC")
    ctx.builder.connect(load, "2", output)
    ctx.builder.connect(transistor, "d", output)
    ctx.builder.connect(transistor, "g", gate)
    ctx.builder.connect(transistor, "s", "0")
    ctx.builder.connect(pulldown, "net", gate)
    ctx.builder.connect(pulldown, "rail", "0")
    ctx.builder.connect(switch, "1", "VCC")
    ctx.builder.connect(switch, "2", gate)
    return True


def _add_pmos_high_side_driver(ctx: MotifContext, remaining: int) -> bool:
    """Add a PMOS high-side driver with a switched gate."""

    if remaining < 4:
        return False
    output = ctx.node()
    gate = ctx.node()
    transistor = ctx.add_part("Q", "mosfet_p", "PMOS", {}, {})
    load = _resistor(ctx)
    pullup = ctx.add_part(
        "RPU",
        "pullup_resistor",
        "100k",
        {"resistance_ohm": 100_000.0},
        {},
    )
    switch = ctx.add_part(
        "S",
        "ideal_switch",
        "open",
        {"state_resistance_ohm": 1e12},
        {"state": "open"},
    )
    ctx.builder.connect(transistor, "s", "VCC")
    ctx.builder.connect(transistor, "d", output)
    ctx.builder.connect(transistor, "g", gate)
    ctx.builder.connect(load, "1", output)
    ctx.builder.connect(load, "2", "0")
    ctx.builder.connect(pullup, "net", gate)
    ctx.builder.connect(pullup, "rail", "VCC")
    ctx.builder.connect(switch, "1", gate)
    ctx.builder.connect(switch, "2", "0")
    return True


def _add_cmos_pair(ctx: MotifContext, remaining: int) -> bool:
    """Add complementary MOSFET pair with passive input bias."""

    if remaining < 5:
        return False
    output = ctx.node()
    gate = ctx.node()
    nmos = ctx.add_part("Q", "mosfet_n", "NMOS", {}, {})
    pmos = ctx.add_part("Q", "mosfet_p", "PMOS", {}, {})
    input_bias = _resistor(ctx)
    output_load = _resistor(ctx)
    switch = ctx.add_part(
        "S",
        "ideal_switch",
        "open",
        {"state_resistance_ohm": 1e12},
        {"state": "open"},
    )
    ctx.builder.connect(pmos, "s", "VCC")
    ctx.builder.connect(pmos, "d", output)
    ctx.builder.connect(pmos, "g", gate)
    ctx.builder.connect(nmos, "d", output)
    ctx.builder.connect(nmos, "g", gate)
    ctx.builder.connect(nmos, "s", "0")
    ctx.builder.connect(input_bias, "1", "VCC")
    ctx.builder.connect(input_bias, "2", gate)
    ctx.builder.connect(switch, "1", gate)
    ctx.builder.connect(switch, "2", "0")
    ctx.builder.connect(output_load, "1", output)
    ctx.builder.connect(output_load, "2", "0")
    return True


def _add_p_jfet_bias(ctx: MotifContext, remaining: int) -> bool:
    """Add a P-channel JFET bias motif."""

    if remaining < 3:
        return False
    drain = ctx.node()
    gate = ctx.node()
    jfet = ctx.add_part("J", "jfet_p", "PJFET", {}, {})
    load = _resistor(ctx)
    pullup = ctx.add_part(
        "RPU",
        "pullup_resistor",
        "1M",
        {"resistance_ohm": 1_000_000.0},
        {},
    )
    ctx.builder.connect(jfet, "s", "VCC")
    ctx.builder.connect(jfet, "d", drain)
    ctx.builder.connect(jfet, "g", gate)
    ctx.builder.connect(load, "1", drain)
    ctx.builder.connect(load, "2", "0")
    ctx.builder.connect(pullup, "net", gate)
    ctx.builder.connect(pullup, "rail", "VCC")
    return True


def _add_dual_jfet_stage(ctx: MotifContext, remaining: int) -> bool:
    """Add complementary JFET bias branches."""

    if remaining < 6:
        return False
    n_out = ctx.node()
    p_out = ctx.node()
    n_gate = ctx.node()
    p_gate = ctx.node()
    n_jfet = ctx.add_part("J", "jfet_n", "NJFET", {}, {})
    p_jfet = ctx.add_part("J", "jfet_p", "PJFET", {}, {})
    n_load = _resistor(ctx)
    p_load = _resistor(ctx)
    pulldown = ctx.add_part(
        "RPD",
        "pulldown_resistor",
        "1M",
        {"resistance_ohm": 1_000_000.0},
        {},
    )
    pullup = ctx.add_part(
        "RPU",
        "pullup_resistor",
        "1M",
        {"resistance_ohm": 1_000_000.0},
        {},
    )
    ctx.builder.connect(n_load, "1", "VCC")
    ctx.builder.connect(n_load, "2", n_out)
    ctx.builder.connect(n_jfet, "d", n_out)
    ctx.builder.connect(n_jfet, "g", n_gate)
    ctx.builder.connect(n_jfet, "s", "0")
    ctx.builder.connect(p_jfet, "s", "VCC")
    ctx.builder.connect(p_jfet, "d", p_out)
    ctx.builder.connect(p_jfet, "g", p_gate)
    ctx.builder.connect(p_load, "1", p_out)
    ctx.builder.connect(p_load, "2", "0")
    ctx.builder.connect(pulldown, "net", n_gate)
    ctx.builder.connect(pulldown, "rail", "0")
    ctx.builder.connect(pullup, "net", p_gate)
    ctx.builder.connect(pullup, "rail", "VCC")
    return True


def _add_controlled_current_load(ctx: MotifContext, remaining: int) -> bool:
    """Add a voltage-controlled current sink with probes."""

    if remaining < 3:
        return False
    output = ctx.node()
    source = ctx.add_part("G", "vccs", "gm=1m", {"gain": 1e-3}, {})
    feed = _resistor(ctx)
    meter = ctx.add_part("VM", "voltmeter", "VM", {"resistance_ohm": 1e12}, {})
    ctx.builder.connect(source, "p", output)
    ctx.builder.connect(source, "n", "0")
    ctx.builder.connect(source, "cp", "VCC")
    ctx.builder.connect(source, "cn", "0")
    ctx.builder.connect(feed, "1", "VCC")
    ctx.builder.connect(feed, "2", output)
    ctx.builder.connect(meter, "p", output)
    ctx.builder.connect(meter, "n", "0")
    return True


def _add_dual_controlled_source(ctx: MotifContext, remaining: int) -> bool:
    """Add coupled voltage- and current-controlled source branches."""

    if remaining < 4:
        return False
    voltage_out = ctx.node()
    current_out = ctx.node()
    voltage_source = ctx.add_part("E", "vcvs", "gain=3", {"gain": 3.0}, {})
    current_source = ctx.add_part("G", "vccs", "gm=2m", {"gain": 2e-3}, {})
    voltage_load = _resistor(ctx)
    current_feed = _resistor(ctx)
    ctx.builder.connect(voltage_source, "p", voltage_out)
    ctx.builder.connect(voltage_source, "n", "0")
    ctx.builder.connect(voltage_source, "cp", "VCC")
    ctx.builder.connect(voltage_source, "cn", "0")
    ctx.builder.connect(current_source, "p", current_out)
    ctx.builder.connect(current_source, "n", "0")
    ctx.builder.connect(current_source, "cp", voltage_out)
    ctx.builder.connect(current_source, "cn", "0")
    ctx.builder.connect(voltage_load, "1", voltage_out)
    ctx.builder.connect(voltage_load, "2", "0")
    ctx.builder.connect(current_feed, "1", "VCC")
    ctx.builder.connect(current_feed, "2", current_out)
    return True


def _add_active_filter_stage(ctx: MotifContext, remaining: int) -> bool:
    """Add a buffered RC low-pass stage."""

    if remaining < 3:
        return False
    input_net = ctx.node()
    output_net = ctx.node()
    resistor = _resistor(ctx)
    capacitor = ctx.add_part("C", "capacitor", "100n", {"capacitance_f": 1e-7}, {})
    opamp = ctx.add_part("U", "op_amp", "ideal", {}, {})
    ctx.builder.connect(resistor, "1", "VCC")
    ctx.builder.connect(resistor, "2", input_net)
    ctx.builder.connect(capacitor, "1", input_net)
    ctx.builder.connect(capacitor, "2", "0")
    ctx.builder.connect(opamp, "noninv", input_net)
    ctx.builder.connect(opamp, "inv", output_net)
    ctx.builder.connect(opamp, "vpos", "VCC")
    ctx.builder.connect(opamp, "vneg", "0")
    ctx.builder.connect(opamp, "out", output_net)
    return True


def _add_threshold_led_monitor(ctx: MotifContext, remaining: int) -> bool:
    """Add a comparator threshold with an LED output monitor."""

    if remaining < 6:
        return False
    sense = ctx.node()
    threshold = ctx.node()
    output = ctx.node()
    led_node = ctx.node()
    sense_feed = _resistor(ctx)
    threshold_top = _resistor(ctx)
    threshold_bottom = _resistor(ctx)
    led_resistor = _resistor(ctx)
    led = ctx.add_part("D", "led", "LED", {}, {})
    comparator = ctx.add_part("U", "comparator", "cmp", {}, {})
    ctx.builder.connect(sense_feed, "1", "VCC")
    ctx.builder.connect(sense_feed, "2", sense)
    ctx.builder.connect(threshold_top, "1", "VCC")
    ctx.builder.connect(threshold_top, "2", threshold)
    ctx.builder.connect(threshold_bottom, "1", threshold)
    ctx.builder.connect(threshold_bottom, "2", "0")
    ctx.builder.connect(comparator, "noninv", sense)
    ctx.builder.connect(comparator, "inv", threshold)
    ctx.builder.connect(comparator, "vpos", "VCC")
    ctx.builder.connect(comparator, "vneg", "0")
    ctx.builder.connect(comparator, "out", output)
    ctx.builder.connect(led_resistor, "1", output)
    ctx.builder.connect(led_resistor, "2", led_node)
    ctx.builder.connect(led, "a", led_node)
    ctx.builder.connect(led, "k", "0")
    return True


def _add_relay_lamp_driver(ctx: MotifContext, remaining: int) -> bool:
    """Add a relay coil driving a lamp load."""

    if remaining < 4:
        return False
    coil = ctx.node()
    lamp_net = ctx.node()
    relay = ctx.add_part("K", "relay", "relay", {}, {})
    switch = ctx.add_part(
        "S",
        "ideal_switch",
        "open",
        {"state_resistance_ohm": 1e12},
        {"state": "open"},
    )
    diode = ctx.add_part("D", "diode", "D", {}, {})
    lamp = ctx.add_part("LA", "lamp", "100", {"resistance_ohm": 100.0}, {})
    ctx.builder.connect(relay, "coil_p", "VCC")
    ctx.builder.connect(relay, "coil_n", coil)
    ctx.builder.connect(relay, "com", lamp_net)
    ctx.builder.connect(relay, "no", "VCC")
    ctx.builder.connect(relay, "nc", "0")
    ctx.builder.connect(switch, "1", coil)
    ctx.builder.connect(switch, "2", "0")
    ctx.builder.connect(diode, "a", coil)
    ctx.builder.connect(diode, "k", "VCC")
    ctx.builder.connect(lamp, "1", lamp_net)
    ctx.builder.connect(lamp, "2", "0")
    return True


def _add_relay_motor_driver(ctx: MotifContext, remaining: int) -> bool:
    """Add a relay coil driving a motor load."""

    if remaining < 4:
        return False
    coil = ctx.node()
    motor_net = ctx.node()
    relay = ctx.add_part("K", "relay", "relay", {}, {})
    switch = ctx.add_part(
        "S",
        "ideal_switch",
        "open",
        {"state_resistance_ohm": 1e12},
        {"state": "open"},
    )
    diode = ctx.add_part("D", "diode", "D", {}, {})
    motor = ctx.add_part("M", "motor", "25", {"resistance_ohm": 25.0}, {})
    ctx.builder.connect(relay, "coil_p", "VCC")
    ctx.builder.connect(relay, "coil_n", coil)
    ctx.builder.connect(relay, "com", motor_net)
    ctx.builder.connect(relay, "no", "VCC")
    ctx.builder.connect(relay, "nc", "0")
    ctx.builder.connect(switch, "1", coil)
    ctx.builder.connect(switch, "2", "0")
    ctx.builder.connect(diode, "a", coil)
    ctx.builder.connect(diode, "k", "VCC")
    ctx.builder.connect(motor, "1", motor_net)
    ctx.builder.connect(motor, "2", "0")
    return True


def _add_motor_current_monitor(ctx: MotifContext, remaining: int) -> bool:
    """Add a local motor supply with current and voltage monitoring."""

    if remaining < 4:
        return False
    supply = ctx.node()
    motor_node = ctx.node()
    source = _voltage_source(ctx)
    ammeter = ctx.add_part("AM", "ammeter", "AM", {"voltage_v": 0.0}, {})
    motor = ctx.add_part("M", "motor", "25", {"resistance_ohm": 25.0}, {})
    voltmeter = ctx.add_part("VM", "voltmeter", "VM", {"resistance_ohm": 1e12}, {})
    ctx.builder.connect(source, "p", supply)
    ctx.builder.connect(source, "n", "VCC")
    ctx.builder.connect(ammeter, "p", supply)
    ctx.builder.connect(ammeter, "n", motor_node)
    ctx.builder.connect(motor, "1", motor_node)
    ctx.builder.connect(motor, "2", "0")
    ctx.builder.connect(voltmeter, "p", motor_node)
    ctx.builder.connect(voltmeter, "n", "0")
    return True


def _add_lamp_indicator_load(ctx: MotifContext, remaining: int) -> bool:
    """Add a lamp load with series feed and voltage probe."""

    if remaining < 3:
        return False
    node = ctx.node()
    feed = _resistor(ctx)
    lamp = ctx.add_part("LA", "lamp", "100", {"resistance_ohm": 100.0}, {})
    meter = ctx.add_part("VM", "voltmeter", "VM", {"resistance_ohm": 1e12}, {})
    ctx.builder.connect(feed, "1", "VCC")
    ctx.builder.connect(feed, "2", node)
    ctx.builder.connect(lamp, "1", node)
    ctx.builder.connect(lamp, "2", "0")
    ctx.builder.connect(meter, "p", node)
    ctx.builder.connect(meter, "n", "0")
    return True


def _add_transformer_symbol_load(ctx: MotifContext, remaining: int) -> bool:
    """Add a transformer symbol feeding a lamp and shunt load."""

    if remaining < 3:
        return False
    secondary = ctx.node()
    transformer = ctx.add_part("T", "transformer", "1:1", {}, {})
    lamp = ctx.add_part("LA", "lamp", "100", {"resistance_ohm": 100.0}, {})
    load = _resistor(ctx)
    ctx.builder.connect(transformer, "p1", "VCC")
    ctx.builder.connect(transformer, "p2", "0")
    ctx.builder.connect(transformer, "s1", secondary)
    ctx.builder.connect(transformer, "s2", "0")
    ctx.builder.connect(lamp, "1", secondary)
    ctx.builder.connect(lamp, "2", "0")
    ctx.builder.connect(load, "1", secondary)
    ctx.builder.connect(load, "2", "0")
    return True


def _add_isolated_transformer_symbol_probe(ctx: MotifContext, remaining: int) -> bool:
    """Add transformer-symbol secondary current and voltage probes."""

    if remaining < 4:
        return False
    secondary_hi = ctx.node()
    secondary_lo = ctx.node()
    load_node = ctx.node()
    transformer = ctx.add_part("T", "transformer", "1:2", {}, {})
    ammeter = ctx.add_part("AM", "ammeter", "AM", {"voltage_v": 0.0}, {})
    load = _resistor(ctx)
    meter = ctx.add_part("VM", "voltmeter", "VM", {"resistance_ohm": 1e12}, {})
    ctx.builder.connect(transformer, "p1", "VCC")
    ctx.builder.connect(transformer, "p2", "0")
    ctx.builder.connect(transformer, "s1", secondary_hi)
    ctx.builder.connect(transformer, "s2", secondary_lo)
    ctx.builder.connect(ammeter, "p", secondary_hi)
    ctx.builder.connect(ammeter, "n", load_node)
    ctx.builder.connect(load, "1", load_node)
    ctx.builder.connect(load, "2", secondary_lo)
    ctx.builder.connect(meter, "p", load_node)
    ctx.builder.connect(meter, "n", secondary_lo)
    return True


def _add_connector_probe(ctx: MotifContext, remaining: int) -> bool:
    """Add a connector with a passive load and voltage probe."""

    if remaining < 3:
        return False
    io = ctx.node()
    connector = ctx.add_part("J", "connector_2", "J2", {}, {})
    load = _resistor(ctx)
    meter = ctx.add_part("VM", "voltmeter", "VM", {"resistance_ohm": 1e12}, {})
    ctx.builder.connect(connector, "1", "VCC")
    ctx.builder.connect(connector, "2", io)
    ctx.builder.connect(load, "1", io)
    ctx.builder.connect(load, "2", "0")
    ctx.builder.connect(meter, "p", io)
    ctx.builder.connect(meter, "n", "0")
    return True


def _add_connector_loopback(ctx: MotifContext, remaining: int) -> bool:
    """Add a connector loopback with current injection."""

    if remaining < 5:
        return False
    source_net = ctx.node()
    pin_one = ctx.node()
    pin_two = ctx.node()
    source = _current_source(ctx)
    ammeter = ctx.add_part("AM", "ammeter", "AM", {"voltage_v": 0.0}, {})
    connector = ctx.add_part("J", "connector_2", "J2", {}, {})
    jumper = ctx.add_part(
        "S",
        "ideal_switch",
        "closed",
        {"state_resistance_ohm": 0.05},
        {"state": "closed"},
    )
    load = _resistor(ctx)
    ctx.builder.connect(source, "p", "VCC")
    ctx.builder.connect(source, "n", source_net)
    ctx.builder.connect(ammeter, "p", source_net)
    ctx.builder.connect(ammeter, "n", pin_one)
    ctx.builder.connect(connector, "1", pin_one)
    ctx.builder.connect(connector, "2", pin_two)
    ctx.builder.connect(jumper, "1", pin_one)
    ctx.builder.connect(jumper, "2", pin_two)
    ctx.builder.connect(load, "1", pin_two)
    ctx.builder.connect(load, "2", "0")
    return True


def _add_generic_ic_interface(ctx: MotifContext, remaining: int) -> bool:
    """Add a generic IC with connector inputs and passive output load."""

    if remaining < 5:
        return False
    in_one = ctx.node()
    in_two = ctx.node()
    output = ctx.node()
    ic = ctx.add_part("U", "generic_ic", "IC", {}, {})
    pullup = ctx.add_part(
        "RPU",
        "pullup_resistor",
        "10k",
        {"resistance_ohm": 10_000.0},
        {},
    )
    pulldown = ctx.add_part(
        "RPD",
        "pulldown_resistor",
        "10k",
        {"resistance_ohm": 10_000.0},
        {},
    )
    connector = ctx.add_part("J", "connector_2", "J2", {}, {})
    load = _resistor(ctx)
    ctx.builder.connect(ic, "in1", in_one)
    ctx.builder.connect(ic, "in2", in_two)
    ctx.builder.connect(ic, "out1", output)
    ctx.builder.connect(ic, "vcc", "VCC")
    ctx.builder.connect(ic, "gnd", "0")
    ctx.builder.connect(pullup, "net", in_one)
    ctx.builder.connect(pullup, "rail", "VCC")
    ctx.builder.connect(pulldown, "net", in_two)
    ctx.builder.connect(pulldown, "rail", "0")
    ctx.builder.connect(connector, "1", in_one)
    ctx.builder.connect(connector, "2", in_two)
    ctx.builder.connect(load, "1", output)
    ctx.builder.connect(load, "2", "0")
    return True


def _add_generic_ic_probe(ctx: MotifContext, remaining: int) -> bool:
    """Add a generic IC with biased inputs and output probe."""

    if remaining < 4:
        return False
    in_one = ctx.node()
    in_two = ctx.node()
    output = ctx.node()
    ic = ctx.add_part("U", "generic_ic", "IC", {}, {})
    input_feed = _resistor(ctx)
    input_shunt = _resistor(ctx)
    meter = ctx.add_part("VM", "voltmeter", "VM", {"resistance_ohm": 1e12}, {})
    ctx.builder.connect(ic, "in1", in_one)
    ctx.builder.connect(ic, "in2", in_two)
    ctx.builder.connect(ic, "out1", output)
    ctx.builder.connect(ic, "vcc", "VCC")
    ctx.builder.connect(ic, "gnd", "0")
    ctx.builder.connect(input_feed, "1", "VCC")
    ctx.builder.connect(input_feed, "2", in_one)
    ctx.builder.connect(input_shunt, "1", in_two)
    ctx.builder.connect(input_shunt, "2", "0")
    ctx.builder.connect(meter, "p", output)
    ctx.builder.connect(meter, "n", "0")
    return True


def _add_logic_and_indicator(ctx: MotifContext, remaining: int) -> bool:
    """Add an AND gate with biased inputs and LED output."""

    if remaining < 5:
        return False
    in_one = ctx.node()
    in_two = ctx.node()
    output = ctx.node()
    led_node = ctx.node()
    gate = ctx.add_part("U", "and_gate", "AND", {}, {})
    pullup = ctx.add_part(
        "RPU",
        "pullup_resistor",
        "10k",
        {"resistance_ohm": 10_000.0},
        {},
    )
    pulldown = ctx.add_part(
        "RPD",
        "pulldown_resistor",
        "10k",
        {"resistance_ohm": 10_000.0},
        {},
    )
    resistor = _resistor(ctx)
    led = ctx.add_part("D", "led", "LED", {}, {})
    _connect_logic_gate(ctx, gate, in_one, in_two, output)
    ctx.builder.connect(pullup, "net", in_one)
    ctx.builder.connect(pullup, "rail", "VCC")
    ctx.builder.connect(pulldown, "net", in_two)
    ctx.builder.connect(pulldown, "rail", "0")
    ctx.builder.connect(resistor, "1", output)
    ctx.builder.connect(resistor, "2", led_node)
    ctx.builder.connect(led, "a", led_node)
    ctx.builder.connect(led, "k", "0")
    return True


def _add_logic_or_lamp(ctx: MotifContext, remaining: int) -> bool:
    """Add an OR gate driving a lamp load."""

    if remaining < 4:
        return False
    in_one = ctx.node()
    in_two = ctx.node()
    output = ctx.node()
    gate = ctx.add_part("U", "or_gate", "OR", {}, {})
    pullup = ctx.add_part(
        "RPU",
        "pullup_resistor",
        "10k",
        {"resistance_ohm": 10_000.0},
        {},
    )
    pulldown = ctx.add_part(
        "RPD",
        "pulldown_resistor",
        "10k",
        {"resistance_ohm": 10_000.0},
        {},
    )
    lamp = ctx.add_part("LA", "lamp", "100", {"resistance_ohm": 100.0}, {})
    _connect_logic_gate(ctx, gate, in_one, in_two, output)
    ctx.builder.connect(pullup, "net", in_one)
    ctx.builder.connect(pullup, "rail", "VCC")
    ctx.builder.connect(pulldown, "net", in_two)
    ctx.builder.connect(pulldown, "rail", "0")
    ctx.builder.connect(lamp, "1", output)
    ctx.builder.connect(lamp, "2", "0")
    return True


def _add_inverter_chain(ctx: MotifContext, remaining: int) -> bool:
    """Add two cascaded inverters with an LED load."""

    if remaining < 5:
        return False
    input_net = ctx.node()
    mid = ctx.node()
    output = ctx.node()
    led_node = ctx.node()
    first = ctx.add_part("U", "not_gate", "NOT", {}, {})
    second = ctx.add_part("U", "not_gate", "NOT", {}, {})
    pullup = ctx.add_part(
        "RPU",
        "pullup_resistor",
        "10k",
        {"resistance_ohm": 10_000.0},
        {},
    )
    resistor = _resistor(ctx)
    led = ctx.add_part("D", "led", "LED", {}, {})
    _connect_not_gate(ctx, first, input_net, mid)
    _connect_not_gate(ctx, second, mid, output)
    ctx.builder.connect(pullup, "net", input_net)
    ctx.builder.connect(pullup, "rail", "VCC")
    ctx.builder.connect(resistor, "1", output)
    ctx.builder.connect(resistor, "2", led_node)
    ctx.builder.connect(led, "a", led_node)
    ctx.builder.connect(led, "k", "0")
    return True


def _add_mixed_logic_chain(ctx: MotifContext, remaining: int) -> bool:
    """Add mixed AND, OR, and inverter logic with a passive output."""

    if remaining < 8:
        return False
    in_one = ctx.node()
    in_two = ctx.node()
    and_out = ctx.node()
    not_out = ctx.node()
    or_out = ctx.node()
    led_node = ctx.node()
    and_gate = ctx.add_part("U", "and_gate", "AND", {}, {})
    or_gate = ctx.add_part("U", "or_gate", "OR", {}, {})
    inverter = ctx.add_part("U", "not_gate", "NOT", {}, {})
    pullup = ctx.add_part(
        "RPU",
        "pullup_resistor",
        "10k",
        {"resistance_ohm": 10_000.0},
        {},
    )
    pulldown = ctx.add_part(
        "RPD",
        "pulldown_resistor",
        "10k",
        {"resistance_ohm": 10_000.0},
        {},
    )
    output_resistor = _resistor(ctx)
    led = ctx.add_part("D", "led", "LED", {}, {})
    meter = ctx.add_part("VM", "voltmeter", "VM", {"resistance_ohm": 1e12}, {})
    _connect_logic_gate(ctx, and_gate, in_one, in_two, and_out)
    _connect_not_gate(ctx, inverter, in_two, not_out)
    _connect_logic_gate(ctx, or_gate, and_out, not_out, or_out)
    ctx.builder.connect(pullup, "net", in_one)
    ctx.builder.connect(pullup, "rail", "VCC")
    ctx.builder.connect(pulldown, "net", in_two)
    ctx.builder.connect(pulldown, "rail", "0")
    ctx.builder.connect(output_resistor, "1", or_out)
    ctx.builder.connect(output_resistor, "2", led_node)
    ctx.builder.connect(led, "a", led_node)
    ctx.builder.connect(led, "k", "0")
    ctx.builder.connect(meter, "p", or_out)
    ctx.builder.connect(meter, "n", "0")
    return True


def _connect_logic_gate(
    ctx: MotifContext, ref: str, in_one: str, in_two: str, output: str
) -> None:
    """Connect a two-input logic gate to power, inputs, and output."""

    ctx.builder.connect(ref, "in1", in_one)
    ctx.builder.connect(ref, "in2", in_two)
    ctx.builder.connect(ref, "out", output)
    ctx.builder.connect(ref, "vcc", "VCC")
    ctx.builder.connect(ref, "gnd", "0")


def _connect_not_gate(ctx: MotifContext, ref: str, input_net: str, output: str) -> None:
    """Connect a single-input inverter to power, input, and output."""

    ctx.builder.connect(ref, "in1", input_net)
    ctx.builder.connect(ref, "out", output)
    ctx.builder.connect(ref, "vcc", "VCC")
    ctx.builder.connect(ref, "gnd", "0")


def _voltage_source(ctx: MotifContext) -> str:
    """Add a generated DC voltage source."""

    voltage = float(ctx.rng.choice((3.3, 5.0, 9.0, 12.0)))
    return ctx.add_part(
        "V",
        "voltage_source_dc",
        f"{voltage:g}V",
        {"voltage_v": voltage},
        {},
    )


def _current_source(ctx: MotifContext) -> str:
    """Add a generated DC current source."""

    current = float(ctx.rng.choice((0.001, 0.002, 0.005, 0.01)))
    return ctx.add_part(
        "I",
        "current_source_dc",
        f"{current:g}A",
        {"current_a": current},
        {},
    )


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
