"""Built-in circuit part catalog."""

from types import MappingProxyType
from typing import Mapping

from rlvr_physics.tasks.physics.circuits.model import (
    ComponentFamily,
    PartSpec,
    PinKind,
    PinSpec,
)


def default_part_catalog() -> Mapping[str, PartSpec]:
    """Return the built-in part catalog.

    Returns
    -------
    Mapping[str, PartSpec]
        Immutable mapping from component kind to part specification.
    """

    return _DEFAULT_PART_CATALOG


def default_catalog() -> Mapping[str, PartSpec]:
    """Return the built-in part catalog.

    Returns
    -------
    Mapping[str, PartSpec]
        Immutable mapping from component kind to part specification.
    """

    return default_part_catalog()


def require_part_spec(catalog: Mapping[str, PartSpec], kind: str) -> PartSpec:
    """Return a part specification or raise a clear error.

    Parameters
    ----------
    catalog:
        Component catalog to search.
    kind:
        Component kind.

    Returns
    -------
    PartSpec
        Matching part specification.

    Raises
    ------
    KeyError
        Raised when ``kind`` is not in ``catalog``.
    """

    try:
        return catalog[kind]
    except KeyError as exc:
        raise KeyError(f"unknown component kind: {kind}") from exc


def _part(
    kind: str,
    display_name: str,
    ref_prefix: str,
    family: ComponentFamily,
    pins: tuple[PinSpec, ...],
    generation_tags: tuple[str, ...],
) -> PartSpec:
    """Build one part specification."""

    return PartSpec(
        kind=kind,
        display_name=display_name,
        ref_prefix=ref_prefix,
        family=family,
        pins=pins,
        generation_tags=generation_tags,
    )


def _two_pin_passive_pins() -> tuple[PinSpec, PinSpec]:
    """Return standard two-terminal passive pins."""

    return (
        PinSpec("1", PinKind.PASSIVE),
        PinSpec("2", PinKind.PASSIVE),
    )


def _polarized_passive_pins() -> tuple[PinSpec, PinSpec]:
    """Return standard polarized passive pins."""

    return (
        PinSpec("p", PinKind.PASSIVE),
        PinSpec("n", PinKind.PASSIVE),
    )


def _two_pin_passive(
    kind: str,
    display_name: str,
    ref_prefix: str,
    tags: tuple[str, ...],
) -> PartSpec:
    """Build a common two-pin passive specification."""

    return _part(
        kind,
        display_name,
        ref_prefix,
        ComponentFamily.PASSIVE,
        _two_pin_passive_pins(),
        tags,
    )


def _power_source(
    kind: str,
    display_name: str,
    ref_prefix: str,
    tags: tuple[str, ...],
    positive_pin_kind: PinKind,
) -> PartSpec:
    """Build a two-terminal independent source specification."""

    return _part(
        kind,
        display_name,
        ref_prefix,
        ComponentFamily.SOURCE,
        (
            PinSpec("p", positive_pin_kind),
            PinSpec("n", PinKind.PASSIVE),
        ),
        tags,
    )


def _diode_like(kind: str, display_name: str, tags: tuple[str, ...]) -> PartSpec:
    """Build a two-pin semiconductor diode specification."""

    return _part(
        kind,
        display_name,
        "D",
        ComponentFamily.SEMICONDUCTOR,
        (
            PinSpec("a", PinKind.PASSIVE),
            PinSpec("k", PinKind.PASSIVE),
        ),
        tags,
    )


def _transistor(
    kind: str,
    display_name: str,
    tags: tuple[str, ...],
    emitter_on_top: bool,
) -> PartSpec:
    """Build a three-pin BJT specification."""

    pins = (
        (
            PinSpec("e", PinKind.PASSIVE),
            PinSpec("b", PinKind.INPUT),
            PinSpec("c", PinKind.PASSIVE),
        )
        if emitter_on_top
        else (
            PinSpec("c", PinKind.PASSIVE),
            PinSpec("b", PinKind.INPUT),
            PinSpec("e", PinKind.PASSIVE),
        )
    )
    return _part(
        kind,
        display_name,
        "Q",
        ComponentFamily.SEMICONDUCTOR,
        pins,
        tags,
    )


def _mosfet(
    kind: str,
    display_name: str,
    tags: tuple[str, ...],
    source_on_top: bool,
) -> PartSpec:
    """Build a three-pin MOSFET specification."""

    pins = (
        (
            PinSpec("s", PinKind.PASSIVE),
            PinSpec("g", PinKind.INPUT),
            PinSpec("d", PinKind.PASSIVE),
        )
        if source_on_top
        else (
            PinSpec("d", PinKind.PASSIVE),
            PinSpec("g", PinKind.INPUT),
            PinSpec("s", PinKind.PASSIVE),
        )
    )
    return _part(
        kind,
        display_name,
        "Q",
        ComponentFamily.SEMICONDUCTOR,
        pins,
        tags,
    )


def _jfet(
    kind: str,
    display_name: str,
    tags: tuple[str, ...],
    source_on_top: bool,
) -> PartSpec:
    """Build a three-pin JFET specification."""

    pins = (
        (
            PinSpec("s", PinKind.PASSIVE),
            PinSpec("g", PinKind.INPUT),
            PinSpec("d", PinKind.PASSIVE),
        )
        if source_on_top
        else (
            PinSpec("d", PinKind.PASSIVE),
            PinSpec("g", PinKind.INPUT),
            PinSpec("s", PinKind.PASSIVE),
        )
    )
    return _part(
        kind,
        display_name,
        "J",
        ComponentFamily.SEMICONDUCTOR,
        pins,
        tags,
    )


def _controlled_source(
    kind: str,
    display_name: str,
    ref_prefix: str,
    tags: tuple[str, ...],
) -> PartSpec:
    """Build a voltage-controlled source specification."""

    return _part(
        kind,
        display_name,
        ref_prefix,
        ComponentFamily.CONTROLLED_SOURCE,
        (
            PinSpec("p", PinKind.OUTPUT),
            PinSpec("n", PinKind.PASSIVE),
            PinSpec("cp", PinKind.INPUT),
            PinSpec("cn", PinKind.INPUT),
        ),
        tags,
    )


def _logic_gate(kind: str, display_name: str, pin_count: int) -> PartSpec:
    """Build a generic logic gate specification."""

    input_pins = tuple(
        PinSpec(f"in{idx}", PinKind.INPUT) for idx in range(1, pin_count + 1)
    )
    pins = input_pins + (
        PinSpec("out", PinKind.OUTPUT),
        PinSpec("vcc", PinKind.POWER_IN),
        PinSpec("gnd", PinKind.POWER_IN),
    )
    return _part(
        kind,
        display_name,
        "U",
        ComponentFamily.LOGIC,
        pins,
        ("logic", "digital"),
    )


def _integrated_part(
    kind: str,
    display_name: str,
    ref_prefix: str,
    pins: tuple[PinSpec, ...],
    tags: tuple[str, ...],
) -> PartSpec:
    """Build an integrated-circuit part specification."""

    return _part(
        kind,
        display_name,
        ref_prefix,
        ComponentFamily.INTEGRATED,
        pins,
        tags,
    )


def _build_default_part_catalog() -> Mapping[str, PartSpec]:
    """Build the immutable built-in part catalog."""

    specs: dict[str, PartSpec] = {}

    def add(spec: PartSpec) -> None:
        specs[spec.kind] = spec

    add(
        _part(
            "ground",
            "Ground",
            "GND",
            ComponentFamily.POWER,
            (PinSpec("0", PinKind.PASSIVE),),
            ("power", "reference"),
        )
    )
    add(
        _part(
            "power_rail",
            "Power Rail",
            "PWR",
            ComponentFamily.POWER,
            (PinSpec("net", PinKind.PASSIVE),),
            ("power", "reference"),
        )
    )
    add(
        _part(
            "test_point",
            "Test Point",
            "TP",
            ComponentFamily.CONNECTOR,
            (PinSpec("net", PinKind.PASSIVE),),
            ("connector", "test_point", "probe"),
        )
    )
    add(
        _two_pin_passive(
            "resistor",
            "Resistor",
            "R",
            ("passive", "load", "divider", "filter"),
        )
    )
    add(
        _two_pin_passive(
            "variable_resistor",
            "Variable Resistor",
            "RV",
            ("passive", "load", "divider", "adjustable"),
        )
    )
    add(
        _two_pin_passive(
            "capacitor",
            "Capacitor",
            "C",
            ("passive", "filter", "decoupling"),
        )
    )
    add(
        _part(
            "polarized_capacitor",
            "Polarized Capacitor",
            "C",
            ComponentFamily.PASSIVE,
            _polarized_passive_pins(),
            ("passive", "filter", "decoupling", "polarized"),
        )
    )
    add(
        _two_pin_passive(
            "inductor",
            "Inductor",
            "L",
            ("passive", "filter"),
        )
    )
    add(
        _two_pin_passive(
            "inductor_looped",
            "Looped Inductor",
            "L",
            ("passive", "filter"),
        )
    )
    add(
        _two_pin_passive(
            "crystal",
            "Crystal",
            "XTAL",
            ("timing", "oscillator"),
        )
    )
    add(
        _two_pin_passive(
            "lamp",
            "Lamp",
            "LA",
            ("load", "electromechanical"),
        )
    )
    add(
        _two_pin_passive(
            "motor",
            "Motor",
            "M",
            ("load", "electromechanical"),
        )
    )
    add(
        _power_source(
            "voltage_source_dc",
            "DC Voltage Source",
            "V",
            ("source", "power"),
            PinKind.POWER_OUT,
        )
    )
    add(
        _power_source(
            "battery",
            "Battery",
            "BT",
            ("source", "power", "battery"),
            PinKind.POWER_OUT,
        )
    )
    add(
        _power_source(
            "voltage_source_ac",
            "AC Voltage Source",
            "VAC",
            ("source", "power", "ac"),
            PinKind.POWER_OUT,
        )
    )
    add(
        _power_source(
            "current_source_dc",
            "DC Current Source",
            "I",
            ("source", "bias"),
            PinKind.PASSIVE,
        )
    )
    add(_diode_like("diode", "Diode", ("semiconductor", "rectifier", "clamp")))
    add(_diode_like("led", "LED", ("semiconductor", "indicator", "load")))
    add(
        _diode_like(
            "zener",
            "Zener Diode",
            ("semiconductor", "clamp", "reference"),
        )
    )
    add(
        _diode_like(
            "photodiode",
            "Photodiode",
            ("semiconductor", "sensor", "photodiode"),
        )
    )
    add(
        _transistor(
            "bjt_npn",
            "NPN BJT",
            ("semiconductor", "switch", "amplifier"),
            False,
        )
    )
    add(_mosfet("mosfet_n", "N-MOSFET", ("semiconductor", "switch"), False))
    add(_mosfet("mosfet_p", "P-MOSFET", ("semiconductor", "switch"), True))
    add(_jfet("jfet_n", "N-JFET", ("semiconductor", "amplifier"), False))
    add(_jfet("jfet_p", "P-JFET", ("semiconductor", "amplifier"), True))
    add(
        _controlled_source(
            "vcvs",
            "Voltage-Controlled Voltage Source",
            "E",
            ("source", "controlled", "linear"),
        )
    )
    add(
        _controlled_source(
            "vccs",
            "Voltage-Controlled Current Source",
            "G",
            ("source", "controlled", "linear"),
        )
    )
    add(
        _part(
            "pullup_resistor",
            "Pull-up Resistor",
            "RPU",
            ComponentFamily.PASSIVE,
            (
                PinSpec("net", PinKind.PULLUP),
                PinSpec("rail", PinKind.POWER_IN),
            ),
            ("passive", "bias", "logic", "pullup"),
        )
    )
    add(
        _part(
            "pulldown_resistor",
            "Pull-down Resistor",
            "RPD",
            ComponentFamily.PASSIVE,
            (
                PinSpec("net", PinKind.PULLDOWN),
                PinSpec("rail", PinKind.POWER_IN),
            ),
            ("passive", "bias", "logic", "pulldown"),
        )
    )
    add(
        _part(
            "ideal_switch",
            "Ideal Switch",
            "S",
            ComponentFamily.SWITCH,
            _two_pin_passive_pins(),
            ("switch", "control"),
        )
    )
    add(
        _part(
            "pushbutton_switch",
            "Pushbutton Switch",
            "S",
            ComponentFamily.SWITCH,
            _two_pin_passive_pins(),
            ("switch", "control", "momentary"),
        )
    )
    add(
        _part(
            "controlled_switch",
            "Controlled Switch",
            "S",
            ComponentFamily.SWITCH,
            (
                PinSpec("in", PinKind.PASSIVE),
                PinSpec("out", PinKind.PASSIVE),
                PinSpec("ctrl", PinKind.INPUT),
            ),
            ("switch", "control"),
        )
    )
    add(
        _part(
            "relay",
            "Relay",
            "K",
            ComponentFamily.ELECTROMECHANICAL,
            (
                PinSpec("coil_p", PinKind.PASSIVE),
                PinSpec("coil_n", PinKind.PASSIVE),
                PinSpec("com", PinKind.PASSIVE),
                PinSpec("no", PinKind.PASSIVE),
                PinSpec("nc", PinKind.PASSIVE),
            ),
            ("switch", "electromechanical", "load"),
        )
    )
    add(
        _integrated_part(
            "op_amp",
            "Ideal Op Amp",
            "U",
            (
                PinSpec("noninv", PinKind.INPUT),
                PinSpec("inv", PinKind.INPUT),
                PinSpec("vpos", PinKind.POWER_IN),
                PinSpec("vneg", PinKind.POWER_IN),
                PinSpec("out", PinKind.OUTPUT),
            ),
            ("amplifier", "filter", "integrated"),
        )
    )
    add(
        _integrated_part(
            "comparator",
            "Comparator",
            "U",
            (
                PinSpec("noninv", PinKind.INPUT),
                PinSpec("inv", PinKind.INPUT),
                PinSpec("vpos", PinKind.POWER_IN),
                PinSpec("vneg", PinKind.POWER_IN),
                PinSpec("out", PinKind.OUTPUT),
            ),
            ("comparator", "integrated", "threshold"),
        )
    )
    add(
        _integrated_part(
            "instrumentation_amplifier",
            "Instrumentation Amplifier",
            "UINA",
            (
                PinSpec("inp", PinKind.INPUT),
                PinSpec("inn", PinKind.INPUT),
                PinSpec("ref", PinKind.INPUT),
                PinSpec("rg1", PinKind.PASSIVE),
                PinSpec("rg2", PinKind.PASSIVE),
                PinSpec("vpos", PinKind.POWER_IN),
                PinSpec("vneg", PinKind.POWER_IN),
                PinSpec("out", PinKind.OUTPUT),
            ),
            ("amplifier", "instrumentation", "integrated"),
        )
    )
    add(
        _integrated_part(
            "timer_555",
            "555 Timer",
            "U",
            (
                PinSpec("gnd", PinKind.POWER_IN),
                PinSpec("vcc", PinKind.POWER_IN),
                PinSpec("reset", PinKind.INPUT),
                PinSpec("ctrl", PinKind.INPUT),
                PinSpec("disch", PinKind.OPEN_COLLECTOR),
                PinSpec("thresh", PinKind.INPUT),
                PinSpec("trig", PinKind.INPUT),
                PinSpec("out", PinKind.OUTPUT),
            ),
            ("timer", "oscillator", "integrated"),
        )
    )
    add(
        _integrated_part(
            "generic_ic",
            "Generic IC",
            "U",
            (
                PinSpec("in1", PinKind.INPUT),
                PinSpec("in2", PinKind.INPUT),
                PinSpec("out1", PinKind.OUTPUT),
                PinSpec("vcc", PinKind.POWER_IN),
                PinSpec("gnd", PinKind.POWER_IN),
            ),
            ("integrated", "block", "generic"),
        )
    )
    add(
        _integrated_part(
            "counter_4bit",
            "4-bit Synchronous Counter",
            "U",
            (
                PinSpec("clk", PinKind.INPUT),
                PinSpec("clr_n", PinKind.INPUT),
                PinSpec("load_n", PinKind.INPUT),
                PinSpec("enp", PinKind.INPUT),
                PinSpec("ent", PinKind.INPUT),
                PinSpec("a", PinKind.INPUT),
                PinSpec("b", PinKind.INPUT),
                PinSpec("c", PinKind.INPUT),
                PinSpec("d", PinKind.INPUT),
                PinSpec("vcc", PinKind.POWER_IN),
                PinSpec("gnd", PinKind.POWER_IN),
                PinSpec("qa", PinKind.OUTPUT),
                PinSpec("qb", PinKind.OUTPUT),
                PinSpec("qc", PinKind.OUTPUT),
                PinSpec("qd", PinKind.OUTPUT),
                PinSpec("rco", PinKind.OUTPUT),
            ),
            ("logic", "counter", "digital"),
        )
    )
    add(
        _part(
            "transformer",
            "Transformer",
            "T",
            ComponentFamily.ELECTROMECHANICAL,
            (
                PinSpec("p1", PinKind.PASSIVE),
                PinSpec("p2", PinKind.PASSIVE),
                PinSpec("s1", PinKind.PASSIVE),
                PinSpec("s2", PinKind.PASSIVE),
            ),
            ("coupled", "power", "ac"),
        )
    )
    add(
        _part(
            "connector_2",
            "Two Pin Connector",
            "J",
            ComponentFamily.CONNECTOR,
            _two_pin_passive_pins(),
            ("connector", "io"),
        )
    )
    add(
        _part(
            "voltmeter",
            "Voltmeter",
            "VM",
            ComponentFamily.MEASUREMENT,
            (
                PinSpec("p", PinKind.INPUT),
                PinSpec("n", PinKind.INPUT),
            ),
            ("measurement", "probe"),
        )
    )
    add(
        _part(
            "ammeter",
            "Ammeter",
            "AM",
            ComponentFamily.MEASUREMENT,
            (
                PinSpec("p", PinKind.PASSIVE),
                PinSpec("n", PinKind.PASSIVE),
            ),
            ("measurement", "probe"),
        )
    )
    add(_logic_gate("and_gate", "AND Gate", 2))
    add(_logic_gate("nand_gate", "NAND Gate", 2))
    add(_logic_gate("or_gate", "OR Gate", 2))
    add(_logic_gate("xor_gate", "XOR Gate", 2))
    add(_logic_gate("not_gate", "NOT Gate", 1))
    return MappingProxyType(specs)


_DEFAULT_PART_CATALOG = _build_default_part_catalog()
