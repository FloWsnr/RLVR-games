"""Built-in circuit part catalog."""

from types import MappingProxyType
from typing import Mapping

from rlvr_physics.tasks.physics.circuits.model import (
    AnalysisSupport,
    ComponentFamily,
    PartSpec,
    PinKind,
    PinSide,
    PinSpec,
    SpiceSpec,
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


def _two_pin_passive(
    kind: str,
    display_name: str,
    ref_prefix: str,
    icon: str,
    spice_prefix: str,
    value_parameter: str,
    default_value: str,
    tags: tuple[str, ...],
    linear_dc: bool,
) -> PartSpec:
    """Build a common two-pin passive specification."""

    support = [AnalysisSupport.SPICE_EXPORT, AnalysisSupport.TRANSIENT_EXPORT]
    if linear_dc:
        support.append(AnalysisSupport.LINEAR_DC)
    return PartSpec(
        kind=kind,
        display_name=display_name,
        ref_prefix=ref_prefix,
        family=ComponentFamily.PASSIVE,
        pins=_two_pin_passive_pins(),
        icon=icon,
        spice=SpiceSpec(
            prefix=spice_prefix,
            pin_order=("1", "2"),
            value_parameter=value_parameter,
            default_value=default_value,
            model_name=None,
            model_definition=None,
        ),
        generation_tags=tags,
        analysis_support=tuple(support),
    )


def _two_pin_passive_pins() -> tuple[PinSpec, PinSpec]:
    """Return standard left/right passive pins."""

    return (
        PinSpec("1", PinKind.PASSIVE, PinSide.LEFT),
        PinSpec("2", PinKind.PASSIVE, PinSide.RIGHT),
    )


def _power_source(
    kind: str,
    display_name: str,
    ref_prefix: str,
    spice_prefix: str,
    value_parameter: str,
    default_value: str,
    tags: tuple[str, ...],
    positive_pin_kind: PinKind,
) -> PartSpec:
    """Build a two-terminal independent source specification."""

    return PartSpec(
        kind=kind,
        display_name=display_name,
        ref_prefix=ref_prefix,
        family=ComponentFamily.SOURCE,
        pins=(
            PinSpec("p", positive_pin_kind, PinSide.TOP),
            PinSpec("n", PinKind.PASSIVE, PinSide.BOTTOM),
        ),
        icon="source",
        spice=SpiceSpec(
            prefix=spice_prefix,
            pin_order=("p", "n"),
            value_parameter=value_parameter,
            default_value=default_value,
            model_name=None,
            model_definition=None,
        ),
        generation_tags=tags,
        analysis_support=(
            AnalysisSupport.SPICE_EXPORT,
            AnalysisSupport.LINEAR_DC,
            AnalysisSupport.TRANSIENT_EXPORT,
        ),
    )


def _diode_like(
    kind: str,
    display_name: str,
    model_name: str,
    model_definition: str,
    tags: tuple[str, ...],
) -> PartSpec:
    """Build a two-pin semiconductor diode specification."""

    return PartSpec(
        kind=kind,
        display_name=display_name,
        ref_prefix="D",
        family=ComponentFamily.SEMICONDUCTOR,
        pins=(
            PinSpec("a", PinKind.PASSIVE, PinSide.LEFT),
            PinSpec("k", PinKind.PASSIVE, PinSide.RIGHT),
        ),
        icon="diode",
        spice=SpiceSpec(
            prefix="D",
            pin_order=("a", "k"),
            value_parameter=None,
            default_value="",
            model_name=model_name,
            model_definition=model_definition,
        ),
        generation_tags=tags,
        analysis_support=(
            AnalysisSupport.SPICE_EXPORT,
            AnalysisSupport.TRANSIENT_EXPORT,
        ),
    )


def _transistor(
    kind: str,
    display_name: str,
    model_name: str,
    model_definition: str,
    tags: tuple[str, ...],
) -> PartSpec:
    """Build a three-pin BJT specification."""

    return PartSpec(
        kind=kind,
        display_name=display_name,
        ref_prefix="Q",
        family=ComponentFamily.SEMICONDUCTOR,
        pins=(
            PinSpec("c", PinKind.OPEN_COLLECTOR, PinSide.TOP),
            PinSpec("b", PinKind.INPUT, PinSide.LEFT),
            PinSpec("e", PinKind.OPEN_EMITTER, PinSide.BOTTOM),
        ),
        icon="transistor",
        spice=SpiceSpec(
            prefix="Q",
            pin_order=("c", "b", "e"),
            value_parameter=None,
            default_value="",
            model_name=model_name,
            model_definition=model_definition,
        ),
        generation_tags=tags,
        analysis_support=(
            AnalysisSupport.SPICE_EXPORT,
            AnalysisSupport.TRANSIENT_EXPORT,
        ),
    )


def _mosfet(
    kind: str,
    display_name: str,
    model_name: str,
    model_definition: str,
    tags: tuple[str, ...],
) -> PartSpec:
    """Build a three-pin MOSFET specification."""

    return PartSpec(
        kind=kind,
        display_name=display_name,
        ref_prefix="Q",
        family=ComponentFamily.SEMICONDUCTOR,
        pins=(
            PinSpec("d", PinKind.PASSIVE, PinSide.TOP),
            PinSpec("g", PinKind.INPUT, PinSide.LEFT),
            PinSpec("s", PinKind.PASSIVE, PinSide.BOTTOM),
        ),
        icon="mosfet",
        spice=SpiceSpec(
            prefix="M",
            pin_order=("d", "g", "s", "s"),
            value_parameter=None,
            default_value="",
            model_name=model_name,
            model_definition=model_definition,
        ),
        generation_tags=tags,
        analysis_support=(
            AnalysisSupport.SPICE_EXPORT,
            AnalysisSupport.TRANSIENT_EXPORT,
        ),
    )


def _jfet(
    kind: str,
    display_name: str,
    model_name: str,
    model_definition: str,
    tags: tuple[str, ...],
) -> PartSpec:
    """Build a three-pin JFET specification."""

    return PartSpec(
        kind=kind,
        display_name=display_name,
        ref_prefix="J",
        family=ComponentFamily.SEMICONDUCTOR,
        pins=(
            PinSpec("d", PinKind.PASSIVE, PinSide.TOP),
            PinSpec("g", PinKind.INPUT, PinSide.LEFT),
            PinSpec("s", PinKind.PASSIVE, PinSide.BOTTOM),
        ),
        icon="jfet",
        spice=SpiceSpec(
            prefix="J",
            pin_order=("d", "g", "s"),
            value_parameter=None,
            default_value="",
            model_name=model_name,
            model_definition=model_definition,
        ),
        generation_tags=tags,
        analysis_support=(
            AnalysisSupport.SPICE_EXPORT,
            AnalysisSupport.TRANSIENT_EXPORT,
        ),
    )


def _controlled_source(
    kind: str,
    display_name: str,
    ref_prefix: str,
    tags: tuple[str, ...],
    linear_dc: bool,
) -> PartSpec:
    """Build a voltage-controlled source specification."""

    support = [AnalysisSupport.SPICE_EXPORT, AnalysisSupport.TRANSIENT_EXPORT]
    if linear_dc:
        support.append(AnalysisSupport.LINEAR_DC)
    return PartSpec(
        kind=kind,
        display_name=display_name,
        ref_prefix=ref_prefix,
        family=ComponentFamily.CONTROLLED_SOURCE,
        pins=(
            PinSpec("p", PinKind.OUTPUT, PinSide.RIGHT),
            PinSpec("n", PinKind.PASSIVE, PinSide.RIGHT),
            PinSpec("cp", PinKind.INPUT, PinSide.LEFT),
            PinSpec("cn", PinKind.INPUT, PinSide.LEFT),
        ),
        icon="controlled_source",
        spice=SpiceSpec(
            prefix=ref_prefix,
            pin_order=("p", "n", "cp", "cn"),
            value_parameter="gain",
            default_value="1",
            model_name=None,
            model_definition=None,
        ),
        generation_tags=tags,
        analysis_support=tuple(support),
    )


def _logic_gate(kind: str, display_name: str, pin_count: int) -> PartSpec:
    """Build a generic logic gate specification."""

    input_pins = tuple(
        PinSpec(f"in{idx}", PinKind.INPUT, PinSide.LEFT)
        for idx in range(1, pin_count + 1)
    )
    pins = input_pins + (
        PinSpec("out", PinKind.OUTPUT, PinSide.RIGHT),
        PinSpec("vcc", PinKind.POWER_IN, PinSide.TOP),
        PinSpec("gnd", PinKind.POWER_IN, PinSide.BOTTOM),
    )
    model_name = f"RLVR_{kind.upper()}"
    return PartSpec(
        kind=kind,
        display_name=display_name,
        ref_prefix="U",
        family=ComponentFamily.LOGIC,
        pins=pins,
        icon="logic",
        spice=SpiceSpec(
            prefix="X",
            pin_order=tuple(pin.name for pin in pins),
            value_parameter=None,
            default_value="",
            model_name=model_name,
            model_definition=f".subckt {model_name} {' '.join(pin.name for pin in pins)}\n.ends {model_name}",
        ),
        generation_tags=("logic", "digital"),
        analysis_support=(
            AnalysisSupport.SPICE_EXPORT,
            AnalysisSupport.TRANSIENT_EXPORT,
        ),
    )


def _build_default_part_catalog() -> Mapping[str, PartSpec]:
    """Build the immutable built-in part catalog."""

    specs: dict[str, PartSpec] = {}

    def add(spec: PartSpec) -> None:
        specs[spec.kind] = spec

    add(
        PartSpec(
            kind="ground",
            display_name="Ground",
            ref_prefix="GND",
            family=ComponentFamily.POWER,
            pins=(PinSpec("0", PinKind.PASSIVE, PinSide.TOP),),
            icon="ground",
            spice=None,
            generation_tags=("power", "reference"),
            analysis_support=(AnalysisSupport.SPICE_EXPORT, AnalysisSupport.LINEAR_DC),
        )
    )
    add(
        _two_pin_passive(
            "resistor",
            "Resistor",
            "R",
            "resistor",
            "R",
            "resistance_ohm",
            "1k",
            ("passive", "load", "divider", "filter"),
            True,
        )
    )
    add(
        _two_pin_passive(
            "capacitor",
            "Capacitor",
            "C",
            "capacitor",
            "C",
            "capacitance_f",
            "1u",
            ("passive", "filter", "decoupling", "transient"),
            False,
        )
    )
    add(
        _two_pin_passive(
            "inductor",
            "Inductor",
            "L",
            "inductor",
            "L",
            "inductance_h",
            "1m",
            ("passive", "filter", "transient"),
            False,
        )
    )
    add(
        _two_pin_passive(
            "lamp",
            "Lamp",
            "LA",
            "lamp",
            "R",
            "resistance_ohm",
            "100",
            ("load", "electromechanical"),
            True,
        )
    )
    add(
        _two_pin_passive(
            "motor",
            "Motor",
            "M",
            "motor",
            "R",
            "resistance_ohm",
            "25",
            ("load", "electromechanical"),
            True,
        )
    )
    add(
        _power_source(
            "voltage_source_dc",
            "DC Voltage Source",
            "V",
            "V",
            "voltage_v",
            "DC 5",
            ("source", "power"),
            PinKind.POWER_OUT,
        )
    )
    add(
        _power_source(
            "current_source_dc",
            "DC Current Source",
            "I",
            "I",
            "current_a",
            "DC 0.001",
            ("source", "bias"),
            PinKind.PASSIVE,
        )
    )
    add(
        _diode_like(
            "diode",
            "Diode",
            "D_RLVR",
            ".model D_RLVR D(Is=1e-14 Rs=0.1 N=1)",
            ("semiconductor", "rectifier", "clamp"),
        )
    )
    add(
        _diode_like(
            "led",
            "LED",
            "D_LED_RLVR",
            ".model D_LED_RLVR D(Is=1e-20 Rs=5 N=2 Bv=6 Ibv=10u)",
            ("semiconductor", "indicator", "load"),
        )
    )
    add(
        _diode_like(
            "zener",
            "Zener Diode",
            "D_ZENER_RLVR",
            ".model D_ZENER_RLVR D(Is=1e-14 Rs=2 Bv=5.1 Ibv=1m)",
            ("semiconductor", "clamp", "reference"),
        )
    )
    add(
        _transistor(
            "bjt_npn",
            "NPN BJT",
            "Q_NPN_RLVR",
            ".model Q_NPN_RLVR NPN(Is=1e-15 Bf=120)",
            ("semiconductor", "switch", "amplifier"),
        )
    )
    add(
        _transistor(
            "bjt_pnp",
            "PNP BJT",
            "Q_PNP_RLVR",
            ".model Q_PNP_RLVR PNP(Is=1e-15 Bf=80)",
            ("semiconductor", "switch", "amplifier"),
        )
    )
    add(
        _mosfet(
            "mosfet_n",
            "N-MOSFET",
            "M_NMOS_RLVR",
            ".model M_NMOS_RLVR NMOS(Level=1 Vto=2 Kp=1m)",
            ("semiconductor", "switch"),
        )
    )
    add(
        _mosfet(
            "mosfet_p",
            "P-MOSFET",
            "M_PMOS_RLVR",
            ".model M_PMOS_RLVR PMOS(Level=1 Vto=-2 Kp=1m)",
            ("semiconductor", "switch"),
        )
    )
    add(
        _jfet(
            "jfet_n",
            "N-JFET",
            "J_NJFET_RLVR",
            ".model J_NJFET_RLVR NJF(Beta=1m Vto=-2 Lambda=0.01)",
            ("semiconductor", "amplifier"),
        )
    )
    add(
        _jfet(
            "jfet_p",
            "P-JFET",
            "J_PJFET_RLVR",
            ".model J_PJFET_RLVR PJF(Beta=1m Vto=2 Lambda=0.01)",
            ("semiconductor", "amplifier"),
        )
    )
    add(
        _controlled_source(
            "vcvs",
            "Voltage-Controlled Voltage Source",
            "E",
            ("source", "controlled", "linear"),
            True,
        )
    )
    add(
        _controlled_source(
            "vccs",
            "Voltage-Controlled Current Source",
            "G",
            ("source", "controlled", "linear"),
            True,
        )
    )
    add(
        PartSpec(
            kind="pullup_resistor",
            display_name="Pull-up Resistor",
            ref_prefix="RPU",
            family=ComponentFamily.PASSIVE,
            pins=(
                PinSpec("net", PinKind.PULLUP, PinSide.LEFT),
                PinSpec("rail", PinKind.POWER_IN, PinSide.TOP),
            ),
            icon="resistor",
            spice=SpiceSpec(
                prefix="R",
                pin_order=("net", "rail"),
                value_parameter="resistance_ohm",
                default_value="10k",
                model_name=None,
                model_definition=None,
            ),
            generation_tags=("passive", "bias", "logic", "pullup"),
            analysis_support=(
                AnalysisSupport.SPICE_EXPORT,
                AnalysisSupport.LINEAR_DC,
                AnalysisSupport.TRANSIENT_EXPORT,
            ),
        )
    )
    add(
        PartSpec(
            kind="pulldown_resistor",
            display_name="Pull-down Resistor",
            ref_prefix="RPD",
            family=ComponentFamily.PASSIVE,
            pins=(
                PinSpec("net", PinKind.PULLDOWN, PinSide.LEFT),
                PinSpec("rail", PinKind.POWER_IN, PinSide.BOTTOM),
            ),
            icon="resistor",
            spice=SpiceSpec(
                prefix="R",
                pin_order=("net", "rail"),
                value_parameter="resistance_ohm",
                default_value="10k",
                model_name=None,
                model_definition=None,
            ),
            generation_tags=("passive", "bias", "logic", "pulldown"),
            analysis_support=(
                AnalysisSupport.SPICE_EXPORT,
                AnalysisSupport.LINEAR_DC,
                AnalysisSupport.TRANSIENT_EXPORT,
            ),
        )
    )
    add(
        PartSpec(
            kind="ideal_switch",
            display_name="Ideal Switch",
            ref_prefix="S",
            family=ComponentFamily.SWITCH,
            pins=(
                PinSpec("1", PinKind.PASSIVE, PinSide.LEFT),
                PinSpec("2", PinKind.PASSIVE, PinSide.RIGHT),
            ),
            icon="switch",
            spice=SpiceSpec(
                prefix="R",
                pin_order=("1", "2"),
                value_parameter="state_resistance_ohm",
                default_value="1e12",
                model_name=None,
                model_definition=None,
            ),
            generation_tags=("switch", "control"),
            analysis_support=(
                AnalysisSupport.SPICE_EXPORT,
                AnalysisSupport.LINEAR_DC,
                AnalysisSupport.TRANSIENT_EXPORT,
            ),
        )
    )
    add(
        PartSpec(
            kind="relay",
            display_name="Relay",
            ref_prefix="K",
            family=ComponentFamily.ELECTROMECHANICAL,
            pins=(
                PinSpec("coil_p", PinKind.PASSIVE, PinSide.LEFT),
                PinSpec("coil_n", PinKind.PASSIVE, PinSide.LEFT),
                PinSpec("com", PinKind.PASSIVE, PinSide.BOTTOM),
                PinSpec("no", PinKind.PASSIVE, PinSide.RIGHT),
                PinSpec("nc", PinKind.PASSIVE, PinSide.RIGHT),
            ),
            icon="relay",
            spice=SpiceSpec(
                prefix="X",
                pin_order=("coil_p", "coil_n", "com", "no", "nc"),
                value_parameter=None,
                default_value="",
                model_name="RLVR_RELAY",
                model_definition=".subckt RLVR_RELAY coil_p coil_n com no nc\n.ends RLVR_RELAY",
            ),
            generation_tags=("switch", "electromechanical", "load"),
            analysis_support=(AnalysisSupport.SPICE_EXPORT,),
        )
    )
    add(
        PartSpec(
            kind="op_amp",
            display_name="Ideal Op Amp",
            ref_prefix="U",
            family=ComponentFamily.INTEGRATED,
            pins=(
                PinSpec("noninv", PinKind.INPUT, PinSide.LEFT),
                PinSpec("inv", PinKind.INPUT, PinSide.LEFT),
                PinSpec("vpos", PinKind.POWER_IN, PinSide.TOP),
                PinSpec("vneg", PinKind.POWER_IN, PinSide.BOTTOM),
                PinSpec("out", PinKind.OUTPUT, PinSide.RIGHT),
            ),
            icon="opamp",
            spice=SpiceSpec(
                prefix="X",
                pin_order=("noninv", "inv", "vpos", "vneg", "out"),
                value_parameter=None,
                default_value="",
                model_name="RLVR_IDEAL_OPAMP",
                model_definition=".subckt RLVR_IDEAL_OPAMP noninv inv vpos vneg out\nEOUT out 0 noninv inv 1e6\n.ends RLVR_IDEAL_OPAMP",
            ),
            generation_tags=("amplifier", "filter", "integrated"),
            analysis_support=(AnalysisSupport.SPICE_EXPORT,),
        )
    )
    add(
        PartSpec(
            kind="comparator",
            display_name="Comparator",
            ref_prefix="U",
            family=ComponentFamily.INTEGRATED,
            pins=(
                PinSpec("noninv", PinKind.INPUT, PinSide.LEFT),
                PinSpec("inv", PinKind.INPUT, PinSide.LEFT),
                PinSpec("vpos", PinKind.POWER_IN, PinSide.TOP),
                PinSpec("vneg", PinKind.POWER_IN, PinSide.BOTTOM),
                PinSpec("out", PinKind.OUTPUT, PinSide.RIGHT),
            ),
            icon="opamp",
            spice=SpiceSpec(
                prefix="X",
                pin_order=("noninv", "inv", "vpos", "vneg", "out"),
                value_parameter=None,
                default_value="",
                model_name="RLVR_COMPARATOR",
                model_definition=(
                    ".subckt RLVR_COMPARATOR noninv inv vpos vneg out\n"
                    "EOUT out 0 noninv inv 1e6\n"
                    ".ends RLVR_COMPARATOR"
                ),
            ),
            generation_tags=("comparator", "integrated", "threshold"),
            analysis_support=(AnalysisSupport.SPICE_EXPORT,),
        )
    )
    add(
        PartSpec(
            kind="generic_ic",
            display_name="Generic IC",
            ref_prefix="U",
            family=ComponentFamily.INTEGRATED,
            pins=(
                PinSpec("in1", PinKind.INPUT, PinSide.LEFT),
                PinSpec("in2", PinKind.INPUT, PinSide.LEFT),
                PinSpec("out1", PinKind.OUTPUT, PinSide.RIGHT),
                PinSpec("vcc", PinKind.POWER_IN, PinSide.TOP),
                PinSpec("gnd", PinKind.POWER_IN, PinSide.BOTTOM),
            ),
            icon="ic",
            spice=SpiceSpec(
                prefix="X",
                pin_order=("in1", "in2", "out1", "vcc", "gnd"),
                value_parameter=None,
                default_value="",
                model_name="RLVR_GENERIC_IC",
                model_definition=(
                    ".subckt RLVR_GENERIC_IC in1 in2 out1 vcc gnd\n"
                    ".ends RLVR_GENERIC_IC"
                ),
            ),
            generation_tags=("integrated", "block", "generic"),
            analysis_support=(AnalysisSupport.SPICE_EXPORT,),
        )
    )
    add(
        PartSpec(
            kind="transformer",
            display_name="Transformer",
            ref_prefix="T",
            family=ComponentFamily.ELECTROMECHANICAL,
            pins=(
                PinSpec("p1", PinKind.PASSIVE, PinSide.LEFT),
                PinSpec("p2", PinKind.PASSIVE, PinSide.LEFT),
                PinSpec("s1", PinKind.PASSIVE, PinSide.RIGHT),
                PinSpec("s2", PinKind.PASSIVE, PinSide.RIGHT),
            ),
            icon="transformer",
            spice=SpiceSpec(
                prefix="X",
                pin_order=("p1", "p2", "s1", "s2"),
                value_parameter=None,
                default_value="",
                model_name="RLVR_TRANSFORMER",
                model_definition=".subckt RLVR_TRANSFORMER p1 p2 s1 s2\n.ends RLVR_TRANSFORMER",
            ),
            generation_tags=("coupled", "power", "ac"),
            analysis_support=(AnalysisSupport.SPICE_EXPORT,),
        )
    )
    add(
        PartSpec(
            kind="connector_2",
            display_name="Two Pin Connector",
            ref_prefix="J",
            family=ComponentFamily.CONNECTOR,
            pins=(
                PinSpec("1", PinKind.PASSIVE, PinSide.LEFT),
                PinSpec("2", PinKind.PASSIVE, PinSide.LEFT),
            ),
            icon="connector",
            spice=None,
            generation_tags=("connector", "io"),
            analysis_support=(),
        )
    )
    add(
        PartSpec(
            kind="voltmeter",
            display_name="Voltmeter",
            ref_prefix="VM",
            family=ComponentFamily.MEASUREMENT,
            pins=(
                PinSpec("p", PinKind.INPUT, PinSide.TOP),
                PinSpec("n", PinKind.INPUT, PinSide.BOTTOM),
            ),
            icon="meter",
            spice=SpiceSpec(
                prefix="R",
                pin_order=("p", "n"),
                value_parameter="resistance_ohm",
                default_value="1e12",
                model_name=None,
                model_definition=None,
            ),
            generation_tags=("measurement", "probe"),
            analysis_support=(AnalysisSupport.SPICE_EXPORT,),
        )
    )
    add(
        PartSpec(
            kind="ammeter",
            display_name="Ammeter",
            ref_prefix="AM",
            family=ComponentFamily.MEASUREMENT,
            pins=(
                PinSpec("p", PinKind.PASSIVE, PinSide.LEFT),
                PinSpec("n", PinKind.PASSIVE, PinSide.RIGHT),
            ),
            icon="meter",
            spice=SpiceSpec(
                prefix="V",
                pin_order=("p", "n"),
                value_parameter="voltage_v",
                default_value="DC 0",
                model_name=None,
                model_definition=None,
            ),
            generation_tags=("measurement", "probe"),
            analysis_support=(AnalysisSupport.SPICE_EXPORT,),
        )
    )
    add(_logic_gate("and_gate", "AND Gate", 2))
    add(_logic_gate("or_gate", "OR Gate", 2))
    add(_logic_gate("not_gate", "NOT Gate", 1))
    return MappingProxyType(specs)


_DEFAULT_PART_CATALOG = _build_default_part_catalog()
