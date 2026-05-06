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


def _polarized_passive_pins() -> tuple[PinSpec, PinSpec]:
    """Return left/right pins for a polarized passive component."""

    return (
        PinSpec("p", PinKind.PASSIVE, PinSide.LEFT),
        PinSpec("n", PinKind.PASSIVE, PinSide.RIGHT),
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
    emitter_on_top: bool,
) -> PartSpec:
    """Build a three-pin BJT specification."""

    pins = (
        (
            PinSpec("e", PinKind.OPEN_EMITTER, PinSide.TOP),
            PinSpec("b", PinKind.INPUT, PinSide.LEFT),
            PinSpec("c", PinKind.OPEN_COLLECTOR, PinSide.BOTTOM),
        )
        if emitter_on_top
        else (
            PinSpec("c", PinKind.OPEN_COLLECTOR, PinSide.TOP),
            PinSpec("b", PinKind.INPUT, PinSide.LEFT),
            PinSpec("e", PinKind.OPEN_EMITTER, PinSide.BOTTOM),
        )
    )
    return PartSpec(
        kind=kind,
        display_name=display_name,
        ref_prefix="Q",
        family=ComponentFamily.SEMICONDUCTOR,
        pins=pins,
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
    source_on_top: bool,
) -> PartSpec:
    """Build a three-pin MOSFET specification."""

    pins = (
        (
            PinSpec("s", PinKind.PASSIVE, PinSide.TOP),
            PinSpec("g", PinKind.INPUT, PinSide.LEFT),
            PinSpec("d", PinKind.PASSIVE, PinSide.BOTTOM),
        )
        if source_on_top
        else (
            PinSpec("d", PinKind.PASSIVE, PinSide.TOP),
            PinSpec("g", PinKind.INPUT, PinSide.LEFT),
            PinSpec("s", PinKind.PASSIVE, PinSide.BOTTOM),
        )
    )
    return PartSpec(
        kind=kind,
        display_name=display_name,
        ref_prefix="Q",
        family=ComponentFamily.SEMICONDUCTOR,
        pins=pins,
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
    source_on_top: bool,
) -> PartSpec:
    """Build a three-pin JFET specification."""

    pins = (
        (
            PinSpec("s", PinKind.PASSIVE, PinSide.TOP),
            PinSpec("g", PinKind.INPUT, PinSide.LEFT),
            PinSpec("d", PinKind.PASSIVE, PinSide.BOTTOM),
        )
        if source_on_top
        else (
            PinSpec("d", PinKind.PASSIVE, PinSide.TOP),
            PinSpec("g", PinKind.INPUT, PinSide.LEFT),
            PinSpec("s", PinKind.PASSIVE, PinSide.BOTTOM),
        )
    )
    return PartSpec(
        kind=kind,
        display_name=display_name,
        ref_prefix="J",
        family=ComponentFamily.SEMICONDUCTOR,
        pins=pins,
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
    input_names = tuple(pin.name for pin in input_pins)
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
            model_definition=_logic_model_definition(model_name, kind, input_names),
        ),
        generation_tags=("logic", "digital"),
        analysis_support=(
            AnalysisSupport.SPICE_EXPORT,
            AnalysisSupport.TRANSIENT_EXPORT,
        ),
    )


def _logic_model_definition(
    model_name: str, kind: str, input_names: tuple[str, ...]
) -> str:
    """Return a simple behavioral SPICE model for a logic gate."""

    pins = " ".join((*input_names, "out", "vcc", "gnd"))
    if kind == "not_gate":
        expression = "limit(V(vcc)-V(in1), 0, V(vcc))"
    elif kind == "nand_gate":
        expression = "limit(V(vcc)-V(in1)*V(in2)/max(V(vcc), 1e-9), 0, V(vcc))"
    elif kind == "and_gate":
        expression = "limit(V(in1)*V(in2)/max(V(vcc), 1e-9), 0, V(vcc))"
    elif kind == "xor_gate":
        expression = "limit(abs(V(in1)-V(in2)), 0, V(vcc))"
    else:
        expression = "limit(max(V(in1), V(in2)), 0, V(vcc))"
    return (
        f".subckt {model_name} {pins}\n"
        "RIN1 in1 gnd 1e12\n"
        f"{'RIN2 in2 gnd 1e12\n' if len(input_names) > 1 else ''}"
        f"BOUT out gnd V = {{{expression}}}\n"
        f".ends {model_name}"
    )


def _subcircuit_part(
    kind: str,
    display_name: str,
    ref_prefix: str,
    pins: tuple[PinSpec, ...],
    icon: str,
    model_name: str,
    model_definition: str,
    tags: tuple[str, ...],
) -> PartSpec:
    """Build a SPICE-exportable symbolic subcircuit part."""

    return PartSpec(
        kind=kind,
        display_name=display_name,
        ref_prefix=ref_prefix,
        family=ComponentFamily.INTEGRATED,
        pins=pins,
        icon=icon,
        spice=SpiceSpec(
            prefix="X",
            pin_order=tuple(pin.name for pin in pins),
            value_parameter=None,
            default_value="",
            model_name=model_name,
            model_definition=model_definition,
        ),
        generation_tags=tags,
        analysis_support=(AnalysisSupport.SPICE_EXPORT,),
    )


def _empty_subcircuit_model(model_name: str, pins: tuple[PinSpec, ...]) -> str:
    """Return a no-op SPICE subcircuit for visual helper parts."""

    pin_names = " ".join(pin.name for pin in pins)
    return f".subckt {model_name} {pin_names}\n.ends {model_name}"


def _dip_20_pins() -> tuple[PinSpec, ...]:
    """Return pin metadata for a generic twenty-pin DIP package."""

    return tuple(
        PinSpec(str(index), PinKind.BIDIRECTIONAL, PinSide.LEFT)
        for index in range(1, 11)
    ) + tuple(
        PinSpec(str(index), PinKind.BIDIRECTIONAL, PinSide.RIGHT)
        for index in range(20, 10, -1)
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
    power_rail_pins = (PinSpec("net", PinKind.PASSIVE, PinSide.BOTTOM),)
    add(
        PartSpec(
            kind="power_rail",
            display_name="Power Rail",
            ref_prefix="PWR",
            family=ComponentFamily.POWER,
            pins=power_rail_pins,
            icon="power_rail",
            spice=SpiceSpec(
                prefix="X",
                pin_order=("net",),
                value_parameter=None,
                default_value="",
                model_name="RLVR_POWER_RAIL",
                model_definition=_empty_subcircuit_model(
                    "RLVR_POWER_RAIL",
                    power_rail_pins,
                ),
            ),
            generation_tags=("power", "reference", "visual"),
            analysis_support=(AnalysisSupport.SPICE_EXPORT,),
        )
    )
    test_point_pins = (PinSpec("net", PinKind.PASSIVE, PinSide.LEFT),)
    add(
        PartSpec(
            kind="test_point",
            display_name="Test Point",
            ref_prefix="TP",
            family=ComponentFamily.CONNECTOR,
            pins=test_point_pins,
            icon="junction_dot",
            spice=SpiceSpec(
                prefix="X",
                pin_order=("net",),
                value_parameter=None,
                default_value="",
                model_name="RLVR_TEST_POINT",
                model_definition=_empty_subcircuit_model(
                    "RLVR_TEST_POINT",
                    test_point_pins,
                ),
            ),
            generation_tags=("connector", "test_point", "probe"),
            analysis_support=(AnalysisSupport.SPICE_EXPORT,),
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
            "variable_resistor",
            "Variable Resistor",
            "RV",
            "variable_resistor",
            "R",
            "resistance_ohm",
            "1k",
            ("passive", "load", "divider", "adjustable"),
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
        PartSpec(
            kind="polarized_capacitor",
            display_name="Polarized Capacitor",
            ref_prefix="C",
            family=ComponentFamily.PASSIVE,
            pins=_polarized_passive_pins(),
            icon="polarized_capacitor",
            spice=SpiceSpec(
                prefix="C",
                pin_order=("p", "n"),
                value_parameter="capacitance_f",
                default_value="1u",
                model_name=None,
                model_definition=None,
            ),
            generation_tags=(
                "passive",
                "filter",
                "decoupling",
                "polarized",
                "transient",
            ),
            analysis_support=(
                AnalysisSupport.SPICE_EXPORT,
                AnalysisSupport.TRANSIENT_EXPORT,
            ),
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
            "inductor_looped",
            "Looped Inductor",
            "L",
            "inductor_looped",
            "L",
            "inductance_h",
            "1m",
            ("passive", "filter", "transient"),
            False,
        )
    )
    add(
        PartSpec(
            kind="crystal",
            display_name="Crystal",
            ref_prefix="XTAL",
            family=ComponentFamily.PASSIVE,
            pins=_two_pin_passive_pins(),
            icon="crystal",
            spice=SpiceSpec(
                prefix="X",
                pin_order=("1", "2"),
                value_parameter=None,
                default_value="",
                model_name="RLVR_CRYSTAL",
                model_definition=(
                    ".subckt RLVR_CRYSTAL 1 2\n"
                    "RLOSS 1 nx 50\n"
                    "LM nx ny 10m\n"
                    "CM ny 2 20f\n"
                    "CP 1 2 5p\n"
                    ".ends RLVR_CRYSTAL"
                ),
            ),
            generation_tags=("timing", "oscillator"),
            analysis_support=(AnalysisSupport.SPICE_EXPORT,),
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
            "battery",
            "Battery",
            "BT",
            "V",
            "voltage_v",
            "DC 9",
            ("source", "power", "battery"),
            PinKind.POWER_OUT,
        )
    )
    add(
        PartSpec(
            kind="voltage_source_ac",
            display_name="AC Voltage Source",
            ref_prefix="VAC",
            family=ComponentFamily.SOURCE,
            pins=(
                PinSpec("p", PinKind.POWER_OUT, PinSide.TOP),
                PinSpec("n", PinKind.PASSIVE, PinSide.BOTTOM),
            ),
            icon="source",
            spice=SpiceSpec(
                prefix="V",
                pin_order=("p", "n"),
                value_parameter="voltage_spec",
                default_value="AC 1",
                model_name=None,
                model_definition=None,
            ),
            generation_tags=("source", "power", "ac"),
            analysis_support=(
                AnalysisSupport.SPICE_EXPORT,
                AnalysisSupport.TRANSIENT_EXPORT,
            ),
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
        _diode_like(
            "photodiode",
            "Photodiode",
            "D_PHOTO_RLVR",
            ".model D_PHOTO_RLVR D(Is=1e-12 Rs=5 Cjo=10p)",
            ("semiconductor", "sensor", "photodiode"),
        )
    )
    add(
        _transistor(
            "bjt_npn",
            "NPN BJT",
            "Q_NPN_RLVR",
            ".model Q_NPN_RLVR NPN(Is=1e-15 Bf=120)",
            ("semiconductor", "switch", "amplifier"),
            False,
        )
    )
    add(
        _transistor(
            "bjt_pnp",
            "PNP BJT",
            "Q_PNP_RLVR",
            ".model Q_PNP_RLVR PNP(Is=1e-15 Bf=80)",
            ("semiconductor", "switch", "amplifier"),
            True,
        )
    )
    add(
        _mosfet(
            "mosfet_n",
            "N-MOSFET",
            "M_NMOS_RLVR",
            ".model M_NMOS_RLVR NMOS(Level=1 Vto=2 Kp=1m)",
            ("semiconductor", "switch"),
            False,
        )
    )
    add(
        _mosfet(
            "mosfet_p",
            "P-MOSFET",
            "M_PMOS_RLVR",
            ".model M_PMOS_RLVR PMOS(Level=1 Vto=-2 Kp=1m)",
            ("semiconductor", "switch"),
            True,
        )
    )
    add(
        _jfet(
            "jfet_n",
            "N-JFET",
            "J_NJFET_RLVR",
            ".model J_NJFET_RLVR NJF(Beta=1m Vto=-2 Lambda=0.01)",
            ("semiconductor", "amplifier"),
            False,
        )
    )
    add(
        _jfet(
            "jfet_p",
            "P-JFET",
            "J_PJFET_RLVR",
            ".model J_PJFET_RLVR PJF(Beta=1m Vto=2 Lambda=0.01)",
            ("semiconductor", "amplifier"),
            True,
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
            kind="pushbutton_switch",
            display_name="Pushbutton Switch",
            ref_prefix="S",
            family=ComponentFamily.SWITCH,
            pins=(
                PinSpec("1", PinKind.PASSIVE, PinSide.LEFT),
                PinSpec("2", PinKind.PASSIVE, PinSide.RIGHT),
            ),
            icon="pushbutton_switch",
            spice=SpiceSpec(
                prefix="R",
                pin_order=("1", "2"),
                value_parameter="state_resistance_ohm",
                default_value="1e12",
                model_name=None,
                model_definition=None,
            ),
            generation_tags=("switch", "control", "momentary"),
            analysis_support=(
                AnalysisSupport.SPICE_EXPORT,
                AnalysisSupport.LINEAR_DC,
                AnalysisSupport.TRANSIENT_EXPORT,
            ),
        )
    )
    add(
        _subcircuit_part(
            "controlled_switch",
            "Controlled Switch",
            "S",
            (
                PinSpec("in", PinKind.PASSIVE, PinSide.LEFT),
                PinSpec("out", PinKind.PASSIVE, PinSide.RIGHT),
                PinSpec("ctrl", PinKind.INPUT, PinSide.TOP),
            ),
            "switch",
            "RLVR_CONTROLLED_SWITCH",
            (
                ".subckt RLVR_CONTROLLED_SWITCH in out ctrl\n"
                "SCTRL in out ctrl 0 RLVR_CONTROLLED_SWITCH_MODEL\n"
                "RCTRL ctrl 0 1e12\n"
                ".model RLVR_CONTROLLED_SWITCH_MODEL SW(Ron=0.1 Roff=1e12 Vt=2 Vh=0.1)\n"
                ".ends RLVR_CONTROLLED_SWITCH"
            ),
            ("switch", "control"),
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
                model_definition=(
                    ".subckt RLVR_RELAY coil_p coil_n com no nc\n"
                    "RCOIL coil_p coil_n 100\n"
                    "SNO com no coil_p coil_n RLVR_RELAY_NO\n"
                    "SNC com nc coil_n coil_p RLVR_RELAY_NC\n"
                    ".model RLVR_RELAY_NO SW(Ron=0.05 Roff=1e12 Vt=2 Vh=0.1)\n"
                    ".model RLVR_RELAY_NC SW(Ron=0.05 Roff=1e12 Vt=-2 Vh=0.1)\n"
                    ".ends RLVR_RELAY"
                ),
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
                model_definition=(
                    ".subckt RLVR_IDEAL_OPAMP noninv inv vpos vneg out\n"
                    "BOUT out vneg V = {limit(1e6*(V(noninv)-V(inv)), 0, V(vpos)-V(vneg))}\n"
                    ".ends RLVR_IDEAL_OPAMP"
                ),
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
                    "BOUT out vneg V = {limit(1e6*(V(noninv)-V(inv)), 0, V(vpos)-V(vneg))}\n"
                    ".ends RLVR_COMPARATOR"
                ),
            ),
            generation_tags=("comparator", "integrated", "threshold"),
            analysis_support=(AnalysisSupport.SPICE_EXPORT,),
        )
    )
    add(
        _subcircuit_part(
            "instrumentation_amplifier",
            "Instrumentation Amplifier",
            "UINA",
            (
                PinSpec("inp", PinKind.INPUT, PinSide.LEFT),
                PinSpec("inn", PinKind.INPUT, PinSide.LEFT),
                PinSpec("ref", PinKind.INPUT, PinSide.LEFT),
                PinSpec("rg1", PinKind.PASSIVE, PinSide.BOTTOM),
                PinSpec("rg2", PinKind.PASSIVE, PinSide.BOTTOM),
                PinSpec("vpos", PinKind.POWER_IN, PinSide.TOP),
                PinSpec("vneg", PinKind.POWER_IN, PinSide.BOTTOM),
                PinSpec("out", PinKind.OUTPUT, PinSide.RIGHT),
            ),
            "ic",
            "RLVR_INSTRUMENTATION_AMP",
            (
                ".subckt RLVR_INSTRUMENTATION_AMP inp inn ref rg1 rg2 vpos vneg out\n"
                "RINP inp ref 1e12\n"
                "RINN inn ref 1e12\n"
                "RGINT rg1 rg2 1e9\n"
                "EOUT out ref inp inn 100\n"
                ".ends RLVR_INSTRUMENTATION_AMP"
            ),
            ("amplifier", "instrumentation", "integrated"),
        )
    )
    add(
        _subcircuit_part(
            "timer_555",
            "555 Timer",
            "U",
            (
                PinSpec("gnd", PinKind.POWER_IN, PinSide.BOTTOM),
                PinSpec("vcc", PinKind.POWER_IN, PinSide.TOP),
                PinSpec("reset", PinKind.INPUT, PinSide.LEFT),
                PinSpec("ctrl", PinKind.INPUT, PinSide.LEFT),
                PinSpec("disch", PinKind.OPEN_COLLECTOR, PinSide.RIGHT),
                PinSpec("thresh", PinKind.INPUT, PinSide.LEFT),
                PinSpec("trig", PinKind.INPUT, PinSide.LEFT),
                PinSpec("out", PinKind.OUTPUT, PinSide.RIGHT),
            ),
            "ic",
            "RLVR_TIMER_555",
            (
                ".subckt RLVR_TIMER_555 gnd vcc reset ctrl disch thresh trig out\n"
                "RRESET reset vcc 1e12\n"
                "RCTRL ctrl gnd 1e12\n"
                "RTH thresh gnd 1e12\n"
                "RTRIG trig gnd 1e12\n"
                "RDISCH disch gnd 1e9\n"
                "VOUT out gnd PULSE(0 5 0 1u 1u 1m 2m)\n"
                ".ends RLVR_TIMER_555"
            ),
            ("timer", "oscillator", "integrated"),
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
                    "RIN1 in1 gnd 1e12\n"
                    "RIN2 in2 gnd 1e12\n"
                    "EOUT out1 gnd in1 in2 1\n"
                    ".ends RLVR_GENERIC_IC"
                ),
            ),
            generation_tags=("integrated", "block", "generic"),
            analysis_support=(AnalysisSupport.SPICE_EXPORT,),
        )
    )
    dip_20_pins = _dip_20_pins()
    add(
        PartSpec(
            kind="dip_20_ic",
            display_name="20-pin DIP IC",
            ref_prefix="U",
            family=ComponentFamily.INTEGRATED,
            pins=dip_20_pins,
            icon="ic",
            spice=SpiceSpec(
                prefix="X",
                pin_order=tuple(pin.name for pin in dip_20_pins),
                value_parameter=None,
                default_value="",
                model_name="RLVR_DIP_20_IC",
                model_definition=_empty_subcircuit_model(
                    "RLVR_DIP_20_IC",
                    dip_20_pins,
                ),
            ),
            generation_tags=("integrated", "block", "dip"),
            analysis_support=(AnalysisSupport.SPICE_EXPORT,),
        )
    )
    add(
        _subcircuit_part(
            "counter_4bit",
            "4-bit Synchronous Counter",
            "U",
            (
                PinSpec("clk", PinKind.INPUT, PinSide.LEFT),
                PinSpec("clr_n", PinKind.INPUT, PinSide.LEFT),
                PinSpec("load_n", PinKind.INPUT, PinSide.LEFT),
                PinSpec("enp", PinKind.INPUT, PinSide.LEFT),
                PinSpec("ent", PinKind.INPUT, PinSide.LEFT),
                PinSpec("a", PinKind.INPUT, PinSide.LEFT),
                PinSpec("b", PinKind.INPUT, PinSide.LEFT),
                PinSpec("c", PinKind.INPUT, PinSide.LEFT),
                PinSpec("d", PinKind.INPUT, PinSide.LEFT),
                PinSpec("vcc", PinKind.POWER_IN, PinSide.TOP),
                PinSpec("gnd", PinKind.POWER_IN, PinSide.BOTTOM),
                PinSpec("qa", PinKind.OUTPUT, PinSide.RIGHT),
                PinSpec("qb", PinKind.OUTPUT, PinSide.RIGHT),
                PinSpec("qc", PinKind.OUTPUT, PinSide.RIGHT),
                PinSpec("qd", PinKind.OUTPUT, PinSide.RIGHT),
                PinSpec("rco", PinKind.OUTPUT, PinSide.RIGHT),
            ),
            "ic",
            "RLVR_COUNTER_4BIT",
            (
                ".subckt RLVR_COUNTER_4BIT clk clr_n load_n enp ent a b c d vcc gnd qa qb qc qd rco\n"
                "RCLK clk gnd 1e12\n"
                "RCLR clr_n vcc 1e12\n"
                "RLOAD load_n vcc 1e12\n"
                "REN1 enp vcc 1e12\n"
                "REN2 ent vcc 1e12\n"
                "RA a gnd 1e12\n"
                "RB b gnd 1e12\n"
                "RC c gnd 1e12\n"
                "RD d gnd 1e12\n"
                "VQA qa gnd PULSE(0 5 0 1n 1n 1u 2u)\n"
                "VQB qb gnd PULSE(0 5 0 1n 1n 2u 4u)\n"
                "VQC qc gnd PULSE(0 5 0 1n 1n 4u 8u)\n"
                "VQD qd gnd PULSE(0 5 0 1n 1n 8u 16u)\n"
                "VRCO rco gnd PULSE(0 5 0 1n 1n 16u 32u)\n"
                ".ends RLVR_COUNTER_4BIT"
            ),
            ("logic", "counter", "digital"),
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
                model_definition=(
                    ".subckt RLVR_TRANSFORMER p1 p2 s1 s2\n"
                    "LPRI p1 p2 1m\n"
                    "LSEC s1 s2 1m\n"
                    "KCOUPLE LPRI LSEC 0.98\n"
                    ".ends RLVR_TRANSFORMER"
                ),
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
            spice=SpiceSpec(
                prefix="X",
                pin_order=("1", "2"),
                value_parameter=None,
                default_value="",
                model_name="RLVR_CONNECTOR_2",
                model_definition=".subckt RLVR_CONNECTOR_2 1 2\n.ends RLVR_CONNECTOR_2",
            ),
            generation_tags=("connector", "io"),
            analysis_support=(AnalysisSupport.SPICE_EXPORT,),
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
    add(_logic_gate("nand_gate", "NAND Gate", 2))
    add(_logic_gate("or_gate", "OR Gate", 2))
    add(_logic_gate("xor_gate", "XOR Gate", 2))
    add(_logic_gate("not_gate", "NOT Gate", 1))
    return MappingProxyType(specs)


_DEFAULT_PART_CATALOG = _build_default_part_catalog()
