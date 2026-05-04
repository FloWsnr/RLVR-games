"""Tests for procedural motif catalog coverage."""

from collections import Counter
from random import Random
from typing import Mapping

from rlvr_physics.tasks.physics.circuits import (
    AnalysisSupport,
    CircuitBuilder,
    check_circuit,
    default_motif_weights,
    default_motifs,
    default_part_catalog,
    export_spice,
    operating_point_analysis,
)
from rlvr_physics.tasks.physics.circuits.motifs import choose_motif


class _MotifTestContext:
    """Minimal motif context for catalog tests."""

    def __init__(self) -> None:
        """Initialize an empty motif test context."""

        self.builder = CircuitBuilder("motif", default_part_catalog())
        self.rng = Random(123)
        self.counters: dict[str, int] = {}
        self.non_ground_count = 0
        self.node_counter = 0

    def add_part(
        self,
        prefix: str,
        kind: str,
        value: str,
        parameters: Mapping[str, object],
        metadata: Mapping[str, object],
    ) -> str:
        """Add a numbered part to the test circuit."""

        number = self.counters.get(prefix, 0) + 1
        self.counters[prefix] = number
        ref = f"{prefix}{number}"
        self.builder.add_part(ref, kind, value, parameters, metadata)
        if kind != "ground":
            self.non_ground_count += 1
        return ref

    def node(self) -> str:
        """Return a fresh generated node name."""

        self.node_counter += 1
        return f"N{self.node_counter}"


def test_default_motif_weights_are_derived_from_catalog() -> None:
    motifs = default_motifs()
    weights = default_motif_weights()

    assert weights == {
        name: motif.default_weight
        for name, motif in motifs.items()
        if motif.default_weight > 0.0
    }


def test_default_motifs_build_declared_element_count() -> None:
    for motif in default_motifs().values():
        ctx = _MotifTestContext()

        assert motif.build(ctx, motif.element_count)
        assert ctx.non_ground_count == motif.element_count, motif.name


def test_choose_motif_is_independent_of_weight_mapping_order() -> None:
    forward_weights = {"divider": 1.0, "rc_lowpass": 1.0, "led_indicator": 1.0}
    reverse_weights = {
        "led_indicator": 1.0,
        "rc_lowpass": 1.0,
        "divider": 1.0,
    }

    forward = [_chosen_motif_name(Random(seed), forward_weights) for seed in range(20)]
    reverse = [_chosen_motif_name(Random(seed), reverse_weights) for seed in range(20)]

    assert forward == reverse


def test_default_motifs_cover_each_non_ground_part_multiple_times() -> None:
    catalog = default_part_catalog()
    part_counts: Counter[str] = Counter()

    for motif in default_motifs().values():
        ctx = _MotifTestContext()
        assert motif.build(ctx, motif.element_count)
        part_counts.update(part.kind for part in ctx.builder.freeze().parts)

    sparse_kinds = {
        kind: part_counts[kind]
        for kind in catalog
        if kind != "ground" and part_counts[kind] < 2
    }
    assert sparse_kinds == {}


def test_default_motifs_pass_erc_without_errors() -> None:
    catalog = default_part_catalog()

    for motif in default_motifs().values():
        ctx = _MotifTestContext()
        assert motif.build(ctx, motif.element_count)

        report = check_circuit(
            ctx.builder.freeze(),
            catalog,
            AnalysisSupport.SPICE_EXPORT,
        )

        assert not report.errors, (motif.name, report.errors)


def test_default_motifs_export_to_spice() -> None:
    catalog = default_part_catalog()

    for motif in default_motifs().values():
        circuit = _build_motif_circuit(motif.name)

        netlist = export_spice(circuit, catalog, operating_point_analysis())

        assert netlist.text.endswith(".op\n.end\n"), motif.name


def test_current_source_motifs_feed_loads_from_vcc() -> None:
    for name in ("current_bias_monitor", "current_led_clamp", "connector_loopback"):
        circuit = _build_motif_circuit(name)

        for part in circuit.parts:
            if part.kind == "current_source_dc":
                assert circuit.net_for_pin(part.ref, "p") == "VCC", name


def test_controlled_current_motifs_have_supply_feed() -> None:
    for name in ("controlled_current_load", "dual_controlled_source"):
        circuit = _build_motif_circuit(name)
        vcc_fed_resistors = [
            part
            for part in circuit.parts
            if part.kind == "resistor" and circuit.net_for_pin(part.ref, "1") == "VCC"
        ]

        assert vcc_fed_resistors, name


def test_active_filter_is_buffered_rc_lowpass() -> None:
    circuit = _build_motif_circuit("active_filter_stage")
    opamp = next(part for part in circuit.parts if part.kind == "op_amp")
    capacitor = next(part for part in circuit.parts if part.kind == "capacitor")

    assert circuit.net_for_pin(opamp.ref, "noninv") == circuit.net_for_pin(
        capacitor.ref, "1"
    )
    assert circuit.net_for_pin(capacitor.ref, "2") == "0"
    assert circuit.net_for_pin(opamp.ref, "inv") == circuit.net_for_pin(
        opamp.ref, "out"
    )


def test_relay_load_motifs_do_not_short_supply_in_nc_state() -> None:
    for name, load_kind in (
        ("relay_lamp_driver", "lamp"),
        ("relay_motor_driver", "motor"),
    ):
        circuit = _build_motif_circuit(name)
        relay = next(part for part in circuit.parts if part.kind == "relay")
        load = next(part for part in circuit.parts if part.kind == load_kind)

        assert circuit.net_for_pin(relay.ref, "com") == circuit.net_for_pin(
            load.ref, "1"
        )
        assert circuit.net_for_pin(relay.ref, "no") == "VCC"
        assert circuit.net_for_pin(relay.ref, "nc") == "0"


def test_connector_loopback_connects_both_connector_pins() -> None:
    circuit = _build_motif_circuit("connector_loopback")
    connector = next(part for part in circuit.parts if part.kind == "connector_2")
    jumper = next(part for part in circuit.parts if part.kind == "ideal_switch")

    assert circuit.net_for_pin(connector.ref, "1") != circuit.net_for_pin(
        connector.ref, "2"
    )
    assert circuit.net_for_pin(jumper.ref, "1") == circuit.net_for_pin(
        connector.ref, "1"
    )
    assert circuit.net_for_pin(jumper.ref, "2") == circuit.net_for_pin(
        connector.ref, "2"
    )


def test_polarity_specific_device_pins_face_supply_rails() -> None:
    catalog = default_part_catalog()

    assert catalog["bjt_pnp"].pin("e").side.value == "top"
    assert catalog["bjt_pnp"].pin("c").side.value == "bottom"
    assert catalog["mosfet_p"].pin("s").side.value == "top"
    assert catalog["mosfet_p"].pin("d").side.value == "bottom"
    assert catalog["jfet_p"].pin("s").side.value == "top"
    assert catalog["jfet_p"].pin("d").side.value == "bottom"


def test_bridge_has_differential_measurement_branch() -> None:
    circuit = _build_motif_circuit("bridge")
    meter = next(part for part in circuit.parts if part.kind == "voltmeter")

    assert circuit.net_for_pin(meter.ref, "p") != circuit.net_for_pin(meter.ref, "n")


def test_lc_tank_has_parallel_reactive_branch() -> None:
    circuit = _build_motif_circuit("lc_tank")
    capacitor = next(part for part in circuit.parts if part.kind == "capacitor")
    inductor = next(part for part in circuit.parts if part.kind == "inductor")

    assert circuit.net_for_pin(capacitor.ref, "1") == circuit.net_for_pin(
        inductor.ref, "1"
    )
    assert circuit.net_for_pin(capacitor.ref, "2") == circuit.net_for_pin(
        inductor.ref, "2"
    )


def _build_motif_circuit(name: str):
    """Build one named default motif circuit."""

    motif = default_motifs()[name]
    ctx = _MotifTestContext()
    assert motif.build(ctx, motif.element_count)
    return ctx.builder.freeze()


def _chosen_motif_name(rng: Random, weights: Mapping[str, float]) -> str:
    """Choose a motif and return its name."""

    motif = choose_motif(rng, default_motifs(), weights, 10)
    assert motif is not None
    return motif.name
