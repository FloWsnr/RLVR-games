"""Tests for procedural motif catalog infrastructure."""

from random import Random
from typing import Mapping

from rlvr_physics.tasks.physics.circuits import (
    Circuit,
    CircuitBuilder,
    CircuitMotif,
    InstantiatedMotif,
    MotifPort,
    MotifPortRole,
    check_circuit,
    default_motif_weights,
    default_motifs,
    default_part_catalog,
)
from rlvr_physics.tasks.physics.circuits.motifs import (
    MotifContext,
    choose_motif,
)


class _MotifTestContext:
    """Minimal motif context for catalog tests."""

    def __init__(self) -> None:
        """Initialize an empty motif test context."""

        self.builder = CircuitBuilder("motif", default_part_catalog())
        self.rng = Random(123)
        self.counters: dict[str, int] = {}
        self.non_ground_count = 0
        self.node_counter = 0
        self.motif_counters: dict[str, int] = {}
        self.negative_supply_nets: set[str] = set()

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

    def motif_instance_id(self, motif_name: str) -> str:
        """Return a fresh deterministic motif instance id."""

        number = self.motif_counters.get(motif_name, 0) + 1
        self.motif_counters[motif_name] = number
        return f"{motif_name}_{number}"

    def add_negative_supply(
        self, net: str, motif_name: str, instance_id: str
    ) -> tuple[str, ...]:
        """Declare one negative supply port per generated net."""

        if net in self.negative_supply_nets:
            return ()
        self.negative_supply_nets.add(net)
        self.builder.add_net(net)
        self.builder.add_net("0")
        return ()


EXPECTED_DEFAULT_MOTIFS = (
    "supply_port",
    "bridge_rectifier",
    "crc_power_filter",
    "zener_shunt_regulator",
    "rc_low_pass_filter",
    "rc_high_pass_filter",
    "passive_rlc_band_pass_filter",
    "twin_t_notch_filter",
    "fixed_bias_bjt_common_emitter_amplifier",
    "voltage_divider_biased_common_emitter_amplifier",
    "common_collector_emitter_follower",
    "bjt_differential_pair",
    "common_source_fet_amplifier",
    "tuned_lc_band_pass_amplifier",
    "inverting_op_amp_amplifier",
    "non_inverting_op_amp_amplifier",
    "differential_op_amp_subtractor",
    "voltage_comparator",
    "schmitt_trigger_comparator",
    "rc_phase_shift_oscillator",
    "wien_bridge_oscillator",
    "colpitts_lc_oscillator",
    "pierce_crystal_oscillator",
    "photodiode_transimpedance_amplifier",
    "wheatstone_bridge_instrumentation_amplifier",
    "timer_555_astable_oscillator",
    "timer_555_monostable_one_shot",
    "npn_low_side_relay_driver",
    "nmos_low_side_pwm_driver",
    "asynchronous_buck_converter",
    "asynchronous_boost_converter",
    "diode_capacitor_voltage_doubler",
    "peak_detector_envelope_detector",
    "precision_half_wave_rectifier",
    "sample_and_hold",
    "two_input_nand_gate",
    "cross_coupled_nand_sr_latch",
    "half_adder",
    "four_bit_synchronous_binary_counter",
    "inline_load_current_meter",
    "battery_powered_led_indicator",
    "two_pin_external_input_connector_rc_filter",
    "generic_ic_powered_logic_block",
    "explicit_ground_reference_node",
    "ideal_switch_power_disconnect",
    "looped_inductor_parallel_resonant_tank",
    "n_jfet_source_follower_buffer",
    "p_jfet_high_side_current_source",
    "led_power_indicator",
    "pmos_high_side_load_switch",
    "two_input_or_gate_with_led_output",
    "bulk_polarized_supply_capacitor",
    "visible_power_rail_distribution",
    "default_low_digital_input",
    "active_low_reset_pullup",
    "rc_debounced_pushbutton_input",
    "test_point_on_filtered_signal",
    "variable_cutoff_rc_low_pass",
    "vccs_voltage_to_current_driver",
    "vcvs_ideal_voltage_gain_block",
    "voltage_divider_with_voltmeter",
)

NEW_MOTIF_ANCHOR_PARTS = {
    "inline_load_current_meter": "ammeter",
    "battery_powered_led_indicator": "battery",
    "two_pin_external_input_connector_rc_filter": "connector_2",
    "generic_ic_powered_logic_block": "generic_ic",
    "explicit_ground_reference_node": "ground",
    "ideal_switch_power_disconnect": "ideal_switch",
    "looped_inductor_parallel_resonant_tank": "inductor_looped",
    "n_jfet_source_follower_buffer": "jfet_n",
    "p_jfet_high_side_current_source": "jfet_p",
    "led_power_indicator": "led",
    "pmos_high_side_load_switch": "mosfet_p",
    "two_input_or_gate_with_led_output": "or_gate",
    "bulk_polarized_supply_capacitor": "polarized_capacitor",
    "visible_power_rail_distribution": "power_rail",
    "default_low_digital_input": "pulldown_resistor",
    "active_low_reset_pullup": "pullup_resistor",
    "rc_debounced_pushbutton_input": "pushbutton_switch",
    "test_point_on_filtered_signal": "test_point",
    "variable_cutoff_rc_low_pass": "variable_resistor",
    "vccs_voltage_to_current_driver": "vccs",
    "vcvs_ideal_voltage_gain_block": "vcvs",
    "voltage_divider_with_voltmeter": "voltmeter",
}


def test_default_motif_catalog_matches_requested_netlist_list() -> None:
    motifs = default_motifs()

    assert tuple(motifs) == EXPECTED_DEFAULT_MOTIFS


def test_default_motif_weights_are_derived_from_catalog() -> None:
    motifs = default_motifs()
    weights = default_motif_weights()

    assert weights == {
        name: motif.default_weight
        for name, motif in motifs.items()
        if motif.default_weight > 0.0
    }


def test_default_motifs_cover_default_part_catalog() -> None:
    catalog = default_part_catalog()
    used_kinds: set[str] = set()

    for motif in default_motifs().values():
        ctx = _MotifTestContext()
        motif.build(ctx, {})
        used_kinds.update(part.kind for part in ctx.builder.freeze().parts)

    assert used_kinds == set(catalog)


def test_new_motifs_contain_their_anchor_parts() -> None:
    motifs = default_motifs()

    for motif_name, anchor_kind in NEW_MOTIF_ANCHOR_PARTS.items():
        ctx = _MotifTestContext()
        motif = motifs[motif_name]
        motif.build(ctx, {})
        part_kinds = {part.kind for part in ctx.builder.freeze().parts}

        assert anchor_kind in part_kinds


def test_reviewed_new_motif_connections_match_intended_biases() -> None:
    pmos = _built_motif_circuit("pmos_high_side_load_switch")
    default_low = _built_motif_circuit("default_low_digital_input")
    vccs = _built_motif_circuit("vccs_voltage_to_current_driver")
    vcvs = _built_motif_circuit("vcvs_ideal_voltage_gain_block")

    assert pmos.net_for_pin("MP1", "s") == "VCC"
    assert pmos.net_for_pin("RGPU1", "1") == "VCC"
    assert default_low.part_by_ref()["SW1"].parameters["state_resistance_ohm"] == 1e12
    assert vccs.net_for_pin("G1", "cp") == "VCC"
    assert vcvs.net_for_pin("E1", "cp") == "VCC"


def test_default_motifs_declare_valid_port_contracts() -> None:
    motifs = default_motifs()

    for motif in motifs.values():
        ctx = _MotifTestContext()
        instance = motif.build(ctx, {})
        local_nets = {
            connection.net.rsplit("_N", maxsplit=1)[0]
            for connection in ctx.builder.freeze().connections
        }
        local_nets.update({"0", "VCC", "VEE"})

        assert motif.ports, motif.name
        assert set(instance.port_nets) == {port.name for port in motif.ports}
        assert all(
            port.net in local_nets
            or {"VDD": "VCC", "VEXC": "VCC", "VBIAS": "VCC"}.get(port.net) in local_nets
            for port in motif.ports
        ), motif.name
        assert any(port.role is MotifPortRole.GROUND for port in motif.ports)
        assert any(
            port.role in {MotifPortRole.SOURCE, MotifPortRole.SINK, MotifPortRole.PROBE}
            for port in motif.ports
        ) or any(port.role is MotifPortRole.SUPPLY for port in motif.ports), motif.name


def test_default_motifs_accept_declared_required_port_bindings() -> None:
    motifs = default_motifs()

    for motif in motifs.values():
        ctx = _MotifTestContext()
        bindings = {
            port.name: _default_port_binding(port)
            for port in motif.ports
            if port.required
        }

        instance = motif.build(ctx, bindings)

        assert set(bindings) <= set(instance.port_nets), motif.name


def test_required_sink_ports_have_compatible_sources() -> None:
    motifs = default_motifs()
    source_signals = {
        port.signal
        for motif in motifs.values()
        for port in motif.ports
        if port.role is MotifPortRole.SOURCE
    }

    for motif in motifs.values():
        for port in motif.ports:
            if port.role is MotifPortRole.SINK and port.required:
                assert port.signal in source_signals, (motif.name, port)


def test_op_amp_feedback_nodes_are_not_external_ports() -> None:
    motif = default_motifs()["inverting_op_amp_amplifier"]
    port_nets = {port.net for port in motif.ports}

    assert "IN" in port_nets
    assert "OUT" in port_nets
    assert "NSUM" not in port_nets


def test_logic_output_pins_can_expose_nonstandard_source_ports() -> None:
    motif = default_motifs()["pierce_crystal_oscillator"]
    source_nets = {
        port.net for port in motif.ports if port.role is MotifPortRole.SOURCE
    }

    assert "NXOUT" in source_nets


def test_internal_excitation_nets_are_not_external_source_ports() -> None:
    motifs = default_motifs()
    bridge_sources = {
        port.net
        for port in motifs["bridge_rectifier"].ports
        if port.role is MotifPortRole.SOURCE
    }
    peak_detector_sources = {
        port.net
        for port in motifs["peak_detector_envelope_detector"].ports
        if port.role is MotifPortRole.SOURCE
    }

    assert "VRAW" in bridge_sources
    assert "ACPRI" not in bridge_sources
    assert "OUT" in peak_detector_sources
    assert "IN" not in peak_detector_sources


def test_choose_motif_returns_none_for_empty_catalog() -> None:
    motif = choose_motif(Random(123), {}, {"removed": 1.0})

    assert motif is None


def test_choose_motif_is_independent_of_weight_mapping_order() -> None:
    catalog = {
        "alpha": CircuitMotif("alpha", 1, 1.0, (), _no_op_motif),
        "beta": CircuitMotif("beta", 1, 1.0, (), _no_op_motif),
        "gamma": CircuitMotif("gamma", 1, 1.0, (), _no_op_motif),
    }
    forward_weights = {"alpha": 1.0, "beta": 1.0, "gamma": 1.0}
    reverse_weights = {
        "gamma": 1.0,
        "beta": 1.0,
        "alpha": 1.0,
    }

    forward = [
        _chosen_motif_name(Random(seed), catalog, forward_weights) for seed in range(20)
    ]
    reverse = [
        _chosen_motif_name(Random(seed), catalog, reverse_weights) for seed in range(20)
    ]

    assert forward == reverse


def test_default_motifs_build_declared_element_count() -> None:
    for motif in default_motifs().values():
        ctx = _MotifTestContext()

        motif.build(ctx, {})
        assert ctx.non_ground_count == motif.element_count, motif.name


def test_default_motifs_pass_erc_without_errors() -> None:
    catalog = default_part_catalog()

    for motif in default_motifs().values():
        ctx = _MotifTestContext()
        motif.build(ctx, {})
        _add_reference_ground(ctx)
        report = check_circuit(
            ctx.builder.freeze(),
            catalog,
        )

        assert not report.errors, (motif.name, report.errors)


def test_relay_driver_does_not_short_supply_through_default_contact() -> None:
    motif = default_motifs()["npn_low_side_relay_driver"]
    ctx = _MotifTestContext()

    motif.build(ctx, {})
    circuit = ctx.builder.freeze()

    assert circuit.net_for_pin("K1", "com") == circuit.net_for_pin("LA1", "1")
    assert circuit.net_for_pin("K1", "no") == "VCC"
    assert circuit.net_for_pin("K1", "nc") == "0"
    assert circuit.net_for_pin("K1", "com") != "VCC"


def _no_op_motif(_ctx: MotifContext, _bindings: Mapping[str, str]) -> InstantiatedMotif:
    """Return success without mutating a circuit."""

    return InstantiatedMotif("noop", "noop_1", {}, ())


def _add_reference_ground(ctx: _MotifTestContext) -> None:
    """Add a reference ground part to a motif-only test circuit."""

    ground = ctx.add_part("GND", "ground", "0", {}, {})
    ctx.builder.connect(ground, "0", "0")


def _built_motif_circuit(motif_name: str) -> Circuit:
    """Return a circuit built from one default motif."""

    motif = default_motifs()[motif_name]
    ctx = _MotifTestContext()
    motif.build(ctx, {})
    return ctx.builder.freeze()


def _chosen_motif_name(
    rng: Random,
    catalog: Mapping[str, CircuitMotif],
    weights: Mapping[str, float],
) -> str:
    """Choose a motif and return its name."""

    motif = choose_motif(rng, catalog, weights)
    assert motif is not None
    return motif.name


def _default_port_binding(port: MotifPort) -> str:
    """Return an external net for binding one declared motif port."""

    if port.role is MotifPortRole.GROUND:
        return "0"
    if port.net in {"VCC", "VDD", "VEXC", "VBIAS"}:
        return "VCC"
    return port.net
