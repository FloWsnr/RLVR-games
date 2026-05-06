"""Tests for procedural motif catalog infrastructure."""

from random import Random
from typing import Mapping

from rlvr_physics.tasks.physics.circuits import (
    AnalysisSupport,
    Circuit,
    CircuitBuilder,
    CircuitMotif,
    check_circuit,
    default_motif_weights,
    default_motifs,
    default_part_catalog,
    export_spice,
    operating_point_analysis,
)
from rlvr_physics.tasks.physics.circuits.motifs import (
    MotifContext,
    add_load_resistor,
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


EXPECTED_DEFAULT_MOTIFS = (
    "bridge_rectifier",
    "crc_power_filter",
    "zener_shunt_regulator",
    "series_pass_transistor_regulator",
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
    "class_ab_push_pull_power_amplifier",
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
    "dip20_ic_minimum_system",
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
    "dip20_ic_minimum_system": "dip_20_ic",
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
        assert motif.build(ctx, motif.element_count)
        used_kinds.update(part.kind for part in ctx.builder.freeze().parts)

    assert used_kinds == set(catalog)


def test_new_motifs_contain_their_anchor_parts() -> None:
    motifs = default_motifs()

    for motif_name, anchor_kind in NEW_MOTIF_ANCHOR_PARTS.items():
        ctx = _MotifTestContext()
        motif = motifs[motif_name]
        assert motif.build(ctx, motif.element_count)
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


def test_choose_motif_returns_none_for_empty_catalog() -> None:
    motif = choose_motif(Random(123), {}, {"removed": 1.0}, 10)

    assert motif is None


def test_choose_motif_is_independent_of_weight_mapping_order() -> None:
    catalog = {
        "alpha": CircuitMotif("alpha", 1, 1.0, _no_op_motif),
        "beta": CircuitMotif("beta", 1, 1.0, _no_op_motif),
        "gamma": CircuitMotif("gamma", 1, 1.0, _no_op_motif),
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


def test_load_resistor_fallback_builds_exportable_branch() -> None:
    catalog = default_part_catalog()
    ctx = _MotifTestContext()

    ctx.add_part("V", "voltage_source_dc", "5V", {"voltage_v": 5.0}, {})
    ctx.builder.connect("V1", "p", "VCC")
    ctx.builder.connect("V1", "n", "0")
    ctx.add_part("GND", "ground", "0", {}, {})
    ctx.builder.connect("GND1", "0", "0")
    add_load_resistor(ctx)
    circuit = ctx.builder.freeze()
    report = check_circuit(circuit, catalog, AnalysisSupport.SPICE_EXPORT)

    assert ctx.non_ground_count == 2
    assert circuit.net_for_pin("R1", "1") == "VCC"
    assert circuit.net_for_pin("R1", "2") == "0"
    assert not report.errors
    assert export_spice(circuit, catalog, operating_point_analysis()).text.endswith(
        ".op\n.end\n"
    )


def test_default_motifs_build_declared_element_count() -> None:
    for motif in default_motifs().values():
        ctx = _MotifTestContext()

        assert motif.build(ctx, motif.element_count)
        assert ctx.non_ground_count == motif.element_count, motif.name


def test_default_motifs_pass_erc_without_errors() -> None:
    catalog = default_part_catalog()

    for motif in default_motifs().values():
        ctx = _MotifTestContext()
        assert motif.build(ctx, motif.element_count)
        _add_reference_ground(ctx)
        report = check_circuit(
            ctx.builder.freeze(),
            catalog,
            AnalysisSupport.SPICE_EXPORT,
        )

        assert not report.errors, (motif.name, report.errors)


def test_default_motifs_export_to_spice() -> None:
    catalog = default_part_catalog()

    for motif in default_motifs().values():
        ctx = _MotifTestContext()
        assert motif.build(ctx, motif.element_count)
        _add_reference_ground(ctx)
        netlist = export_spice(
            ctx.builder.freeze(),
            catalog,
            operating_point_analysis(),
        )

        assert netlist.text.endswith(".op\n.end\n"), motif.name


def test_relay_driver_does_not_short_supply_through_default_contact() -> None:
    motif = default_motifs()["npn_low_side_relay_driver"]
    ctx = _MotifTestContext()

    assert motif.build(ctx, motif.element_count)
    circuit = ctx.builder.freeze()

    assert circuit.net_for_pin("K1", "com") == circuit.net_for_pin("LA1", "1")
    assert circuit.net_for_pin("K1", "no") == "VCC"
    assert circuit.net_for_pin("K1", "nc") == "0"
    assert circuit.net_for_pin("K1", "com") != "VCC"


def _no_op_motif(_ctx: MotifContext, remaining: int) -> bool:
    """Return success without mutating a circuit."""

    return remaining >= 0


def _add_reference_ground(ctx: _MotifTestContext) -> None:
    """Add a reference ground part to a motif-only test circuit."""

    ground = ctx.add_part("GND", "ground", "0", {}, {})
    ctx.builder.connect(ground, "0", "0")


def _built_motif_circuit(motif_name: str) -> Circuit:
    """Return a circuit built from one default motif."""

    motif = default_motifs()[motif_name]
    ctx = _MotifTestContext()
    assert motif.build(ctx, motif.element_count)
    return ctx.builder.freeze()


def _chosen_motif_name(
    rng: Random,
    catalog: Mapping[str, CircuitMotif],
    weights: Mapping[str, float],
) -> str:
    """Choose a motif and return its name."""

    motif = choose_motif(rng, catalog, weights, 10)
    assert motif is not None
    return motif.name
