"""Tests for canonical circuit construction."""

import pytest

from rlvr_physics.tasks.physics.circuits import (
    CircuitBuilder,
    CircuitTopologyError,
    default_catalog,
)


def divider_circuit():
    """Build a deterministic voltage divider test circuit."""

    builder = CircuitBuilder("divider", default_catalog())
    builder.add_part("V1", "voltage_source_dc", "12V", {"voltage_v": 12.0}, {})
    builder.add_part("GND1", "ground", "0", {}, {})
    builder.add_part("R1", "resistor", "1k", {"resistance_ohm": 1000.0}, {})
    builder.add_part("R2", "resistor", "2k", {"resistance_ohm": 2000.0}, {})
    builder.connect("V1", "p", "VIN")
    builder.connect("V1", "n", "0")
    builder.connect("GND1", "0", "0")
    builder.connect("R1", "1", "VIN")
    builder.connect("R1", "2", "MID")
    builder.connect("R2", "1", "MID")
    builder.connect("R2", "2", "0")
    return builder.freeze()


def test_circuit_builder_creates_stable_plain_data() -> None:
    circuit = divider_circuit()
    repeated = divider_circuit()

    assert circuit.to_plain_data() == repeated.to_plain_data()
    assert circuit.content_hash() == repeated.content_hash()
    assert circuit.net_for_pin("R1", "2") == "MID"
    assert tuple(part.ref for part in circuit.parts) == ("GND1", "R1", "R2", "V1")


def test_circuit_builder_rejects_unknown_pin() -> None:
    builder = CircuitBuilder("bad", default_catalog())
    builder.add_part("R1", "resistor", "1k", {"resistance_ohm": 1000.0}, {})

    with pytest.raises(CircuitTopologyError, match="unknown pin"):
        builder.connect("R1", "bad", "N1")


def test_circuit_builder_rejects_duplicate_reference() -> None:
    builder = CircuitBuilder("bad", default_catalog())
    builder.add_part("R1", "resistor", "1k", {"resistance_ohm": 1000.0}, {})

    with pytest.raises(CircuitTopologyError, match="duplicate"):
        builder.add_part("R1", "resistor", "2k", {"resistance_ohm": 2000.0}, {})
