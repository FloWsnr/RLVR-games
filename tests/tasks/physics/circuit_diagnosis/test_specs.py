"""Tests for circuit diagnosis task specs and configuration."""

from dataclasses import replace

import pytest

from rlvr_physics.tasks.physics.circuit_diagnosis.specs import (
    DEFAULT_CONFIG,
    MAX_COMPONENT_COUNT,
    validate_circuit_diagnosis_config,
)


def test_config_rejects_non_single_fault_counts_for_procedural_v1() -> None:
    """Procedural circuit diagnosis v1 samples exactly one hidden fault."""

    config = replace(
        DEFAULT_CONFIG,
        max_fault_count=2,
    )

    with pytest.raises(ValueError, match="exactly one hidden fault"):
        validate_circuit_diagnosis_config(config)


def test_config_rejects_probe_budget_below_diagnosis_depth() -> None:
    """The probe budget must cover source setup and generated diagnosis depth."""

    config = replace(
        DEFAULT_CONFIG,
        probe_budget=3,
        max_diagnosis_measurements=3,
    )

    with pytest.raises(ValueError, match="probe_budget"):
        validate_circuit_diagnosis_config(config)


def test_config_rejects_unbounded_component_count() -> None:
    """Generator component count should stay inside bounded search limits."""

    config = replace(DEFAULT_CONFIG, component_count=MAX_COMPONENT_COUNT + 1)

    with pytest.raises(ValueError, match="component_count"):
        validate_circuit_diagnosis_config(config)
