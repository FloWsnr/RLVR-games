"""Tests for circuit diagnosis task specs and configuration."""

from dataclasses import replace

import pytest

from rlvr_physics.tasks.physics.circuit_diagnosis.specs import (
    DEFAULT_CONFIG,
    validate_circuit_diagnosis_config,
)


def test_config_rejects_more_than_two_faults_for_v1() -> None:
    """Circuit diagnosis v1 keeps the public one-or-two fault invariant."""

    config = replace(
        DEFAULT_CONFIG,
        min_fault_count=3,
        max_fault_count=3,
        repair_budget=3,
        turn_budget=16,
    )

    with pytest.raises(ValueError, match="max_fault_count must be at most 2"):
        validate_circuit_diagnosis_config(config)
