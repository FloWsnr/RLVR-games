"""Tests for circuit diagnosis renderers."""

from importlib import resources

import pytest

from rlvr_physics.tasks.physics.circuit_diagnosis.instances import (
    build_circuit_diagnosis_instance,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.prompting import (
    circuit_text_prompt_template,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.sessions import (
    CircuitDiagnosisSession,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.specs import (
    CIRCUIT_TEXT_RENDERER,
    DEFAULT_CONFIG,
    validate_circuit_renderer_type,
)

_CIRCUIT_PACKAGE = "rlvr_physics.tasks.physics.circuit_diagnosis"


def test_text_renderer_includes_public_netlist_topology() -> None:
    instance = build_circuit_diagnosis_instance(seed=1, config=DEFAULT_CONFIG)
    session = CircuitDiagnosisSession(
        instance, CIRCUIT_TEXT_RENDERER, DEFAULT_CONFIG.reward
    )

    reset = session.reset(seed=2)
    text = reset.turn.observation.text()

    assert reset.turn.observation.renderer_name == CIRCUIT_TEXT_RENDERER
    assert "Circuit topology:" in text
    assert "- nodes:" in text
    assert "component_id" not in text
    assert "D1_reversed" not in text


def test_text_renderer_includes_public_final_answer_vocabulary() -> None:
    instance = build_circuit_diagnosis_instance(seed=4, config=DEFAULT_CONFIG)
    session = CircuitDiagnosisSession(
        instance, CIRCUIT_TEXT_RENDERER, DEFAULT_CONFIG.reward
    )

    reset = session.reset(seed=2)
    text = reset.turn.observation.text()

    assert "Diagnosis answer vocabulary:" in text
    assert "R1_wrong_low" in text
    assert "R1 resistance is 156.667 ohm" in text
    assert "replace_R1_470_ohm" in text
    assert '"node_plus":"VIN"' in text
    assert '"node_a":"VIN"' in text
    assert '"node_plus":"A"' not in text


def test_circuit_prompt_template_is_task_local_file() -> None:
    assert circuit_text_prompt_template() == _circuit_prompt_file_text(
        "text_observation.md"
    )


def test_circuit_renderer_rejects_image_renderer_for_now() -> None:
    with pytest.raises(ValueError, match="unsupported circuit diagnosis renderer"):
        validate_circuit_renderer_type("circuit_diagnosis.image")


def _circuit_prompt_file_text(filename: str) -> str:
    """Return text from a circuit task prompt asset.

    Parameters
    ----------
    filename:
        Prompt asset filename inside the circuit task ``prompts`` directory.

    Returns
    -------
    str
        Prompt asset text.
    """

    return (
        resources.files(_CIRCUIT_PACKAGE)
        .joinpath("prompts", filename)
        .read_text(encoding="utf-8")
    )
