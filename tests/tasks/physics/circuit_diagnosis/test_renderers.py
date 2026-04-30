"""Tests for circuit diagnosis renderers."""

from struct import unpack

from rlvr_physics.core.rendering import ImageContent, PNG_MIME_TYPE, TextContent
from rlvr_physics.tasks.physics.circuit_diagnosis.instances import (
    build_circuit_diagnosis_instance,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.sessions import (
    CircuitDiagnosisSession,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.specs import (
    CIRCUIT_IMAGE_RENDERER,
    CIRCUIT_TEXT_RENDERER,
    DEFAULT_CONFIG,
)


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


def test_image_renderer_returns_png_and_text_omits_netlist() -> None:
    instance = build_circuit_diagnosis_instance(seed=1, config=DEFAULT_CONFIG)
    session = CircuitDiagnosisSession(
        instance, CIRCUIT_IMAGE_RENDERER, DEFAULT_CONFIG.reward
    )

    reset = session.reset(seed=2)
    observation = reset.turn.observation

    assert observation.renderer_name == CIRCUIT_IMAGE_RENDERER
    assert len(observation.contents) == 2
    assert isinstance(observation.contents[0], ImageContent)
    assert isinstance(observation.contents[1], TextContent)
    assert observation.contents[0].mime_type == PNG_MIME_TYPE
    assert _png_size(observation.contents[0].data) == (960, 640)
    assert "- nodes:" not in observation.text()
    assert "Inspect the schematic image" in observation.text()
    assert "D1_reversed" not in observation.text()


def _png_size(data: bytes) -> tuple[int, int]:
    """Return PNG width and height from the IHDR chunk."""

    return unpack(">II", data[16:24])
