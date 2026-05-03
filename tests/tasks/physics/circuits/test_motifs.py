"""Tests for procedural motif catalog coverage."""

from random import Random
from typing import Mapping

from rlvr_physics.tasks.physics.circuits import (
    CircuitBuilder,
    default_motif_weights,
    default_motifs,
    default_part_catalog,
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
