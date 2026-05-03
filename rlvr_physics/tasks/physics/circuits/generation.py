"""Procedural circuit generation from reusable motifs."""

from dataclasses import dataclass
from random import Random
from typing import Mapping

from rlvr_physics.tasks.physics.circuits.motifs import (
    add_load_resistor,
    choose_motif,
    default_motifs,
)
from rlvr_physics.tasks.physics.circuits.model import (
    Circuit,
    CircuitBuilder,
    PartSpec,
)


@dataclass(frozen=True)
class GeneratorConfig:
    """Configuration for deterministic procedural circuit generation.

    Parameters
    ----------
    seed:
        Deterministic generator seed.
    element_count:
        Requested count of non-ground parts.
    motif_weights:
        Relative motif weights by motif name.
    """

    seed: int
    element_count: int
    motif_weights: Mapping[str, float]


@dataclass(frozen=True)
class GeneratedCircuit:
    """Generated circuit plus provenance metadata."""

    circuit: Circuit
    motif_names: tuple[str, ...]
    seed: int


def generate_circuit(
    config: GeneratorConfig, catalog: Mapping[str, PartSpec]
) -> GeneratedCircuit:
    """Generate a deterministic circuit.

    Parameters
    ----------
    config:
        Generation configuration.
    catalog:
        Component catalog.

    Returns
    -------
    GeneratedCircuit
        Generated canonical circuit and motif provenance.
    """

    if config.element_count < 2:
        raise ValueError("element_count must be at least 2")
    rng = Random(config.seed)
    ctx = _GenerationContext(
        builder=CircuitBuilder("generated-circuit", catalog),
        rng=rng,
        target_count=config.element_count,
    )
    ctx.add_part("V", "voltage_source_dc", "5V", {"voltage_v": 5.0}, {})
    ctx.builder.connect("V1", "p", "VCC")
    ctx.builder.connect("V1", "n", "0")
    ctx.add_part("GND", "ground", "0", {}, {})
    ctx.builder.connect("GND1", "0", "0")

    motif_names: list[str] = []
    motif_catalog = default_motifs()
    while ctx.non_ground_count < config.element_count:
        remaining = config.element_count - ctx.non_ground_count
        motif = choose_motif(rng, motif_catalog, config.motif_weights, remaining)
        if motif is not None and motif.build(ctx, remaining):
            motif_names.append(motif.name)
        else:
            add_load_resistor(ctx)
            motif_names.append("load_resistor")

    ctx.builder.set_metadata(
        {
            "source": "procedural",
            "target_element_count": config.element_count,
            "motifs": tuple(motif_names),
        }
    )
    return GeneratedCircuit(
        circuit=ctx.builder.freeze(),
        motif_names=tuple(motif_names),
        seed=config.seed,
    )


class _GenerationContext:
    """Mutable generator context."""

    def __init__(self, builder: CircuitBuilder, rng: Random, target_count: int) -> None:
        """Initialize context state."""

        self.builder = builder
        self.rng = rng
        self.target_count = target_count
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
        """Add a new numbered part and return its reference."""

        number = self.counters.get(prefix, 0) + 1
        self.counters[prefix] = number
        ref = f"{prefix}{number}"
        self.builder.add_part(ref, kind, value, parameters, metadata)
        if kind != "ground":
            self.non_ground_count += 1
        return ref

    def node(self) -> str:
        """Return a fresh node name."""

        self.node_counter += 1
        return f"N{self.node_counter}"
