"""Immutable instance construction for the circuit diagnosis task."""

from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.core.payloads import stable_hash
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.payloads import (
    circuit_definition_payload,
    fault_payload,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.schema import (
    CircuitTruth,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.budgets import (
    circuit_budget_limits,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.generation import (
    GeneratedCircuit,
    build_generated_circuit,
    diagnosis_options_payload,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.specs import (
    CIRCUIT_DIAGNOSIS_DOMAIN,
    CIRCUIT_DIAGNOSIS_KIND,
    CircuitDiagnosisConfig,
    config_parameters,
    validate_circuit_diagnosis_config,
)


def build_circuit_diagnosis_instance(
    seed: int, config: CircuitDiagnosisConfig
) -> TaskInstance:
    """Build one deterministic circuit diagnosis task instance.

    Parameters
    ----------
    seed:
        Generator seed for reproducible public and privileged payloads.
    config:
        Public generation, rollout, and verifier configuration.

    Returns
    -------
    TaskInstance
        Immutable scalar task instance ready for session creation.
    """

    validate_circuit_diagnosis_config(config)
    generated = build_generated_circuit(seed, config)
    return _instance_from_generated(seed=seed, config=config, generated=generated)


def _instance_from_generated(
    seed: int,
    config: CircuitDiagnosisConfig,
    generated: GeneratedCircuit,
) -> TaskInstance:
    """Build a task instance from a validated generated circuit."""

    truth = CircuitTruth(
        public_definition=generated.definition,
        hidden_faults=(generated.hidden_fault,),
        fault_count_range=(config.min_fault_count, config.max_fault_count),
    )
    public_view = truth.public_view
    public_payload: dict[str, object] = {
        "circuit": circuit_definition_payload(public_view.definition),
        "fault_count_range": {
            "min": public_view.fault_count_range[0],
            "max": public_view.fault_count_range[1],
        },
        "generation": dict(generated.public_metrics),
        "diagnosis_options": diagnosis_options_payload(
            public_view.definition, generated.eligible_faults
        ),
        "required_answer": {
            "action": "final_answer",
            "fields": ("faults", "repairs"),
        },
    }
    privileged_payload: dict[str, object] = {
        "faults": [fault_payload(fault) for fault in truth.hidden_faults],
        "generation_debug": dict(generated.debug_metrics),
    }
    task_hash = stable_hash(
        {
            "kind": CIRCUIT_DIAGNOSIS_KIND,
            "seed": seed,
            "config": config_parameters(config),
            "public_payload": public_payload,
            "privileged_payload": privileged_payload,
        }
    )[:16]
    return TaskInstance(
        task_id=f"circuit-diagnosis-v1-{task_hash}",
        kind=CIRCUIT_DIAGNOSIS_KIND,
        domain=CIRCUIT_DIAGNOSIS_DOMAIN,
        seed=seed,
        public_payload=public_payload,
        privileged_payload=privileged_payload,
        budget_limits=circuit_budget_limits(
            turn_budget=config.turn_budget,
            probe_budget=config.probe_budget,
            repair_budget=config.repair_budget,
            final_answer_budget=config.final_answer_budget,
        ),
        timeout_seconds=config.timeout_seconds,
        token_budget=config.token_budget,
        metadata={
            "task_family": "circuit_diagnosis",
            "generator": "procedural_passive_resistor_network_v1",
            "difficulty": "diagnostic",
        },
    )
