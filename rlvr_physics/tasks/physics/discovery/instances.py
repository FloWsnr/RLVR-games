"""Instance construction for physics discovery."""

from random import Random
import re
from typing import Mapping

from rlvr_physics.core.instances import (
    TaskInstance,
    TaskLimits,
    freeze_mapping,
    stable_hash,
)
from rlvr_physics.tasks.physics.discovery.constants import (
    PHYSICS_DISCOVERY_KIND,
    PHYSICS_DISCOVERY_SOURCE,
    PHYSICS_DOMAIN,
)
from rlvr_physics.tasks.physics.discovery.expressions import evaluate_expression
from rlvr_physics.tasks.physics.discovery.records import record_by_source_id
from rlvr_physics.tasks.physics.discovery.types import PhysicsDiscoveryRecord
from rlvr_physics.tasks.physics.discovery.utils import (
    float_mapping,
    range_pair,
    single_mapping_key,
    validate_positive_quota,
    validate_prior_mode,
)


def make_physics_discovery_instance(
    source_id: int,
    seed: int,
    prior_mode: str,
    sample_quota: int,
    hypothesis_quota: int,
) -> TaskInstance:
    """Create one immutable interactive physics discovery task instance.

    Parameters
    ----------
    source_id:
        Embedded PhysGym-derived source law id.
    seed:
        Seed used for deterministic hidden verification points.
    prior_mode:
        Information exposure mode.
    sample_quota:
        Maximum accepted experiments for the session.
    hypothesis_quota:
        Maximum accepted hypothesis tests for the session.
    """

    validate_positive_quota(sample_quota, "sample_quota")
    validate_positive_quota(hypothesis_quota, "hypothesis_quota")
    validate_prior_mode(prior_mode)
    record = record_by_source_id(source_id)
    variable_names = tuple(str(name) for name in record.input_variables.keys())
    output_name = single_mapping_key(record.output_variable, "output_variable")

    if prior_mode == "no_description_anonymous":
        variable_mapping = {
            original: f"var_{index + 1}"
            for index, original in enumerate(variable_names)
        }
        visible_output_name = "var_obs"
    else:
        variable_mapping = {original: original for original in variable_names}
        visible_output_name = output_name

    visible_inputs = _visible_input_variables(record, prior_mode, variable_mapping)
    visible_output = _visible_output_variable(record, prior_mode, visible_output_name)
    visible_equation = _rename_symbols(record.equation, variable_mapping)
    parameter_ranges = _visible_parameter_ranges(record, variable_mapping)
    hidden_points = make_hidden_points(
        seed=seed,
        source_id=source_id,
        variable_names=tuple(visible_inputs.keys()),
        count=24,
        parameter_ranges=parameter_ranges,
        equation=visible_equation,
    )
    context = _visible_context(record.context, prior_mode)
    task_id = (
        "physics-discovery-"
        + stable_hash(
            {
                "source_id": source_id,
                "seed": seed,
                "prior_mode": prior_mode,
                "sample_quota": sample_quota,
                "hypothesis_quota": hypothesis_quota,
                "visible_equation": visible_equation,
                "hidden_points": hidden_points,
            }
        )[:16]
    )
    return TaskInstance(
        task_id=task_id,
        kind=PHYSICS_DISCOVERY_KIND,
        domain=PHYSICS_DOMAIN,
        seed=seed,
        public_payload={
            "source_id": source_id,
            "source": PHYSICS_DISCOVERY_SOURCE,
            "tag": record.tag,
            "prior_mode": prior_mode,
            "problem": context,
            "input_variables": visible_inputs,
            "output_variable": visible_output,
            "parameter_ranges": parameter_ranges,
            "sample_quota": sample_quota,
            "hypothesis_quota": hypothesis_quota,
        },
        privileged_payload={
            "equation": visible_equation,
            "source_equation": record.equation,
            "variable_mapping": {
                visible: original for original, visible in variable_mapping.items()
            },
            "hidden_points": hidden_points,
        },
        limits=TaskLimits(
            max_turns=sample_quota + hypothesis_quota,
            action_budget=sample_quota + hypothesis_quota,
        ),
        metadata={
            "source": PHYSICS_DISCOVERY_SOURCE,
            "source_id": source_id,
            "tag": record.tag,
            "prior_mode": prior_mode,
        },
    )


def make_hidden_points(
    seed: int,
    source_id: int,
    variable_names: tuple[str, ...],
    count: int,
    parameter_ranges: Mapping[str, object],
    equation: str,
) -> tuple[Mapping[str, object], ...]:
    """Create deterministic hidden verification points."""

    rng_seed = int(
        stable_hash(
            {
                "seed": seed,
                "source_id": source_id,
                "variable_names": variable_names,
                "count": count,
            }
        )[:16],
        16,
    )
    rng = Random(rng_seed)
    points: list[Mapping[str, object]] = []
    attempts = 0
    max_attempts = count * 500
    while len(points) < count and attempts < max_attempts:
        attempts += 1
        point = _sample_point(rng, variable_names, parameter_ranges)
        try:
            evaluate_expression(equation, float_mapping(point))
        except (ArithmeticError, ValueError, TypeError, OverflowError):
            continue
        points.append(freeze_mapping(point))
    if len(points) < count:
        raise ValueError(
            f"could not sample valid hidden points for physics source id {source_id}"
        )
    return tuple(points)


def _sample_point(
    rng: Random,
    variable_names: tuple[str, ...],
    parameter_ranges: Mapping[str, object],
) -> dict[str, object]:
    point: dict[str, object] = {}
    for name in variable_names:
        low, high = range_pair(parameter_ranges[name], f"range for {name}")
        point[name] = round(rng.uniform(low, high), 6)
    return point


def _visible_context(context: str, prior_mode: str) -> str:
    if prior_mode == "default":
        return context
    return "Unknown context."


def _visible_input_variables(
    record: PhysicsDiscoveryRecord,
    prior_mode: str,
    variable_mapping: Mapping[str, str],
) -> Mapping[str, object]:
    visible: dict[str, object] = {}
    for original, description in record.input_variables.items():
        visible_name = variable_mapping[str(original)]
        if prior_mode in ("no_description", "no_description_anonymous"):
            visible[visible_name] = "Some variable."
        else:
            visible[visible_name] = description
    return freeze_mapping(visible)


def _visible_output_variable(
    record: PhysicsDiscoveryRecord, prior_mode: str, visible_output_name: str
) -> Mapping[str, object]:
    description = record.output_variable[
        single_mapping_key(record.output_variable, "output_variable")
    ]
    if prior_mode in ("no_description", "no_description_anonymous"):
        description = "Some variable."
    return freeze_mapping({visible_output_name: description})


def _visible_parameter_ranges(
    record: PhysicsDiscoveryRecord, variable_mapping: Mapping[str, str]
) -> Mapping[str, object]:
    visible: dict[str, object] = {}
    for original in record.input_variables.keys():
        original_name = str(original)
        visible_name = variable_mapping[original_name]
        visible[visible_name] = record.parameter_ranges[original_name]
    return freeze_mapping(visible)


def _rename_symbols(expression: str, variable_mapping: Mapping[str, str]) -> str:
    renamed = expression
    for original, visible in sorted(
        variable_mapping.items(), key=lambda item: len(item[0]), reverse=True
    ):
        renamed = re.sub(rf"\b{re.escape(original)}\b", visible, renamed)
    return renamed
