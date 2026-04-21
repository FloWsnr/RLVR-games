"""Text renderer for physics discovery."""

import json
from typing import Mapping

from rlvr_physics.core.instances import (
    TaskInstance,
    mapping_to_dict,
    require_int,
    require_mapping,
    require_str,
)
from rlvr_physics.core.rendering import RenderedObservation, text_observation
from rlvr_physics.tasks.physics.discovery.constants import (
    ALLOWED_CONSTANT_NAMES,
    ALLOWED_FUNCTION_NAMES,
)
from rlvr_physics.tasks.physics.discovery.types import (
    ExperimentObservation,
    HypothesisAttempt,
)
from rlvr_physics.tasks.physics.discovery.utils import range_pair, single_mapping_key


def render_physics_discovery_text(
    instance: TaskInstance,
    observations: tuple[ExperimentObservation, ...],
    hypotheses: tuple[HypothesisAttempt, ...],
    samples_used: int,
    hypotheses_used: int,
) -> RenderedObservation:
    """Render a physics discovery state as text.

    Parameters
    ----------
    instance:
        Immutable discovery instance.
    observations:
        Accepted experiment observations.
    hypotheses:
        Tested hypothesis records.
    samples_used:
        Number of accepted experiments.
    hypotheses_used:
        Number of accepted hypothesis tests.
    """

    input_variables = require_mapping(
        instance.public_payload["input_variables"], "input_variables"
    )
    output_variable = require_mapping(
        instance.public_payload["output_variable"], "output_variable"
    )
    parameter_ranges = require_mapping(
        instance.public_payload["parameter_ranges"], "parameter_ranges"
    )
    sample_quota = require_int(instance.public_payload["sample_quota"], "sample_quota")
    hypothesis_quota = require_int(
        instance.public_payload["hypothesis_quota"], "hypothesis_quota"
    )
    problem = require_str(instance.public_payload["problem"], "problem")
    prior_mode = require_str(instance.public_payload["prior_mode"], "prior_mode")
    output_name = single_mapping_key(output_variable, "output_variable")
    lines = [
        "Physics discovery",
        f"Prior mode: {prior_mode}",
        "",
        "Problem:",
        problem,
        "",
        "Controllable variables:",
    ]
    for name, description in input_variables.items():
        low, high = range_pair(parameter_ranges[name], f"range for {name}")
        lines.append(f"- {name}: {description} | allowed range [{low}, {high}]")
    lines.extend(
        [
            "",
            "Observable:",
            f"- {output_name}: {output_variable[output_name]}",
            "",
            "Budget:",
            f"- Experiments used: {samples_used}/{sample_quota}",
            f"- Hypotheses tested: {hypotheses_used}/{hypothesis_quota}",
            "",
            "Submit JSON to run an experiment:",
            _example_experiment_action(tuple(input_variables.keys()), parameter_ranges),
            "",
            "Submit JSON or final text to propose a hypothesis:",
            '{"action": "submit_hypothesis", "equation": "expression using the variables"}',
            (
                "Allowed expression syntax: +, -, *, /, **, parentheses, "
                f"constants {ALLOWED_CONSTANT_NAMES}, and functions "
                f"{ALLOWED_FUNCTION_NAMES}."
            ),
            "",
            "Observation history:",
        ]
    )
    if observations:
        for observation in observations:
            lines.append(_format_observation(observation, output_name))
    else:
        lines.append("- none")
    lines.append("")
    lines.append("Hypothesis tests:")
    if hypotheses:
        for hypothesis in hypotheses:
            lines.append(
                f"- {hypothesis.hypothesis_id}: {hypothesis.expression} "
                f"| score={hypothesis.score:.6f} | correct={hypothesis.correct}"
            )
    else:
        lines.append("- none")
    return text_observation("text", "\n".join(lines))


def _format_observation(observation: ExperimentObservation, output_name: str) -> str:
    inputs = mapping_to_dict(observation.inputs)
    input_text = ", ".join(f"{name}={inputs[name]}" for name in sorted(inputs))
    return f"- {observation.sample_id}: inputs({input_text}) -> {output_name}={observation.output:.10g}"


def _example_experiment_action(
    variable_names: tuple[str, ...], parameter_ranges: Mapping[str, object]
) -> str:
    values: dict[str, object] = {}
    for name in variable_names:
        low, high = range_pair(parameter_ranges[name], f"range for {name}")
        values[name] = round((low + high) / 2.0, 3)
    return json.dumps({"action": "run_experiment", "inputs": values})
