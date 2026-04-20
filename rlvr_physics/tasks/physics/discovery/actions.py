"""Action parsing and schemas for physics discovery."""

import json
from typing import Mapping

from rlvr_physics.core.instances import TaskInstance, require_mapping
from rlvr_physics.core.session import TaskSubmission
from rlvr_physics.tasks.physics.discovery.expressions import (
    extract_hypothesis_expression,
)
from rlvr_physics.tasks.physics.discovery.types import ParsedDiscoveryAction
from rlvr_physics.tasks.physics.discovery.utils import coerce_float, range_pair


def validate_experiment_inputs(
    instance: TaskInstance, inputs: Mapping[str, object]
) -> Mapping[str, float]:
    """Validate controlled experiment inputs against instance ranges."""

    input_variables = require_mapping(
        instance.public_payload["input_variables"], "input_variables"
    )
    parameter_ranges = require_mapping(
        instance.public_payload["parameter_ranges"], "parameter_ranges"
    )
    expected_names = set(str(name) for name in input_variables.keys())
    provided_names = set(str(name) for name in inputs.keys())
    missing = sorted(expected_names - provided_names)
    extra = sorted(provided_names - expected_names)
    if missing:
        raise ValueError(f"missing inputs: {', '.join(missing)}")
    if extra:
        raise ValueError(f"unknown inputs: {', '.join(extra)}")
    validated: dict[str, float] = {}
    for name in sorted(expected_names):
        value = coerce_float(inputs[name], name)
        low, high = range_pair(parameter_ranges[name], f"range for {name}")
        if value < low or value > high:
            raise ValueError(f"{name} must be in [{low}, {high}]")
        validated[name] = value
    return validated


def parse_discovery_action(submission: TaskSubmission) -> ParsedDiscoveryAction:
    """Parse a discovery action from a task submission."""

    if submission.kind == "final_text":
        return ParsedDiscoveryAction(
            action_type="submit_hypothesis",
            equation=extract_hypothesis_expression(submission.raw),
        )
    raw = submission.raw.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("action_must_be_json") from error
    if not isinstance(parsed, dict):
        raise ValueError("action_must_be_json_object")
    action_value = parsed.get("action", parsed.get("type"))
    if not isinstance(action_value, str):
        if isinstance(parsed.get("equation"), str):
            action_value = "submit_hypothesis"
        else:
            raise ValueError("missing_action")
    action_type = action_value.strip()
    if action_type == "run_experiment":
        inputs = parsed.get("inputs")
        if not isinstance(inputs, dict):
            raise ValueError("experiment_inputs_must_be_object")
        return ParsedDiscoveryAction(action_type=action_type, inputs=inputs)
    if action_type == "submit_hypothesis":
        equation = parsed.get("equation", parsed.get("hypothesis"))
        if not isinstance(equation, str):
            raise ValueError("hypothesis_equation_must_be_string")
        return ParsedDiscoveryAction(action_type=action_type, equation=equation)
    return ParsedDiscoveryAction(action_type=action_type)


def make_action_schema(instance: TaskInstance) -> Mapping[str, object]:
    """Build the public JSON action schema for a discovery instance."""

    input_variables = require_mapping(
        instance.public_payload["input_variables"], "input_variables"
    )
    parameter_ranges = require_mapping(
        instance.public_payload["parameter_ranges"], "parameter_ranges"
    )
    input_properties: dict[str, object] = {}
    for name in input_variables.keys():
        low, high = range_pair(parameter_ranges[name], f"range for {name}")
        input_properties[str(name)] = {
            "type": "number",
            "minimum": low,
            "maximum": high,
        }
    return {
        "type": "object",
        "oneOf": (
            {
                "properties": {
                    "action": {"const": "run_experiment"},
                    "inputs": {
                        "type": "object",
                        "properties": input_properties,
                        "required": tuple(input_properties.keys()),
                        "additionalProperties": False,
                    },
                },
                "required": ("action", "inputs"),
                "additionalProperties": False,
            },
            {
                "properties": {
                    "action": {"const": "submit_hypothesis"},
                    "equation": {"type": "string"},
                },
                "required": ("action", "equation"),
                "additionalProperties": False,
            },
        ),
    }
