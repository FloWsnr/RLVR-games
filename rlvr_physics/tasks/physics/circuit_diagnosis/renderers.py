"""Text renderer for the circuit diagnosis task."""

from collections.abc import Mapping
from dataclasses import dataclass
import json

from rlvr_physics.core.payloads import to_plain_data
from rlvr_physics.core.rendering import RenderedObservation, text_observation
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.schema import (
    CircuitComponent,
    CircuitDefinition,
    CircuitPublicView,
    ReplacementSpec,
    SourceSetting,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.prompting import (
    circuit_text_prompt_template,
    render_prompt_template,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.specs import (
    CIRCUIT_TEXT_RENDERER,
    validate_circuit_renderer_type,
)


@dataclass(frozen=True)
class CircuitRenderContext:
    """Public current-turn circuit state used by renderers.

    Parameters
    ----------
    public_view:
        Trainer-safe public circuit view.
    diagnosis_options:
        Public final-answer fault and repair vocabulary.
    feedback:
        Latest public feedback shown to the model.
    source_setting:
        Current bench source setting, when connected.
    repairs:
        Accepted public repair overlays.
    budget_status:
        Public budget status lines.
    """

    public_view: CircuitPublicView
    diagnosis_options: Mapping[str, object]
    feedback: str
    source_setting: SourceSetting | None
    repairs: Mapping[str, ReplacementSpec]
    budget_status: str


def render_circuit_observation(
    renderer_type: str, context: CircuitRenderContext
) -> RenderedObservation:
    """Render one circuit diagnosis observation.

    Parameters
    ----------
    renderer_type:
        Supported renderer identifier.
    context:
        Public circuit rollout state to render.

    Returns
    -------
    RenderedObservation
        Text-only observation for the requested renderer.
    """

    validate_circuit_renderer_type(renderer_type)
    return text_observation(CIRCUIT_TEXT_RENDERER, render_circuit_text(context))


def render_circuit_text(context: CircuitRenderContext) -> str:
    """Build the text-only observation for one circuit diagnosis turn.

    Parameters
    ----------
    context:
        Public circuit rollout state to render.

    Returns
    -------
    str
        Model-facing text prompt.
    """

    return _render_prompt(
        circuit_text_prompt_template(),
        context,
        netlist=_render_netlist(context.public_view.definition),
    )


def _render_prompt(template: str, context: CircuitRenderContext, netlist: str) -> str:
    """Render a circuit prompt template."""

    return render_prompt_template(
        template,
        {
            "feedback": context.feedback,
            "netlist": netlist,
            "target_behavior": _render_target_behavior(context.public_view.definition),
            "diagnosis_options": _render_diagnosis_options(context.diagnosis_options),
            "source_setting": _render_source_setting(context.source_setting),
            "repair_state": _render_repair_state(context.repairs),
            "budget_status": context.budget_status,
            "submission_examples": _render_submission_examples(context),
        },
    )


def _render_netlist(definition: CircuitDefinition) -> str:
    """Return a compact text netlist for the public circuit."""

    lines = [
        f"- nodes: {', '.join(definition.nodes)}",
        f"- ground: {definition.ground_node}",
    ]
    for component in definition.components:
        lines.append(
            "- "
            f"{component.component_id}: {component.kind} "
            f"{component.node_a} -> {component.node_b}; "
            f"{_component_parameter_text(component)}"
        )
    return "\n".join(lines)


def _render_target_behavior(definition: CircuitDefinition) -> str:
    """Return public target behavior text."""

    lines: list[str] = [definition.description]
    if definition.target_source is None:
        lines.append("- target source: use the circuit's internal source")
    else:
        source = definition.target_source
        lines.append(
            "- target source: "
            f"{source.node_plus} relative to {source.node_minus} = "
            f"{_fmt(source.voltage_V)} V"
        )
    for check in definition.target_checks:
        if check.kind == "voltage_between":
            lines.append(
                "- "
                f"{check.check_id}: V({_param(check.parameters, 'node_a')}, "
                f"{_param(check.parameters, 'node_b')}) in "
                f"[{_fmt(_num(check.parameters, 'min_V'))}, "
                f"{_fmt(_num(check.parameters, 'max_V'))}] V"
            )
        elif check.kind == "current_range":
            lines.append(
                "- "
                f"{check.check_id}: I({_param(check.parameters, 'component')}) in "
                f"[{_fmt(_num(check.parameters, 'min_A'))}, "
                f"{_fmt(_num(check.parameters, 'max_A'))}] A"
            )
        elif check.kind == "power_max":
            lines.append(
                "- "
                f"{check.check_id}: P({_param(check.parameters, 'component')}) <= "
                f"{_fmt(_num(check.parameters, 'max_W'))} W"
            )
    return "\n".join(lines)


def _render_source_setting(source_setting: SourceSetting | None) -> str:
    """Return text for the current bench source."""

    if source_setting is None:
        return "- no external bench source connected"
    return (
        "- "
        f"{source_setting.node_plus} relative to {source_setting.node_minus} = "
        f"{_fmt(source_setting.voltage_V)} V"
    )


def _render_repair_state(repairs: Mapping[str, ReplacementSpec]) -> str:
    """Return text for accepted repair overlays."""

    if len(repairs) == 0:
        return "- no components replaced"
    lines: list[str] = []
    for component_id in sorted(repairs):
        repair = repairs[component_id]
        parameters = ", ".join(
            f"{key}={_fmt_value(value)}" for key, value in repair.parameters.items()
        )
        lines.append(f"- {component_id}: {repair.kind}, {parameters}")
    return "\n".join(lines)


def _render_diagnosis_options(options: Mapping[str, object]) -> str:
    """Return public final-answer vocabulary text."""

    faults = _mapping_tuple_field(options, "faults")
    repairs = _mapping_tuple_field(options, "repairs")
    repair_by_component = {
        _str_field(repair, "component"): repair for repair in repairs
    }
    lines: list[str] = []
    for repair in repairs:
        component_id = _str_field(repair, "component")
        component_faults = [
            fault for fault in faults if _str_field(fault, "component") == component_id
        ]
        fault_text = "; ".join(_fault_option_text(fault) for fault in component_faults)
        lines.append(
            "- "
            f"{component_id}: faults [{fault_text}]; "
            f"repair {_str_field(repair_by_component[component_id], 'repair_code')}"
        )
    return "\n".join(lines)


def _render_submission_examples(context: CircuitRenderContext) -> str:
    """Return valid public JSON action examples for one circuit instance."""

    definition = context.public_view.definition
    first_component = definition.components[0]
    examples = (
        (
            "source",
            _action_payload("set_source", _source_example_arguments(definition)),
        ),
        (
            "voltage",
            _action_payload("measure_voltage", _voltage_example_arguments(definition)),
        ),
        (
            "current",
            _action_payload(
                "measure_current", {"component": first_component.component_id}
            ),
        ),
        ("repair", _first_repair_action(context.diagnosis_options)),
        ("final", _final_answer_example(context.diagnosis_options)),
    )
    return "\n".join(
        f"- {label} example: {_compact_json(payload)}" for label, payload in examples
    )


def _source_example_arguments(definition: CircuitDefinition) -> dict[str, object]:
    """Return a valid source action example for a public circuit."""

    if definition.target_source is not None:
        source = definition.target_source
        return {
            "node_plus": source.node_plus,
            "node_minus": source.node_minus,
            "voltage_V": source.voltage_V,
        }
    non_ground = next(
        node for node in definition.nodes if node != definition.ground_node
    )
    return {
        "node_plus": non_ground,
        "node_minus": definition.ground_node,
        "voltage_V": 5.0,
    }


def _voltage_example_arguments(definition: CircuitDefinition) -> dict[str, object]:
    """Return a valid voltage measurement example for a public circuit."""

    if definition.target_source is not None:
        return {
            "node_a": definition.target_source.node_plus,
            "node_b": definition.target_source.node_minus,
        }
    non_ground = next(
        node for node in definition.nodes if node != definition.ground_node
    )
    return {"node_a": non_ground, "node_b": definition.ground_node}


def _first_repair_action(options: Mapping[str, object]) -> Mapping[str, object]:
    """Return the first public repair action from diagnosis options."""

    repairs = _mapping_tuple_field(options, "repairs")
    first_repair = repairs[0]
    action = first_repair["action"]
    if isinstance(action, Mapping):
        return action
    raise TypeError("repair action must be a mapping")


def _final_answer_example(options: Mapping[str, object]) -> dict[str, object]:
    """Return a final-answer example using public allowed labels."""

    fault_ids = _str_tuple_field(options, "fault_ids")
    repair_codes = _str_tuple_field(options, "repair_codes")
    return _action_payload(
        "final_answer",
        {"faults": [fault_ids[0]], "repairs": [repair_codes[0]]},
    )


def _action_payload(action: str, arguments: Mapping[str, object]) -> dict[str, object]:
    """Return one JSON action payload."""

    return {"action": action, "arguments": dict(arguments)}


def _compact_json(value: object) -> str:
    """Return deterministic compact JSON for prompt examples."""

    return json.dumps(to_plain_data(value), sort_keys=True, separators=(",", ":"))


def _fault_option_text(fault: Mapping[str, object]) -> str:
    """Return one compact fault option description."""

    return f"{_str_field(fault, 'fault_id')} ({_str_field(fault, 'description')})"


def _component_parameter_text(component: CircuitComponent) -> str:
    """Return public component parameter text."""

    if component.kind == "resistor":
        return f"value={_fmt(_num(component.parameters, 'value_ohm'))} ohm"
    if component.kind == "capacitor":
        return f"value={_fmt(_num(component.parameters, 'value_F'))} F"
    if component.kind == "diode":
        return (
            f"anode={component.node_a}, cathode={component.node_b}, "
            f"forward_drop={_fmt(_num(component.parameters, 'forward_drop_V'))} V"
        )
    if component.kind == "switch":
        state = "closed" if bool(component.parameters["closed"]) else "open"
        return f"nominal_state={state}"
    if component.kind == "voltage_source":
        return f"value={_fmt(_num(component.parameters, 'voltage_V'))} V"
    return ", ".join(
        f"{key}={_fmt_value(value)}" for key, value in component.parameters.items()
    )


def _param(values: Mapping[str, object], name: str) -> str:
    """Return a string parameter for rendering."""

    value = values[name]
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _str_field(values: Mapping[str, object], name: str) -> str:
    """Return a string field from public payload data."""

    value = values[name]
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _mapping_tuple_field(
    values: Mapping[str, object], name: str
) -> tuple[Mapping[str, object], ...]:
    """Return a tuple of mapping fields from public payload data."""

    raw_values = values[name]
    if not isinstance(raw_values, tuple):
        raise TypeError(f"{name} must be a tuple")
    result: list[Mapping[str, object]] = []
    for raw_value in raw_values:
        if not isinstance(raw_value, Mapping):
            raise TypeError(f"{name} values must be mappings")
        result.append(raw_value)
    return tuple(result)


def _str_tuple_field(values: Mapping[str, object], name: str) -> tuple[str, ...]:
    """Return a tuple of string fields from public payload data."""

    raw_values = values[name]
    if not isinstance(raw_values, tuple):
        raise TypeError(f"{name} must be a tuple")
    result: list[str] = []
    for raw_value in raw_values:
        if not isinstance(raw_value, str):
            raise TypeError(f"{name} values must be strings")
        result.append(raw_value)
    return tuple(result)


def _num(values: Mapping[str, object], name: str) -> float:
    """Return a numeric parameter for rendering."""

    value = values[name]
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    if isinstance(value, int | float):
        return float(value)
    raise TypeError(f"{name} must be numeric")


def _fmt_value(value: object) -> str:
    """Return compact rendered text for a value."""

    if isinstance(value, int | float) and not isinstance(value, bool):
        return _fmt(float(value))
    return str(value)


def _fmt(value: float) -> str:
    """Return a compact numeric string."""

    return f"{value:.6g}"
