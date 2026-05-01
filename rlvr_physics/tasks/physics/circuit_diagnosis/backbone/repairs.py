"""Repair parsing and canonical repair labels."""

from rlvr_physics.core.submissions import ParsedAction
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.actions import (
    positive_numeric_argument,
    required_numeric_argument,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.errors import (
    SubmissionParseError,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.schema import (
    CircuitComponent,
    ReplacementSpec,
)
from rlvr_physics.tasks.physics.circuit_diagnosis.backbone.values import (
    bool_parameter,
    float_parameter,
    format_code_number,
)


def nominal_replacement_for_component(component: CircuitComponent) -> ReplacementSpec:
    """Return the nominal repair overlay for a public component."""

    return ReplacementSpec(
        component_id=component.component_id,
        kind=component.kind,
        parameters=nominal_replacement_parameters(component),
    )


def canonical_repair_code(component: CircuitComponent) -> str:
    """Return the canonical repair code for a component."""

    if component.kind == "resistor":
        return (
            f"replace_{component.component_id}_"
            f"{format_code_number(float_parameter(component.parameters, 'value_ohm'))}"
            "_ohm"
        )
    if component.kind == "capacitor":
        return (
            f"replace_{component.component_id}_"
            f"{format_code_number(float_parameter(component.parameters, 'value_F'))}"
            "_F"
        )
    if component.kind == "switch":
        closed = bool_parameter(component.parameters, "closed")
        state = "closed" if closed else "open"
        return f"replace_{component.component_id}_{state}"
    if component.kind == "voltage_source":
        return (
            f"replace_{component.component_id}_"
            f"{format_code_number(float_parameter(component.parameters, 'voltage_V'))}"
            "_V"
        )
    return f"replace_{component.component_id}_{component.kind}"


def replacement_from_action(
    nominal: CircuitComponent, action: ParsedAction
) -> ReplacementSpec:
    """Build a replacement spec from action arguments."""

    if nominal.kind == "resistor":
        return ReplacementSpec(
            component_id=nominal.component_id,
            kind=nominal.kind,
            parameters={
                "value_ohm": positive_numeric_argument(action, "value_ohm"),
            },
        )
    if nominal.kind == "capacitor":
        return ReplacementSpec(
            component_id=nominal.component_id,
            kind=nominal.kind,
            parameters={"value_F": positive_numeric_argument(action, "value_F")},
        )
    if nominal.kind == "diode":
        return ReplacementSpec(
            component_id=nominal.component_id,
            kind=nominal.kind,
            parameters={
                "forward_drop_V": positive_numeric_argument(action, "forward_drop_V"),
            },
        )
    if nominal.kind == "switch":
        value = action.arguments.get("closed")
        if not isinstance(value, bool):
            raise SubmissionParseError("closed must be a boolean")
        return ReplacementSpec(
            component_id=nominal.component_id,
            kind=nominal.kind,
            parameters={"closed": value},
        )
    if nominal.kind == "voltage_source":
        return ReplacementSpec(
            component_id=nominal.component_id,
            kind=nominal.kind,
            parameters={
                "voltage_V": required_numeric_argument(action, "voltage_V"),
                "internal_resistance_ohm": 0.0,
            },
        )
    if nominal.kind == "current_source":
        return ReplacementSpec(
            component_id=nominal.component_id,
            kind=nominal.kind,
            parameters={"current_A": required_numeric_argument(action, "current_A")},
        )
    raise SubmissionParseError(f"unsupported replacement kind: {nominal.kind}")


def validate_nominal_replacement(
    nominal: CircuitComponent, replacement: ReplacementSpec
) -> None:
    """Reject replacements that do not match the nominal schematic component."""

    nominal_parameters = nominal_replacement_parameters(nominal)
    if replacement.kind != nominal.kind:
        raise SubmissionParseError(
            f"replacement kind for {nominal.component_id} must be {nominal.kind}"
        )
    for name, expected_value in nominal_parameters.items():
        submitted_value = replacement.parameters.get(name)
        if not parameter_values_match(submitted_value, expected_value):
            raise SubmissionParseError(
                f"{name} for {nominal.component_id} must match nominal value "
                f"{format_code_number(float(expected_value))}"
                if isinstance(expected_value, int | float)
                and not isinstance(expected_value, bool)
                else f"{name} for {nominal.component_id} must match nominal value"
            )


def parameter_values_match(submitted_value: object, expected_value: object) -> bool:
    """Return whether a repair parameter exactly matches the nominal value."""

    if isinstance(expected_value, bool):
        return submitted_value is expected_value
    if isinstance(expected_value, int | float) and not isinstance(expected_value, bool):
        if isinstance(submitted_value, bool) or not isinstance(
            submitted_value, int | float
        ):
            return False
        return abs(float(submitted_value) - float(expected_value)) <= 1.0e-9
    return submitted_value == expected_value


def nominal_replacement_parameters(component: CircuitComponent) -> dict[str, object]:
    """Return the repair parameters that restore one nominal component."""

    if component.kind == "resistor":
        return {"value_ohm": float_parameter(component.parameters, "value_ohm")}
    if component.kind == "capacitor":
        return {"value_F": float_parameter(component.parameters, "value_F")}
    if component.kind == "diode":
        return {
            "forward_drop_V": float_parameter(component.parameters, "forward_drop_V")
        }
    if component.kind == "switch":
        return {"closed": bool_parameter(component.parameters, "closed")}
    if component.kind == "voltage_source":
        return {
            "voltage_V": float_parameter(component.parameters, "voltage_V"),
            "internal_resistance_ohm": 0.0,
        }
    if component.kind == "current_source":
        return {"current_A": float_parameter(component.parameters, "current_A")}
    raise ValueError(f"unsupported component kind: {component.kind}")
