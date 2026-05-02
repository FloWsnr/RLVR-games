"""Prompt resource helpers for circuit diagnosis."""

from importlib import resources
import re

_PROMPT_PACKAGE = "rlvr_physics.tasks.physics.circuit_diagnosis.prompts"
_MARKER_PATTERN = re.compile(r"{{([a-zA-Z0-9_]+)}}")


def circuit_initial_feedback(fault_count_range: tuple[int, int]) -> str:
    """Return the first public feedback message for circuit diagnosis."""

    return render_prompt_template(
        _prompt_file_text("initial_feedback.md"),
        {"fault_count_text": _fault_count_text(fault_count_range)},
    )


def circuit_text_prompt_template() -> str:
    """Return the task-local text observation prompt template."""

    return _prompt_file_text("text_observation.md")


def render_prompt_template(template: str, values: dict[str, str]) -> str:
    """Render a simple ``{{name}}`` prompt template.

    Parameters
    ----------
    template:
        Template text containing marker names.
    values:
        Replacement values keyed by marker name.

    Returns
    -------
    str
        Rendered prompt text.
    """

    def replace_marker(match: re.Match[str]) -> str:
        marker = match.group(1)
        try:
            return values[marker]
        except KeyError as error:
            raise ValueError(f"missing prompt template value: {marker}") from error

    return _MARKER_PATTERN.sub(replace_marker, template)


def _prompt_file_text(name: str) -> str:
    """Load one task-local prompt resource."""

    return resources.files(_PROMPT_PACKAGE).joinpath(name).read_text(encoding="utf-8")


def _fault_count_text(fault_count_range: tuple[int, int]) -> str:
    """Return readable public hidden-fault count text."""

    min_fault_count, max_fault_count = fault_count_range
    if min_fault_count == max_fault_count:
        if min_fault_count == 1:
            return "One hidden fault is present"
        return f"{min_fault_count} hidden faults are present"
    if min_fault_count == 1 and max_fault_count == 2:
        return "One or two hidden faults are present"
    return f"Between {min_fault_count} and {max_fault_count} hidden faults are present"
