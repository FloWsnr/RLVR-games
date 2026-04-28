"""Prompt resource loading for the cart inference task."""

from collections.abc import Mapping
from functools import lru_cache
from importlib import resources
import re

_CART_PACKAGE = "rlvr_physics.tasks.physics.cart_inference"
_PROMPTS_DIR = "prompts"
_TEXT_OBSERVATION_PROMPT = "text_observation.md"
_INITIAL_FEEDBACK_PROMPT = "initial_feedback.md"
_PROMPT_MARKER_PATTERN = re.compile(r"{{([A-Za-z_][A-Za-z0-9_]*)}}")
_PROMPT_BRACE_PATTERN = re.compile(r"{{([^{}]+)}}")
_PROMPT_MARKER_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@lru_cache(maxsize=1)
def cart_text_prompt_template() -> str:
    """Return the text observation prompt template.

    Returns
    -------
    str
        Template text with ``{{marker}}`` fields.
    """

    return _read_cart_prompt(_TEXT_OBSERVATION_PROMPT).rstrip()


@lru_cache(maxsize=1)
def cart_initial_feedback() -> str:
    """Return the initial cart task feedback line.

    Returns
    -------
    str
        First public feedback line shown at reset.
    """

    return _read_cart_prompt(_INITIAL_FEEDBACK_PROMPT).rstrip()


def render_prompt_template(template: str, values: Mapping[str, object]) -> str:
    """Render a cart prompt template with explicit marker replacement.

    Parameters
    ----------
    template:
        Prompt template text with ``{{marker}}`` fields.
    values:
        Marker values keyed by marker name.

    Returns
    -------
    str
        Rendered prompt text.

    Raises
    ------
    ValueError
        Raised when the template contains malformed or unknown markers.
    """

    malformed_markers = _malformed_prompt_markers(template)
    if malformed_markers:
        marker_list = ", ".join(malformed_markers)
        raise ValueError(f"prompt template has malformed markers: {marker_list}")

    marker_names = set(_PROMPT_MARKER_PATTERN.findall(template))
    unknown_markers = sorted(marker for marker in marker_names if marker not in values)
    if unknown_markers:
        marker_list = ", ".join(unknown_markers)
        raise ValueError(f"prompt template has unresolved markers: {marker_list}")

    return _PROMPT_MARKER_PATTERN.sub(
        lambda match: str(values[match.group(1)]),
        template,
    )


def _malformed_prompt_markers(template: str) -> list[str]:
    """Return malformed double-brace markers from a prompt template.

    Parameters
    ----------
    template:
        Prompt template text.

    Returns
    -------
    list of str
        Invalid marker bodies in deterministic order.
    """

    marker_bodies = set(_PROMPT_BRACE_PATTERN.findall(template))
    return sorted(
        marker_body
        for marker_body in marker_bodies
        if _PROMPT_MARKER_NAME_PATTERN.fullmatch(marker_body) is None
    )


def _read_cart_prompt(filename: str) -> str:
    """Read one cart prompt asset from the task package.

    Parameters
    ----------
    filename:
        Prompt asset filename inside the cart task ``prompts`` directory.

    Returns
    -------
    str
        Prompt asset text.
    """

    return (
        resources.files(_CART_PACKAGE)
        .joinpath(_PROMPTS_DIR, filename)
        .read_text(encoding="utf-8")
    )
