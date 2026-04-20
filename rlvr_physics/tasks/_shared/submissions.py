"""Submission text helpers shared by task verifiers."""


def nonempty_stripped_lines(text: str) -> tuple[str, ...]:
    """Return stripped nonempty lines from submission text.

    Parameters
    ----------
    text:
        Raw text to split.
    """

    return tuple(line.strip() for line in text.splitlines() if line.strip())


def strip_code_fence_lines(text: str) -> str:
    """Remove standalone Markdown code fence lines from text.

    Parameters
    ----------
    text:
        Raw text that may include fenced code blocks.
    """

    lines = []
    for line in text.replace("```python", "```").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("```"):
            lines.append(stripped)
    return "\n".join(lines)
