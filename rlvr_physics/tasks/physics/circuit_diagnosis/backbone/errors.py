"""Circuit diagnosis backbone exceptions."""


class SubmissionParseError(ValueError):
    """Raised when a model submission cannot be interpreted for this task."""


class CircuitSimulationError(RuntimeError):
    """Raised when a physical circuit cannot be solved."""
