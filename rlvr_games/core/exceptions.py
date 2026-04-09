"""Project-specific exceptions."""


class RLVRGamesError(Exception):
    """Base exception for the project."""


class EnvironmentNotResetError(RLVRGamesError):
    """Raised when `step()` is called before `reset()`."""


class EpisodeFinishedError(RLVRGamesError):
    """Raised when `step()` is called after the episode has ended."""


class InvalidActionError(RLVRGamesError):
    """Raised when an action cannot be parsed or applied."""
