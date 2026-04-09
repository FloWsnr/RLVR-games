"""Project-specific exceptions."""


class RLVRGamesError(Exception):
    """Base exception for project-specific runtime errors.

    All framework-defined exceptions inherit from this type so callers can
    catch RLVR-specific failures without swallowing unrelated exceptions.
    """


class EnvironmentNotResetError(RLVRGamesError):
    """Raised when episode state is accessed before initialization.

    This typically indicates that `reset()` has not been called before using
    `step()`, `state`, or `trajectory`.
    """


class EpisodeFinishedError(RLVRGamesError):
    """Raised when a transition is requested after episode completion.

    Environments use this to enforce the `reset()`/`step()` lifecycle once an
    episode has terminated or been truncated.
    """


class InvalidActionError(RLVRGamesError):
    """Raised when an action is malformed or illegal for the current state.

    Backends should use this exception when model output cannot be mapped onto
    a valid, rule-compliant environment action.
    """
