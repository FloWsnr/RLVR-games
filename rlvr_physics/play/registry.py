"""Registry of task families available through the play CLI."""

from dataclasses import dataclass
from importlib import import_module

from rlvr_physics.play.task import PlayableTask


@dataclass(frozen=True)
class PlayableTaskRegistration:
    """Lazy registration for one playable task family.

    Parameters
    ----------
    name:
        Stable playable task name.
    aliases:
        Additional accepted CLI aliases.
    object_path:
        Import path of the registered :class:`PlayableTask` object in
        ``module:attribute`` form.

    Attributes
    ----------
    name:
        Stable playable task name.
    aliases:
        Additional accepted CLI aliases.
    object_path:
        Import path of the registered :class:`PlayableTask` object in
        ``module:attribute`` form.
    """

    name: str
    aliases: tuple[str, ...]
    object_path: str


_PLAYABLE_REGISTRATIONS = (
    PlayableTaskRegistration(
        name="physics.cart_inference",
        aliases=("cart_inference",),
        object_path="rlvr_physics.tasks.physics.cart_inference.play:CART_PLAYABLE",
    ),
    PlayableTaskRegistration(
        name="physics.circuit_diagnosis",
        aliases=("circuit_diagnosis",),
        object_path=(
            "rlvr_physics.tasks.physics.circuit_diagnosis.play:CIRCUIT_PLAYABLE"
        ),
    ),
)


def registered_playable_tasks() -> tuple[PlayableTaskRegistration, ...]:
    """Return task families available through the play CLI.

    Returns
    -------
    tuple of PlayableTaskRegistration
        Registered playable task descriptors.
    """

    return _PLAYABLE_REGISTRATIONS


def playable_task_registry() -> dict[str, PlayableTaskRegistration]:
    """Return playable task aliases mapped to lazy registrations.

    Returns
    -------
    dict[str, PlayableTaskRegistration]
        Registry keyed by stable task names and short aliases.

    Raises
    ------
    RuntimeError
        Raised when two playable tasks claim the same alias.
    """

    registry: dict[str, PlayableTaskRegistration] = {}
    for registration in registered_playable_tasks():
        for alias in _registration_aliases(registration):
            existing = registry.get(alias)
            if existing is not None and existing != registration:
                raise RuntimeError(f"duplicate playable task alias: {alias}")
            registry[alias] = registration
    return registry


def playable_task_names() -> tuple[str, ...]:
    """Return all accepted playable task names and aliases.

    Returns
    -------
    tuple of str
        Sorted accepted task names.
    """

    return tuple(sorted(playable_task_registry()))


def get_playable_task(name: str) -> PlayableTask | None:
    """Return the playable task matching a name or alias.

    Parameters
    ----------
    name:
        Stable task name or short alias.

    Returns
    -------
    PlayableTask or None
        Matching playable task descriptor, or ``None`` when not registered.
    """

    registration = playable_task_registry().get(name)
    if registration is None:
        return None
    playable = _load_playable_task(registration)
    if playable.name != registration.name:
        raise RuntimeError(
            "registered playable name mismatch: "
            f"{registration.name!r} != {playable.name!r}"
        )
    return playable


def _load_playable_task(registration: PlayableTaskRegistration) -> PlayableTask:
    """Load a playable task object from a lazy registration."""

    module_name, separator, attribute_name = registration.object_path.partition(":")
    if separator == "" or module_name == "" or attribute_name == "":
        raise RuntimeError(
            f"invalid playable task object path: {registration.object_path}"
        )
    module = import_module(module_name)
    value = getattr(module, attribute_name)
    if not isinstance(value, PlayableTask):
        raise RuntimeError(
            f"registered object is not a PlayableTask: {registration.object_path}"
        )
    return value


def _registration_aliases(
    registration: PlayableTaskRegistration,
) -> tuple[str, ...]:
    """Return accepted names for a playable task registration."""

    return tuple(dict.fromkeys((registration.name, *registration.aliases)))
