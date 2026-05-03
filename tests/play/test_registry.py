"""Tests for the playable task registry."""

from collections.abc import Mapping
from types import ModuleType

import pytest

from rlvr_physics.core.factory import ConfiguredTask
from rlvr_physics.play import registry
from rlvr_physics.play.task import PlayableTask


def test_playable_task_names_do_not_import_registered_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Listing names uses lazy registration metadata only."""

    def fail_import(module_name: str) -> ModuleType:
        """Fail if a task module import is attempted."""

        raise AssertionError(f"unexpected import: {module_name}")

    monkeypatch.setattr(registry, "import_module", fail_import)

    assert registry.playable_task_names() == (
        "cart_inference",
        "physics.cart_inference",
    )


def test_get_playable_task_imports_only_on_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Task lookup loads the registered object lazily."""

    calls: list[str] = []

    def fake_import(module_name: str) -> ModuleType:
        """Return a fake module with a playable task."""

        calls.append(module_name)
        return _module_with_playable(
            registry.PlayableTaskRegistration(
                name="physics.cart_inference",
                aliases=("cart_inference",),
                object_path=(
                    "rlvr_physics.tasks.physics.cart_inference.play:CART_PLAYABLE"
                ),
            )
        )

    monkeypatch.setattr(registry, "import_module", fake_import)

    playable = registry.get_playable_task("cart_inference")

    assert isinstance(playable, PlayableTask)
    assert playable.name == "physics.cart_inference"
    assert calls == ["rlvr_physics.tasks.physics.cart_inference.play"]


def _module_with_playable(
    registration: registry.PlayableTaskRegistration,
) -> ModuleType:
    """Return a module-like object containing a matching playable."""

    module = ModuleType("fake_playable_module")

    def build_task(_: Mapping[str, object], __: str) -> ConfiguredTask:
        """Unused task builder for registry tests."""

        raise AssertionError("task builder should not be called")

    setattr(
        module,
        "CART_PLAYABLE",
        PlayableTask(
            name=registration.name,
            default_renderer="text",
            renderers=("text",),
            default_parameters={},
            build_task=build_task,
            public_info_excluded_keys=frozenset(),
        ),
    )
    return module
