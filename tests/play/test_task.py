"""Tests for reusable play-test descriptors."""

from collections.abc import Mapping

import pytest

from rlvr_physics.core.factory import ConfiguredTask
from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.play.interaction import DEFAULT_PUBLIC_INFO_EXCLUDED_KEYS
from rlvr_physics.play.task import (
    PlayableTask,
    build_playable_interaction_config,
    parameters_with_overrides,
)
from rlvr_physics.core.specs import (
    RendererSpec,
    RewardSpec,
    SourceSpec,
    TaskSpec,
    VerifierSpec,
)
from tests.conftest import ExampleScalarSession


def test_playable_task_rejects_default_renderer_outside_supported_set() -> None:
    """Playable descriptors require a usable default renderer."""

    with pytest.raises(ValueError, match="default_renderer"):
        PlayableTask(
            name="tests.playable",
            default_renderer="missing",
            renderers=("text",),
            default_parameters={},
            build_task=_build_task,
            public_info_excluded_keys=DEFAULT_PUBLIC_INFO_EXCLUDED_KEYS,
        )


def test_parameters_with_overrides_parses_json_values() -> None:
    """CLI parameter overrides preserve defaults and parse JSON values."""

    parameters = parameters_with_overrides(
        default_parameters={"count": 1, "enabled": False},
        overrides=("count=3", 'name="cart"', "enabled=true"),
    )

    assert parameters == {"count": 3, "enabled": True, "name": "cart"}


@pytest.mark.parametrize("override", ("count=NaN", "count=Infinity", "count=1e309"))
def test_parameters_with_overrides_rejects_non_finite_numbers(
    override: str,
) -> None:
    """CLI parameter overrides reject non-standard or non-finite numbers."""

    with pytest.raises(ValueError, match="count"):
        parameters_with_overrides(
            default_parameters={"count": 1}, overrides=(override,)
        )


def test_build_playable_interaction_config_uses_descriptor_builder() -> None:
    """Playable descriptors build generic task interaction configs."""

    playable = PlayableTask(
        name="tests.playable",
        default_renderer="text",
        renderers=("text",),
        default_parameters={"suffix": "ok"},
        build_task=_build_task,
        public_info_excluded_keys=DEFAULT_PUBLIC_INFO_EXCLUDED_KEYS,
    )

    config = build_playable_interaction_config(
        playable=playable,
        parameters={"suffix": "ok"},
        renderer_type="text",
        instance_seed=5,
        session_seed=6,
    )
    instance = config.task.build_instance(seed=5)

    assert instance.task_id == "playable-test-ok-5"
    assert config.session_seed == 6


def _build_task(parameters: Mapping[str, object], renderer_type: str) -> ConfiguredTask:
    """Build a configured task from test play parameters.

    Parameters
    ----------
    parameters:
        Public task parameters.
    renderer_type:
        Renderer identifier selected for emitted observations.

    Returns
    -------
    ConfiguredTask
        Configured test task.
    """

    suffix = parameters["suffix"] if "suffix" in parameters else "default"
    if not isinstance(suffix, str):
        raise TypeError("suffix must be a string")

    def build_instance(seed: int) -> TaskInstance:
        """Build one immutable test instance.

        Parameters
        ----------
        seed:
            Deterministic instance seed.

        Returns
        -------
        TaskInstance
            Immutable test instance.
        """

        return TaskInstance(
            task_id=f"playable-test-{suffix}-{seed}",
            kind="tests.playable.v1",
            domain="tests",
            seed=seed,
            public_payload={},
            privileged_payload={},
            max_turns=1,
        )

    return ConfiguredTask(
        spec=TaskSpec(
            kind="tests.playable.v1",
            domain="tests",
            source=SourceSpec(source_type="tests.playable", seed=0),
            renderers=(RendererSpec(renderer_type=renderer_type),),
            verifier=VerifierSpec(verifier_type="tests.playable"),
            reward=RewardSpec(reward_type="tests.playable", parameters={}),
            max_turns=1,
        ),
        instance_builder=build_instance,
        session_builder=ExampleScalarSession,
    )
