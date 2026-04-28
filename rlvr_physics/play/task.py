"""Reusable play-test descriptors and CLI helpers."""

from argparse import ArgumentParser
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
from math import isfinite
from typing import TextIO

from rlvr_physics.core.factory import ConfiguredTask
from rlvr_physics.play.interaction import TaskInteractionConfig, run_task_interaction
from rlvr_physics.core.payloads import freeze_mapping, mapping_to_dict

PlayableTaskBuilder = Callable[[Mapping[str, object], str], ConfiguredTask]


@dataclass(frozen=True)
class PlayableTask:
    """Task family descriptor for public interaction play-tests.

    Parameters
    ----------
    name:
        Stable play-test task family name.
    default_renderer:
        Renderer selected when the CLI does not receive ``--renderer``.
    renderers:
        Supported renderer identifiers for public play-tests.
    default_parameters:
        Public task construction parameters used by default.
    build_task:
        Callable that builds a configured task from public parameters and a
        renderer identifier.
    public_info_excluded_keys:
        ``public_info`` keys omitted from public interaction events.

    Attributes
    ----------
    name:
        Stable play-test task family name.
    default_renderer:
        Renderer selected when the CLI does not receive ``--renderer``.
    renderers:
        Supported renderer identifiers for public play-tests.
    default_parameters:
        Public task construction parameters used by default.
    build_task:
        Callable that builds a configured task from public parameters and a
        renderer identifier.
    public_info_excluded_keys:
        ``public_info`` keys omitted from public interaction events.
    """

    name: str
    default_renderer: str
    renderers: tuple[str, ...]
    default_parameters: Mapping[str, object]
    build_task: PlayableTaskBuilder
    public_info_excluded_keys: frozenset[str]

    def __post_init__(self) -> None:
        """Validate and freeze descriptor payloads."""

        if self.default_renderer not in self.renderers:
            raise ValueError("default_renderer must be one of renderers")
        object.__setattr__(
            self, "default_parameters", freeze_mapping(self.default_parameters)
        )


def build_playable_interaction_config(
    playable: PlayableTask,
    parameters: Mapping[str, object],
    renderer_type: str,
    instance_seed: int,
    session_seed: int,
) -> TaskInteractionConfig:
    """Build a generic interaction config from a playable task descriptor.

    Parameters
    ----------
    playable:
        Playable task family descriptor.
    parameters:
        Public task construction parameters.
    renderer_type:
        Public renderer identifier selected for this rollout.
    instance_seed:
        Private seed used to build the immutable task instance.
    session_seed:
        Seed used to reset the scalar session.

    Returns
    -------
    TaskInteractionConfig
        Generic interaction configuration ready for
        :func:`run_task_interaction`.
    """

    if renderer_type not in playable.renderers:
        raise ValueError(f"unsupported renderer for {playable.name}: {renderer_type}")
    return TaskInteractionConfig(
        task=playable.build_task(parameters, renderer_type),
        instance_seed=instance_seed,
        session_seed=session_seed,
        public_info_excluded_keys=playable.public_info_excluded_keys,
    )


def run_playable_interaction_cli(
    playable: PlayableTask,
    argv: Sequence[str] | None,
    input_stream: TextIO,
    output_stream: TextIO,
    error_stream: TextIO,
    program_name: str,
) -> int:
    """Run a playable task through the shared JSONL CLI protocol.

    Parameters
    ----------
    playable:
        Playable task family descriptor.
    argv:
        Command line arguments excluding the program name. ``None`` reads
        arguments from :data:`sys.argv`.
    input_stream:
        Stream containing one model submission per line.
    output_stream:
        Stream receiving one public JSON event per line.
    error_stream:
        Stream receiving process-level errors that are not task observations.
    program_name:
        Program name shown in CLI help and errors.

    Returns
    -------
    int
        Process-style status code from :func:`run_task_interaction`.
    """

    parser = playable_argument_parser(playable, program_name)
    args = parser.parse_args(argv)
    try:
        parameters = parameters_with_overrides(
            playable.default_parameters, args.parameter
        )
        config = build_playable_interaction_config(
            playable=playable,
            parameters=parameters,
            renderer_type=args.renderer,
            instance_seed=args.instance_seed,
            session_seed=args.session_seed,
        )
    except ValueError as error:
        parser.error(str(error))
    return run_task_interaction(
        config=config,
        input_stream=input_stream,
        output_stream=output_stream,
        error_stream=error_stream,
    )


def playable_argument_parser(
    playable: PlayableTask, program_name: str
) -> ArgumentParser:
    """Build a command line parser for a playable task.

    Parameters
    ----------
    playable:
        Playable task family descriptor.
    program_name:
        Program name shown in CLI help and errors.

    Returns
    -------
    ArgumentParser
        Parser for the task's JSONL interaction command.
    """

    parser = ArgumentParser(
        prog=program_name,
        description=(
            f"Run {playable.name} as a JSONL interaction. "
            "Private instance seeds are not emitted in public events."
        ),
    )
    parser.add_argument(
        "--instance-seed",
        type=int,
        required=True,
        help="private deterministic seed for building the task instance",
    )
    parser.add_argument(
        "--session-seed",
        type=int,
        required=True,
        help="deterministic seed used to reset the scalar session",
    )
    parser.add_argument(
        "--renderer",
        choices=playable.renderers,
        default=playable.default_renderer,
        help="public observation renderer",
    )
    parser.add_argument(
        "--parameter",
        action="append",
        default=[],
        metavar="KEY=JSON",
        help="override one public task parameter with a JSON value",
    )
    return parser


def parameters_with_overrides(
    default_parameters: Mapping[str, object], overrides: Sequence[str]
) -> dict[str, object]:
    """Return task parameters with CLI overrides applied.

    Parameters
    ----------
    default_parameters:
        Default public task construction parameters.
    overrides:
        Strings of the form ``KEY=JSON``.

    Returns
    -------
    dict[str, object]
        Plain parameter mapping with override values applied.
    """

    parameters = mapping_to_dict(default_parameters)
    for override in overrides:
        key, value = _parse_parameter_override(override)
        parameters[key] = value
    return parameters


def _parse_parameter_override(override: str) -> tuple[str, object]:
    """Parse one ``KEY=JSON`` parameter override."""

    key, separator, raw_value = override.partition("=")
    if separator == "" or key == "":
        raise ValueError("parameter overrides must use KEY=JSON")
    try:
        value = json.loads(
            raw_value,
            parse_constant=_reject_json_constant,
            parse_float=_parse_json_float,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"parameter {key!r} must use a JSON value") from error
    return key, value


def _parse_json_float(raw: str) -> float:
    """Parse a JSON float and reject non-finite values."""

    value = float(raw)
    if not isfinite(value):
        raise ValueError(f"parameter float must be finite: {raw}")
    return value


def _reject_json_constant(raw: str) -> object:
    """Reject non-standard JSON numeric constants."""

    raise ValueError(f"invalid JSON numeric constant: {raw}")
