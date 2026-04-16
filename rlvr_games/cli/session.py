"""Interactive CLI session runners."""

import json
from pathlib import Path
from typing import Any, TextIO

from rlvr_games.cli.specs import GameCliSpec, InteractiveCommandSpec
from rlvr_games.core.exceptions import InvalidActionError
from rlvr_games.core.protocol import Environment
from rlvr_games.core.rollout import build_action_context
from rlvr_games.core.types import Observation, RenderedImage, StepResult

_BUILT_IN_COMMAND_USAGES: tuple[str, ...] = (
    "help",
    "state",
    "show <key>",
    "debug-state",
    "debug-show <key>",
    "debug-legal",
    "trajectory",
    "quit",
    "exit",
)
_BUILT_IN_COMMAND_NAMES = frozenset(
    {
        "help",
        "state",
        "show",
        "debug-state",
        "debug-show",
        "debug-legal",
        "trajectory",
        "quit",
        "exit",
    }
)


def run_play_session(
    *,
    env: Environment[Any, Any],
    game_spec: GameCliSpec,
    seed: int,
    image_output_dir: Path | None,
    input_stream: TextIO,
    output_stream: TextIO,
) -> int:
    """Run an interactive play session against an in-process environment.

    Parameters
    ----------
    env : Environment[Any, Any]
        Environment to reset and step during the session.
    game_spec : GameCliSpec
        CLI specification describing game-specific session behavior.
    seed : int
        Explicit scenario seed passed into ``env.reset(...)``.
    image_output_dir : Path | None
        Optional directory where any rendered observation images should be
        persisted as PNG files for inspection.
    input_stream : TextIO
        Input stream used to read commands or raw actions.
    output_stream : TextIO
        Output stream used to print observations and transition summaries.

    Returns
    -------
    int
        Process-style exit code, where ``0`` denotes a clean session exit.
    """
    extra_command_map = _build_extra_command_map(game_spec=game_spec)
    try:
        observation, reset_info = env.reset(seed=seed)
        _write_line(output_stream, f"Reset info: {_format_json(reset_info)}")
        _write_observation(
            output_stream,
            observation,
            image_output_dir=image_output_dir,
        )
        if env.episode_finished:
            _write_line(output_stream, "Episode finished.")
            return 0

        while True:
            context = build_action_context(env=env)
            output_stream.write(f"turn[{context.turn_index}]> ")
            output_stream.flush()

            raw_input = input_stream.readline()
            if raw_input == "":
                _write_line(output_stream, "")
                return 0

            command_text = raw_input.strip()
            if not command_text:
                continue

            command_tokens = tuple(command_text.split())
            command_name = command_tokens[0]
            command_arguments = command_tokens[1:]

            if command_name in {"quit", "exit"}:
                _write_line(output_stream, "Session ended.")
                return 0

            if command_name == "help":
                _write_help(output_stream=output_stream, game_spec=game_spec)
                continue

            if command_name == "debug-legal":
                legal_actions = " ".join(env.legal_actions())
                _write_line(
                    output_stream,
                    f"Legal actions ({len(env.legal_actions())}): {legal_actions}",
                )
                continue

            if command_name == "state":
                _write_line(
                    output_stream, f"State: {_format_json(observation.metadata)}"
                )
                continue

            if command_name == "show":
                _write_metadata_value(
                    output_stream=output_stream,
                    state_view=observation.metadata,
                    arguments=command_arguments,
                    command_name="show",
                )
                continue

            if command_name == "debug-state":
                _write_line(
                    output_stream,
                    f"Canonical state: {_format_json(env.inspect_canonical_state())}",
                )
                continue

            if command_name == "debug-show":
                _write_metadata_value(
                    output_stream=output_stream,
                    state_view=env.inspect_canonical_state(),
                    arguments=command_arguments,
                    command_name="debug-show",
                )
                continue

            if command_name == "trajectory":
                _write_trajectory(output_stream, env)
                continue

            extra_command = extra_command_map.get(command_name)
            if extra_command is not None:
                extra_command.handler(
                    env=env,
                    observation=observation,
                    context=context,
                    arguments=command_arguments,
                    output_stream=output_stream,
                )
                continue

            try:
                step_result = env.step(command_text)
            except InvalidActionError as exc:
                _write_line(output_stream, f"Invalid action: {exc}")
                continue

            observation = step_result.observation
            _write_step_result(
                output_stream=output_stream,
                step_result=step_result,
                game_spec=game_spec,
            )
            _write_observation(
                output_stream,
                observation,
                image_output_dir=image_output_dir,
            )

            if step_result.terminated or step_result.truncated:
                _write_line(output_stream, "Episode finished.")
                return 0
    finally:
        env.close()


def _build_extra_command_map(
    *,
    game_spec: GameCliSpec,
) -> dict[str, InteractiveCommandSpec]:
    """Validate and index game-specific interactive commands.

    Parameters
    ----------
    game_spec : GameCliSpec
        CLI specification whose extra commands should be indexed.

    Returns
    -------
    dict[str, InteractiveCommandSpec]
        Mapping from command token to command specification.

    Raises
    ------
    ValueError
        If a game registers duplicate commands or collides with built-ins.
    """
    command_map: dict[str, InteractiveCommandSpec] = {}
    for command_spec in game_spec.interactive_commands:
        if command_spec.name in _BUILT_IN_COMMAND_NAMES:
            raise ValueError(
                f"Interactive command {command_spec.name!r} collides with a built-in "
                "CLI command."
            )
        if command_spec.name in command_map:
            raise ValueError(
                f"Interactive command {command_spec.name!r} was registered twice."
            )
        command_map[command_spec.name] = command_spec
    return command_map


def _format_json(payload: dict[str, object]) -> str:
    """Serialize a metadata dictionary for CLI output.

    Parameters
    ----------
    payload : dict[str, object]
        Dictionary payload to serialize.

    Returns
    -------
    str
        Stable JSON representation of the payload.
    """
    return json.dumps(payload, default=str, sort_keys=True)


def _write_help(
    *,
    output_stream: TextIO,
    game_spec: GameCliSpec,
) -> None:
    """Print the supported interactive commands.

    Parameters
    ----------
    output_stream : TextIO
        Stream receiving help output.
    game_spec : GameCliSpec
        CLI specification whose extra commands should be included.
    """
    command_usages = list(_BUILT_IN_COMMAND_USAGES)
    command_usages.extend(
        command_spec.usage for command_spec in game_spec.interactive_commands
    )
    _write_line(output_stream, f"Commands: {' '.join(command_usages)}")


def _write_metadata_value(
    *,
    output_stream: TextIO,
    state_view: dict[str, object],
    arguments: tuple[str, ...],
    command_name: str,
) -> None:
    """Print one observation metadata value selected by key.

    Parameters
    ----------
    output_stream : TextIO
        Stream receiving command output.
    state_view : dict[str, object]
        Observation or canonical debug summary to query.
    arguments : tuple[str, ...]
        Command arguments following the metadata inspection command.
    """
    if len(arguments) != 1:
        _write_line(output_stream, f"Usage: {command_name} <key>")
        return

    key = arguments[0]
    if key not in state_view:
        _write_line(output_stream, f"Metadata key unavailable: {key}")
        return

    value = state_view[key]
    _write_line(output_stream, f"{key}: {_format_metadata_value(value)}")


def _format_metadata_value(value: object) -> str:
    """Format one observation metadata value for CLI output.

    Parameters
    ----------
    value : object
        Metadata value to format.

    Returns
    -------
    str
        Readable serialized value.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, int, float)) or value is None:
        return str(value)
    return json.dumps(value, default=str, sort_keys=True)


def _write_line(output_stream: TextIO, line: str) -> None:
    """Write one newline-terminated line to the output stream.

    Parameters
    ----------
    output_stream : TextIO
        Stream receiving the line.
    line : str
        Line content to write without a trailing newline.
    """
    output_stream.write(f"{line}\n")


def _write_observation(
    output_stream: TextIO,
    observation: Observation,
    *,
    image_output_dir: Path | None,
) -> None:
    """Write an observation's text and any persisted image paths.

    Parameters
    ----------
    output_stream : TextIO
        Stream receiving formatted observation output.
    observation : Observation
        Observation to print and optionally persist.
    image_output_dir : Path | None
        Optional directory where in-memory images should be saved as PNG files.
    """
    if observation.text is not None:
        _write_line(output_stream, observation.text)
    if observation.images and image_output_dir is not None:
        image_paths = ", ".join(
            str(path)
            for path in _persist_rendered_images(
                image_output_dir=image_output_dir,
                images=observation.images,
            )
        )
        _write_line(output_stream, f"Image paths: {image_paths}")


def _persist_rendered_images(
    *,
    image_output_dir: Path,
    images: tuple[RenderedImage, ...],
) -> tuple[Path, ...]:
    """Persist rendered images to PNG files and return their paths.

    Parameters
    ----------
    image_output_dir : Path
        Directory where PNG files should be written.
    images : tuple[RenderedImage, ...]
        In-memory images to persist.

    Returns
    -------
    tuple[Path, ...]
        Persisted PNG paths in the same order as the input images.
    """
    image_output_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    for image_index, rendered_image in enumerate(images):
        suffix = "" if image_index == 0 else f"-{image_index}"
        image_path = image_output_dir / f"{rendered_image.key}{suffix}.png"
        if not image_path.exists():
            rendered_image.image.save(image_path, format="PNG")
        image_paths.append(image_path)
    return tuple(image_paths)


def _write_step_result(
    *,
    output_stream: TextIO,
    step_result: StepResult,
    game_spec: GameCliSpec,
) -> None:
    """Write a step result summary to the output stream.

    Parameters
    ----------
    output_stream : TextIO
        Stream receiving formatted step output.
    step_result : StepResult
        Step result to summarize.
    game_spec : GameCliSpec
        CLI specification that provides any game-specific summary lines.
    """
    _write_line(output_stream, f"Accepted: {step_result.accepted}")
    for summary_line in game_spec.format_step_result(step_result):
        _write_line(output_stream, summary_line)
    _write_line(output_stream, f"Reward: {step_result.reward}")
    _write_line(output_stream, f"Terminated: {step_result.terminated}")
    _write_line(output_stream, f"Truncated: {step_result.truncated}")
    _write_line(output_stream, f"Info: {_format_json(step_result.info)}")


def _write_trajectory(output_stream: TextIO, env: Environment[Any, Any]) -> None:
    """Write a short trajectory summary for the active episode.

    Parameters
    ----------
    output_stream : TextIO
        Stream receiving formatted trajectory output.
    env : Environment[Any, Any]
        Active environment whose trajectory should be summarized.
    """
    _write_line(
        output_stream,
        f"Reset events: {len(env.trajectory.reset_events)}",
    )
    for event_index, event in enumerate(
        env.trajectory.reset_events,
        start=1,
    ):
        info_text = _format_json(event.info)
        _write_line(
            output_stream,
            (
                f"reset[{event_index}]. source={event.source!r} "
                f"label={event.label!r} info={info_text}"
            ),
        )
    _write_line(output_stream, f"Trajectory steps: {len(env.trajectory.steps)}")
    for turn_index, step in enumerate(env.trajectory.steps, start=1):
        info_text = _format_json(step.info)
        _write_line(
            output_stream,
            (
                f"{turn_index}. raw_action={step.raw_action!r} "
                f"accepted={step.accepted} "
                f"reward={step.reward} terminated={step.terminated} "
                f"truncated={step.truncated} info={info_text}"
            ),
        )
