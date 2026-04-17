"""Shared parser helpers for interactive play CLIs."""

from argparse import Action, ArgumentParser, Namespace
from pathlib import Path
import sys
from typing import Any, Sequence, cast

from rlvr_games.core.protocol import Environment
from rlvr_games.task_specs import (
    TaskSpec,
    build_environment_from_task_spec,
    load_task_spec,
)
from rlvr_games.core.types import (
    EpisodeConfig,
    InvalidActionMode,
    InvalidActionPolicy,
)

COMMON_TASK_SPEC_DISALLOWED_ARGUMENT_NAMES = (
    "max_attempts",
    "max_transitions",
    "image_size",
    "invalid_action_policy",
    "invalid_action_penalty",
)


def add_common_play_arguments(parser: ArgumentParser) -> None:
    """Attach play-session arguments shared by supported games.

    Parameters
    ----------
    parser : ArgumentParser
        Parser to extend with common play-session arguments.
    """
    ensure_argument_tracking(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--max-transitions", type=int)
    parser.add_argument("--image-output-dir", type=Path)
    parser.add_argument("--image-size", type=int, default=360)
    parser.add_argument("--task-spec", type=Path)
    parser.add_argument(
        "--invalid-action-policy",
        choices=tuple(mode.value for mode in InvalidActionMode),
        default=InvalidActionMode.RAISE.value,
    )
    parser.add_argument("--invalid-action-penalty", type=float)


def ensure_argument_tracking(parser: ArgumentParser) -> None:
    """Install supplied-argument tracking on one parser.

    Parameters
    ----------
    parser : ArgumentParser
        Parser whose `parse_args` method should annotate explicitly supplied
        options on the returned namespace.
    """
    parser.allow_abbrev = False
    if getattr(parser, "_rlvr_argument_tracking_installed", False):
        return

    original_parse_args = parser.parse_args

    def parse_args(
        args: Sequence[str] | None = None,
        namespace: Namespace | None = None,
    ) -> Namespace:
        raw_argv = tuple(sys.argv[1:] if args is None else args)
        parsed_args = cast(
            Namespace,
            original_parse_args(args=args, namespace=namespace),
        )
        setattr(
            parsed_args,
            "_supplied_argument_names",
            frozenset(
                _collect_supplied_argument_names(
                    parser=parser,
                    args=parsed_args,
                    raw_argv=raw_argv,
                )
            ),
        )
        return parsed_args

    setattr(parser, "parse_args", parse_args)
    setattr(parser, "_rlvr_argument_tracking_installed", True)


def build_episode_config(
    *,
    args: Namespace,
    parser: ArgumentParser,
) -> EpisodeConfig:
    """Build an episode configuration from parsed CLI arguments.

    Parameters
    ----------
    args : Namespace
        Parsed CLI arguments.
    parser : ArgumentParser
        Parser used to raise argument errors when the configuration is
        inconsistent.

    Returns
    -------
    EpisodeConfig
        Episode configuration implied by the parsed arguments.
    """
    return EpisodeConfig(
        max_attempts=args.max_attempts,
        max_transitions=args.max_transitions,
        invalid_action_policy=build_invalid_action_policy(args=args, parser=parser),
    )


def build_invalid_action_policy(
    *,
    args: Namespace,
    parser: ArgumentParser,
) -> InvalidActionPolicy:
    """Build the invalid-action policy implied by parsed CLI arguments.

    Parameters
    ----------
    args : Namespace
        Parsed CLI arguments.
    parser : ArgumentParser
        Parser used to raise argument errors when the configuration is
        inconsistent.

    Returns
    -------
    InvalidActionPolicy
        Validated invalid-action policy for the environment.
    """
    invalid_action_mode = InvalidActionMode(args.invalid_action_policy)
    if invalid_action_mode == InvalidActionMode.RAISE:
        if args.invalid_action_penalty is not None:
            parser.error(
                "--invalid-action-penalty requires a penalize invalid-action policy."
            )
        return InvalidActionPolicy(
            mode=invalid_action_mode,
            penalty=None,
        )

    if args.invalid_action_penalty is None:
        parser.error(
            "--invalid-action-penalty is required for penalize invalid-action policies."
        )

    return InvalidActionPolicy(
        mode=invalid_action_mode,
        penalty=args.invalid_action_penalty,
    )


def load_task_spec_argument(
    *,
    args: Namespace,
    parser: ArgumentParser,
    expected_game: str,
) -> TaskSpec | None:
    """Load a task spec from CLI arguments when one was supplied.

    Parameters
    ----------
    args : Namespace
        Parsed CLI arguments.
    parser : ArgumentParser
        Parser used to raise user-facing argument errors.
    expected_game : str
        Game name implied by the CLI subcommand.

    Returns
    -------
    TaskSpec | None
        Parsed task spec or `None` when `--task-spec` was not supplied.
    """
    task_spec_path = args.task_spec
    if task_spec_path is None:
        return None
    try:
        task_spec = load_task_spec(path=task_spec_path)
    except Exception as exc:  # pragma: no cover - exercised through parser.error
        parser.error(f"--task-spec could not be loaded: {exc}")
    if task_spec.game != expected_game:
        parser.error(
            f"--task-spec game {task_spec.game!r} does not match CLI game "
            f"{expected_game!r}."
        )
    return task_spec


def build_environment_from_task_spec_argument(
    *,
    args: Namespace,
    parser: ArgumentParser,
    expected_game: str,
    disallowed_argument_names: tuple[str, ...],
) -> Environment[Any, Any] | None:
    """Load and build one task-spec environment from CLI arguments.

    Parameters
    ----------
    args : Namespace
        Parsed CLI arguments.
    parser : ArgumentParser
        Parser used to raise user-facing argument errors.
    expected_game : str
        Game name implied by the CLI subcommand.
    disallowed_argument_names : tuple[str, ...]
        CLI argument names that cannot be supplied when ``--task-spec`` is
        supplied.

    Returns
    -------
    Environment[Any, Any] | None
        Built environment or `None` when ``--task-spec`` was not supplied.
    """
    task_spec = load_task_spec_argument(
        args=args,
        parser=parser,
        expected_game=expected_game,
    )
    if task_spec is None:
        return None
    reject_task_spec_argument_overrides(
        args=args,
        parser=parser,
        disallowed_argument_names=disallowed_argument_names,
    )
    validate_task_spec_session_arguments(
        args=args,
        parser=parser,
        task_spec=task_spec,
    )
    try:
        return build_environment_from_task_spec(task_spec=task_spec)
    except Exception as exc:  # pragma: no cover - exercised through parser.error
        parser.error(f"--task-spec could not be built: {exc}")


def reject_task_spec_argument_overrides(
    *,
    args: Namespace,
    parser: ArgumentParser,
    disallowed_argument_names: tuple[str, ...],
) -> None:
    """Reject CLI env overrides that conflict with ``--task-spec``.

    Parameters
    ----------
    args : Namespace
        Parsed CLI arguments.
    parser : ArgumentParser
        Parser used to raise user-facing argument errors.
    disallowed_argument_names : tuple[str, ...]
        CLI argument names that cannot be supplied.
    """
    supplied_argument_names = frozenset(getattr(args, "_supplied_argument_names", ()))
    overridden_flags: list[str] = []
    for argument_name in disallowed_argument_names:
        if argument_name not in supplied_argument_names:
            continue
        option_strings = _argument_option_strings_from_parser(
            parser=parser,
            args=args,
            argument_name=argument_name,
        )
        overridden_flags.append(
            option_strings[0]
            if option_strings
            else f"--{argument_name.replace('_', '-')}"
        )
    if overridden_flags:
        joined_flags = ", ".join(overridden_flags)
        parser.error(f"{joined_flags} cannot be used together with --task-spec.")


def validate_task_spec_session_arguments(
    *,
    args: Namespace,
    parser: ArgumentParser,
    task_spec: TaskSpec,
) -> None:
    """Reject session flags that are incompatible with one task spec.

    Parameters
    ----------
    args : Namespace
        Parsed CLI arguments.
    parser : ArgumentParser
        Parser used to raise user-facing argument errors.
    task_spec : TaskSpec
        Parsed task specification referenced by the CLI invocation.
    """
    if args.image_output_dir is None:
        return
    if _task_spec_includes_images(task_spec=task_spec):
        return
    parser.error(
        "--image-output-dir requires task-spec observation.include_images: true."
    )


def _argument_option_strings_from_parser(
    *,
    parser: ArgumentParser,
    args: Namespace,
    argument_name: str,
) -> tuple[str, ...]:
    """Return the CLI option strings for one parsed argument.

    Parameters
    ----------
    parser : ArgumentParser
        Root parser used to parse the current arguments.
    args : Namespace
        Parsed CLI arguments.
    argument_name : str
        Argument destination whose option strings should be resolved.

    Returns
    -------
    tuple[str, ...]
        CLI option strings for the active parser path, or an empty tuple when
        the argument is not defined there.
    """
    for action in parser._actions:
        if action.dest == argument_name:
            return tuple(action.option_strings)
        nested_parser = _selected_subparser(action=action, args=args)
        if nested_parser is not None:
            nested_option_strings = _argument_option_strings_from_parser(
                parser=nested_parser,
                args=args,
                argument_name=argument_name,
            )
            if nested_option_strings:
                return nested_option_strings
    return ()


def _selected_subparser(
    *,
    action: Action,
    args: object,
) -> ArgumentParser | None:
    """Return the selected nested parser for one subparser action.

    Parameters
    ----------
    action : Action
        Candidate parser action to inspect.
    args : object
        Parsed CLI arguments.

    Returns
    -------
    ArgumentParser | None
        Selected nested parser or ``None`` when the action is not an active
        subparser branch.
    """
    choices = getattr(action, "choices", None)
    selected_choice = getattr(args, action.dest, None)
    if not isinstance(choices, dict):
        return None
    nested_parser = choices.get(selected_choice)
    if isinstance(nested_parser, ArgumentParser):
        return nested_parser
    return None


def _task_spec_includes_images(
    *,
    task_spec: TaskSpec,
) -> bool:
    """Return whether one task spec requests image observations.

    Parameters
    ----------
    task_spec : TaskSpec
        Parsed task specification to inspect.

    Returns
    -------
    bool
        `True` when the task spec observation config enables image rendering.
    """
    observation_spec = getattr(task_spec, "observation", None)
    return bool(getattr(observation_spec, "include_images", False))


def _collect_supplied_argument_names(
    *,
    parser: ArgumentParser,
    args: object,
    raw_argv: tuple[str, ...],
) -> tuple[str, ...]:
    """Return the argument destinations explicitly supplied on the CLI.

    Parameters
    ----------
    parser : ArgumentParser
        Parser that processed the CLI invocation.
    args : object
        Parsed CLI arguments.
    raw_argv : tuple[str, ...]
        Raw CLI tokens supplied to the parser.

    Returns
    -------
    tuple[str, ...]
        Parsed argument destination names explicitly present in the raw CLI
        token stream.
    """
    supplied_argument_names: list[str] = []
    for action in _iter_active_actions(parser=parser, args=args):
        if not action.option_strings:
            continue
        if _raw_argv_contains_option(
            raw_argv=raw_argv,
            option_strings=tuple(action.option_strings),
        ):
            supplied_argument_names.append(action.dest)
    return tuple(supplied_argument_names)


def _iter_active_actions(
    *,
    parser: ArgumentParser,
    args: object,
) -> tuple[Action, ...]:
    """Return parser actions along the active subparser path.

    Parameters
    ----------
    parser : ArgumentParser
        Parser whose actions should be traversed.
    args : object
        Parsed CLI arguments used to select active nested subparsers.

    Returns
    -------
    tuple[Action, ...]
        Actions attached to the active parser path.
    """
    actions: list[Action] = []
    for action in parser._actions:
        actions.append(action)
        nested_parser = _selected_subparser(action=action, args=args)
        if nested_parser is not None:
            actions.extend(_iter_active_actions(parser=nested_parser, args=args))
    return tuple(actions)


def _raw_argv_contains_option(
    *,
    raw_argv: tuple[str, ...],
    option_strings: tuple[str, ...],
) -> bool:
    """Return whether the raw argument vector supplied one option.

    Parameters
    ----------
    raw_argv : tuple[str, ...]
        Raw CLI tokens supplied to the parser.
    option_strings : tuple[str, ...]
        Accepted option spellings for one action.

    Returns
    -------
    bool
        `True` when any option spelling was explicitly present in the raw CLI
        token stream.
    """
    for raw_token in raw_argv:
        for option_string in option_strings:
            if raw_token == option_string or raw_token.startswith(f"{option_string}="):
                return True
    return False
