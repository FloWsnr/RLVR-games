"""Shared specification types for interactive play CLIs."""

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Any, Callable, Protocol, TextIO

from rlvr_games.core.protocol import Environment
from rlvr_games.core.rollout import ActionContext
from rlvr_games.core.types import Observation, StepResult

ParserRegistrar = Callable[[ArgumentParser], None]
EnvironmentBuilder = Callable[[Namespace, ArgumentParser], Environment[Any, Any]]
StepResultFormatter = Callable[[StepResult], tuple[str, ...]]


class InteractiveCommandHandler(Protocol):
    """Protocol for a game-specific interactive CLI command."""

    def __call__(
        self,
        *,
        env: Environment[Any, Any],
        observation: Observation,
        context: ActionContext,
        arguments: tuple[str, ...],
        output_stream: TextIO,
    ) -> None:
        """Handle one interactive CLI command.

        Parameters
        ----------
        env : Environment[Any, Any]
            Active environment being driven by the CLI session.
        observation : Observation
            Most recent observation emitted by the environment.
        context : ActionContext
            Turn context computed for the current state.
        arguments : tuple[str, ...]
            Space-delimited command arguments entered after the command name.
        output_stream : TextIO
            Stream receiving command output.
        """


@dataclass(frozen=True, slots=True)
class InteractiveCommandSpec:
    """Game-specific interactive CLI command registration.

    Attributes
    ----------
    name : str
        First token that invokes the command.
    usage : str
        Short usage text shown in the interactive help output.
    handler : InteractiveCommandHandler
        Callable that executes the command for the current session.
    """

    name: str
    usage: str
    handler: InteractiveCommandHandler


@dataclass(frozen=True, slots=True)
class GameCliSpec:
    """Per-game CLI registration for parser and session behavior.

    Attributes
    ----------
    name : str
        Command name used under ``rlvr-games play <game>``.
    register_arguments : ParserRegistrar
        Callable that attaches game-specific CLI arguments to a subparser.
    build_environment : EnvironmentBuilder
        Callable that constructs the environment for parsed CLI arguments.
    format_step_result : StepResultFormatter
        Callable that renders any game-specific step summary lines.
    interactive_commands : tuple[InteractiveCommandSpec, ...]
        Extra interactive commands supported by the game in addition to the
        built-in generic session commands.
    """

    name: str
    register_arguments: ParserRegistrar
    build_environment: EnvironmentBuilder
    format_step_result: StepResultFormatter
    interactive_commands: tuple[InteractiveCommandSpec, ...]
