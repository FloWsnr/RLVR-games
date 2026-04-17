"""Mastermind-specific CLI registration."""

from argparse import ArgumentParser, ArgumentTypeError, Namespace
from enum import StrEnum
from typing import Any

from rlvr_games.cli.common import (
    COMMON_TASK_SPEC_DISALLOWED_ARGUMENT_NAMES,
    build_environment_from_task_spec_argument,
    build_episode_config,
)
from rlvr_games.cli.specs import GameCliSpec
from rlvr_games.core.protocol import Environment, RewardFn
from rlvr_games.core.types import StepResult
from rlvr_games.games.mastermind.actions import MastermindAction
from rlvr_games.games.mastermind.engine import (
    STANDARD_MASTERMIND_MIN_IMAGE_SIZE,
    MastermindCode,
)
from rlvr_games.games.mastermind.factory import make_mastermind_env
from rlvr_games.games.mastermind.rewards import (
    CandidateReductionDenseReward,
    TerminalOutcomeReward,
)
from rlvr_games.games.mastermind.scenarios import (
    FixedCodeScenario,
    StandardGameScenario,
    normalize_initial_code,
)
from rlvr_games.games.mastermind.state import MastermindState


class MastermindRewardKind(StrEnum):
    """Supported Mastermind reward policies exposed through the CLI."""

    OUTCOME = "outcome"
    CANDIDATE_REDUCTION_DENSE = "candidate-reduction-dense"


def register_mastermind_arguments(parser: ArgumentParser) -> None:
    """Attach Mastermind-specific CLI arguments to a play subparser."""
    parser.add_argument(
        "--reward",
        choices=tuple(kind.value for kind in MastermindRewardKind),
        default=MastermindRewardKind.OUTCOME.value,
    )
    parser.add_argument("--code", type=parse_mastermind_code_argument)


def build_mastermind_environment(
    args: Namespace,
    parser: ArgumentParser,
) -> Environment[Any, Any]:
    """Construct a Mastermind environment from parsed CLI arguments."""
    task_spec_environment = build_environment_from_task_spec_argument(
        args=args,
        parser=parser,
        expected_game="mastermind",
        disallowed_argument_names=COMMON_TASK_SPEC_DISALLOWED_ARGUMENT_NAMES
        + (
            "reward",
            "code",
        ),
    )
    if task_spec_environment is not None:
        return task_spec_environment

    if (
        args.image_output_dir is not None
        and args.image_size < STANDARD_MASTERMIND_MIN_IMAGE_SIZE
    ):
        parser.error(
            "Mastermind image_size must be >= "
            f"{STANDARD_MASTERMIND_MIN_IMAGE_SIZE} when --image-output-dir is used."
        )

    if args.code is None:
        scenario = StandardGameScenario()
    else:
        scenario = FixedCodeScenario(secret_code=args.code)

    return make_mastermind_env(
        scenario=scenario,
        reward_fn=build_mastermind_reward(args=args),
        config=build_episode_config(args=args, parser=parser),
        include_images=args.image_output_dir is not None,
        image_size=args.image_size,
    )


def build_mastermind_reward(
    *,
    args: Namespace,
) -> RewardFn[MastermindState, MastermindAction]:
    """Construct a Mastermind reward function from parsed CLI arguments."""
    reward_kind = MastermindRewardKind(args.reward)
    if reward_kind == MastermindRewardKind.OUTCOME:
        return TerminalOutcomeReward(win_reward=1.0, loss_reward=-1.0)
    return CandidateReductionDenseReward()


def format_mastermind_step_result(step_result: StepResult) -> tuple[str, ...]:
    """Render Mastermind-specific transition summary lines."""
    summary_lines: list[str] = []
    guess_text = step_result.info.get("guess_text")
    black_pegs = step_result.info.get("black_pegs")
    white_pegs = step_result.info.get("white_pegs")
    candidate_count = step_result.info.get("candidate_count")
    guesses_remaining = step_result.info.get("guesses_remaining")
    if guess_text is not None:
        summary_lines.append(f"Guess: {guess_text}")
    if black_pegs is not None and white_pegs is not None:
        summary_lines.append(f"Feedback: {black_pegs} black, {white_pegs} white")
    if candidate_count is not None:
        summary_lines.append(f"Consistent candidates: {candidate_count}")
    if guesses_remaining is not None:
        summary_lines.append(f"Guesses remaining: {guesses_remaining}")
    return tuple(summary_lines)


def parse_mastermind_code_argument(raw_code: str) -> MastermindCode:
    """Parse a CLI Mastermind code argument into canonical form."""
    try:
        return normalize_initial_code(code=raw_code)
    except ValueError as exc:
        raise ArgumentTypeError(str(exc)) from exc


MASTERMIND_CLI_SPEC = GameCliSpec(
    name="mastermind",
    register_arguments=register_mastermind_arguments,
    build_environment=build_mastermind_environment,
    format_step_result=format_mastermind_step_result,
    interactive_commands=(),
)
