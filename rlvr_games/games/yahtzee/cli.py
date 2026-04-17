"""Yahtzee-specific CLI registration."""

from argparse import ArgumentParser, Namespace
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
from rlvr_games.games.yahtzee.actions import YahtzeeAction
from rlvr_games.games.yahtzee.factory import make_yahtzee_env
from rlvr_games.games.yahtzee.rewards import FinalScoreReward, ScoreDeltaReward
from rlvr_games.games.yahtzee.state import YahtzeeState


class YahtzeeRewardKind(StrEnum):
    """Supported Yahtzee reward policies exposed through the CLI."""

    FINAL_SCORE = "final-score"
    SCORE_DELTA_DENSE = "score-delta-dense"


def register_yahtzee_arguments(parser: ArgumentParser) -> None:
    """Attach Yahtzee-specific CLI arguments to a play subparser.

    Parameters
    ----------
    parser : ArgumentParser
        Yahtzee play subparser to configure.
    """
    parser.add_argument(
        "--reward",
        choices=tuple(kind.value for kind in YahtzeeRewardKind),
        default=YahtzeeRewardKind.FINAL_SCORE.value,
    )


def build_yahtzee_environment(
    args: Namespace,
    parser: ArgumentParser,
) -> Environment[Any, Any]:
    """Construct a Yahtzee environment from parsed CLI arguments.

    Parameters
    ----------
    args : Namespace
        Parsed CLI arguments for a Yahtzee play session.
    parser : ArgumentParser
        Parser used to validate common episode configuration.

    Returns
    -------
    Environment[Any, Any]
        Fully configured Yahtzee environment.
    """
    task_spec_environment = build_environment_from_task_spec_argument(
        args=args,
        parser=parser,
        expected_game="yahtzee",
        disallowed_argument_names=COMMON_TASK_SPEC_DISALLOWED_ARGUMENT_NAMES
        + ("reward",),
    )
    if task_spec_environment is not None:
        return task_spec_environment

    return make_yahtzee_env(
        initial_state=None,
        reward_fn=build_yahtzee_reward(args=args),
        config=build_episode_config(args=args, parser=parser),
        include_images=args.image_output_dir is not None,
        image_size=args.image_size,
    )


def build_yahtzee_reward(
    *,
    args: Namespace,
) -> RewardFn[YahtzeeState, YahtzeeAction]:
    """Construct a Yahtzee reward function from parsed CLI arguments.

    Parameters
    ----------
    args : Namespace
        Parsed CLI arguments for a Yahtzee play session.

    Returns
    -------
    RewardFn[YahtzeeState, YahtzeeAction]
        Reward function implied by the parsed arguments.
    """
    reward_kind = YahtzeeRewardKind(args.reward)
    if reward_kind == YahtzeeRewardKind.FINAL_SCORE:
        return FinalScoreReward()
    return ScoreDeltaReward()


def format_yahtzee_step_result(step_result: StepResult) -> tuple[str, ...]:
    """Render Yahtzee-specific transition summary lines.

    Parameters
    ----------
    step_result : StepResult
        Step result whose transition metadata should be summarized.

    Returns
    -------
    tuple[str, ...]
        Human-readable summary lines derived from Yahtzee transition info.
    """
    summary_lines: list[str] = []
    action_kind = step_result.info.get("action_kind")
    internal_transitions = step_result.info.get("internal_transitions")
    if action_kind == "reroll":
        rerolled_positions = step_result.info.get("rerolled_positions")
        if isinstance(rerolled_positions, tuple):
            summary_lines.append(
                "Rerolled positions: "
                + " ".join(str(position) for position in rerolled_positions)
            )
        dice = step_result.info.get("dice")
        if isinstance(dice, tuple):
            summary_lines.append("Dice: " + " ".join(str(value) for value in dice))

    if action_kind == "score":
        scored_category = step_result.info.get("scored_category")
        score_value = step_result.info.get("score_value")
        if isinstance(scored_category, str) and isinstance(score_value, int):
            summary_lines.append(f"Scored: {scored_category} = {score_value}")
        if isinstance(internal_transitions, tuple):
            for transition in internal_transitions:
                if not isinstance(transition, dict):
                    continue
                transition_info = transition.get("info")
                if not isinstance(transition_info, dict):
                    continue
                if transition_info.get("event_kind") != "opening_roll":
                    continue
                dice = transition_info.get("dice")
                if isinstance(dice, tuple):
                    summary_lines.append(
                        "Next roll: " + " ".join(str(value) for value in dice)
                    )
                break

    total_score = step_result.info.get("total_score")
    if isinstance(total_score, int):
        summary_lines.append(f"Total score: {total_score}")
    return tuple(summary_lines)


YAHTZEE_CLI_SPEC = GameCliSpec(
    name="yahtzee",
    register_arguments=register_yahtzee_arguments,
    build_environment=build_yahtzee_environment,
    format_step_result=format_yahtzee_step_result,
    interactive_commands=(),
)
