"""Shared parser helpers for the RLVR CLI."""

from argparse import ArgumentParser, Namespace
from pathlib import Path

from rlvr_games.datasets import SplitPercentages
from rlvr_games.core.types import (
    EpisodeConfig,
    InvalidActionMode,
    InvalidActionPolicy,
)


def add_common_play_arguments(parser: ArgumentParser) -> None:
    """Attach play-session arguments shared by supported games.

    Parameters
    ----------
    parser : ArgumentParser
        Parser to extend with common play-session arguments.
    """
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int)
    parser.add_argument("--max-transitions", type=int)
    parser.add_argument("--image-output-dir", type=Path)
    parser.add_argument("--image-size", type=int, default=360)
    parser.add_argument(
        "--invalid-action-policy",
        choices=tuple(mode.value for mode in InvalidActionMode),
        default=InvalidActionMode.RAISE.value,
    )
    parser.add_argument("--invalid-action-penalty", type=float)


def add_common_dataset_download_arguments(parser: ArgumentParser) -> None:
    """Attach dataset-download arguments shared by supported corpora.

    Parameters
    ----------
    parser : ArgumentParser
        Parser to extend with common raw-download arguments.
    """
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--overwrite", action="store_true")


def add_common_dataset_preprocess_arguments(parser: ArgumentParser) -> None:
    """Attach dataset-preprocess arguments shared by supported corpora.

    Parameters
    ----------
    parser : ArgumentParser
        Parser to extend with common preprocessing arguments.
    """
    parser.add_argument("--source-file", type=Path)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--train-percentage", type=int, default=98)
    parser.add_argument("--val-percentage", type=int, default=1)
    parser.add_argument("--test-percentage", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")


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


def build_split_percentages(
    *,
    args: Namespace,
    parser: ArgumentParser,
) -> SplitPercentages:
    """Build validated split percentages from parsed CLI arguments.

    Parameters
    ----------
    args : Namespace
        Parsed CLI arguments.
    parser : ArgumentParser
        Parser used to raise argument errors when the configuration is
        inconsistent.

    Returns
    -------
    SplitPercentages
        Validated deterministic split percentages.
    """
    try:
        return SplitPercentages(
            train=args.train_percentage,
            val=args.val_percentage,
            test=args.test_percentage,
        )
    except ValueError as exc:
        parser.error(str(exc))
