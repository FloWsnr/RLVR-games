"""Generic rollout helpers for scalar task sessions."""

from collections.abc import Callable

from rlvr_games.core.session import TaskSessionProtocol, TaskTrajectory, TaskTurn

TaskSessionPolicy = Callable[[TaskTurn], str]


def rollout_task_session(
    *,
    session: TaskSessionProtocol,
    seed: int,
    policy: TaskSessionPolicy,
    max_submissions: int | None = None,
) -> TaskTrajectory:
    """Run one scalar task session until completion.

    Parameters
    ----------
    session : TaskSessionProtocol
        Scalar task session to reset and drive.
    seed : int
        Seed passed to ``session.reset(...)``.
    policy : TaskSessionPolicy
        Callable that maps each actionable task turn to one assistant output.
    max_submissions : int | None
        Optional guard against non-terminating sessions.

    Returns
    -------
    TaskTrajectory
        Recorded task-level trajectory after rollout completion.

    Raises
    ------
    RuntimeError
        If ``max_submissions`` is reached before the task session finishes.
    ValueError
        If ``max_submissions`` is not positive when provided.
    """
    if max_submissions is not None and max_submissions <= 0:
        raise ValueError("max_submissions must be positive when provided.")

    reset_result = session.reset(seed=seed)
    turn = reset_result.turn
    submission_count = 0
    while turn is not None:
        if max_submissions is not None and submission_count >= max_submissions:
            raise RuntimeError("Task session did not finish before max_submissions.")
        assistant_output = policy(turn)
        submission_result = session.submit(assistant_output)
        submission_count += 1
        turn = submission_result.turn
    return session.trajectory


__all__ = ["TaskSessionPolicy", "rollout_task_session"]
