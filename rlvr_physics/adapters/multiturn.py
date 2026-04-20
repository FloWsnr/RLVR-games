"""Generic scalar-session rollout helpers for adapter code."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from rlvr_physics.adapters.datasets import task_id_from_mapping
from rlvr_physics.core.factory import TaskFactory
from rlvr_physics.core.instances import TaskInstance, freeze_mapping, mapping_to_dict
from rlvr_physics.core.session import TaskSession, TaskStepResult, TaskSubmission


@dataclass(frozen=True)
class SessionStepRecord:
    """Trainer-safe record for one scalar session step.

    Parameters
    ----------
    submission_kind:
        Kind of submitted model output or action.
    raw_submission:
        Raw submitted text or action.
    accepted:
        Whether the submission was well formed and applied.
    reward:
        Scalar reward emitted by this step.
    score:
        Optional task score after this step.
    done:
        Whether the rollout ended after this step.
    terminal:
        Whether the task naturally ended after this step.
    truncated:
        Whether limits ended the rollout after this step.
    observation:
        Next model-facing observation text, when the task continues.
    public_info:
        Trainer-safe step metadata.
    debug_info:
        Local debug metadata emitted by the task.
    """

    submission_kind: str
    raw_submission: str
    accepted: bool
    reward: float
    score: float | None
    done: bool
    terminal: bool
    truncated: bool
    observation: str | None
    public_info: Mapping[str, object] = field(default_factory=dict)
    debug_info: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze nested metadata after construction."""

        object.__setattr__(self, "public_info", freeze_mapping(self.public_info))
        object.__setattr__(self, "debug_info", freeze_mapping(self.debug_info))

    def as_dict(self) -> dict[str, object]:
        """Return this step as plain trainer-safe containers."""

        return {
            "submission_kind": self.submission_kind,
            "raw_submission": self.raw_submission,
            "accepted": self.accepted,
            "reward": self.reward,
            "score": self.score,
            "done": self.done,
            "terminal": self.terminal,
            "truncated": self.truncated,
            "observation": self.observation,
            "public_info": mapping_to_dict(self.public_info),
            "debug_info": mapping_to_dict(self.debug_info),
        }


class ScalarSessionEnvironment:
    """Mutable adapter-side environment around one scalar task session.

    Parameters
    ----------
    instances:
        Immutable task instances keyed by task id.
    task_factory:
        Factory that creates scalar sessions for the configured task family.
    seed:
        Deterministic session seed used for resets.
    """

    def __init__(
        self,
        instances: Mapping[str, TaskInstance],
        task_factory: TaskFactory,
        seed: int,
    ) -> None:
        self._instances = dict(instances)
        self._task_factory = task_factory
        self._seed = seed
        self._task_id: str | None = None
        self._session: TaskSession | None = None
        self._initial_observation: str | None = None
        self._steps: list[SessionStepRecord] = []

    def reset(self, task_id: str) -> str:
        """Reset this environment for one immutable task instance.

        Parameters
        ----------
        task_id:
            Stable task instance identifier.
        """

        instance = self._instances[task_id]
        self._task_id = task_id
        self._session = self._task_factory.create_session(instance)
        reset = self._session.reset(seed=self._seed)
        self._initial_observation = reset.turn.observation.text()
        self._steps = []
        return self._initial_observation

    def reset_from_mapping(self, fields: Mapping[str, object]) -> str:
        """Reset using a row mapping containing a task id.

        Parameters
        ----------
        fields:
            Row fields containing a task id, label, ground truth, reward model,
            or nested ``extra_info`` task id.
        """

        return self.reset(task_id_from_mapping(fields))

    def submit(self, submission: TaskSubmission) -> SessionStepRecord:
        """Submit one model output or action and record the step result.

        Parameters
        ----------
        submission:
            Raw and optionally parsed model submission.
        """

        if self._session is None:
            raise ValueError("environment must be reset before submitting")
        result = self._session.submit(submission)
        record = _step_record_from_result(submission, result)
        self._steps.append(record)
        return record

    def submit_action(self, action: str) -> SessionStepRecord:
        """Submit one action string and record the step result.

        Parameters
        ----------
        action:
            Raw task action.
        """

        return self.submit(TaskSubmission.action(action))

    def submit_action_text(self, action: str) -> str:
        """Submit one action and return trainer-facing feedback text.

        Parameters
        ----------
        action:
            Raw task action.
        """

        return format_step_feedback(self.submit_action(action))

    @property
    def task_id(self) -> str | None:
        """Return the current task id, if this environment was reset."""

        return self._task_id

    @property
    def initial_observation(self) -> str | None:
        """Return the initial observation text, if this environment was reset."""

        return self._initial_observation

    @property
    def steps(self) -> tuple[SessionStepRecord, ...]:
        """Return recorded step results."""

        return tuple(self._steps)

    @property
    def step_rewards(self) -> tuple[float, ...]:
        """Return one scalar reward per submitted step."""

        return tuple(step.reward for step in self._steps)

    @property
    def total_reward(self) -> float:
        """Return the sum of step rewards."""

        return sum(self.step_rewards)

    @property
    def final_score(self) -> float | None:
        """Return the latest non-null task score."""

        for step in reversed(self._steps):
            if step.score is not None:
                return step.score
        return None

    @property
    def done(self) -> bool:
        """Return whether the latest step ended the rollout."""

        return bool(self._steps and self._steps[-1].done)


def format_step_feedback(step: SessionStepRecord) -> str:
    """Return a text feedback message for a session step.

    Parameters
    ----------
    step:
        Step record to expose to a model or trainer.
    """

    parts = [
        f"accepted: {step.accepted}",
        f"reward: {step.reward}",
        f"score: {step.score}",
        f"done: {step.done}",
        f"reason: {step.public_info.get('reason', 'unknown')}",
    ]
    if step.observation:
        parts.append("observation:\n" + step.observation)
    return "\n".join(parts)


def run_action_rollout(
    instance: TaskInstance,
    task_factory: TaskFactory,
    seed: int,
    actions: Sequence[str],
) -> tuple[str, tuple[SessionStepRecord, ...]]:
    """Run a deterministic action rollout through a scalar task session.

    Parameters
    ----------
    instance:
        Immutable task instance to roll out.
    task_factory:
        Factory that creates scalar sessions for the configured task family.
    seed:
        Deterministic session seed used for reset.
    actions:
        Raw actions to submit in order.
    """

    environment = ScalarSessionEnvironment(
        instances={instance.task_id: instance},
        task_factory=task_factory,
        seed=seed,
    )
    initial_observation = environment.reset(instance.task_id)
    for action in actions:
        step = environment.submit_action(action)
        if step.done:
            break
    return initial_observation, environment.steps


def _step_record_from_result(
    submission: TaskSubmission, result: TaskStepResult
) -> SessionStepRecord:
    observation = None
    if result.observation is not None:
        observation = result.observation.observation.text()
    return SessionStepRecord(
        submission_kind=submission.kind,
        raw_submission=submission.raw,
        accepted=result.accepted,
        reward=result.reward,
        score=result.score,
        done=result.done,
        terminal=result.terminal,
        truncated=result.truncated,
        observation=observation,
        public_info=result.public_info,
        debug_info=result.debug_info,
    )
