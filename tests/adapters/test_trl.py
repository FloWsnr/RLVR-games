"""Tests for TRL adapter helpers."""

from tempfile import TemporaryDirectory

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.chat_template_utils import qwen3_schema
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import GRPOTrainer

from rlvr_physics.adapters.datasets import (
    make_prompt_row,
    make_task_instance_registry,
)
from rlvr_physics.adapters.trl import (
    TrlDenseRewardFunction,
    TrlRewardFunction,
    TrlTaskEnvironment,
    make_trl_dataset,
    make_trl_environment_factory,
    make_trl_multiturn_dataset,
    to_trl_row,
)
from rlvr_physics.core.factory import ConfiguredTaskFactory
from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.core.session import TaskSession
from rlvr_physics.tasks.games.countdown import (
    CountdownSession,
    countdown_task_spec,
    make_countdown_instance,
)
from rlvr_physics.tasks.games.game2048 import (
    Game2048Session,
    game2048_task_spec,
    make_2048_instance,
)


def _countdown_text_session(instance: TaskInstance) -> TaskSession:
    return CountdownSession(instance, "text")


def _game2048_text_session(instance: TaskInstance) -> TaskSession:
    return Game2048Session(instance, "text")


def _countdown_factory() -> ConfiguredTaskFactory:
    return ConfiguredTaskFactory(
        spec=countdown_task_spec(seed=17, size=1),
        session_builder=_countdown_text_session,
    )


def _game2048_factory() -> ConfiguredTaskFactory:
    return ConfiguredTaskFactory(
        spec=game2048_task_spec(seed=5, max_turns=2, target_tile=2048),
        session_builder=_game2048_text_session,
    )


def test_trl_dataset_row_and_reward_function() -> None:
    instance = make_countdown_instance(seed=17, source_index=0)
    factory = _countdown_factory()
    row = make_prompt_row(
        instance=instance,
        task_factory=factory,
        seed=3,
        extra_info={"split": "train", "ability": "countdown"},
    )

    trl_row = to_trl_row(row)
    dataset = make_trl_dataset((row,))
    completion = [{"role": "assistant", "content": "answer: 1 + 2"}]
    reward_function = TrlRewardFunction(
        instances=make_task_instance_registry((instance,)),
        task_factory=factory,
        seed=4,
    )
    rewards = reward_function(
        prompts=[row.prompt],
        completions=[completion],
        task_id=[row.task_id],
    )

    assert dataset.num_rows == 1
    assert isinstance(dataset, Dataset)
    assert trl_row["prompt"] == row.prompt
    assert trl_row["task_id"] == instance.task_id
    assert rewards == [0.05]


def test_trl_grpo_trainer_runs_one_step_with_rlvr_reward() -> None:
    instance = make_countdown_instance(seed=17, source_index=0)
    factory = _countdown_factory()
    row = make_prompt_row(
        instance=instance,
        task_factory=factory,
        seed=3,
        extra_info={"split": "train", "ability": "countdown"},
    )
    dataset = make_trl_dataset((row,))
    reward_function = TrlRewardFunction(
        instances=make_task_instance_registry((instance,)),
        task_factory=factory,
        seed=4,
    )
    model_name = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="float32")

    with TemporaryDirectory() as output_dir:
        args = GRPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            num_generations=2,
            max_completion_length=4,
            max_steps=1,
            report_to="none",
            logging_strategy="no",
            save_strategy="no",
        )
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_function,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        result = trainer.train()

    assert trainer.state.global_step == 1
    assert result.training_loss is not None
    assert any(
        "rewards/rlvr_physics_reward/mean" in entry
        for entry in trainer.state.log_history
    )


def test_trl_environment_factory_tracks_step_rewards() -> None:
    instance = make_2048_instance(seed=5, max_turns=2, target_tile=2048)
    factory = _game2048_factory()
    row = make_prompt_row(
        instance=instance,
        task_factory=factory,
        seed=3,
        extra_info={"split": "train", "ability": "2048"},
    )
    dataset = make_trl_multiturn_dataset((row,))
    reward_function = TrlDenseRewardFunction()
    model_name = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    setattr(tokenizer, "response_schema", qwen3_schema)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype="float32")

    with TemporaryDirectory() as output_dir:
        args = GRPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            num_generations=2,
            max_completion_length=4,
            max_steps=1,
            report_to="none",
            logging_strategy="no",
            save_strategy="no",
            steps_per_generation=1,
        )
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_function,
            args=args,
            train_dataset=dataset,
            processing_class=tokenizer,
            environment_factory=make_trl_environment_factory(
                make_task_instance_registry((instance,)),
                factory,
                seed=4,
            ),
        )
        result = trainer.train()

    environments = trainer.environments
    assert environments is not None
    assert trainer.state.global_step == 1
    assert result.training_loss is not None
    environment = environments[0]
    assert isinstance(environment, TrlTaskEnvironment)
    observation = environment.reset(**dataset[0])
    first_feedback = environment.submit_action("up")
    second_feedback = environment.submit_action("right")
    rewards = reward_function(
        prompts=[dataset[0]["prompt"]],
        completions=[""],
        environments=[environment],
    )

    assert observation.startswith("\n\n2048")
    assert "reward: 4.0" in first_feedback
    assert "reward: 0.0" in second_feedback
    assert environment.step_rewards == (4.0, 0.0)
    assert environment.done
    assert rewards == [4.0]
    assert "submit_action" in {tool.__name__ for tool in trainer.tools}
