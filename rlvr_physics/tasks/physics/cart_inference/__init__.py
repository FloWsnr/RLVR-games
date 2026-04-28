"""One-dimensional cart inference task."""

from rlvr_physics.core.factory import ConfiguredTask
from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.core.rewards import RewardResult
from rlvr_physics.tasks.physics.cart_inference.backbone import (
    FINAL_ANSWER_ACTION,
    MEASURE_POSITION_ACTION,
    ActionBudgetExceeded,
    CartInferenceBackbone,
    FinalAnswerEvaluation,
)
from rlvr_physics.tasks.physics.cart_inference.instances import (
    build_cart_inference_instance,
)
from rlvr_physics.tasks.physics.cart_inference.sessions import CartInferenceSession
from rlvr_physics.tasks.physics.cart_inference.rewards import (
    reward_final_answer,
)
from rlvr_physics.tasks.physics.cart_inference.specs import (
    CART_INFERENCE_DOMAIN,
    CART_INFERENCE_KIND,
    DEFAULT_CONFIG,
    CartInferenceConfig,
    cart_inference_spec,
)


def cart_inference_task(config: CartInferenceConfig) -> ConfiguredTask:
    """Build a configured cart inference task.

    Parameters
    ----------
    config:
        Public generation, rollout, and verifier configuration.

    Returns
    -------
    ConfiguredTask
        Configured task that builds cart inference instances and sessions from
        the same configuration.
    """

    def build_instance(seed: int) -> TaskInstance:
        """Build a cart inference instance from the configured task."""

        return build_cart_inference_instance(seed, config)

    return ConfiguredTask(
        spec=cart_inference_spec(config),
        instance_builder=build_instance,
        session_builder=CartInferenceSession,
    )


__all__ = [
    "CART_INFERENCE_DOMAIN",
    "CART_INFERENCE_KIND",
    "DEFAULT_CONFIG",
    "FINAL_ANSWER_ACTION",
    "MEASURE_POSITION_ACTION",
    "ActionBudgetExceeded",
    "CartInferenceConfig",
    "CartInferenceBackbone",
    "CartInferenceSession",
    "FinalAnswerEvaluation",
    "RewardResult",
    "build_cart_inference_instance",
    "cart_inference_spec",
    "cart_inference_task",
    "reward_final_answer",
]
