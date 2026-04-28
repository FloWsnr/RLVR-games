"""Configured task builder for the cart inference task."""

from rlvr_physics.core.factory import ConfiguredTask
from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.tasks.physics.cart_inference.instances import (
    build_cart_inference_instance,
)
from rlvr_physics.tasks.physics.cart_inference.sessions import CartInferenceSession
from rlvr_physics.tasks.physics.cart_inference.specs import (
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
        """Build a cart inference instance from the configured task.

        Parameters
        ----------
        seed:
            Deterministic seed passed to the instance builder.

        Returns
        -------
        TaskInstance
            Immutable cart inference task instance.
        """

        return build_cart_inference_instance(seed, config)

    return ConfiguredTask(
        spec=cart_inference_spec(config),
        instance_builder=build_instance,
        session_builder=CartInferenceSession,
    )
