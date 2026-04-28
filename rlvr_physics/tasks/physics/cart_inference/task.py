"""Configured task builder for the cart inference task."""

from rlvr_physics.core.factory import ConfiguredTask
from rlvr_physics.core.instances import TaskInstance
from rlvr_physics.tasks.physics.cart_inference.instances import (
    build_cart_inference_instance,
)
from rlvr_physics.tasks.physics.cart_inference.renderers import (
    CART_TEXT_RENDERER,
    validate_cart_renderer_type,
)
from rlvr_physics.tasks.physics.cart_inference.sessions import CartInferenceSession
from rlvr_physics.tasks.physics.cart_inference.specs import (
    CartInferenceConfig,
    cart_inference_spec,
)


def cart_inference_task(
    config: CartInferenceConfig, renderer_type: str = CART_TEXT_RENDERER
) -> ConfiguredTask:
    """Build a configured cart inference task.

    Parameters
    ----------
    config:
        Public generation, rollout, and verifier configuration.
    renderer_type:
        Renderer identifier captured by sessions created from this task.

    Returns
    -------
    ConfiguredTask
        Configured task that builds cart inference instances and sessions from
        the same configuration.
    """

    validate_cart_renderer_type(renderer_type)

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

    def build_session(instance: TaskInstance) -> CartInferenceSession:
        """Build a cart inference session with the configured renderer.

        Parameters
        ----------
        instance:
            Immutable cart inference task instance.

        Returns
        -------
        CartInferenceSession
            Fresh cart inference session.
        """

        return CartInferenceSession(instance, renderer_type)

    return ConfiguredTask(
        spec=cart_inference_spec(config),
        instance_builder=build_instance,
        session_builder=build_session,
    )
