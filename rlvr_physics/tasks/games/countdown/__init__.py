"""Reasoning Gym Countdown task package."""

from rlvr_physics.tasks.games.countdown.constants import (
    COUNTDOWN_DOMAIN,
    COUNTDOWN_KIND,
)
from rlvr_physics.tasks.games.countdown.instances import make_countdown_instance
from rlvr_physics.tasks.games.countdown.renderers import (
    render_countdown_image,
    render_countdown_text,
)
from rlvr_physics.tasks.games.countdown.session import CountdownSession
from rlvr_physics.tasks.games.countdown.spec import countdown_task_spec
from rlvr_physics.tasks.games.countdown.types import CountdownVerification
from rlvr_physics.tasks.games.countdown.verifier import verify_countdown_submission

__all__ = [
    "COUNTDOWN_DOMAIN",
    "COUNTDOWN_KIND",
    "CountdownSession",
    "CountdownVerification",
    "countdown_task_spec",
    "make_countdown_instance",
    "render_countdown_image",
    "render_countdown_text",
    "verify_countdown_submission",
]
