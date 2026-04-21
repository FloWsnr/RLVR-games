"""Constants for physics discovery."""

PHYSICS_DISCOVERY_KIND = "physics.discovery.v1"
PHYSICS_DOMAIN = "physics"
PHYSICS_DISCOVERY_PRIOR_MODES = (
    "default",
    "no_context",
    "no_description",
    "no_description_anonymous",
)
PHYSICS_DISCOVERY_SOURCE = "physgym.full"
PHYSICS_DISCOVERY_RECORDS_FILE = "physgym_records.json"
DEFAULT_RANGE = (0.5, 5.0)
ALLOWED_CONSTANT_NAMES = "pi, e, c, mu_0"
ALLOWED_FUNCTION_NAMES = "sqrt, sin, cos, tan, exp, log, abs, arccos, arccosh, arctan"
