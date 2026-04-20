"""Constants for physics discovery."""

PHYSICS_DISCOVERY_KIND = "physics.discovery.v1"
PHYSICS_DOMAIN = "physics"
PHYSICS_DISCOVERY_PRIOR_MODES = (
    "default",
    "no_context",
    "no_description",
    "no_description_anonymous",
)
PHYSICS_DISCOVERY_SOURCE = "physgym.curated_subset"
PHYSICS_DISCOVERY_RECORDS_FILE = "physgym_curated_records.json"
DEFAULT_RANGE = (0.5, 5.0)
ALLOWED_FUNCTION_NAMES = "sqrt, sin, cos, tan, exp, log, abs"
