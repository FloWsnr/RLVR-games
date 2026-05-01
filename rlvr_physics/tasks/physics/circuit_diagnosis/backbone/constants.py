"""Constants for the circuit diagnosis backbone."""

from rlvr_physics.core.submissions import ACTION_ARGUMENTS_FIELD, ACTION_NAME_FIELD

SET_SOURCE_ACTION = "set_source"
MEASURE_VOLTAGE_ACTION = "measure_voltage"
MEASURE_CURRENT_ACTION = "measure_current"
REPLACE_COMPONENT_ACTION = "replace_component"
FINAL_ANSWER_ACTION = "final_answer"
ACTION_SUBMISSION_PARSE_ERROR = (
    "could not parse action submission; expected one JSON line with fields "
    f'"{ACTION_NAME_FIELD}" and "{ACTION_ARGUMENTS_FIELD}"'
)

GROUND_NODE = "GND"
MAX_SOURCE_ABS_VOLTAGE = 24.0
MIN_RESISTANCE_OHM = 1.0e-6
DIODE_STATE_TOLERANCE = 1.0e-7
CHECK_TOLERANCE = 1.0e-9
