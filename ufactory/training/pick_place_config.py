"""Dependency-free contract for the end-to-end pick-place policy."""

PICK_PLACE_ENV_VERSION = 3
PICK_PLACE_NUM_OBS = 70
PICK_PLACE_NUM_ACTIONS = 4
PICK_PLACE_CONTROL = "cartesian_and_g2_gap_increment"
PICK_PLACE_ACTION_CONTRACT = "cartesian_and_g2_gap_delta_v3"
