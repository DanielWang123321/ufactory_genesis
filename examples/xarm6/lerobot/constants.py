"""Shared constants for xArm6 + G2 + UMI fisheye LeRobot pipeline."""

from __future__ import annotations

# Simulation / dataset timing (FastUMI Pro cropped RGB stream)
CTRL_DT = 1.0 / 60.0
DATASET_FPS = 60

# Fisheye (https://lumosumi.lumosbot.tech/pro/)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 1280
CAMERA_FOV_DEG = 172.52
CAMERA_MODEL = "fisheye"
CAMERA_HEIGHT_M = 0.085  # along link6 +Z (calibrated open/closed)
CAMERA_LATERAL_Y_M = -0.10  # forward on gripper; both fingertips land in bottom 20%
CAMERA_FEATURE_KEY = "observation.images.wrist"

# Image framing: bottom 20% must show G2 fingers (y=0 is top)
FINGER_BAND_Y_START = int(CAMERA_HEIGHT * 0.8)  # 1024
FINGER_BAND_Y_END = CAMERA_HEIGHT

# Gripper G2 drive_joint limits (URDF)
DRIVE_OPEN = 0.0
DRIVE_CLOSED = 0.85
GRIPPER_MAX_WIDTH_MM = 84.0

# Default scene
TABLE_HEIGHT = 0.4
OBJ_SIZE = (0.04, 0.04, 0.04)
OBJ_XY_DEFAULT = (0.30, 0.00)
PLACE_XY_DEFAULT = (0.30, 0.30)
LIFT_Z = 0.30
HOME_Z = 0.30

# EE state/action vector layout
STATE_DIM = 7
POS_SLICE = slice(0, 3)
ROTVEC_SLICE = slice(3, 6)
GRIPPER_IDX = 6

# LeRobot dataset
ROBOT_TYPE = "xarm6_g2_umi"
DEFAULT_TASK = "pick cube and place at target"

# Training modalities whitelist (ToF / depth excluded by design)
ALLOWED_FEATURE_KEYS = frozenset({
    "observation.state",
    "action",
    CAMERA_FEATURE_KEY,
})
