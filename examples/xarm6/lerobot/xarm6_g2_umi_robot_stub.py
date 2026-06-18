"""
Stub LeRobot Robot config for xArm6 + Gripper G2 + UMI fisheye (Path A).

Install as separate package or copy into your teleop stack.
Implements the feature schema from xarm6_lerobot_features (7D EE + wrist RGB).
ToF is NOT exposed to the policy — only used offline for pose when Vive is absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# When integrated with huggingface/lerobot:
# from lerobot.cameras import CameraConfig
# from lerobot.robots.config import RobotConfig


@dataclass
class XArm6G2UmiRobotConfig:
    """Placeholder matching local dataset schema."""

    ip: str = "192.168.1.184"
    # Camera: FastUMI Pro cropped fisheye 1280x1280 @ 60fps
    camera_width: int = 1280
    camera_height: int = 1280
    camera_fps: int = 60
    # Mount: fixed on G2, optical axis down; bottom 20% shows fingers
    use_vive: bool = True
    use_tof_for_pose: bool = False  # only when use_vive=False
    ee_feature_dim: int = 7
    feature_keys: tuple[str, ...] = (
        "observation.state",
        "action",
        "observation.images.wrist",
    )

    def observation_features(self) -> dict[str, tuple]:
        return {
            "observation.state": (self.ee_feature_dim,),
            "observation.images.wrist": (self.camera_height, self.camera_width, 3),
        }

    def action_features(self) -> dict[str, tuple]:
        return {"action": (self.ee_feature_dim,)}

    def assert_no_tof_in_policy_io(self) -> None:
        forbidden = ("tof", "depth", "pointcloud", "pcd")
        for key in (*self.feature_keys,):
            if any(f in key.lower() for f in forbidden):
                raise ValueError(f"ToF/depth must not be policy I/O: {key}")

    def pose_source(self) -> str:
        """Label-only pose source for dataset recording (not ACT inputs)."""
        if self.use_vive:
            return "vive"
        if self.use_tof_for_pose:
            return "tof_slam"
        return "unset"


@dataclass
class XArm6G2UmiRobot:
    """Minimal real-robot I/O skeleton for Path A integration."""

    config: XArm6G2UmiRobotConfig = field(default_factory=XArm6G2UmiRobotConfig)

    def connect(self) -> None:
        self.config.assert_no_tof_in_policy_io()
        raise NotImplementedError(
            "Wire xArm SDK + UMI fisheye capture here. See convert_fastumi_to_lerobot.py for Path B."
        )

    def get_observation(self) -> dict:
        raise NotImplementedError("Return observation.state + observation.images.wrist (7D + RGB).")

    def send_action(self, action) -> None:
        raise NotImplementedError("Map 7D EE action to xArm IK + G2 drive_joint.")


# Path A (xArm6+G2+UMI module): implement XArm6G2UmiRobot, then:
# Path B (FastUMI Pro export): python convert_fastumi_to_lerobot.py --session-root /path/to/export
