"""Project path helpers for xArm URDF assets."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
XARM6_ASSETS = PROJECT_ROOT / "assets" / "urdf" / "xarm6"

XARM6_1305_VISUAL_GLB_URDF = "xarm6_1305_visual.glb.urdf"
XARM6_1305_G2_VISUAL_URDF = "xarm6_1305_g2_visual.urdf"


def xarm6_urdf(name: str = "xarm6.urdf") -> str:
    """Return absolute path to an xArm 6 URDF file in project assets."""
    path = XARM6_ASSETS / name
    if not path.exists():
        raise FileNotFoundError(path)
    return str(path.resolve())


def xarm6_1305_visual_glb_urdf(with_g2: bool = False) -> str:
    """Return URDF path for xArm6 1305 GLB visual model (optionally with G2 gripper)."""
    name = XARM6_1305_G2_VISUAL_URDF if with_g2 else XARM6_1305_VISUAL_GLB_URDF
    return xarm6_urdf(name)
