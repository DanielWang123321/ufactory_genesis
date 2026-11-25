"""Robotiq 2F-85 and 2F-140 geometry, specs and conversions across metres, radians and SDK registers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET

ROBOTIQ_2F_85_PREFIX = "robotiq_2f_85"
ROBOTIQ_2F_140_PREFIX = "robotiq_2f_140"
ROBOTIQ_ADAPTERS = frozenset((ROBOTIQ_2F_85_PREFIX, ROBOTIQ_2F_140_PREFIX))

# Modes that touch a controller: sdk-sim may physically preposition the arm (--confirm-real) and real moves both.
ROBOTIQ_BLOCKED_MODES = frozenset(("sdk-sim", "real"))
ROBOTIQ_BLOCKED_MODE_REASON = (
    "currently requires --mode sim or dry-run: the controller TCP offset/load (including the 11 mm flange adapter) "
    "and the real Robotiq command path have not been validated on hardware"
)

ROBOTIQ_JOINT_SUFFIXES = (
    "drive_joint",
    "left_finger_joint",
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_finger_joint",
    "right_inner_knuckle_joint",
)


def robotiq_joint_names(prefix: str) -> tuple[str, ...]:
    """Return the 6 active + mimic joint names for a prefixed Robotiq gripper."""
    return tuple(f"{prefix}_{suffix}" for suffix in ROBOTIQ_JOINT_SUFFIXES)


ROBOTIQ_2F_85_JOINT_NAMES = robotiq_joint_names(ROBOTIQ_2F_85_PREFIX)
ROBOTIQ_2F_140_JOINT_NAMES = robotiq_joint_names(ROBOTIQ_2F_140_PREFIX)

ROBOTIQ_2F_85_CLOSED_GAP_MM = 0.0
ROBOTIQ_2F_85_OPEN_GAP_MM = 85.0
ROBOTIQ_2F_85_OPEN_GAP_M = ROBOTIQ_2F_85_OPEN_GAP_MM / 1000.0
ROBOTIQ_2F_85_SIM_OPEN_DRIVE = 0.0
ROBOTIQ_2F_85_SIM_CLOSE_DRIVE = 0.8014334346423151

ROBOTIQ_2F_140_CLOSED_GAP_MM = 0.0
ROBOTIQ_2F_140_OPEN_GAP_MM = 140.0
ROBOTIQ_2F_140_OPEN_GAP_M = ROBOTIQ_2F_140_OPEN_GAP_MM / 1000.0
ROBOTIQ_2F_140_SIM_OPEN_DRIVE = 0.0038593285902723817
ROBOTIQ_2F_140_SIM_CLOSE_DRIVE = 0.7665940156296729


@dataclass(frozen=True)
class RobotiqLinkageGeometry:
    """Four-bar parallel-pad linkage dimensions from a Robotiq URDF and pad mesh."""

    base_joint_y_m: float
    finger_joint_y_m: float
    finger_joint_z_m: float
    pad_inner_y_m: float
    pad_center_z_m: float
    drive_lower_rad: float
    drive_upper_rad: float

    @classmethod
    def from_urdf(cls, path: str | Path, *, prefix: str) -> RobotiqLinkageGeometry:
        """Read linkage dimensions and pad inner surface directly from a prefixed URDF."""
        import numpy as np
        import trimesh

        path = Path(path)
        root = ET.parse(path).getroot()
        drive = root.find(f"./joint[@name='{prefix}_drive_joint']")
        finger = root.find(f"./joint[@name='{prefix}_left_finger_joint']")
        collision = root.find(f"./link[@name='{prefix}_left_finger']/collision")
        if drive is None or finger is None or collision is None:
            raise ValueError(f"Robotiq URDF ({prefix}) is missing drive, finger joint or pad collision mesh")

        def vector(element: ET.Element, tag: str, attribute: str, default: str) -> tuple[float, ...]:
            child = element.find(tag)
            return tuple(map(float, (default if child is None else child.get(attribute, default)).split()))

        if (
            vector(drive, "axis", "xyz", "0 0 0") != (1.0, 0.0, 0.0)
            or vector(finger, "axis", "xyz", "0 0 0") != (-1.0, 0.0, 0.0)
            or vector(drive, "origin", "rpy", "0 0 0") != (0.0, 0.0, 0.0)
            or vector(finger, "origin", "rpy", "0 0 0") != (0.0, 0.0, 0.0)
            or vector(collision, "origin", "rpy", "0 0 0") != (0.0, 0.0, 0.0)
        ):
            raise ValueError(f"Robotiq calibration requires parallel-pad geometry for {prefix}")
        mimic = finger.find("mimic")
        if (
            mimic is None
            or mimic.get("joint") != f"{prefix}_drive_joint"
            or float(mimic.get("multiplier", "1")) != 1.0
            or float(mimic.get("offset", "0")) != 0.0
        ):
            raise ValueError(f"Robotiq calibration requires a 1:1 finger mimic joint for {prefix}")
        mesh_element = collision.find("geometry/mesh")
        limit = drive.find("limit")
        if mesh_element is None or limit is None:
            raise ValueError(f"Robotiq URDF ({prefix}) is missing pad collision mesh or drive limits")
        mesh = trimesh.load_mesh(path.parent / mesh_element.attrib["filename"])
        vertices = np.asarray(mesh.vertices) * np.asarray(tuple(map(float, mesh_element.get("scale", "1 1 1").split())))
        vertices += np.asarray(vector(collision, "origin", "xyz", "0 0 0"))
        inner_y = float(vertices[:, 1].min())
        pad = vertices[np.isclose(vertices[:, 1], inner_y, atol=1e-6)]
        drive_xyz = vector(drive, "origin", "xyz", "0 0 0")
        finger_xyz = vector(finger, "origin", "xyz", "0 0 0")
        return cls(
            base_joint_y_m=drive_xyz[1],
            finger_joint_y_m=finger_xyz[1],
            finger_joint_z_m=finger_xyz[2],
            pad_inner_y_m=inner_y,
            pad_center_z_m=float((pad[:, 2].min() + pad[:, 2].max()) / 2.0),
            drive_lower_rad=float(limit.attrib["lower"]),
            drive_upper_rad=float(limit.attrib["upper"]),
        )

    @property
    def radius_m(self) -> float:
        return math.hypot(self.finger_joint_y_m, self.finger_joint_z_m)

    @property
    def phase_rad(self) -> float:
        return math.atan2(self.finger_joint_z_m, self.finger_joint_y_m)

    def pad_gap_m(self, drive_rad: float) -> float:
        q = float(drive_rad)
        return 2.0 * (
            self.base_joint_y_m
            + self.finger_joint_y_m * math.cos(q)
            - self.finger_joint_z_m * math.sin(q)
            + self.pad_inner_y_m
        )

    def gap_jacobian_m_per_rad(self, drive_rad: float) -> float:
        """Absolute derivative of the full pad gap with respect to drive joint angle."""
        q = float(drive_rad)
        return abs(2.0 * (self.finger_joint_y_m * math.sin(q) + self.finger_joint_z_m * math.cos(q)))

    def drive_for_gap_m(self, gap_m: float) -> float:
        gap = float(gap_m)
        if (
            not math.isfinite(gap)
            or not self.pad_gap_m(self.drive_upper_rad) - 1e-6 <= gap <= self.pad_gap_m(self.drive_lower_rad) + 1e-6
        ):
            raise ValueError("calibration gap is outside the collision pads' physical range")
        cosine = (gap / 2.0 - self.base_joint_y_m - self.pad_inner_y_m) / self.radius_m
        q = math.acos(max(-1.0, min(1.0, cosine))) - self.phase_rad
        return max(self.drive_lower_rad, min(self.drive_upper_rad, q))


ROBOTIQ_2F_85_LINKAGE = RobotiqLinkageGeometry(
    base_joint_y_m=0.03060114,
    finger_joint_y_m=0.0371575,
    finger_joint_z_m=0.04342168,
    pad_inner_y_m=-0.02525864541530609,
    pad_center_z_m=0.032010000000000004,
    drive_lower_rad=ROBOTIQ_2F_85_SIM_OPEN_DRIVE,
    drive_upper_rad=ROBOTIQ_2F_85_SIM_CLOSE_DRIVE,
)

ROBOTIQ_2F_140_LINKAGE = RobotiqLinkageGeometry(
    base_joint_y_m=0.030601,
    finger_joint_y_m=0.066855,
    finger_joint_z_m=0.074366,
    pad_inner_y_m=-0.02716850000000001,
    pad_center_z_m=0.04772499999999999,
    drive_lower_rad=ROBOTIQ_2F_140_SIM_OPEN_DRIVE,
    drive_upper_rad=ROBOTIQ_2F_140_SIM_CLOSE_DRIVE,
)


@dataclass(frozen=True)
class RobotiqSpec:
    """Variant specification and conversions for one Robotiq 2F adaptive gripper."""

    adapter: str
    display_name: str
    prefix: str
    closed_gap_mm: float
    open_gap_mm: float
    speed_min_mm_s: float
    speed_max_mm_s: float
    force_min_n: float
    force_max_n: float
    linkage: RobotiqLinkageGeometry

    @property
    def closed_gap_m(self) -> float:
        return self.closed_gap_mm / 1000.0

    @property
    def open_gap_m(self) -> float:
        return self.open_gap_mm / 1000.0

    @property
    def joint_names(self) -> tuple[str, ...]:
        return robotiq_joint_names(self.prefix)

    @property
    def drive_joint_name(self) -> str:
        return f"{self.prefix}_drive_joint"

    def clamp_gap_m(self, gap_m: float) -> float:
        value = float(gap_m)
        if not math.isfinite(value):
            raise ValueError("gap_m must be finite")
        return max(self.closed_gap_m, min(self.open_gap_m, value))

    def gap_m_to_sim_drive(self, gap_m: float) -> float:
        return self.linkage.drive_for_gap_m(self.clamp_gap_m(gap_m))

    def sim_drive_to_gap_m(self, drive: float) -> float:
        value = float(drive)
        if not math.isfinite(value):
            raise ValueError("drive must be finite")
        clipped = max(self.linkage.drive_lower_rad, min(self.linkage.drive_upper_rad, value))
        return self.clamp_gap_m(self.linkage.pad_gap_m(clipped))

    def gap_m_to_register(self, gap_m: float) -> int:
        """Map pad gap in metres to xArm SDK Robotiq register [0=open .. 255=closed]."""
        clamped = self.clamp_gap_m(gap_m)
        ratio_closed = 1.0 - (clamped - self.closed_gap_m) / (self.open_gap_m - self.closed_gap_m)
        return int(max(0, min(255, round(ratio_closed * 255.0))))

    def register_to_gap_m(self, register: int) -> float:
        """Map xArm SDK Robotiq register [0=open .. 255=closed] to pad gap in metres."""
        clamped = max(0, min(255, int(register)))
        ratio_open = 1.0 - float(clamped) / 255.0
        return self.closed_gap_m + ratio_open * (self.open_gap_m - self.closed_gap_m)

    def speed_percent_to_mm_s(self, speed_percent: float) -> float:
        pct = max(0.0, min(100.0, float(speed_percent)))
        return self.speed_min_mm_s + (pct / 100.0) * (self.speed_max_mm_s - self.speed_min_mm_s)

    def force_percent_to_n(self, force_percent: float) -> float:
        pct = max(1.0, min(100.0, float(force_percent)))
        return self.force_min_n + ((pct - 1.0) / 99.0) * (self.force_max_n - self.force_min_n)


ROBOTIQ_2F_85_SPEC = RobotiqSpec(
    adapter="robotiq_2f_85",
    display_name="Robotiq 2F-85",
    prefix=ROBOTIQ_2F_85_PREFIX,
    closed_gap_mm=ROBOTIQ_2F_85_CLOSED_GAP_MM,
    open_gap_mm=ROBOTIQ_2F_85_OPEN_GAP_MM,
    speed_min_mm_s=20.0,
    speed_max_mm_s=150.0,
    force_min_n=20.0,
    force_max_n=235.0,
    linkage=ROBOTIQ_2F_85_LINKAGE,
)

ROBOTIQ_2F_140_SPEC = RobotiqSpec(
    adapter="robotiq_2f_140",
    display_name="Robotiq 2F-140",
    prefix=ROBOTIQ_2F_140_PREFIX,
    closed_gap_mm=ROBOTIQ_2F_140_CLOSED_GAP_MM,
    open_gap_mm=ROBOTIQ_2F_140_OPEN_GAP_MM,
    speed_min_mm_s=30.0,
    speed_max_mm_s=250.0,
    force_min_n=10.0,
    force_max_n=125.0,
    linkage=ROBOTIQ_2F_140_LINKAGE,
)

ROBOTIQ_SPECS: dict[str, RobotiqSpec] = {
    "robotiq_2f_85": ROBOTIQ_2F_85_SPEC,
    "robotiq_2f_140": ROBOTIQ_2F_140_SPEC,
}


def get_robotiq_spec(variant: str) -> RobotiqSpec:
    """Return the Robotiq variant spec for ``robotiq_2f_85`` or ``robotiq_2f_140``."""
    try:
        return ROBOTIQ_SPECS[str(variant)]
    except KeyError as exc:
        raise ValueError(f"unknown Robotiq variant: {variant}") from exc


def robotiq_percent_to_register(percent: float) -> int:
    """Convert a percentage [0..100] to a Robotiq 8-bit register value [0..255]."""
    value = float(percent)
    if not math.isfinite(value):
        raise ValueError("percent must be finite")
    return int(max(0, min(255, round(max(0.0, min(100.0, value)) * 255.0 / 100.0))))


def robotiq_gap_m_to_sim_drive(variant: str, gap_m: float) -> float:
    return get_robotiq_spec(variant).gap_m_to_sim_drive(gap_m)


def robotiq_sim_drive_to_gap_m(variant: str, drive: float) -> float:
    return get_robotiq_spec(variant).sim_drive_to_gap_m(drive)


def robotiq_gap_m_to_register(variant: str, gap_m: float) -> int:
    return get_robotiq_spec(variant).gap_m_to_register(gap_m)


def robotiq_register_to_gap_m(variant: str, register: int) -> float:
    return get_robotiq_spec(variant).register_to_gap_m(register)
