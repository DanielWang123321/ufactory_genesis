"""Bio Gripper G2 controller for Genesis robot entities.

Provides a reusable :class:`BioGripperG2` class that discovers Bio Gripper G2
joints on a Genesis robot entity and exposes position control, PD setup, and
open/close demo helpers.  The same class works for any supported UFACTORY arm
(xArm5 / xArm6 / xArm7 / UF850) because joint names are identical across all
combo URDFs.
"""

from __future__ import annotations

import numpy as np

# Joint-space constants (prismatic, meters).
#
# The right-finger prismatic dof drives the gripper; the left finger mimics it (-1).
# Joint zero is the CLOSED pose (real 71 mm two-finger gap) and the dof opens the jaws
# symmetrically up to STROKE (real 150 mm gap).  Per finger the travel is
# (150 - 71) / 2 = 39.5 mm, so the full finger stroke is 0.0395 m.
STROKE = 0.0395      # Per-finger prismatic travel: closed (0) → open (0.0395 m).
CLOSE_POS = 0.0      # Right-finger joint value when gripper is fully closed (71 mm gap).
OPEN_POS = 0.0395    # Right-finger joint value when gripper is fully open (150 mm gap).

# Physical two-finger gap (meters) at each stroke extreme — the real Bio Gripper G2
# mechanical range.  Gap scales linearly: GAP ≈ CLOSED_GAP + 2 * right_joint_value.
CLOSED_GAP = 0.071
OPEN_GAP = 0.150

RIGHT_JOINT = "bio_gripper_g2_right_finger_joint"
LEFT_JOINT = "bio_gripper_g2_left_finger_joint"
_JOINT_NAMES = (RIGHT_JOINT, LEFT_JOINT)

_DEFAULT_HOLD_STEPS = 200

# PD gains (tuned for visual-only GLB previews).
_KP = 500.0
_KV = 50.0
_FORCE_LIMIT = 20.0  # N
_DAMPING = 0.05


class BioGripperG2:
    """Reusable Bio Gripper G2 controller bound to a Genesis robot entity.

    Class-level constants (also available as module-level names)::

        BioGripperG2.RIGHT_JOINT   # "bio_gripper_g2_right_finger_joint"
        BioGripperG2.LEFT_JOINT    # "bio_gripper_g2_left_finger_joint"
        BioGripperG2.STROKE        # 0.0395 m (per-finger travel)
        BioGripperG2.CLOSE_POS     # 0.0     (closed, 71 mm two-finger gap)
        BioGripperG2.OPEN_POS      # 0.0395  (open, 150 mm two-finger gap)
        BioGripperG2.CLOSED_GAP    # 0.071 m
        BioGripperG2.OPEN_GAP      # 0.150 m

    Usage::

        from ufactory.bio_gripper_g2 import BioGripperG2

        gripper = BioGripperG2(robot)
        if gripper.found:
            gripper.setup_pd()
            gripper.open()

        # Demo loop:
        for step in range(1000):
            target = gripper.demo_target(step)
            gripper.control_pose(target)
            scene.step()
    """

    # Re-export module constants at the class level for convenience.
    RIGHT_JOINT = RIGHT_JOINT
    LEFT_JOINT = LEFT_JOINT
    STROKE = STROKE
    OPEN_POS = OPEN_POS
    CLOSE_POS = CLOSE_POS
    CLOSED_GAP = CLOSED_GAP
    OPEN_GAP = OPEN_GAP

    def __init__(self, robot):
        """Discover Bio Gripper G2 joints on *robot*.

        Parameters:
            robot: A ``gs.Entity`` loaded from a combo URDF that includes the
                   Bio Gripper G2 kinematic subtree.
        """
        self._robot = robot
        joint_map = {j.name.split("/")[-1]: j for j in robot.joints}
        self._right_joint = joint_map.get(RIGHT_JOINT)
        self._left_joint = joint_map.get(LEFT_JOINT)

    # -- discovery ----------------------------------------------------------

    @property
    def found(self) -> bool:
        """``True`` if the robot entity has Bio Gripper G2 joints."""
        return self._right_joint is not None

    @property
    def drive_dof_idx(self) -> list[int]:
        """DOF index of the drive joint (right finger)."""
        if self._right_joint is None:
            return []
        return [self._right_joint.dofs_idx_local[0]]

    @property
    def all_dof_idx(self) -> list[int]:
        """DOF indices of both finger joints (drive + mimic)."""
        idx: list[int] = []
        if self._right_joint is not None:
            idx.append(self._right_joint.dofs_idx_local[0])
        if self._left_joint is not None:
            idx.append(self._left_joint.dofs_idx_local[0])
        return idx

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def demo_target(step: int, hold_steps: int = _DEFAULT_HOLD_STEPS) -> float:
        """Alternating open/close target value for demo loops.

        Returns:
            ``CLOSE_POS`` during even phases, ``OPEN_POS`` during odd phases.
        """
        phase = (step // hold_steps) % 2
        return CLOSE_POS if phase else OPEN_POS

    # -- PD control ---------------------------------------------------------

    def setup_pd(self) -> None:
        """Configure PD gains for visual-only GLB preview.

        Sets kp/kv, force range, damping, and friction loss on the gripper
        DOFs.  Call once after scene is built and before the main loop.
        """
        pd_idx = self.drive_dof_idx or self.all_dof_idx
        damp_idx = self.all_dof_idx or self.drive_dof_idx
        if not pd_idx:
            return
        robot = self._robot
        robot.set_dofs_kp(np.full(len(pd_idx), _KP), pd_idx)
        robot.set_dofs_kv(np.full(len(pd_idx), _KV), pd_idx)
        robot.set_dofs_force_range(
            np.full(len(pd_idx), -_FORCE_LIMIT),
            np.full(len(pd_idx), _FORCE_LIMIT),
            pd_idx,
        )
        robot.set_dofs_damping(np.full(len(damp_idx), _DAMPING), damp_idx)
        robot.set_dofs_frictionloss(np.zeros(len(damp_idx)), damp_idx)

    def set_pose(self, right_value: float) -> None:
        """Teleport gripper joints to *right_value* (kinematic, no forces).

        Parameters:
            right_value: Target position for the right-finger prismatic joint
                         (m).  The left finger receives ``-right_value``.
        """
        active = self.all_dof_idx or self.drive_dof_idx
        if not active:
            return
        target = self._targets(active, right_value)
        self._robot.set_dofs_position(target, active)
        self._robot.control_dofs_position(target, active)

    def control_pose(self, right_value: float) -> None:
        """PD-control gripper toward *right_value* (applies forces).

        Parameters:
            right_value: Target position for the right-finger prismatic joint.
        """
        active = self.all_dof_idx or self.drive_dof_idx
        if not active:
            return
        target = self._targets(active, right_value)
        self._robot.set_dofs_position(target, active, zero_velocity=False)
        self._robot.control_dofs_position(target, active)

    # -- convenience --------------------------------------------------------

    def open(self) -> None:
        """Open the gripper fully (kinematic teleport)."""
        self.set_pose(OPEN_POS)

    def close(self) -> None:
        """Close the gripper fully (kinematic teleport)."""
        self.set_pose(CLOSE_POS)

    # -- internal -----------------------------------------------------------

    def _targets(self, active_dof_idx: list[int], right_value: float) -> np.ndarray:
        """Build a position target array for *active_dof_idx*.

        ``right_value`` is used for the right-finger DOF;
        ``-right_value`` for the left-finger DOF (mirrored opening).
        """
        values: list[float] = []
        for dof_i in active_dof_idx:
            if (
                self._right_joint is not None
                and dof_i == self._right_joint.dofs_idx_local[0]
            ):
                values.append(right_value)
            elif (
                self._left_joint is not None
                and dof_i == self._left_joint.dofs_idx_local[0]
            ):
                values.append(-right_value)
            else:
                values.append(0.0)
        return np.array(values, dtype=np.float64)
