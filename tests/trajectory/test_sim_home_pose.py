"""The Genesis start pose is the program's first Cartesian point for every robot and gripper (GPU).

``build_scene`` solves the home pose with ``solve_home_ik``. With Genesis's default IK tolerances it stopped up to
0.445 mm from the target (xArm5 / xArm6 + Robotiq 2F-140), and preflight read that step into the first MoveL as
1.40 m/s^2 against the 0.8 m/s^2 Cartesian limit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

CASES = [
    *(
        (robot, gripper)
        for robot in ("xarm5", "xarm6", "xarm7", "uf850")
        for gripper in ("g2", "robotiq_2f_85", "robotiq_2f_140")
    ),
    ("lite6", "lite6"),
]


@pytest.mark.parametrize(("robot", "gripper"), CASES)
def test_sim_home_pose_is_the_program_start_point(robot: str, gripper: str) -> None:
    import genesis as gs
    from ufactory.cli.pick_place import _backends, _build_ik_scene, _build_program, _home_xyz_m
    from ufactory.config import load_runtime_config
    from ufactory.safety.statistics import compute_motion_statistics
    from ufactory.simulation import GenesisRuntimeManager
    from ufactory.trajectory.preflight import create_safety_gate

    config = load_runtime_config(robot, gripper=gripper)
    with GenesisRuntimeManager(config.simulation):
        assert str(gs.device).startswith("cuda")
        ctx = _build_ik_scene(config, calibration=None, calibrated_g2=True)
        q_home = np.asarray(ctx.home_qpos, dtype=np.float64)[list(ctx.arm_dof_idx)]
        kinematics, _collision = _backends(config, Path(ctx.robot_urdf_path), calibrated_g2=True)

        offset_m = np.asarray(kinematics.forward(q_home), dtype=np.float64)[:3] - _home_xyz_m(config)
        assert np.linalg.norm(offset_m) <= 1e-5, offset_m

        gate = create_safety_gate(
            config,
            kinematics=kinematics,
            collision=None,
            urdf_path=ctx.robot_urdf_path,
            calibration_sha256="0" * 64,
            scene_sha256="0" * 64,
        )
        program = _build_program(config, q_home=q_home)
        timeline, violations = gate._build_timeline(program)
        assert timeline is not None and not violations, violations
        stats = compute_motion_statistics(timeline.q_rad, timeline.xyz_m, rate_hz=program.rate)
        limit = config.motion.cartesian_acceleration_m_s2
        assert stats.max_cartesian_acceleration_m_s2 <= limit * 1.001 + 1e-12
