"""
xArm 6 Dynamics Verification Script for Genesis Simulation.

Thin runner over the ``ufactory.dynamics`` probe layer: model parameter readback,
gravity free-fall, static torques / gravity compensation, PD step response,
energy dissipation and mass-matrix plausibility (Tests 5-10, continuing the
numbering from verify_xarm6.py which has Tests 1-4).

No hardcoded mass/PD tables: gains and effort limits come from
``ufactory.robot_params`` and link masses from Genesis ``get_mass`` / ``link``.

Usage:
    /opt/conda_envs/py313/bin/python examples/xarm6/verify_xarm6_dynamics.py        # headless (URDF default)
    /opt/conda_envs/py313/bin/python examples/xarm6/verify_xarm6_dynamics.py -v     # with viewer
"""

import argparse
import sys

import _bootstrap  # noqa: F401  (repo-root sys.path setup)
from ufactory.dynamics import (
    build_genesis_scene,
    test_energy_dissipation,
    test_gravity_compensation_torques,
    test_gravity_freefall,
    test_mass_matrix_plausibility,
    test_mass_parameters,
    test_pd_step_response,
)
from ufactory.paths import xarm6_urdf
from ufactory.robot_params import get_robot_runtime_profile


def main() -> int:
    parser = argparse.ArgumentParser(description="xArm 6 Dynamics Verification")
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument(
        "--robot-model", type=str, default=None,
        help="Robot URDF model path. Default: xarm6_1305.urdf",
    )
    args = parser.parse_args()

    urdf_path = args.robot_model or xarm6_urdf()
    runtime = get_robot_runtime_profile("xarm6")

    print(f"Building Genesis scene for {urdf_path}")
    scene, robot, ee_link, dof_idx = build_genesis_scene(
        urdf_path,
        runtime_profile=runtime,
        show_viewer=args.vis,
    )
    print(f"Loaded robot: DOFs={robot.n_dofs} Links={robot.n_links}")

    checks = [
        ("Test 5:  Model Parameters", test_mass_parameters(robot, dof_idx, runtime)),
        ("Test 6:  Gravity Freefall", test_gravity_freefall(robot, scene, dof_idx, runtime)),
        ("Test 7:  Gravity Comp Torques", test_gravity_compensation_torques(robot, scene, dof_idx, runtime)),
        ("Test 8:  PD Step Response", test_pd_step_response(robot, scene, dof_idx, runtime)),
        ("Test 9:  Energy Dissipation", test_energy_dissipation(robot, scene, dof_idx, runtime)),
        ("Test 10: Mass Matrix", test_mass_matrix_plausibility(robot, scene, dof_idx, runtime)),
    ]

    print("\n" + "=" * 60)
    print("DYNAMICS VALIDATION SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, (passed, lines) in checks:
        for line in lines:
            print(line)
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False
    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()