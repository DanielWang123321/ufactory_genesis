"""Offline Pinocchio/Coal collision audit for recorded Bio G2 simulation trajectories."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from ufactory.simulation.bio_g2_model import LEFT_JOINT, RIGHT_JOINT, gap_to_q
from ufactory.trajectory.bio_g2_frames import BioG2TaskFrames


def audit_collisions(frames: BioG2TaskFrames, joint_artifact: dict, *, exhaustive: bool = False) -> dict:
    """Check initial, commanded, and measured states with both finger joints expanded.

    The simulation retains its mimic equality. This independent model expands it so
    each collision geometry follows its own finger; commanded states use q_left=-q_right.
    Adjacent links and geometries on the same rigid body are omitted. Only the
    mounted base is omitted from table checks; all moving links and nests are checked.
    """
    import coal
    import pinocchio as pin

    from ufactory.trajectory.bio_g2_replay import (
        NEST_SIZE_M,
        PICK_XY,
        PLACE_XY,
        ROBOT_BASE_POS,
        TABLE_TOP_Z,
    )

    model = pin.buildModelFromUrdf(str(frames.urdf_path), mimic=False)
    data = model.createData()
    geometry = pin.buildGeomFromUrdf(
        model, str(frames.urdf_path), pin.GeometryType.COLLISION, package_dirs=[str(frames.urdf_path.parent)]
    )
    geometry.addAllCollisionPairs()
    root = ET.parse(frames.urdf_path).getroot()
    adjacent = {frozenset((j.find("parent").get("link"), j.find("child").get("link"))) for j in root.findall("joint")}

    def link_name(index):
        return str(model.frames[int(geometry.geometryObjects[index].parentFrame)].name)

    for pair in list(geometry.collisionPairs):
        a, b = geometry.geometryObjects[pair.first], geometry.geometryObjects[pair.second]
        if a.parentJoint == b.parentJoint or frozenset((link_name(pair.first), link_name(pair.second))) in adjacent:
            geometry.removeCollisionPair(pair)
    robot_count = len(geometry.geometryObjects)
    for name, size, center in (
        ("table", (0.65, 0.70, TABLE_TOP_Z), (0.25, 0, -TABLE_TOP_Z / 2)),
        ("pick_nest", NEST_SIZE_M, (*PICK_XY, NEST_SIZE_M[2] / 2)),
        ("place_nest", NEST_SIZE_M, (*PLACE_XY, NEST_SIZE_M[2] / 2)),
    ):
        idx = geometry.addGeometryObject(
            pin.GeometryObject(name, 0, 0, pin.SE3(np.eye(3), np.array(center)), coal.Box(*size))
        )
        for robot_idx in range(robot_count):
            if name == "table" and link_name(robot_idx) == "link_base":
                continue
            geometry.addCollisionPair(pin.CollisionPair(robot_idx, idx))
    geom_data = geometry.createData()
    local_corners = []
    for obj in geometry.geometryObjects:
        obj.geometry.computeLocalAABB()
        bounds = np.array([obj.geometry.aabb_local.min_, obj.geometry.aabb_local.max_])
        local_corners.append([bounds[[i, j, k], [0, 1, 2]] for i in (0, 1) for j in (0, 1) for k in (0, 1)])
    local_corners = np.asarray(local_corners)
    pairs = list(geometry.collisionPairs)
    pair_indices = np.array([[pair.first, pair.second] for pair in pairs])
    indices = {
        name: int(model.joints[model.getJointId(name)].idx_q)
        for name in (*frames.runtime.arm.joint_names, RIGHT_JOINT, LEFT_JOINT)
    }
    for finger in (RIGHT_JOINT, LEFT_JOINT):
        if model.joints[model.getJointId(finger)].nq != 1:
            raise RuntimeError("Collision audit requires independently expanded finger joints")
    samples = []
    if joint_artifact["initial_arm_q_rad"] is not None:
        samples.append(("initial", -1, "initial", joint_artifact["initial_arm_q_rad"], gap_to_q(150), -gap_to_q(150)))
    for row in joint_artifact["executed"]:
        right, left = row["right_finger_m"], row["left_finger_m"]
        for kind in ("target", "actual"):
            arm = [row[f"{kind}_{name}_rad"] for name in frames.runtime.arm.joint_names]
            samples.append((kind, row["target_index"], row["phase"], arm, right, -right if kind == "target" else left))
    minima = {"self": float("inf"), "table": float("inf"), "nest": float("inf")}
    collisions = []
    max_tcp_error_mm = 0.0
    distance_queries = 0
    flange_id = model.getFrameId(frames.runtime.arm.ee_link)
    gripper_id = model.getFrameId("bio_gripper_g2_base_link")
    for kind, index, phase, arm, right, left in samples:
        q = pin.neutral(model)
        for name, value in zip(frames.runtime.arm.joint_names, arm):
            q[indices[name]] = value
        q[indices[RIGHT_JOINT]], q[indices[LEFT_JOINT]] = right, left
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        # Independent FK also verifies the mounting transform used for world TCP conversion.
        predicted = data.oMf[flange_id].homogeneous @ frames.flange_to_gripper
        max_tcp_error_mm = max(
            max_tcp_error_mm, float(np.linalg.norm(predicted[:3, 3] - data.oMf[gripper_id].translation) * 1000)
        )
        pin.updateGeometryPlacements(model, data, geometry, geom_data)
        transforms = np.array([placement.homogeneous for placement in geom_data.oMg])
        corners = np.einsum("gij,gkj->gki", transforms[:, :3, :3], local_corners) + transforms[:, None, :3, 3]
        lower, upper = corners.min(axis=1), corners.max(axis=1)
        a, b = pair_indices.T
        bounds = np.linalg.norm(np.maximum(0, np.maximum(lower[a] - upper[b], lower[b] - upper[a])), axis=1)
        for pair_idx, pair in enumerate(pairs):
            second = str(geometry.geometryObjects[pair.second].name)
            category = "self" if pair.second < robot_count else "table" if second == "table" else "nest"
            # Disjoint world AABBs cannot penetrate. Their separation is also a
            # lower bound on mesh distance, so a pair farther than the current
            # category minimum cannot change it. Every overlapping pair is queried.
            if not exhaustive and bounds[pair_idx] > 1e-12 and bounds[pair_idx] * 1000 >= minima[category]:
                continue
            distance = float(pin.computeDistance(geometry, geom_data, pair_idx).min_distance)
            distance_queries += 1
            minima[category] = min(minima[category], distance * 1000)
            if distance < -1e-6:
                collisions.append(
                    {
                        "kind": kind,
                        "target_index": index,
                        "phase": phase,
                        "category": category,
                        "first": str(geometry.geometryObjects[pair.first].name),
                        "second": second,
                        "distance_mm": distance * 1000,
                    }
                )
    return {
        "passed": not collisions,
        "sample_count": len(samples),
        "mimic_expanded": True,
        "base_frame_world_xyz_m": list(ROBOT_BASE_POS),
        "penetration_tolerance_mm": 0.001,
        "min_distance_mm": minima,
        "collision_count": len(collisions),
        "collisions": collisions[:30],
        "max_mounting_fk_error_mm": max_tcp_error_mm,
        "exact_distance_queries": distance_queries,
        "total_pair_samples": len(pairs) * len(samples),
    }
