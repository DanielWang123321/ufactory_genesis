"""Isolated natural-drop measurement for the packaging task."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

import genesis as gs
from ufactory.config import ResolvedRuntimeConfig
from ufactory.manipulation.packaging.core import packaging_layout
from ufactory.manipulation.packaging.scene import build_packaging_scene
from ufactory.simulation import GenesisRuntimeManager


@dataclass(frozen=True)
class NaturalDropReport:
    impact_time_s: float
    impact_velocity_m_s: float
    post_impact_velocity_m_s: float
    maximum_post_impact_velocity_m_s: float
    rebound_height_m: float
    settled_time_s: float | None
    samples: int


def measure_natural_drop(
    config: ResolvedRuntimeConfig,
    *,
    max_steps: int = 250,
    settle_velocity_m_s: float = 0.002,
    settle_samples: int = 20,
) -> NaturalDropReport:
    """Drop the configured cube into its box without correcting its trajectory.

    The sole pose write places the cube at the configured release point before
    sampling starts. Every subsequent state is produced by Genesis contact
    dynamics; no weld, reset, velocity clear, or rebound scaling is applied.
    """

    if max_steps < 1 or settle_samples < 1:
        raise ValueError("drop measurement step counts must be positive")
    task = packaging_layout(config)
    dt = 1.0 / float(config.motion.rate_hz)
    with GenesisRuntimeManager(config.simulation):
        scene, _robot, block, display = build_packaging_scene(
            sim_dt=dt,
            show_viewer=False,
            runtime_config=config,
        )
        release = task.release_object_center_m
        block.set_pos(
            torch.tensor(
                [[display.place_xy[0], display.place_xy[1], display.table_top_z + release[2]]],
                device=gs.device,
                dtype=gs.tc_float,
            ),
            zero_velocity=True,
        )
        block.set_quat(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=gs.device, dtype=gs.tc_float),
            zero_velocity=True,
        )

        rest_center_z = display.box_inner_floor_z + display.obj_size[2] / 2.0
        impact_step: int | None = None
        impact_velocity = float("nan")
        post_impact_velocity = float("nan")
        maximum_upward_velocity = 0.0
        rebound_peak_z = rest_center_z
        settled_step: int | None = None
        quiet = 0
        previous_vz = 0.0
        steps_run = 0
        for step in range(1, max_steps + 1):
            scene.step()
            steps_run = step
            center_z = float(block.get_pos()[0, 2].item())
            velocity = block.get_vel()[0]
            vz = float(velocity[2].item())
            speed = float(torch.linalg.norm(velocity).item())
            if impact_step is None and center_z <= rest_center_z + 0.001:
                impact_step = step
                impact_velocity = previous_vz
                post_impact_velocity = vz
            if impact_step is not None:
                maximum_upward_velocity = max(maximum_upward_velocity, vz)
                if vz > 0.0:
                    rebound_peak_z = max(rebound_peak_z, center_z)
                if speed <= settle_velocity_m_s and abs(center_z - rest_center_z) <= 0.002:
                    quiet += 1
                    if quiet >= settle_samples:
                        settled_step = step
                        break
                else:
                    quiet = 0
            previous_vz = vz

    if impact_step is None or not math.isfinite(impact_velocity):
        raise RuntimeError("configured cube did not reach the box floor during the measurement window")
    return NaturalDropReport(
        impact_time_s=impact_step * dt,
        impact_velocity_m_s=impact_velocity,
        post_impact_velocity_m_s=post_impact_velocity,
        maximum_post_impact_velocity_m_s=maximum_upward_velocity,
        rebound_height_m=max(0.0, rebound_peak_z - rest_center_z),
        settled_time_s=None if settled_step is None else settled_step * dt,
        samples=steps_run,
    )
