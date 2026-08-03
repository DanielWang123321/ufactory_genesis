"""Scripted state-machine expert for the xArm6 + Gripper G2 pick-place environment.

The expert emits the same bounded 4-dim action the learned policy emits (relative
Cartesian delta on the IK link plus a gripper-gap delta), so it answers two questions
the learned runs never separated:

* Are the strict quality targets reachable at all inside the current action limits,
  gripper command range and contact physics? If a hand-written controller with perfect
  state access cannot pass, no amount of reward tuning will.
* What does a clean trajectory look like, as supervision for behaviour cloning?

Actions displace the IK link, and the wrist is held gripper-down, so a commanded delta
translates the finger centre by the same vector. Every phase target below is therefore
expressed directly in finger-centre coordinates.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from ufactory.training.logic import cartesian_delta_to_action


PHASE_APPROACH = 0
PHASE_DESCEND = 1
PHASE_CLOSE = 2
PHASE_LIFT = 3
PHASE_TRANSPORT = 4
PHASE_SETDOWN = 5
PHASE_SETTLE = 6
PHASE_RELEASE = 7
PHASE_RETREAT = 8

PHASE_NAMES = {
    PHASE_APPROACH: "approach",
    PHASE_DESCEND: "descend",
    PHASE_CLOSE: "close",
    PHASE_LIFT: "lift",
    PHASE_TRANSPORT: "transport",
    PHASE_SETDOWN: "setdown",
    PHASE_SETTLE: "settle",
    PHASE_RELEASE: "release",
    PHASE_RETREAT: "retreat",
}

SCRIPTED_EXPERT_CONTROLLER = "scripted_pick_place_expert_v1"
SCRIPTED_EXPERT_CONTROLLERS = {
    SCRIPTED_EXPERT_CONTROLLER,
    "scripted_pick_place_expert_v2",
}


def scripted_pick_place_expert_from_env(env, **overrides) -> "ScriptedPickPlaceExpert":
    """Build the versioned expert declared by an environment artifact.

    Guided checkpoints depend on this controller at inference time, so its selected
    parameters must live in the hashed training configuration instead of being hidden
    in call-site defaults. Legacy recipes without these fields retain the v1 defaults.
    """

    controller = str(env.env_cfg.get("scripted_action_hint_controller", SCRIPTED_EXPERT_CONTROLLER))
    if controller not in SCRIPTED_EXPERT_CONTROLLERS:
        raise ValueError(f"unsupported scripted pick-place controller: {controller!r}")
    raw_parameters = env.env_cfg.get("scripted_action_hint_config", {})
    if not isinstance(raw_parameters, Mapping):
        raise ValueError("scripted_action_hint_config must be a mapping")
    parameters = dict(raw_parameters)
    parameters.update(overrides)
    return ScriptedPickPlaceExpert(env, **parameters)


def expert_phase_sample_weights(
    phases: torch.Tensor,
    *,
    near_table_weight: float = 4.0,
    release_retreat_weight: float = 2.0,
    balance_phases: bool = False,
) -> torch.Tensor:
    """Return BC weights for expert phases and optionally remove duration bias.

    The set-down phase is deliberately slow because its velocity limit is strict. If
    every time step carries equal mass, that long phase can dominate the supervised
    loss and leave too little capacity for approach, grasp and transport. Phase
    balancing gives every observed phase equal base mass while retaining the explicit
    near-table and release priorities.
    """

    if near_table_weight <= 0.0 or release_retreat_weight <= 0.0:
        raise ValueError("phase weights must be positive")
    weights = torch.ones_like(phases, dtype=torch.float32)
    near_table = (phases == PHASE_SETDOWN) | (phases == PHASE_SETTLE)
    release_retreat = (phases == PHASE_RELEASE) | (phases == PHASE_RETREAT)
    weights[near_table] = float(near_table_weight)
    weights[release_retreat] = float(release_retreat_weight)
    if balance_phases and phases.numel() > 0:
        for phase in PHASE_NAMES:
            mask = phases == phase
            if bool(mask.any()):
                weights[mask] /= mask.sum()
        weights *= weights.numel() / weights.sum()
    return weights


class ScriptedPickPlaceExpert:
    """Phase-sequenced controller producing bounded actions for the pick-place env."""

    def __init__(
        self,
        env,
        *,
        hover_m: float = 0.06,
        lift_height_m: float = 0.08,
        retreat_height_m: float = 0.05,
        pre_grasp_gap_m: float | None = None,
        close_gap_m: float | None = None,
        grasp_y_compensation_m: float = -0.001,
        align_tolerance_m: float = 0.002,
        descend_tolerance_m: float = 0.0015,
        close_dwell_steps: int = 12,
        settle_dwell_steps: int = 2,
        settle_speed_m_s: float = 0.010,
        coarse_step_m: float = 0.005,
        descend_step_m: float = 0.003,
        lift_step_m: float = 0.004,
        setdown_ramp_gain: float = 0.06,
        setdown_ramp_offset_m: float = 0.014,
        grasp_ramp_gain: float = 0.15,
        near_table_margin_m: float = 0.025,
        near_table_xy_step_m: float = 0.0001,
        near_table_z_step_m: float = 0.0001,
        landing_brake_speed_m_s: float | None = None,
        landing_brake_max_step_m: float = 0.0005,
        release_hover_m: float = 0.00002,
        hold_bias_rate: float | None = None,
        hold_bias_limit_m: float = 0.02,
        release_open_step_m: float = 0.0005,
        retreat_step_m: float = 0.001,
        recover_phase_from_state: bool = False,
    ) -> None:
        self.env = env
        self.device = env.device
        self.num_envs = int(env.num_envs)

        self.hover_m = float(hover_m)
        self.lift_height_m = float(lift_height_m)
        self.retreat_height_m = float(retreat_height_m)
        # Wide enough that a 30 mm cube passes between the pads untouched on the way down.
        self.pre_grasp_gap_m = float(pre_grasp_gap_m if pre_grasp_gap_m is not None else env.obj_size[1] + 0.020)
        # Gripper G2's coupled mimic chain closes with a small repeatable +Y bias.
        # Approaching 1 mm toward -Y keeps the cube inside the 5 mm pre-lift drag
        # contract at the far +Y edge without changing the learned observation API.
        self.grasp_y_compensation_m = float(grasp_y_compensation_m)
        self.align_tolerance_m = float(align_tolerance_m)
        self.descend_tolerance_m = float(descend_tolerance_m)
        self.close_dwell_steps = int(close_dwell_steps)
        self.settle_dwell_steps = int(settle_dwell_steps)
        self.settle_speed_m_s = float(settle_speed_m_s)
        self.coarse_step_m = float(coarse_step_m)
        self.descend_step_m = float(descend_step_m)
        self.lift_step_m = float(lift_step_m)
        self.setdown_ramp_gain = float(setdown_ramp_gain)
        self.setdown_ramp_offset_m = float(setdown_ramp_offset_m)
        self.grasp_ramp_gain = float(grasp_ramp_gain)
        self.near_table_margin_m = float(near_table_margin_m)
        self.near_table_xy_step_m = float(near_table_xy_step_m)
        self.near_table_z_step_m = float(near_table_z_step_m)
        self.landing_brake_speed_m_s = None if landing_brake_speed_m_s is None else float(landing_brake_speed_m_s)
        self.landing_brake_max_step_m = float(landing_brake_max_step_m)
        if self.landing_brake_speed_m_s is not None and self.landing_brake_speed_m_s <= 0.0:
            raise ValueError("landing_brake_speed_m_s must be positive when enabled")
        if self.landing_brake_max_step_m <= 0.0:
            raise ValueError("landing_brake_max_step_m must be positive")
        self.release_hover_m = float(release_hover_m)
        # The feedforward only exists to cancel the droop that measured-pose integration
        # subtracts from every command. With a commanded setpoint a zero action already
        # holds, so leaving the estimator running just adds a feedback loop that rings.
        integrates_command = getattr(env, "ee_command_integration", "measured") == "commanded"
        self.hold_bias_rate = float(
            hold_bias_rate if hold_bias_rate is not None else (0.0 if integrates_command else 0.15)
        )
        self.hold_bias_limit_m = float(hold_bias_limit_m)
        self.release_open_step_m = float(release_open_step_m)
        self.retreat_step_m = float(retreat_step_m)
        self.recover_phase_from_state = bool(recover_phase_from_state)

        self.grasp_offset_z = float(env.grasp_center_offset_z)
        self.obj_rest_z = float(env.obj_rest_z_base)
        self.object_width_m = float(env.obj_size[1])
        minimum_gap = float(getattr(env, "gripper_min_command_gap_m", env.gripper_close_gap_m))
        self.close_gap_m = float(
            close_gap_m if close_gap_m is not None else max(minimum_gap, float(env.gripper_close_gap_m) - 0.004)
        )
        self.open_gap_m = float(env.gripper_open_gap_m)
        self.release_gap_m = self.object_width_m + 0.030

        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.dwell = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Deltas are added to the *measured* link pose, so joint droop under gravity is
        # subtracted from every command and a zero action makes the wrist sink instead of
        # hold. The bias tracks that shortfall so the commanded motion is what actually
        # happens; without it no phase target is ever reached.
        self.hold_bias = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_finger = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.bias_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.max_command_m = min(float(env.action_scale), float(env.max_cartesian_delta_m))

    def reset_idx(self, envs_mask: torch.Tensor) -> None:
        """Restart the state machine for the selected environments."""
        if not bool(envs_mask.any()):
            return
        # Phase-bootstrap resets hand the expert a cube that is already held; those
        # episodes rejoin the sequence at transport instead of reaching for the table.
        already_held = self.env.ever_grasped & envs_mask
        self.phase[envs_mask] = PHASE_APPROACH
        self.phase[already_held] = PHASE_TRANSPORT
        self.dwell[envs_mask] = 0
        self.hold_bias[envs_mask] = 0.0
        self.prev_command[envs_mask] = 0.0
        self.bias_valid[envs_mask] = False

    def phase_counts(self) -> dict[str, int]:
        return {name: int((self.phase == index).sum().item()) for index, name in PHASE_NAMES.items()}

    def __call__(self, observations=None, *, recover_phase_from_state: bool | None = None) -> torch.Tensor:
        del observations  # The expert reads simulator state directly.
        env = self.env

        self.reset_idx(env.episode_length_buf == 0)

        finger = env.finger_center_base()
        obj = env.obj_pos_base()
        target = env.target_pos
        gap = env.gripper_gap_m()
        obj_speed = torch.norm(env.obj.get_vel(), dim=-1)
        height_error = obj[:, 2] - self.obj_rest_z

        self._update_hold_bias(finger)
        recover = self.recover_phase_from_state if recover_phase_from_state is None else bool(recover_phase_from_state)
        if recover:
            self._recover_phases(finger, obj, gap, obj_speed, height_error)
        self._advance_phases(finger, obj, target, gap, obj_speed, height_error)

        desired, max_xy_step, max_z_step, gap_target = self._phase_setpoints(
            finger,
            obj,
            target,
            env.obj.get_vel(),
            height_error,
            gap,
        )

        delta = desired - finger
        motion = torch.cat(
            [
                torch.clamp(delta[:, :2], -max_xy_step, max_xy_step),
                torch.clamp(delta[:, 2:3], -max_z_step, max_z_step),
            ],
            dim=-1,
        )
        command = torch.clamp(motion + self.hold_bias, -self.max_command_m, self.max_command_m)
        self.prev_command = command

        cartesian = cartesian_delta_to_action(
            command,
            env.action_scale,
            getattr(env, "action_response_exponent", 1.0),
        )
        gripper = (gap_target - env.commanded_gap) / env.gripper_delta_m
        actions = torch.cat([cartesian, gripper.unsqueeze(-1)], dim=-1)
        return torch.clamp(actions, -1.0, 1.0)

    def _recover_phases(
        self,
        finger: torch.Tensor,
        obj: torch.Tensor,
        gap: torch.Tensor,
        obj_speed: torch.Tensor,
        height_error: torch.Tensor,
    ) -> None:
        """Re-label arbitrary student states with a coherent recoverable phase.

        Stateful expert rollouts retain their original sequencing. DAgger rollouts can
        drift away from that history, so carrying the old phase forward produces
        contradictory labels (for example, transport commands after a dropped grasp).
        This reconstruction uses only state already represented in the policy contract.
        """

        env = self.env
        previous = self.phase
        recovered = torch.full_like(previous, PHASE_APPROACH)
        xy_to_obj = torch.norm(finger[:, :2] - obj[:, :2], dim=-1)
        grasp_z = self.obj_rest_z + self.grasp_offset_z
        approach_z = grasp_z + self.hover_m
        aligned = xy_to_obj < self.align_tolerance_m
        gripper_ready = gap > (self.pre_grasp_gap_m - 0.003)
        below_hover = finger[:, 2] <= approach_z + 0.005
        at_grasp = (finger[:, 2] - grasp_z).abs() < self.descend_tolerance_m

        descending = aligned & gripper_ready & below_hover
        recovered[descending] = PHASE_DESCEND
        recovered[descending & at_grasp] = PHASE_CLOSE

        holding = env.holding.bool()
        recovered[holding & (height_error <= 0.9 * self.lift_height_m)] = PHASE_LIFT
        recovered[holding & (height_error > 0.9 * self.lift_height_m)] = PHASE_TRANSPORT

        carried_near = env.ever_carried_near.bool()
        recovered[carried_near] = PHASE_SETDOWN
        table_ready = carried_near & (height_error.abs() < 0.001) & (obj_speed < self.settle_speed_m_s)
        recovered[table_ready] = PHASE_SETTLE

        released = env.release_started.bool()
        recovered[released] = PHASE_RELEASE
        recovered[released & (gap > self.release_gap_m)] = PHASE_RETREAT

        # Preserve the two deliberate dwell windows instead of jumping as soon as a
        # contact bit or table-ready bit first appears.
        closing_dwell = (previous == PHASE_CLOSE) & holding & (self.dwell < self.close_dwell_steps)
        settle_dwell = (previous == PHASE_SETTLE) & table_ready & (self.dwell < self.settle_dwell_steps)
        recovered[closing_dwell] = PHASE_CLOSE
        recovered[settle_dwell] = PHASE_SETTLE

        changed = recovered != previous
        self.phase = recovered
        self.dwell = torch.where(changed, torch.zeros_like(self.dwell), self.dwell)

    def _update_hold_bias(self, finger: torch.Tensor) -> None:
        """Track how much of the previous command the arm actually executed."""
        achieved = finger - self.prev_finger
        shortfall = torch.clamp(
            self.prev_command - achieved,
            -self.max_command_m,
            self.max_command_m,
        )
        updated = torch.lerp(self.hold_bias, shortfall, self.hold_bias_rate)
        self.hold_bias = torch.where(
            self.bias_valid.unsqueeze(-1),
            updated.clamp(-self.hold_bias_limit_m, self.hold_bias_limit_m),
            self.hold_bias,
        )
        self.prev_finger = finger.clone()
        self.bias_valid.fill_(True)

    def _advance_phases(
        self,
        finger: torch.Tensor,
        obj: torch.Tensor,
        target: torch.Tensor,
        gap: torch.Tensor,
        obj_speed: torch.Tensor,
        height_error: torch.Tensor,
    ) -> None:
        env = self.env
        phase = self.phase
        xy_to_obj = torch.norm(finger[:, :2] - obj[:, :2], dim=-1)
        xy_to_target = torch.norm(obj[:, :2] - target[:, :2], dim=-1)
        grasp_z = self.obj_rest_z + self.grasp_offset_z
        approach_z = grasp_z + self.hover_m

        aligned = xy_to_obj < self.align_tolerance_m
        advance = torch.zeros_like(phase, dtype=torch.bool)

        at_hover = (finger[:, 2] - approach_z).abs() < 0.005
        gripper_ready = gap > (self.pre_grasp_gap_m - 0.003)
        advance |= (phase == PHASE_APPROACH) & aligned & at_hover & gripper_ready

        at_grasp = (finger[:, 2] - grasp_z).abs() < self.descend_tolerance_m
        advance |= (phase == PHASE_DESCEND) & aligned & at_grasp

        closed = env.holding & (self.dwell >= self.close_dwell_steps)
        advance |= (phase == PHASE_CLOSE) & closed

        advance |= (phase == PHASE_LIFT) & (height_error > 0.9 * self.lift_height_m)

        advance |= (phase == PHASE_TRANSPORT) & (xy_to_target < 0.003)

        # Release just above the table instead of holding the cube against it. A long
        # dwell under contact turns tiny action perturbations into horizontal impulses;
        # a 20 um drop has only about 0.02 m/s of ideal free-fall speed and stays below
        # the landing limit with useful margin.
        at_release_hover = ((height_error - self.release_hover_m).abs() < 0.00004) & (obj_speed < self.settle_speed_m_s)
        advance |= (phase == PHASE_SETDOWN) & at_release_hover

        # Everything the strict release gate checks must already hold before the pads
        # start opening: the first opening step is the one that gets graded.
        release_ready = (
            (xy_to_target < 0.008)
            & (height_error.abs() < 0.001)
            & (obj_speed < self.settle_speed_m_s)
            & env.ever_carried_near
            & (self.dwell >= self.settle_dwell_steps)
        )
        advance |= (phase == PHASE_SETTLE) & release_ready

        advance |= (phase == PHASE_RELEASE) & (gap > self.release_gap_m)

        self.phase = torch.where(advance & (phase < PHASE_RETREAT), phase + 1, phase)
        self.dwell = torch.where(advance, torch.zeros_like(self.dwell), self.dwell + 1)

    def _phase_setpoints(
        self,
        finger: torch.Tensor,
        obj: torch.Tensor,
        target: torch.Tensor,
        obj_vel: torch.Tensor,
        height_error: torch.Tensor,
        gap: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        phase = self.phase
        grasp_z = self.obj_rest_z + self.grasp_offset_z
        carry_z = grasp_z + self.lift_height_m

        desired = finger.clone()
        max_xy_step = torch.full((self.num_envs, 1), self.coarse_step_m, device=self.device)
        max_z_step = torch.full((self.num_envs, 1), self.coarse_step_m, device=self.device)
        gap_target = torch.full((self.num_envs,), self.close_gap_m, device=self.device)

        approach = phase == PHASE_APPROACH
        desired[approach, 0] = obj[approach, 0]
        desired[approach, 1] = obj[approach, 1] + self.grasp_y_compensation_m
        desired[approach, 2] = grasp_z + self.hover_m
        gap_target[approach] = self.pre_grasp_gap_m

        # Both descents brake proportionally to the distance still to travel. A late
        # step change cannot be tracked: the wrist keeps coasting for several control
        # steps, which is what pushed the cube's entry speed over the landing limit.
        descend = phase == PHASE_DESCEND
        desired[descend, 0] = obj[descend, 0]
        desired[descend, 1] = obj[descend, 1] + self.grasp_y_compensation_m
        desired[descend, 2] = grasp_z
        gap_target[descend] = self.pre_grasp_gap_m
        grasp_remaining = (finger[:, 2] - grasp_z).abs().unsqueeze(-1)
        grasp_ramp = torch.clamp(
            self.grasp_ramp_gain * grasp_remaining,
            self.near_table_xy_step_m,
            self.descend_step_m,
        )
        max_xy_step[descend] = self.descend_step_m
        max_z_step[descend] = grasp_ramp[descend]

        # Closing, settling and releasing all hold the wrist still: any residual motion
        # while a pad touches the cube shows up directly as drag or post-release drift.
        frozen = (phase == PHASE_CLOSE) | (phase == PHASE_SETTLE) | (phase == PHASE_RELEASE)
        max_xy_step[frozen] = 0.0
        max_z_step[frozen] = 0.0

        # A firm grasp stores elastic energy in the contact; dumping it in one 4 mm
        # command step kicks the cube downward hard enough to trip the landing limit.
        # Bleed the command out until the pads clear the faces, then open freely.
        release = phase == PHASE_RELEASE
        gap_target[release] = self.open_gap_m
        bleeding = release & (gap < self.object_width_m + 0.008)
        gap_target[bleeding] = torch.minimum(
            self.env.commanded_gap[bleeding] + self.release_open_step_m,
            torch.full_like(gap_target[bleeding], self.open_gap_m),
        )

        lift = phase == PHASE_LIFT
        desired[lift, 2] = carry_z
        max_xy_step[lift] = 0.0
        max_z_step[lift] = self.lift_step_m

        # Steering by the cube's own error rather than the wrist's keeps an off-centre
        # grasp from turning into an equal placement offset.
        cube_to_target = target[:, :2] - obj[:, :2]
        transport = phase == PHASE_TRANSPORT
        desired[transport, 0] = finger[transport, 0] + cube_to_target[transport, 0]
        desired[transport, 1] = finger[transport, 1] + cube_to_target[transport, 1]
        desired[transport, 2] = carry_z

        setdown = phase == PHASE_SETDOWN
        desired[setdown, 0] = finger[setdown, 0] + cube_to_target[setdown, 0]
        desired[setdown, 1] = finger[setdown, 1] + cube_to_target[setdown, 1]
        release_descent = (height_error - self.release_hover_m).clamp(min=0.0)
        desired[setdown, 2] = finger[setdown, 2] - release_descent[setdown]
        setdown_ramp = torch.clamp(
            self.setdown_ramp_gain * (height_error - self.setdown_ramp_offset_m).unsqueeze(-1),
            self.near_table_z_step_m,
            self.descend_step_m,
        )
        # A firmly gripped cube inherits every bit of wrist motion, so the whole set-down
        # keeps horizontal corrections at the landing-speed budget rather than only the
        # last two centimetres. Transport already closes the gap to a few millimetres.
        max_xy_step[setdown] = self.near_table_xy_step_m
        max_z_step[setdown] = setdown_ramp[setdown]
        # Transport already gets the cube within 3 mm of target. Stop correcting XY in
        # the last 5 mm of descent so table proximity cannot amplify a harmless
        # correction into a lateral contact impulse.
        final_descent = setdown & (height_error <= 0.005)
        desired[final_descent, :2] = finger[final_descent, :2]
        max_xy_step[final_descent] = 0.0

        # v2 guide: brake measured downward momentum before contact rather than relying
        # only on a position ramp. A short upward setpoint pulse is proportional to the
        # excess velocity, capped well below the coarse motion step. It activates only
        # inside the declared near-table band and therefore cannot perturb transport.
        if self.landing_brake_speed_m_s is not None:
            down_speed = torch.clamp(-obj_vel[:, 2], min=0.0)
            brake = setdown & (height_error <= self.near_table_margin_m) & (down_speed > self.landing_brake_speed_m_s)
            brake_step = torch.clamp(
                (down_speed - self.landing_brake_speed_m_s) * float(self.env.ctrl_dt),
                min=self.near_table_z_step_m,
                max=self.landing_brake_max_step_m,
            ).unsqueeze(-1)
            desired[brake, 2] = finger[brake, 2] + brake_step[brake, 0]
            max_z_step[brake] = brake_step[brake]

        retreat = phase == PHASE_RETREAT
        desired[retreat, 2] = grasp_z + self.retreat_height_m
        max_xy_step[retreat] = 0.0
        max_z_step[retreat] = self.retreat_step_m
        gap_target[retreat] = self.open_gap_m

        return desired, max_xy_step, max_z_step, gap_target
