"""Regressions for Bio Gripper G2 pick-and-place deferred startup, rounded transport, and level return."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ufactory.trajectory import bio_g2_replay as pick_place


@pytest.fixture
def transport_phases():
    return tuple(p for p in pick_place.default_phases() if p.name in pick_place.TRANSPORT_PHASE_NAMES)


@pytest.fixture
def ik_sim():
    sim = object.__new__(pick_place.XArm6BioG2PickPlaceSimulation)
    sim.arm_dof_idx = list(range(6))
    sim.transport_speed_mm_s = pick_place.DEFAULT_TRANSPORT_SPEED_MM_S

    def solve(grasp_xyz, seed_q=None):
        xyz = np.asarray(grasp_xyz)
        return np.r_[xyz, xyz * 0.5]

    sim.solve_arm_ik = solve
    return sim


@pytest.mark.parametrize("substeps", [8, 32])
def test_scene_is_hidden_even_for_visual_run(monkeypatch, substeps):
    import genesis as gs

    class SceneRequested(Exception):
        pass

    def capture_scene(**kwargs):
        assert kwargs["show_viewer"] is False
        assert kwargs["sim_options"].substeps == substeps
        assert kwargs["sim_options"].dt == 0.02
        assert kwargs["viewer_options"].refresh_rate == 30
        assert kwargs["viewer_options"].realtime_factor is None
        assert kwargs["rigid_options"].constraint_timeconst == 0.005
        assert kwargs["rigid_options"].tolerance == 1e-6
        raise SceneRequested

    monkeypatch.setattr(gs, "_initialized", True)
    monkeypatch.setattr(gs, "backend", gs.cpu, raising=False)
    monkeypatch.setattr(gs, "device", SimpleNamespace(type="cpu"), raising=False)
    monkeypatch.setattr(gs, "np_float", np.float64, raising=False)
    monkeypatch.setattr(gs, "Scene", capture_scene)
    with pytest.raises(SceneRequested):
        pick_place.XArm6BioG2PickPlaceSimulation(backend="cpu", headless=False, substeps=substeps)


def test_prepare_resets_exact_home_before_opening_viewer(ik_sim):
    events = []
    compile_trajectory = ik_sim.compile_trajectory

    def compile_hidden(phases):
        events.append("compile")
        return compile_trajectory(phases)

    def reset(home, initial_gap_mm):
        events.append("reset_and_warmup")
        np.testing.assert_array_equal(home, ik_sim.solve_arm_ik(pick_place.HOME_GRASP_POS))
        assert initial_gap_mm == 150.0

    ik_sim.compile_trajectory = compile_hidden
    ik_sim.reset = reset
    ik_sim.start_viewer = lambda: events.append("present_initial_frame")
    ticks = ik_sim.prepare(pick_place.default_phases())
    assert events == ["compile", "reset_and_warmup", "present_initial_frame"]
    assert np.linalg.norm(ticks[0]["target_arm_q"] - ik_sim.home_arm_q) > 1e-6


def test_reset_warms_up_without_visual_updates(ik_sim):
    calls = []

    def record(name):
        return lambda *args, **kwargs: calls.append((name, args, kwargs))

    ik_sim.headless = False
    ik_sim.scene = SimpleNamespace(reset=record("scene_reset"), step=record("step"))
    ik_sim.robot = SimpleNamespace(
        geoms=[],
        set_dofs_position=record("arm_position"),
        set_dofs_velocity=record("arm_velocity"),
        control_dofs_position_velocity=record("arm_hold"),
    )
    ik_sim.reagent_kit = SimpleNamespace(
        geoms=[],
        set_pos=record("kit_position"),
        set_quat=record("kit_quat"),
        set_dofs_velocity=record("kit_velocity"),
    )
    ik_sim.controller = SimpleNamespace(reset=record("gripper_reset"), apply=record("gripper_apply"))
    ik_sim.control_arm = record("compensated_arm_hold")
    home = ik_sim.solve_arm_ik(pick_place.HOME_GRASP_POS)
    ik_sim.reset(home)
    np.testing.assert_array_equal(next(c[1][0] for c in calls if c[0] == "arm_position"), home)
    steps = [c for c in calls if c[0] == "step"]
    assert len(steps) == 10
    assert all(c[2] == {"update_visualizer": False} for c in steps)
    assert ik_sim.time_s == 0.0


@pytest.mark.parametrize("headless", [False, True])
def test_viewer_waits_for_first_present_without_stepping(monkeypatch, headless):
    from ufactory.visualization import viewer as viewer_module

    events = []
    window = SimpleNamespace(_uf_frame_ready=False)

    def update(*, force):
        assert force
        events.append("refresh")
        if events.count("refresh") == 2:
            window._uf_frame_ready = True

    viewer = SimpleNamespace(_pyrender_viewer=window, is_alive=lambda: True, update=update)
    sim = object.__new__(pick_place.XArm6BioG2PickPlaceSimulation)
    sim.headless = headless
    sim.scene = SimpleNamespace(visualizer=SimpleNamespace(viewer=None))

    def start(scene):
        events.append("start_deferred_viewer")
        scene.visualizer.viewer = viewer

    monkeypatch.setattr(viewer_module, "start_deferred_viewer", start)
    monkeypatch.setattr(pick_place.time, "sleep", lambda _: None)
    sim.start_viewer()
    assert events == ([] if headless else ["start_deferred_viewer", "refresh", "refresh"])
    assert window._uf_frame_ready is (not headless)


@pytest.mark.parametrize("closed", [False, True])
def test_viewer_fails_if_initial_frame_cannot_be_presented(monkeypatch, closed):
    from ufactory.visualization import viewer as viewer_module

    sim = object.__new__(pick_place.XArm6BioG2PickPlaceSimulation)
    sim.headless = False
    viewer = SimpleNamespace(_pyrender_viewer=SimpleNamespace(), is_alive=lambda: not closed)
    sim.scene = SimpleNamespace(visualizer=SimpleNamespace(viewer=viewer))
    monkeypatch.setattr(viewer_module, "start_deferred_viewer", lambda _: None)
    times = iter((0.0, 31.0))
    monkeypatch.setattr(pick_place.time, "monotonic", lambda: next(times))
    with pytest.raises(RuntimeError, match="Viewer closed" if closed else "Timed out"):
        sim.start_viewer()


def test_transport_bezier_joins_are_c2_and_trimmed_by_20mm(transport_phases):
    segments = pick_place.transport_path_segments(transport_phases)
    assert len(segments) == 5
    for idx in (1, 3):
        controls = segments[idx]
        corner = np.asarray(transport_phases[idx // 2].end_grasp_xyz)
        assert np.linalg.norm(controls[0] - corner) == pytest.approx(0.020)
        assert np.linalg.norm(controls[-1] - corner) == pytest.approx(0.020)
        np.testing.assert_array_equal(segments[idx - 1][-1], controls[0])
        np.testing.assert_array_equal(segments[idx + 1][0], controls[-1])
        incoming = segments[idx - 1][-1] - segments[idx - 1][0]
        outgoing = segments[idx + 1][-1] - segments[idx + 1][0]
        start_velocity = 5 * (controls[1] - controls[0])
        end_velocity = 5 * (controls[-1] - controls[-2])
        for tangent, line in ((start_velocity, incoming), (end_velocity, outgoing)):
            assert np.linalg.norm(tangent) > 0
            np.testing.assert_allclose(tangent / np.linalg.norm(tangent), line / np.linalg.norm(line), atol=1e-12)
        np.testing.assert_allclose(20 * (controls[2] - 2 * controls[1] + controls[0]), 0, atol=1e-12)
        np.testing.assert_allclose(20 * (controls[-1] - 2 * controls[-2] + controls[-3]), 0, atol=1e-12)
        curve = pick_place.bezier_positions(controls, np.linspace(0, 1, 1001))
        bounds = np.array([controls[0], corner, controls[-1]])
        assert np.all(curve >= bounds.min(axis=0) - 1e-12)
        assert np.all(curve <= bounds.max(axis=0) + 1e-12)


def test_transport_arc_length_and_vertical_approach(transport_phases):
    xyz = pick_place.sample_transport_path(transport_phases, np.linspace(0, 1, 4001))
    np.testing.assert_array_equal(xyz[0], transport_phases[0].start_grasp_xyz)
    np.testing.assert_array_equal(xyz[-1], transport_phases[-1].end_grasp_xyz)
    step_lengths = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    assert step_lengths.max() / step_lengths.min() < 1.001
    blend_entry_z = transport_phases[0].end_grasp_xyz[2] - 0.020
    near_pick = xyz[(xyz[:, 2] < blend_entry_z - 1e-8) & (xyz[:, 1] < 0)]
    near_place = xyz[(xyz[:, 2] < blend_entry_z - 1e-8) & (xyz[:, 1] > 0)]
    assert len(near_pick) > 100 and len(near_place) > 100
    np.testing.assert_allclose(near_pick[:, :2], np.tile(pick_place.PICK_XY, (len(near_pick), 1)), atol=1e-12)
    np.testing.assert_allclose(near_place[:, :2], np.tile(pick_place.PLACE_XY, (len(near_place), 1)), atol=1e-12)
    assert xyz[:, 2].max() == pytest.approx(0.523)


def test_transport_duration_and_nonzero_phase_boundary_velocities(ik_sim):
    ticks = ik_sim.compile_trajectory(pick_place.default_phases())
    transport = [t for t in ticks if t["phase"] in pick_place.TRANSPORT_PHASE_NAMES]
    phases = tuple(p for p in pick_place.default_phases() if p.name in pick_place.TRANSPORT_PHASE_NAMES)
    timing = pick_place.transport_timing(phases, 250)
    assert len(transport) * pick_place.DT == pytest.approx(timing["duration_s"])
    assert len(transport) * pick_place.DT == pytest.approx(sum(p.duration_s for p in phases))
    np.testing.assert_array_equal(transport[0]["target_arm_dq"], 0)
    np.testing.assert_array_equal(transport[-1]["target_arm_dq"], 0)
    for idx in (i for i in range(1, len(transport)) if transport[i]["phase"] != transport[i - 1]["phase"]):
        assert transport[idx]["phase_step"] == 0
        before, after = transport[idx - 1]["target_arm_dq"], transport[idx]["target_arm_dq"]
        assert np.linalg.norm(before) > 0.05
        assert np.linalg.norm(after) > 0.05
        assert np.dot(before, after) > 0
        assert transport[idx - 1]["target_speed_mm_s"] == pytest.approx(250)
        assert transport[idx]["target_speed_mm_s"] == pytest.approx(250)


def test_retreat_reaches_548mm_before_level_return(ik_sim):
    phases = {p.name: p for p in pick_place.default_phases()}
    retreat, home = phases["retreat_up"], phases["return_home"]
    assert retreat.start_grasp_xyz[:2] == retreat.end_grasp_xyz[:2] == pick_place.PLACE_XY
    assert retreat.end_grasp_xyz == home.start_grasp_xyz
    assert home.start_grasp_xyz[2] == home.end_grasp_xyz[2] == pytest.approx(0.548)
    ticks = ik_sim.compile_trajectory(pick_place.default_phases())
    for phase_name in ("retreat_up", "return_home"):
        speeds = np.array([t["target_speed_mm_s"] for t in ticks if t["phase"] == phase_name])
        assert speeds.max() == pytest.approx(250.0)
    return_xyz = np.array([t["target_grasp_xyz"] for t in ticks if t["phase"] == "return_home"])
    np.testing.assert_allclose(return_xyz[:, 2], 0.548, atol=1e-12)
    np.testing.assert_allclose(return_xyz[-1], pick_place.HOME_GRASP_POS, atol=1e-12)


@pytest.fixture
def run_rows():
    rows = []
    for phase in pick_place.default_phases():
        for idx in range(3 if phase.name == "grasp_align" else 2):
            kit_z = pick_place.REAGENT_KIT_REST_Z * 1000
            grasp_z = phase.end_grasp_xyz[2] * 1000
            if phase.name in pick_place.TRANSPORT_PHASE_NAMES:
                kit_z = grasp_z = {
                    "lift": (470, 510),
                    "transit": (523, 503),
                    "place_descend": (480, 438.5),
                }[phase.name][idx]
            opened = phase.name in ("release", "retreat_up", "return_home", "home_settle")
            rows.append(
                {
                    "phase": phase.name,
                    "kit_x_mm": pick_place.PLACE_XY[0] * 1000,
                    "kit_y_mm": pick_place.PLACE_XY[1] * 1000,
                    "kit_z_mm": kit_z,
                    "grasp_x_mm": phase.end_grasp_xyz[0] * 1000,
                    "grasp_y_mm": phase.end_grasp_xyz[1] * 1000,
                    "grasp_z_mm": grasp_z,
                    "gap_mm": (114.0 if idx == 0 else 150.0) if opened else 110.0,
                    "left_normal_n": 0.0 if opened else pick_place.finger_force_n(50),
                    "right_normal_n": 0.0 if opened else pick_place.finger_force_n(50),
                    "nest_finger_contacts": 0,
                    "position_resets": 1,
                    "left_gap_mm": 32.26,
                    "right_gap_mm": 32.26,
                    "gap_difference_mm": 0.0,
                    "ee_error_mm": 0.0,
                    "relative_x_mm": 0.0,
                    "relative_y_mm": 0.0,
                    "relative_z_mm": 0.0,
                }
            )
    return rows


def test_slip_ignores_normal_vertical_path_motion(run_rows):
    evaluation = pick_place.evaluate_run(run_rows, force_percent=50)
    assert evaluation["passed"]
    assert evaluation["metrics"]["transport_vertical_slip_mm"] == pytest.approx(0)


@pytest.mark.parametrize("phase", pick_place.TRANSPORT_PHASE_NAMES)
def test_slip_detects_kit_moving_relative_to_grasp_center(run_rows, phase):
    next(row for row in run_rows if row["phase"] == phase)["kit_z_mm"] -= 3.1
    evaluation = pick_place.evaluate_run(run_rows, force_percent=50)
    assert not evaluation["checks"]["transit_vertical_slip_le_3mm"]
    assert evaluation["metrics"]["transport_vertical_slip_mm"] == pytest.approx(3.1)


@pytest.mark.parametrize("speed", [50, 166, 250, 500, 2000])
def test_transport_speed_limit_plateau_and_recomputed_duration(transport_phases, speed):
    timing = pick_place.transport_timing(transport_phases, speed)
    distances = timing["progress"] * timing["length_m"]
    average_speed = np.diff(np.r_[0.0, distances]) / pick_place.DT * 1000
    assert average_speed.max() <= speed + 1e-8
    assert timing["speeds_m_s"].max() * 1000 <= speed + 1e-8
    assert timing["speeds_m_s"][-1] == pytest.approx(0, abs=1e-12)
    assert np.all(np.diff(distances) >= 0)
    assert timing["progress"][-1] == pytest.approx(1)
    if speed <= 500:
        plateau = average_speed[np.isclose(average_speed, speed, atol=1e-8)]
        assert len(plateau) * pick_place.DT > 0.1
        assert 0.40 <= timing["ramp_s"] < 0.421
    phases = pick_place.default_phases(speed)
    assert sum(p.duration_s for p in phases if p.name in pick_place.TRANSPORT_PHASE_NAMES) == pytest.approx(
        timing["duration_s"]
    )


def alignment_row(left=32.26, right=32.26, ee=0.1):
    return {"phase": "grasp_align", "left_gap_mm": left, "right_gap_mm": right, "ee_error_mm": ee}


def test_alignment_requires_three_consecutive_samples():
    guard = pick_place.AlignmentGuard()
    for row in (alignment_row(), alignment_row(), alignment_row(right=31.0), alignment_row(), alignment_row()):
        guard.update(row)
        assert not guard.ready
    guard.update(alignment_row())
    assert guard.ready


@pytest.mark.parametrize(
    "row",
    [
        alignment_row(right=31.0),
        alignment_row(ee=0.501),
        alignment_row(left=-0.1, right=-0.1),
        alignment_row(ee=float("nan")),
    ],
)
def test_alignment_timeout_prevents_any_close_command(row):
    calls = []

    def step(tick, **kwargs):
        calls.append(tick)
        return row

    sim = SimpleNamespace(headless=True, step=step)
    ticks = [{"phase": "grasp_align"}] * 20 + [{"phase": "grip", "gripper_target_mm": 71}]
    with pytest.raises(pick_place.AlignmentError, match="timeout after 0.40s"):
        list(pick_place.execute_trajectory(sim, ticks, 166, 50))
    assert len(calls) == 20
    assert all(t["phase"] == "grasp_align" for t in calls)


def test_alignment_closes_as_soon_as_three_samples_pass():
    calls = []

    def step(tick, **kwargs):
        calls.append(tick)
        return alignment_row() | {"phase": tick["phase"]}

    sim = SimpleNamespace(headless=True, step=step)
    ticks = [{"phase": "grasp_align"}] * 20 + [{"phase": "grip", "gripper_target_mm": 71}]
    rows = list(pick_place.execute_trajectory(sim, ticks, 166, 50))
    assert len(rows) == 4
    assert calls[-1]["gripper_target_mm"] == 71


def test_grip_force_guard_advances_to_lift_after_sustained_hold():
    calls = []

    def step(tick, **kwargs):
        calls.append(tick)
        return alignment_row() | {
            "phase": tick["phase"],
            "left_normal_n": 11.8,
            "right_normal_n": 11.9,
            "gap_speed_mm_s": 0.0,
        }

    sim = SimpleNamespace(headless=True, step=step)
    ticks = (
        [{"phase": "grasp_align"}] * 20
        + [{"phase": "grip", "gripper_target_mm": 71}]
        + [{"phase": "grip", "gripper_target_mm": None}] * 79
        + [{"phase": "lift", "gripper_target_mm": None}] * 2
    )
    rows = list(pick_place.execute_trajectory(sim, ticks, 200, 50))
    grip_rows = [r for r in rows if r["phase"] == "grip"]
    lift_rows = [r for r in rows if r["phase"] == "lift"]
    assert len(grip_rows) == pick_place.GRIP_HOLD_STEPS
    assert len(lift_rows) == 2


def test_grip_timeout_prevents_lift_when_contact_force_is_insufficient():
    def step(tick, **kwargs):
        return alignment_row() | {
            "phase": tick["phase"],
            "left_normal_n": 2.0,
            "right_normal_n": 2.0,
            "gap_speed_mm_s": 0.0,
        }

    sim = SimpleNamespace(headless=True, step=step)
    ticks = (
        [{"phase": "grasp_align"}] * 3
        + [{"phase": "grip", "gripper_target_mm": 71}] * 10
        + [{"phase": "lift", "gripper_target_mm": None}]
    )
    with pytest.raises(pick_place.GripTimeoutError, match="grip timeout"):
        list(pick_place.execute_trajectory(sim, ticks, 200, 50))


def test_closing_push_and_3d_relative_motion_are_measured(run_rows):
    row = next(r for r in run_rows if r["phase"] == "grip")
    row["kit_x_mm"] += 0.31
    row = next(r for r in run_rows if r["phase"] == "transit")
    row["relative_x_mm"] = 1.2
    row["relative_y_mm"] = 1.6
    result = pick_place.evaluate_run(run_rows, 50)
    assert not result["checks"]["closing_horizontal_push_le_0_30mm"]
    assert result["metrics"]["transport_relative_displacement_3d_mm"] == pytest.approx(2)


@pytest.mark.parametrize(
    "error",
    [np.zeros(6), np.array([1.1e-5, 0, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 1.1e-4])],
)
def test_ik_tolerances_and_error_validation(error):
    sim = object.__new__(pick_place.XArm6BioG2PickPlaceSimulation)
    sim.arm_dof_idx = list(range(6))
    sim.adapter = SimpleNamespace(indices=[6, 7])
    sim.controller = SimpleNamespace(reference_q_m=0.02)
    sim.gripper_base_link = object()
    sim.last_arm_q = np.zeros(6)
    sim.max_ik_position_error_m = sim.max_ik_rotation_error_rad = 0.0

    def solve(**kwargs):
        assert kwargs["pos_tol"] == 1e-5
        assert kwargs["rot_tol"] == 1e-4
        assert kwargs["return_error"] is True
        return np.zeros(8), error

    sim.robot = SimpleNamespace(n_dofs=8, inverse_kinematics=solve)
    if np.any(error):
        with pytest.raises(RuntimeError, match="IK failed"):
            sim.solve_arm_ik(pick_place.HOME_GRASP_POS)
    else:
        np.testing.assert_array_equal(sim.solve_arm_ik(pick_place.HOME_GRASP_POS), 0.0)


def test_gravity_is_applied_as_position_bias_at_measured_configuration():
    calls = []
    sim = object.__new__(pick_place.XArm6BioG2PickPlaceSimulation)
    sim.arm_dof_idx = list(range(6))
    measured = np.arange(8) * 0.01
    sim.adapter = SimpleNamespace(indices=[7, 6], positions=lambda: measured)

    def gravity(positions):
        assert positions[pick_place.RIGHT] == measured[7]
        assert positions[pick_place.LEFT] == measured[6]
        assert positions["joint2"] == measured[1]
        return np.arange(6) + 1.0, 0.0

    sim.gravity = SimpleNamespace(compute=gravity)
    sim.robot = SimpleNamespace(
        get_dofs_position=lambda: measured,
        control_dofs_position_velocity=lambda *args: calls.append(args),
    )
    target = np.ones(6)
    dq = np.full(6, 0.1)
    sim.control_arm(target, dq)
    np.testing.assert_allclose(calls[0][0], target + (np.arange(6) + 1) / pick_place.XARM6_KP)
    np.testing.assert_array_equal(calls[0][1], dq)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_transport_cli_rejects_invalid_speed(value):
    with pytest.raises(SystemExit) as exc:
        pick_place.parse_args(["--transport-speed-mm-s", value])
    assert exc.value.code == 2


def test_substeps_cli_default_and_allowed_values():
    assert pick_place.parse_args([]).substeps == 8
    assert pick_place.parse_args([]).transport_speed_mm_s == 250
    assert pick_place.parse_args(["--substeps", "32"]).substeps == 32
    with pytest.raises(SystemExit):
        pick_place.parse_args(["--substeps", "16"])


@pytest.mark.parametrize(
    "position_delta, force_delta, passed",
    [(0.49, 0.19, True), (0.51, 0.19, False), (0.49, 0.21, False)],
)
def test_substep_comparison_enforces_position_and_force_thresholds(position_delta, force_delta, passed):
    first = {
        "substeps": 8,
        "backend": "gpu",
        "speed_mm_s": 166,
        "transport_speed_mm_s": 250,
        "force_percent": 50,
        "passed": True,
        "microplate": {"sha256": {"model": "same"}},
        "metrics": {
            "final_kit_xyz_mm": [380.0, 120.0, 437.175],
            "grip_steady_left_normal_n": 11.9,
            "grip_steady_right_normal_n": 11.9,
        },
    }
    second = first | {
        "substeps": 32,
        "metrics": first["metrics"]
        | {
            "final_kit_xyz_mm": [380.0 + position_delta, 120.0, 437.175],
            "grip_steady_left_normal_n": 11.9 + force_delta,
        },
    }
    assert pick_place.compare_substeps(first, second)["passed"] is passed


def test_joint_sample_is_reused_only_within_same_physics_step():
    sim_clock = SimpleNamespace(cur_step_global=0)
    reads = []
    adapter = object.__new__(pick_place.PickPlaceAdapter)
    adapter.scene = SimpleNamespace(sim=sim_clock)
    adapter.indices, adapter.right, adapter.left = [7, 6], 7, 6
    adapter._sample_step = None

    def positions():
        reads.append(sim_clock.cur_step_global)
        return np.arange(8) * 0.01 + sim_clock.cur_step_global

    adapter.entity = SimpleNamespace(
        get_dofs_position=positions,
        get_dofs_velocity=lambda _: np.array([0.1, -0.1]),
    )
    np.testing.assert_allclose(adapter.positions(), np.arange(8) * 0.01)
    assert adapter.state() == pytest.approx((0.07, 0.1, 0.06, -0.1))
    assert reads == [0]
    sim_clock.cur_step_global = 1
    assert adapter.state() == pytest.approx((1.07, 0.1, 1.06, -0.1))
    assert reads == [0, 1]


def test_playback_does_not_yield_or_sleep_when_computation_is_late(monkeypatch):
    clock_values = iter([0.0, 0.03, 0.06, 0.09, 0.12])
    sleeps = []
    monkeypatch.setattr(pick_place.time, "perf_counter", lambda: next(clock_values))
    monkeypatch.setattr(pick_place.time, "sleep", sleeps.append)
    sim = SimpleNamespace(headless=False, step=lambda tick, **kwargs: alignment_row() | {"phase": tick["phase"]})
    ticks = [{"phase": "grasp_align"}] * 20 + [{"phase": "grip", "gripper_target_mm": 71}]
    assert len(list(pick_place.execute_trajectory(sim, ticks, 166, 50))) == 4
    assert sleeps == []


def test_playback_resets_debt_after_slow_step_to_prevent_return_fast_forward(monkeypatch):
    # Step 1 is slow (takes 0.10s > DT=0.02s); step 2 is fast (takes 0.008s < DT=0.02s).
    # The pacer must sleep 0.012s after step 2 instead of fast-forwarding to pay back step 1's debt.
    clock_values = iter([0.0, 0.10, 0.108, 0.128, 0.148])
    sleeps = []
    monkeypatch.setattr(pick_place.time, "perf_counter", lambda: next(clock_values))
    monkeypatch.setattr(pick_place.time, "sleep", sleeps.append)
    sim = SimpleNamespace(headless=False, step=lambda tick, **kwargs: alignment_row() | {"phase": tick["phase"]})
    ticks = [{"phase": "grasp_align"}] * 20 + [{"phase": "grip", "gripper_target_mm": 71}]
    assert len(list(pick_place.execute_trajectory(sim, ticks, 166, 50))) == 4
    assert len(sleeps) == 3
    assert sleeps[0] == pytest.approx(0.012)
