from types import SimpleNamespace
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ufactory.config.models import G2SimulationConfig
from ufactory.trajectory.g2_replay import replay_g2
from ufactory.trajectory.segments import Program, Segment


@pytest.mark.parametrize("fault", [None, "grip", "release"])
def test_confirmation_controls_next_tick_and_timeout_saves_failure(fault, monkeypatch, tmp_path):
    import ufactory.trajectory.g2_replay as replay

    state = SimpleNamespace(command=84, phase_ticks={})
    position = torch.zeros(1, 3)
    robot = SimpleNamespace(
        get_dofs_position=lambda idx: torch.zeros(1, len(idx)), control_dofs_position=lambda *_: None
    )
    obj = SimpleNamespace(get_pos=lambda: position)
    adapter = SimpleNamespace(
        controller=SimpleNamespace(reference_gap_mm=torch.tensor([84.0], dtype=torch.float64)),
        reset=lambda *_: None,
        gap_mm=lambda: 84,
        apply=lambda *_: None,
        command=lambda p, *_: setattr(state, "command", p),
    )
    ctx = SimpleNamespace(
        robot=robot,
        scene=SimpleNamespace(step=lambda: None),
        obj=obj,
        ik_link=SimpleNamespace(get_pos=lambda: position),
        arm_dof_idx=[0],
        gripper_dof_idx=[1],
        home_qpos=np.zeros(2),
        home_pos_base=(0, 0, 0),
        obj_pos_base=(0, 0, 0),
        place_pos_base=(0, 0, 0),
        obj_size=(0.03, 0.03, 0.03),
        base_to_world=lambda p: p,
        g2_settings=G2SimulationConfig(),
        g2_controller=adapter,
    )

    def record(ctx, adapter, output, phase, time_s, previous, q_arm, dt):
        state.phase_ticks[phase] = state.phase_ticks.get(phase, 0) + 1
        closed = state.command == 0
        gap = 30 if closed else 84
        force = 10 if closed else 0
        if fault == "grip":
            force = 0
        if fault == "release" and phase == "release":
            gap, force = 30, 10
        row = dict(
            time_s=time_s,
            phase=phase,
            gap_mm=gap,
            left_normal_n=force,
            right_normal_n=force,
            expected_static_force_n=10,
            sensor_force_n=force,
            drive_torque_nm=1,
            torque_limit_nm=2,
            motor_torque_estimate_nm=0.1,
            object_speed_mm_s=0,
            home_error_mm=0,
        )
        for prefix in ("object_base", "object_in_pad", "object_in_tool"):
            row.update({f"{prefix}_{axis}_mm": 0 for axis in "xyz"})
        return row

    monkeypatch.setattr(replay, "_record", record)
    monkeypatch.setattr("ufactory.trajectory.sim_executor._assert_finite_replay_state", lambda *_a, **_kw: None)
    segments = [
        Segment("gripper", 2, 0, 0, label="grip", gap_start=0.084, gap_end=0),
        Segment("movej", 0.02, 1, 1, label="lift", q_start=np.zeros(1), q_end=np.zeros(1)),
        Segment("gripper", 2, 0, 0, label="release", gap_start=0, gap_end=0.084),
        Segment("movej", 0.02, 1, 1, label="retreat", q_start=np.zeros(1), q_end=np.zeros(1)),
    ]
    path = tmp_path / "result.json"
    report = replay_g2(
        Program(rate=50, segments=segments),
        ctx,
        final_metric_settle_s=0,
        csv_path=path.with_suffix(".csv"),
        sim_report_path=path,
    )
    saved = json.loads(path.read_text())
    if fault:
        assert not report.completed and report.timeout_phase == fault
        assert report.records[-1]["phase"] == fault
        assert state.phase_ticks[fault] == 100
        assert "retreat" not in state.phase_ticks
        if fault == "grip":
            assert "lift" not in state.phase_ticks
        assert not saved["completed"]
    else:
        assert report.completed
        assert state.phase_ticks["grip"] == state.phase_ticks["release"] == 2
        assert report.confirmations["grip"]["next_action_delay_s"] == 0.02
        assert report.confirmations["release"]["next_action_delay_s"] == 0.02
    assert path.with_suffix(".csv").read_text().count("\n") == len(report.records) + 1
