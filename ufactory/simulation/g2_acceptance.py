"""Standalone G2 grasp acceptance: 33-case suite, free-cube grasp and CSV samples."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ufactory.simulation.g2_model import G2Model, checked, sdk_speed_rpm, speed_mm_s

if TYPE_CHECKING:
    from ufactory.simulation.g2_scene import Sample


def summarize(samples, *, support_removed: bool, release_z_mm: float | None) -> dict:
    hold = [sample for sample in samples if sample.phase == "hold"]
    steady = hold[-50:]
    force = [sample.sensor_force_n for sample in steady]
    expected = steady[-1].expected_static_force_n if steady else 0.0
    stable_force = sum(force) / len(force) if force else 0.0
    drop = release_z_mm - hold[-1].object_z_mm if hold and release_z_mm is not None else None
    held = bool(support_removed and hold and drop < 15 and min(hold[-1].left_normal_n, hold[-1].right_normal_n) > 0.2)
    return {
        "held": held,
        "support_removed": support_removed,
        "steady_window_s": 1.0,
        "stable_force_n": stable_force,
        "stable_force_min_n": min(force, default=0.0),
        "stable_force_max_n": max(force, default=0.0),
        "expected_static_force_n": expected,
        "stable_force_error_n": abs(stable_force - expected),
        "contact_peak_n": max((sample.sensor_force_n for sample in samples), default=0.0),
        "left_contact_peak_n": max((sample.left_normal_n for sample in samples), default=0.0),
        "right_contact_peak_n": max((sample.right_normal_n for sample in samples), default=0.0),
        "drive_torque_peak_nm": max((abs(sample.drive_torque_nm) for sample in samples), default=0.0),
        "motor_torque_estimate_peak_nm": max((abs(sample.motor_torque_estimate_nm) for sample in samples), default=0.0),
        "drop_mm": drop,
    }


def grasp_cube_samples(
    model: G2Model,
    *,
    position_mm: float = 0,
    speed_percent: float = 50,
    force_percent: float = 50,
    cube_size_mm: float = 30,
    show_viewer: bool = False,
    timeout_s: float | None = None,
) -> tuple[Any, list[Sample], dict]:
    """Close on a free cube, remove the support after a held contact, then hold."""

    from ufactory.simulation.g2_scene import DT, G2Scene

    simulation = G2Scene(model, show_viewer=show_viewer, object_gap_mm=cube_size_mm, free_cube=True)
    simulation.command(84, speed_percent, force_percent)
    samples = simulation.advance(0.4, "settle")
    simulation.command(position_mm, speed_percent, force_percent)
    if timeout_s is None:
        speed = speed_mm_s(speed_percent)
        timeout_s = max(3.0, 84 / speed + 2.0) if speed else 3.0
    stable_steps = 0
    release_z_mm = None
    for _ in range(round(timeout_s / DT)):
        sample = simulation.tick("close")
        samples.append(sample)
        bilateral = min(sample.left_normal_n, sample.right_normal_n) >= 0.2
        stable_steps = stable_steps + 1 if bilateral else 0
        if stable_steps * DT >= 0.4:
            release_z_mm = sample.object_z_mm
            simulation.remove_support()
            print(
                f"撤支撑 t={sample.time_s:.2f} s，较弱法向力 {min(sample.left_normal_n, sample.right_normal_n):.2f} N",
                flush=True,
            )
            break
    if simulation.support_removed:
        samples.extend(simulation.advance(3.0, "hold"))
    result = summarize(samples, support_removed=simulation.support_removed, release_z_mm=release_z_mm)
    return simulation, samples, result


def run(
    *,
    show_viewer: bool,
    model: G2Model,
    position_mm: float = 0,
    speed_percent: float = 50,
    force_percent: float = 50,
    cube_size_mm: float = 30,
    csv_path: Path | None = None,
) -> dict:
    from ufactory.simulation.g2_scene import write_samples

    simulation, samples, result = grasp_cube_samples(
        model,
        position_mm=position_mm,
        speed_percent=speed_percent,
        force_percent=force_percent,
        cube_size_mm=cube_size_mm,
        show_viewer=show_viewer,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if csv_path is not None:
        write_samples(csv_path, samples)
        summary_path = csv_path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"记录：{csv_path}")
    if show_viewer:
        print("保持位置指令，Ctrl+C 结束。")
        try:
            while True:
                simulation.tick("hold")
        except KeyboardInterrupt:
            pass
    return result


def force_result(samples, *, reference=False) -> dict:
    steady = samples[-50:]
    expected = steady[-1].curve_force_n if reference else steady[-1].expected_static_force_n
    force = np.array([s.sensor_force_n for s in steady])
    error = abs(float(force.mean()) - expected)
    tolerance = max(2.0, 0.05 * expected)
    torque_ok = all(abs(s.drive_torque_nm) <= s.torque_limit_nm + 1e-5 for s in samples)
    bilateral = all(min(s.left_normal_n, s.right_normal_n) > 0.2 for s in steady)
    return {
        "kind": "force",
        "passed": error <= tolerance and torque_ok and bilateral,
        "speed_percent": steady[-1].speed_percent,
        "force_percent": steady[-1].force_percent,
        "stable_force_n": float(force.mean()),
        "stable_force_min_n": float(force.min()),
        "stable_force_max_n": float(force.max()),
        "stable_force_std_n": float(force.std()),
        "stable_left_normal_n": float(np.mean([s.left_normal_n for s in steady])),
        "stable_right_normal_n": float(np.mean([s.right_normal_n for s in steady])),
        "bilateral_contact": bilateral,
        "expected_force_n": expected,
        "force_error_n": error,
        "force_tolerance_n": tolerance,
        "contact_peak_n": max(s.sensor_force_n for s in samples),
        "left_contact_peak_n": max(s.left_normal_n for s in samples),
        "right_contact_peak_n": max(s.right_normal_n for s in samples),
        "gap_mm": steady[-1].gap_mm,
        "torque_within_limit": torque_ok,
        "torque_peak_nm": max(abs(s.drive_torque_nm) for s in samples),
        "calibration_speed_rpm": steady[-1].calibration_speed_rpm,
        "outside_calibration": steady[-1].outside_calibration,
    }


def free_cases(model, save):
    from ufactory.simulation.g2_scene import G2Scene

    simulation = G2Scene(model)
    for speed in (10, 50, 100):
        simulation.reset()
        simulation.command(84, speed, 100)
        simulation.advance(0.5, "settle")
        for direction, target in (("close", 0), ("open", 84)):
            simulation.command(target, speed, 100)
            samples = simulation.advance(84 / speed_mm_s(speed) + 1.5, direction)
            central = [s for s in samples if 25 <= s.gap_mm <= 65]
            measured_speed = (
                abs(float(np.polyfit([s.time_s for s in central], [s.gap_mm for s in central], 1)[0]))
                if len(central) >= 3
                else 0.0
            )
            speed_error = abs(measured_speed / speed_mm_s(speed) - 1)
            central_peak_error = max(
                (abs(abs(s.gap_velocity_mm_s) / speed_mm_s(speed) - 1) for s in central), default=1.0
            )
            position_error = abs(samples[-1].gap_mm - target)
            save(
                f"free_s{speed}_{direction}",
                samples,
                {
                    "kind": "free",
                    "passed": speed_error <= 0.10 and central_peak_error <= 0.10 and position_error <= 1,
                    "speed_percent": speed,
                    "position_mm": target,
                    "measured_speed_mm_s": measured_speed,
                    "requested_speed_mm_s": speed_mm_s(speed),
                    "speed_relative_error": speed_error,
                    "central_speed_max_relative_error": central_peak_error,
                    "position_error_mm": position_error,
                    "gap_mm": samples[-1].gap_mm,
                    "outside_calibration": samples[-1].outside_calibration,
                },
            )
    simulation.reset()
    simulation.command(0, 50, 100)
    before = simulation.advance(0.3, "moving")
    simulation.command(0, 0, 100)
    paused = simulation.advance(1.5, "pause")
    pause_drift = max(abs(s.gap_mm - before[-1].gap_mm) for s in paused)
    reference_frozen = max(s.reference_gap_mm for s in paused) - min(s.reference_gap_mm for s in paused) <= 1e-9
    stopped = abs(paused[-1].gap_velocity_mm_s) <= 1.0
    simulation.command(0, 50, 100)
    resumed = simulation.advance(2.0, "resume")
    save(
        "free_pause_resume",
        before + paused + resumed,
        {
            "kind": "behavior",
            "passed": reference_frozen and stopped and resumed[-1].gap_mm <= 1,
            "pause_drift_mm": pause_drift,
            "position_error_mm": resumed[-1].gap_mm,
            "reference_frozen": reference_frozen,
            "pause_final_velocity_mm_s": paused[-1].gap_velocity_mm_s,
        },
    )


def reference_cases(model, save):
    from ufactory.simulation.g2_scene import G2Scene

    # Let the specimen center itself along Y; a fixed specimen would absorb
    # unequal lateral reactions, despite having the same average force.
    simulation = G2Scene(model, object_gap_mm=model.reference_gap_mm, free_cube=True, sensor_block=True)
    for speed in (10, 50, 90):
        for force in (10, 30, 50, 80, 100):
            simulation.reset()
            simulation.command(84, speed, force)
            simulation.advance(0.2, "settle")
            simulation.command(0, speed, force)
            samples = simulation.advance(84 / speed_mm_s(speed) + 3.0, "hold")
            save(f"reference_s{speed}_f{force}", samples, force_result(samples, reference=True))
    simulation.reset()
    simulation.command(0, 50, 30)
    samples = simulation.advance(4.0, "close")
    for force in (80, 10):
        simulation.command(0, 50, force)
        changed = simulation.advance(2.0, "force_change")
        samples.extend(changed)
        save(f"change_force_f{force}", changed, force_result(changed, reference=True))
    simulation.command(0, 0, 10)
    paused = simulation.advance(1.5, "pause")
    samples.extend(paused)
    result = force_result(paused, reference=True)
    result["pause_drift_mm"] = abs(paused[-1].gap_mm - paused[0].gap_mm)
    result["passed"] = result["passed"] and result["pause_drift_mm"] <= 1 and paused[-1].motor_speed_command_rpm == 0
    save("grasp_pause", paused, result)
    simulation.command(0, 0, 80)
    changed = simulation.advance(2.0, "paused_force_change")
    samples.extend(changed)
    save("paused_change_force_f80", changed, force_result(changed, reference=True))
    simulation.command(84, 0, 80)
    queued = simulation.advance(1.5, "paused_open_command")
    samples.extend(queued)
    result = force_result(queued, reference=True)
    result["reference_frozen"] = (
        max(s.reference_gap_mm for s in queued) - min(s.reference_gap_mm for s in queued) <= 1e-9
    )
    result["passed"] = result["passed"] and result["reference_frozen"]
    save("paused_position_change", queued, result)
    simulation.command(0, 50, 50)
    resumed = simulation.advance(1.5, "resume")
    samples.extend(resumed)
    save("grasp_resume", resumed, force_result(resumed, reference=True))
    simulation.command(84, 50, 50)
    opened = simulation.advance(2.0, "open")
    samples.extend(opened)
    save(
        "grasp_reverse_open",
        samples,
        {
            "kind": "behavior",
            "passed": abs(opened[-1].gap_mm - 84) <= 1 and opened[-1].sensor_force_n <= 0.2,
            "position_error_mm": abs(opened[-1].gap_mm - 84),
            "stable_force_n": opened[-1].sensor_force_n,
        },
    )


def thickness_cases(model, save):
    from ufactory.simulation.g2_scene import G2Scene

    for thickness in (10, 50, 70):
        simulation = G2Scene(model, object_gap_mm=thickness, free_cube=True, sensor_block=True)
        simulation.command(0, 50, 80)
        samples = simulation.advance(4.0, "hold")
        result = force_result(samples)
        result["object_gap_mm"] = thickness
        save(f"thickness_{thickness}mm", samples, result)


def cube_case(model, save):
    """Free-cube grasp with support removal, on the same path as the grasp CLI."""
    _, samples, result = grasp_cube_samples(model, timeout_s=4.0)
    result["kind"] = "cube"
    result["passed"] = result["held"] and result["stable_force_error_n"] <= max(
        2, result["expected_static_force_n"] * 0.05
    )
    save("cube_30mm_support_removed", samples, result)


def grasp_main() -> None:
    parser = argparse.ArgumentParser(description="Gripper G2 百分比控制，夹持方块后撤支撑")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--position-mm", type=float, default=0, help="指垫间隙目标，0–84 mm")
    parser.add_argument("--speed-percent", type=float, default=50, help="0–100%%；0 暂停位置进给")
    parser.add_argument("--force-percent", type=float, default=50, help="1–100%%，按官方曲线限制关节力矩")
    parser.add_argument("--cube-size-mm", type=float, default=30)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    try:
        checked(args.position_mm, 0, 84, "position_mm")
        speed = speed_mm_s(args.speed_percent)
        checked(args.force_percent, 1, 100, "force_percent")
        checked(args.cube_size_mm, 1, 83, "cube_size_mm")
    except ValueError as exc:
        parser.error(str(exc))
    model = G2Model.load()
    rpm = sdk_speed_rpm(speed)
    calibration_rpm = rpm or sdk_speed_rpm(speed_mm_s(50))
    lookup = model.curve.lookup(calibration_rpm, args.force_percent)
    print(
        f"位置 {args.position_mm:g} mm，速度 {args.speed_percent:g}% → {speed:g} mm/s / {rpm} rpm，力 {args.force_percent:g}%"
    )
    print(
        f"参考原始位置 230 → {model.reference_gap_mm:.6f} mm（SDK 源码推断）；标定稳定力 {lookup.force_n:.2f} N，关节限幅 ±{model.torque_limit_nm(lookup):.3f} N·m"
    )
    if lookup.outside_calibration:
        print("转速超出标定范围：力标定使用 3500 rpm 边界曲线，运动速度按输入执行。")
    import genesis as gs
    from ufactory.simulation import genesis_backend_constant
    from ufactory.simulation.compat import require_genesis_runtime

    require_genesis_runtime(gs)
    gs.init(backend=genesis_backend_constant(gs, args.backend), logging_level="error")
    result = run(
        show_viewer=not args.headless,
        model=model,
        position_mm=args.position_mm,
        speed_percent=args.speed_percent,
        force_percent=args.force_percent,
        cube_size_mm=args.cube_size_mm,
        csv_path=args.csv,
    )
    if args.headless:
        raise SystemExit(0 if result["held"] else 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="G2 百分比控制仿真验收")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--suite", choices=("all", "free", "reference", "thickness", "cube"), default="all")
    parser.add_argument("--out", type=Path, default=Path("reports/g2/validation"))
    args = parser.parse_args()
    model = G2Model.load()
    import genesis as gs
    from ufactory.simulation import genesis_backend_constant
    from ufactory.simulation.compat import require_genesis_runtime
    from ufactory.simulation.g2_scene import write_samples

    require_genesis_runtime(gs)
    gs.init(backend=genesis_backend_constant(gs, args.backend), logging_level="error")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.backend == "gpu" and str(gs.device) == "cpu":
        raise RuntimeError("GPU acceptance cannot fall back to CPU")
    from importlib import metadata
    from ufactory.simulation.g2_model import g2_physics_summary

    (args.out / "run.json").write_text(
        json.dumps(
            {
                "genesis_version": metadata.version("genesis-world"),
                "device": str(gs.device),
                "requested_backend": args.backend,
                "physics": g2_physics_summary(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    results = []

    def save(name, samples, result):
        result["case"] = name
        result["csv"] = f"{name}.csv"
        write_samples(args.out / result["csv"], samples)
        results.append(result)
        (args.out / "results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        fields = sorted({key for row in results for key in row})
        with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    suites = {"free": free_cases, "reference": reference_cases, "thickness": thickness_cases, "cube": cube_case}
    for name, suite in suites.items():
        if args.suite in ("all", name):
            suite(model, save)
    passed = all(row["passed"] for row in results)
    print(f"验收：{sum(row['passed'] for row in results)}/{len(results)} 通过；记录 {args.out}", flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
