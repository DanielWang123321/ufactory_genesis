"""Architectural boundaries for the task-oriented v0.2.7 examples."""

from __future__ import annotations

from pathlib import Path
import tomllib

from conftest import PROJECT_ROOT


def _python_sources(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)


def test_production_package_does_not_import_examples() -> None:
    failures = []
    for source in _python_sources(PROJECT_ROOT / "ufactory"):
        text = source.read_text(encoding="utf-8")
        if "from examples" in text or "import examples" in text:
            failures.append(source.relative_to(PROJECT_ROOT).as_posix())
    assert failures == []


def test_public_examples_do_not_modify_python_path_or_keep_bootstraps() -> None:
    failures = []
    for source in _python_sources(PROJECT_ROOT / "examples"):
        text = source.read_text(encoding="utf-8")
        if "sys.path" in text or source.name == "_bootstrap.py":
            failures.append(source.relative_to(PROJECT_ROOT).as_posix())
    assert failures == []


def test_task_oriented_examples_exist_and_legacy_paths_are_removed() -> None:
    expected = {
        "examples/visualization/view_robot.py",
        "examples/visualization/view_gripper_g2.py",
        "examples/visualization/view_bio_gripper_g2.py",
        "examples/visualization/view_lite6_gripper.py",
        "examples/kinematics/verify_robot.py",
        "examples/kinematics/verify_fk.py",
        "examples/kinematics/verify_ik.py",
        "examples/pick_place/run.py",
        "examples/pick_place/runtime.example.yaml",
        "examples/packaging/run.py",
        "examples/packaging/runtime.example.yaml",
    }
    assert all((PROJECT_ROOT / relative).is_file() for relative in expected)
    assert not (PROJECT_ROOT / "examples" / "manipulation").exists()
    assert not (PROJECT_ROOT / "examples" / "_grasp_place_traj.py").exists()
    assert not (PROJECT_ROOT / "examples" / "_pick_place_traj.py").exists()
    assert not (PROJECT_ROOT / "examples" / "xarm6" / "xarm6_reach_deploy.py").exists()
    assert not any((PROJECT_ROOT / "examples").glob("*/_bootstrap.py"))
    assert not (PROJECT_ROOT / "examples" / "reinforcement_learning").exists()


def test_sim_and_rl_extras_do_not_install_pinocchio_or_coal() -> None:
    pyproject = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    for name in ("sim", "rl"):
        normalized = " ".join(extras[name]).lower()
        assert "pin==" not in normalized
        assert "coal==" not in normalized
    for name in ("real", "dynamics"):
        normalized = " ".join(extras[name]).lower()
        assert "pin==4.0.0" in normalized
        assert "coal==3.0.3" in normalized
