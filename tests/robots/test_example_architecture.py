"""Architectural boundaries for the task-oriented public examples."""

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
        "examples/rl/pick_place/env.py",
        "examples/rl/pick_place/contacts.py",
        "examples/rl/pick_place/expert.py",
        "examples/rl/pick_place/evaluate.py",
        "examples/rl/pick_place/metrics.py",
        "examples/rl/pick_place/recipe.yaml",
        "examples/rl/pick_place/runtime.py",
        "examples/rl/pick_place/run_seeds.py",
        "examples/rl/pick_place/train.py",
        "examples/rl/pick_place/visualize.py",
        "examples/rl/pick_place/README.md",
        "examples/rl/pick_place/README_cn.md",
        "examples/rl/reach/env.py",
        "examples/rl/reach/evaluate.py",
        "examples/rl/reach/recipe.yaml",
        "examples/rl/reach/train.py",
        "examples/rl/reach/visualize.py",
        "examples/rl/reach/realtime_env.py",
        "examples/rl/reach/realtime_viewer.py",
        "examples/rl/reach/runtime.py",
        "examples/rl/reach/run_experiments.py",
        "examples/rl/reach/README.md",
        "examples/rl/reach/README_cn.md",
    }
    assert all((PROJECT_ROOT / relative).is_file() for relative in expected)
    assert not (PROJECT_ROOT / "examples" / "rl" / "pick_place" / "pretrained").exists()


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


def test_public_runtime_has_no_development_imports():
    import ast

    root = Path(__file__).resolve().parents[2]
    for path in (root / "ufactory").rglob("*.py"):
        if path.name == "legacy_warm_start.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("dev."), path
            elif isinstance(node, ast.Import):
                assert all(not alias.name.startswith("dev.") for alias in node.names), path
