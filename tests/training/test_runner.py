"""Shared runner metrics and saved artifact behavior for both RL tasks."""

from types import SimpleNamespace
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")

from ufactory.training import load_training_config, write_training_config, validate_checkpoint_artifacts
from ufactory.training.runner import ArtifactOnPolicyRunner, OnPolicyRunner
from ufactory.training.pick_place_config import PICK_PLACE_ACTION_CONTRACT
from ufactory.config.models import G2_PHYSICS_PROFILE


@pytest.mark.parametrize(
    "task,contract,obs,actions",
    [
        ("reach", "normalized_joint_position_delta", 24, 6),
        ("pick_place", PICK_PLACE_ACTION_CONTRACT, 70, 4),
    ],
)
def test_shared_runner_logs_completed_episodes_and_saves_task_contract(
    tmp_path, monkeypatch, task, contract, obs, actions
):
    logged = []
    logger = SimpleNamespace(
        log_dir=str(tmp_path),
        ep_extras=[
            {"success_rate": torch.tensor([1.0, 0.0]), "distance": torch.tensor([0.002, 0.1])},
            {"success_rate": torch.tensor(1.0)},
        ],
        log=lambda **values: logged.append(values),
    )
    monkeypatch.setattr(OnPolicyRunner, "__init__", lambda self: setattr(self, "logger", logger))
    monkeypatch.setattr(OnPolicyRunner, "save", lambda self, path, infos: torch.save({"infos": infos}, path))
    runner = ArtifactOnPolicyRunner(executor_action_contract=contract)
    runner.checkpoint_env = SimpleNamespace(training_state_dict=lambda: {"counter": torch.tensor([7])})
    config_path = tmp_path / "config.yaml"
    write_training_config(
        config_path,
        task=task,
        robot_key="xarm6_1305",
        env={
            "task_name": task,
            "env_version": 3 if task == "pick_place" else 2,
            "num_obs": obs,
            "num_actions": actions,
            "physics_profile": G2_PHYSICS_PROFILE,
            "success_distance_m": 0.005,
        },
        reward={},
        robot={},
        train={},
    )
    runner.checkpoint_training_config = load_training_config(config_path)
    metrics = dict(
        it=4,
        collect_time=0.1,
        learn_time=0.2,
        learning_rate=3e-4,
        action_std=torch.ones(actions),
        loss_dict={"value": 0.5},
    )
    runner.logger.log(**metrics)
    report = json.loads((tmp_path / "training_metrics.jsonl").read_text())
    assert logged == [metrics]
    assert report["completed_episodes"] == 3
    assert report["episode"]["success_rate"] == pytest.approx(2 / 3)
    assert report["episode"]["distance"] == pytest.approx(0.051)
    infos = {"validation": "passed"}
    checkpoint = tmp_path / "model_4.pt"
    runner.save(str(checkpoint), infos)
    assert infos == {"validation": "passed"}
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["infos"]["validation"] == "passed"
    assert saved["infos"]["ufactory_env_state"]["counter"].tolist() == [7]
    _, manifest = validate_checkpoint_artifacts(checkpoint, config_path)
    assert manifest.executor_action_contract == contract
