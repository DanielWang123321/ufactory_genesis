"""RSL-RL runner with task-independent metrics and checkpoint manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rsl_rl.runners import OnPolicyRunner
import torch

from ufactory.training.artifacts import write_checkpoint_manifest


class ArtifactOnPolicyRunner(OnPolicyRunner):
    """Write the integrity manifest immediately after every RSL-RL checkpoint."""

    checkpoint_training_config: dict[str, Any] | None = None
    checkpoint_env: Any | None = None

    def __init__(self, *args: Any, executor_action_contract: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.executor_action_contract = executor_action_contract
        original_log = self.logger.log

        def log_with_metrics(**metrics: Any) -> None:
            episodes = self.logger.ep_extras
            keys = {key for episode in episodes for key in episode}
            values = {
                key: torch.cat(
                    [torch.as_tensor(episode[key]).flatten().cpu() for episode in episodes if key in episode]
                )
                for key in keys
            }
            report = {
                "iteration": metrics["it"],
                "completed_episodes": values["success_rate"].numel() if "success_rate" in values else 0,
                "collection_s": metrics["collect_time"],
                "learning_s": metrics["learn_time"],
                "learning_rate": metrics["learning_rate"],
                "action_std": metrics["action_std"].mean().item(),
                "loss": metrics["loss_dict"],
                "episode": {key: value.float().mean().item() for key, value in values.items()},
            }
            original_log(**metrics)
            with (Path(self.logger.log_dir) / "training_metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(report, allow_nan=False) + "\n")

        self.logger.log = log_with_metrics

    def save(self, path: str, infos: dict[str, Any] | None = None) -> None:
        saved_infos = {} if infos is None else dict(infos)
        if self.checkpoint_env is not None:
            saved_infos["ufactory_env_state"] = self.checkpoint_env.training_state_dict()
        super().save(path, saved_infos)
        if self.checkpoint_training_config is not None:
            write_checkpoint_manifest(
                Path(path),
                training_config=self.checkpoint_training_config,
                executor_action_contract=self.executor_action_contract,
            )
