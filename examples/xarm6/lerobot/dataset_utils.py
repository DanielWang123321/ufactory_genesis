"""LeRobotDataset helpers (HuggingFace lerobot package)."""

from __future__ import annotations

from pathlib import Path

import constants
from xarm6_lerobot_features import assert_allowed_features

CAMERA_FEATURE_KEY = constants.CAMERA_FEATURE_KEY
CAMERA_HEIGHT = constants.CAMERA_HEIGHT
CAMERA_WIDTH = constants.CAMERA_WIDTH
DATASET_FPS = constants.DATASET_FPS
DEFAULT_TASK = constants.DEFAULT_TASK
ROBOT_TYPE = constants.ROBOT_TYPE
STATE_DIM = constants.STATE_DIM


def build_dataset_features() -> dict:
  features = {
      "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": None},
      "action": {"dtype": "float32", "shape": (STATE_DIM,), "names": None},
      CAMERA_FEATURE_KEY: {
          "dtype": "video",
          "shape": (CAMERA_HEIGHT, CAMERA_WIDTH, 3),
          "names": ["height", "width", "channels"],
      },
  }
  assert_allowed_features(features)
  return features


def create_lerobot_dataset(repo_id: str, root: Path | None = None, *, resume: bool = False):
    from hf_lerobot import get_lerobot_dataset_class

    LeRobotDataset = get_lerobot_dataset_class()
    if resume and root is not None and (root / "meta" / "info.json").is_file():
        ds = LeRobotDataset.resume(repo_id, root=root)
        meta_keys = set(ds.meta.features.keys())
        assert not any("tof" in k.lower() or "depth" in k.lower() or "point" in k.lower() for k in meta_keys)
        return ds

    features = build_dataset_features()
    kwargs = {
        "repo_id": repo_id,
        "fps": DATASET_FPS,
        "robot_type": ROBOT_TYPE,
        "features": features,
        "use_videos": True,
    }
    if root is not None:
        kwargs["root"] = root
    ds = LeRobotDataset.create(**kwargs)
    meta_keys = set(ds.meta.features.keys())
    assert not any("tof" in k.lower() or "depth" in k.lower() or "point" in k.lower() for k in meta_keys)
    return ds


def add_frame_to_dataset(dataset, rgb: object, state, action, task: str = DEFAULT_TASK) -> None:
    import constants

    dataset.add_frame(
        {
            "observation.state": state,
            "action": action,
            constants.CAMERA_FEATURE_KEY: rgb,
            "task": task,
        }
    )
