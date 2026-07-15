#!/usr/bin/env python3
"""One-shot migration for explicitly trusted local cfgs.pkl artifacts."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys

from ufactory.training import load_training_config, write_checkpoint_manifest, write_training_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate a trusted legacy cfgs.pkl to v0.2.5 safe artifacts")
    parser.add_argument("legacy_cfgs")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", required=True, choices=("reach", "pick_place"))
    parser.add_argument("--robot-key", required=True)
    parser.add_argument("--action-contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--trusted-input",
        action="store_true",
        help="Required acknowledgement: pickle can execute arbitrary code",
    )
    args = parser.parse_args()
    if not args.trusted_input:
        parser.error("--trusted-input is required; never migrate an untrusted pickle")
    source = Path(args.legacy_cfgs).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(
        f"WARNING: unpickling explicitly trusted local input: {source}",
        file=sys.stderr,
    )
    with source.open("rb") as stream:
        loaded = pickle.load(stream)  # noqa: S301 - isolated, explicit trust boundary
    if not isinstance(loaded, (list, tuple)) or len(loaded) != 4:
        raise ValueError("legacy cfgs.pkl must contain env/reward/robot/train mappings")
    env, reward, robot, train = loaded
    write_training_config(
        output / "config.yaml",
        task=args.task,
        robot_key=args.robot_key,
        env=env,
        reward=reward,
        robot=robot,
        train=train,
    )
    artifact = load_training_config(output / "config.yaml")
    checkpoint = Path(args.checkpoint).resolve()
    manifest = write_checkpoint_manifest(
        checkpoint,
        training_config=artifact,
        executor_action_contract=args.action_contract,
        output_path=output / "checkpoint_manifest.json",
    )
    print(f"wrote {output / 'config.yaml'}")
    print(f"wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
