"""Visualize or record the xArm6 + Gripper G2 pick-and-place policy with GLB PBR rendering."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from .runtime import (
    DEFAULT_RECIPE,
    OUTPUT_ROOT,
    configure_runtime,
    patch_pyglet_screen,
    require_backend,
    require_checkpoint,
    require_rsl_runtime,
)

if __name__ == "__main__":
    configure_runtime(viewer=True)
else:
    configure_runtime()
    patch_pyglet_screen()
require_rsl_runtime()

import genesis as gs
from rsl_rl.runners import OnPolicyRunner
import torch

from ufactory.simulation.compat import require_genesis_runtime
from ufactory.training import (
    PICK_PLACE_RL_ROBOTS,
    build_pick_place_task_configs,
    load_pick_place_rl_runtime_config,
    validate_and_load_rsl_checkpoint,
)

from .env import XArm6PickPlaceEnv
from .evaluate import acceptance_passed, evaluate, save_results
from .expert import PickPlaceExpert


def _iter_num(path: Path) -> int:
    try:
        return int(path.stem.split("_", 1)[1])
    except ValueError:
        return -1


def latest_checkpoint(exp_name: str) -> Path:
    directory = OUTPUT_ROOT / exp_name
    for name in ("model_best.pt", "model_final.pt"):
        if (directory / name).is_file():
            return directory / name
    files = sorted(directory.glob("model_*.pt"), key=_iter_num)
    if not files:
        raise FileNotFoundError(f"no checkpoints found in {directory}")
    return files[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", "--ckpt", dest="checkpoint", type=Path)
    parser.add_argument("--expert", action="store_true", help="Visualize the heuristic expert policy")
    parser.add_argument("-e", "--exp-name", default="pick_place_e2e_s1")
    parser.add_argument("--robot", default="xarm6", choices=PICK_PLACE_RL_ROBOTS)
    parser.add_argument("--recipe", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("-B", "-b", "--num-envs", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=18000)
    parser.add_argument("--headless", action="store_true", help="Run without an interactive GUI window")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument(
        "--high-fidelity",
        dest="high_fidelity",
        action="store_true",
        default=True,
        help="Enable high-fidelity GLB PBR robot rendering (default: True)",
    )
    parser.add_argument("--low-poly", dest="high_fidelity", action="store_false")
    parser.add_argument("--record", type=Path, help="Optional output MP4 video path")
    parser.add_argument("--out", "--summary-json", dest="out", type=Path)
    args = parser.parse_args()

    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    torch.set_num_threads(4)
    runtime_config = load_pick_place_rl_runtime_config(args.robot, config_path=args.runtime_config)
    expected_env, expected_reward, expected_robot = build_pick_place_task_configs(
        args.robot,
        recipe_path=args.recipe,
        runtime_config_path=args.runtime_config,
    )

    loaded_state = None
    ckpt: Path | None = None
    if not args.expert:
        ckpt = require_checkpoint(args.checkpoint if args.checkpoint is not None else latest_checkpoint(args.exp_name))
        loaded_state, artifact, _manifest = validate_and_load_rsl_checkpoint(
            ckpt,
            ckpt.parent / "config.yaml",
            map_location="cpu",
            expected_task="pick_place",
            expected_robot_key=runtime_config.robot.key,
            expected_runtime_config_sha256=runtime_config.sha256,
            expected_runtime_env=expected_env,
        )
        env_cfg = deepcopy(artifact["env"])
        reward_cfg = deepcopy(artifact["reward"])
        robot_cfg = deepcopy(artifact["robot"])
        train_cfg = deepcopy(artifact["train"])
    else:
        env_cfg = deepcopy(expected_env)
        reward_cfg = deepcopy(expected_reward)
        robot_cfg = deepcopy(expected_robot)
        train_cfg = None

    env_cfg["num_envs"] = int(args.num_envs)
    require_genesis_runtime(gs)
    gs.init(
        backend=gs.cpu if args.backend == "cpu" else gs.gpu,
        precision="32",
        logging_level="warning",
        seed=int(args.seed),
        performance_mode=bool(env_cfg.get("genesis_performance_mode", False)),
    )
    require_backend(gs, args.backend)
    env = XArm6PickPlaceEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=not args.headless and args.record is None,
        capture=args.record is not None,
        high_fidelity=args.high_fidelity,
    )

    if args.expert:
        policy = PickPlaceExpert(env)
    else:
        assert train_cfg is not None and loaded_state is not None
        train_cfg["seed"] = int(args.seed)
        train_cfg["runner"]["max_iterations"] = 0
        train_cfg["runner"]["resume"] = False
        runner = OnPolicyRunner(env, train_cfg, log_dir=None, device=gs.device)
        runner.alg.load(loaded_state, None, strict=True)
        actor = runner.get_inference_policy(device=gs.device)

        def _policy(obs: Any) -> torch.Tensor:
            return actor(obs, stochastic_output=args.stochastic)

        policy = _policy

    if args.record is not None:
        assert env.camera is not None
        record_path = args.record.expanduser().resolve()
        record_path.parent.mkdir(parents=True, exist_ok=True)
        env.camera.start_recording(save_to_filename=str(record_path), fps=round(1 / env.ctrl_dt))
        try:
            summary, rows = evaluate(env, policy, episodes=args.episodes, seed=args.seed)
            summary["passed"] = acceptance_passed(summary)
            save_results(
                args.out or record_path.with_suffix(".json"),
                summary,
                rows,
                extra_metadata={
                    "checkpoint": None if ckpt is None else str(ckpt),
                    "video": str(record_path),
                    "high_fidelity": args.high_fidelity,
                    "stochastic": args.stochastic,
                },
            )
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        finally:
            env.camera.stop_recording()
            gs.destroy()
        print(f"Video saved to {record_path}", flush=True)
        return

    if args.headless:
        summary, rows = evaluate(env, policy, episodes=args.episodes, seed=args.seed)
        summary["passed"] = acceptance_passed(summary)
        gs.destroy()
        if args.out is not None:
            save_results(
                args.out,
                summary,
                rows,
                extra_metadata={"checkpoint": None if ckpt is None else str(ckpt), "stochastic": args.stochastic},
            )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    obs, _ = env.reset()
    try:
        with torch.inference_mode():
            while env.scene.viewer.is_alive():
                actions = policy(obs)
                obs, _, done, info = env.step(actions)
                for i in done.nonzero(as_tuple=False).flatten().tolist():
                    t = info["terminal"]
                    print(
                        f"env={i} success={bool(t['success'][i])} "
                        f"pos=({float(t['obj_x'][i]) * 1000:.1f}, {float(t['obj_y'][i]) * 1000:.1f})mm "
                        f"err={float(t['target_error_mm'][i]):.2f}mm "
                        f"drag={float(t['max_drag_mm'][i]):.2f}mm tilt={float(t['max_tilt_deg'][i]):.2f}deg",
                        flush=True,
                    )
    except KeyboardInterrupt:
        pass
    finally:
        gs.destroy()


if __name__ == "__main__":
    main()
