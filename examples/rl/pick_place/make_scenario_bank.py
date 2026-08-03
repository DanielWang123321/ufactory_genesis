"""Create a deterministic pick-place evaluation scenario bank."""

from __future__ import annotations

import argparse
from pathlib import Path

from ufactory.training import (
    PICK_PLACE_SCENARIO_MODES,
    build_pick_place_task_configs,
    generate_pick_place_scenario_bank,
    scenario_bank_sha256,
    write_scenario_bank,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=Path(__file__).with_name("recipe.yaml"))
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument(
        "--mode",
        choices=PICK_PLACE_SCENARIO_MODES,
        default="fixed",
        help="Layout distribution; object_* modes keep the target fixed.",
    )
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    env_cfg, _reward_cfg, _robot_cfg = build_pick_place_task_configs(
        "xarm6",
        recipe_path=args.recipe,
        runtime_config_path=args.runtime_config,
    )
    bank = generate_pick_place_scenario_bank(
        count=args.count,
        seed=args.seed,
        mode=args.mode,
        runtime_config_sha256=env_cfg["runtime_config_sha256"],
        fixed_obj=env_cfg["fixed_obj_pos"],
        fixed_target=env_cfg["fixed_target_pos"],
        obj_spawn_lower=env_cfg["obj_spawn_lower"],
        obj_spawn_upper=env_cfg["obj_spawn_upper"],
        target_spawn_lower=env_cfg["target_spawn_lower"],
        target_spawn_upper=env_cfg["target_spawn_upper"],
    )
    output = write_scenario_bank(args.output, bank)
    print(f"Wrote {bank['count']} {bank['mode']} scenarios to {output}")
    print(f"SHA256: {scenario_bank_sha256(output)}")


if __name__ == "__main__":
    main()
