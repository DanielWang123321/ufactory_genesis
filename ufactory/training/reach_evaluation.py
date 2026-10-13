"""Fixed-target reach evaluation, independent of Genesis and policy libraries."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def sample_reach_targets(lower: list[float], upper: list[float], count: int, seed: int) -> torch.Tensor:
    """Use a dedicated CPU generator so targets do not depend on batch size or GPU."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    lo, hi = torch.tensor(lower, device="cpu"), torch.tensor(upper, device="cpu")
    return lo + torch.rand(count, 3, generator=generator, device="cpu") * (hi - lo)


def evaluate_fixed_targets(env: Any, policy: Callable[[Any], torch.Tensor], targets: torch.Tensor) -> list[dict]:
    """Run every supplied target exactly once, including the slowest episodes."""
    reports = []
    for start in range(0, len(targets), env.num_envs):
        batch = targets[start : start + env.num_envs]
        count = len(batch)
        env.reset()
        padded = batch[-1:].expand(env.num_envs, -1).clone()
        padded[:count] = batch
        env.target_pos.copy_(padded.to(env.device))
        env.target_marker.set_pos(env.target_pos)
        observation = env.get_observations()
        active = torch.arange(env.num_envs, device=env.device) < count
        for step in range(1, env.max_episode_length + 1):
            actions = policy(observation)
            actions = torch.where(active[:, None], actions, 0.0)
            observation, _reward, dones, infos = env.step(actions)
            done_idx = dones.nonzero(as_tuple=False).flatten()
            if done_idx.numel():
                keep = active[done_idx]
                selected = done_idx[keep].cpu().tolist()
                distances = infos["reach_done_distance_m"][keep].cpu().tolist()
                successes = infos["reach_done_success"][keep].cpu().tolist()
                for idx, distance, success in zip(selected, distances, successes, strict=True):
                    reports.append(
                        {
                            "index": start + idx,
                            "target_m": batch[idx].tolist(),
                            "success": bool(success),
                            "timeout": not bool(success),
                            "terminal_distance_m": distance,
                            "steps": step,
                            "elapsed_s": step * env.ctrl_dt,
                        }
                    )
                active[done_idx] = False
            if not active.any():
                break
        if active.any():
            raise RuntimeError("fixed-target evaluation left unfinished episodes at the time limit")
    return sorted(reports, key=lambda report: report["index"])


def summarize_reach_reports(reports: list[dict]) -> dict:
    if not reports:
        raise RuntimeError("evaluation finished no episodes")
    distances = torch.tensor([report["terminal_distance_m"] for report in reports], dtype=torch.float64, device="cpu")
    successes = [report for report in reports if report["success"]]
    quantiles = torch.quantile(distances, torch.tensor([0.5, 0.9, 0.99], dtype=torch.float64, device="cpu"))
    return {
        "episodes": len(reports),
        "successes": len(successes),
        "success_rate": len(successes) / len(reports),
        "mean_m": distances.mean().item(),
        "median_m": quantiles[0].item(),
        "p90_m": quantiles[1].item(),
        "p99_m": quantiles[2].item(),
        "mean_elapsed_s": sum(report["elapsed_s"] for report in reports) / len(reports),
        "mean_success_elapsed_s": (
            sum(report["elapsed_s"] for report in successes) / len(successes) if successes else None
        ),
    }
