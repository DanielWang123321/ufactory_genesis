"""HuggingFace checkpoint upload/download helpers (Genesis/torch free at import)."""

from __future__ import annotations

import os
from pathlib import Path

from ufactory.training.artifacts import validate_checkpoint_artifacts


def collect_checkpoint_bundle(log_dir: Path | str, checkpoint: Path | str) -> list[Path]:
    """Return existing files to upload for a checkpoint bundle."""
    log_dir = Path(log_dir)
    checkpoint = Path(checkpoint)
    candidates = [
        checkpoint,
        checkpoint.with_suffix(".checkpoint_manifest.json"),
        log_dir / "config.yaml",
        log_dir / "artifacts.yaml",
    ]
    return [path for path in candidates if path.is_file()]


def upload_checkpoint_bundle(
    log_dir: Path | str,
    checkpoint: Path | str,
    *,
    repo_id: str,
    path_in_repo: str,
    token: str | None = None,
    dry_run: bool = False,
) -> list[str]:
    """Upload (or dry-run list) the checkpoint bundle. Token from arg or HF_TOKEN."""
    files = collect_checkpoint_bundle(log_dir, checkpoint)
    prefix = path_in_repo.strip("/")
    remote_names: list[str] = []
    for path in files:
        remote = f"{prefix}/{path.name}" if prefix else path.name
        remote_names.append(remote)

    if dry_run:
        return remote_names

    auth = token if token is not None else os.environ.get("HF_TOKEN")
    if not auth:
        raise RuntimeError("HF_TOKEN is not set and no token argument was provided")

    from huggingface_hub import HfApi  # lazy import

    api = HfApi(token=auth)
    for path, remote in zip(files, remote_names, strict=True):
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=remote,
            repo_id=repo_id,
            repo_type="model",
        )
    return remote_names


def download_and_validate(
    repo_id: str,
    path_in_repo: str,
    dest_dir: Path | str,
    *,
    expected_task: str | None = None,
    expected_robot_key: str | None = None,
    token: str | None = None,
    checkpoint_name: str | None = None,
) -> tuple:
    """Download a checkpoint bundle and validate hashes."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    auth = token if token is not None else os.environ.get("HF_TOKEN")

    from huggingface_hub import hf_hub_download  # lazy import

    prefix = path_in_repo.strip("/")
    names = ["config.yaml", "artifacts.yaml"]
    # Discover checkpoint: caller may pass explicit name; otherwise try common pattern later.
    if checkpoint_name is None:
        raise ValueError("checkpoint_name is required (e.g. model_599.pt)")
    names = [checkpoint_name, Path(checkpoint_name).with_suffix(".checkpoint_manifest.json").name, *names]

    local_paths: dict[str, Path] = {}
    for name in names:
        remote = f"{prefix}/{name}" if prefix else name
        local = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=remote,
                local_dir=str(dest),
                token=auth,
            )
        )
        local_paths[name] = local

    checkpoint = local_paths[checkpoint_name]
    # hf_hub_download may nest under path_in_repo; normalize by searching dest
    if not checkpoint.is_file():
        matches = list(dest.rglob(checkpoint_name))
        if not matches:
            raise FileNotFoundError(f"downloaded checkpoint not found: {checkpoint_name}")
        checkpoint = matches[0]
    config = next(dest.rglob("config.yaml"))
    return validate_checkpoint_artifacts(
        checkpoint,
        config,
        expected_task=expected_task,
        expected_robot_key=expected_robot_key,
    )
