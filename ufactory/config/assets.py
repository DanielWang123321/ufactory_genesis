"""Asset discovery for source checkouts installed with ``pip install -e .``."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Protocol, runtime_checkable, cast


class AssetLayoutError(RuntimeError):
    """Raised when a runtime is detached from the required source assets."""


@runtime_checkable
class AssetStore(Protocol):
    """Locate runtime assets relative to a discovery root."""

    @property
    def root(self) -> Path:
        """Root used to resolve repository-relative asset paths."""

    @property
    def assets_dir(self) -> Path:
        """Directory that contains the tracked ``assets/`` tree."""

    def require(self, relative_path: str | Path) -> Path:
        """Return an existing file path for a repository-relative asset."""

    def validate_manifest(self, *, verify_paths: Iterable[str] = ()) -> dict[str, object]:
        """Load and validate ``assets/manifest.json``."""


def _validate_relative_asset_path(relative_path: str | Path) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise AssetLayoutError(f"asset path must be repository-relative: {relative}")
    return relative


def _load_and_validate_manifest(
    store: AssetStore,
    *,
    verify_paths: Iterable[str] = (),
) -> dict[str, object]:
    manifest_path = store.require("assets/manifest.json")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AssetLayoutError("asset manifest root must be a mapping")
    if data.get("schema_version") != 1:
        raise AssetLayoutError("unsupported assets/manifest.json schema_version")
    declared = data.get("required_paths")
    if not isinstance(declared, list) or not all(isinstance(item, str) for item in declared):
        raise AssetLayoutError("asset manifest required_paths must be a string list")
    for item in (*declared, *verify_paths):
        store.require(item)
    return cast(dict[str, object], data)


@dataclass(frozen=True)
class RepositoryAssetStore:
    """Locate tracked assets relative to a cloned source checkout."""

    root: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> RepositoryAssetStore:
        return discover_asset_store(start=start)

    @classmethod
    def try_discover(cls, start: Path | None = None) -> RepositoryAssetStore | None:
        candidate = (start or Path(__file__).resolve()).resolve()
        for root in (candidate, *candidate.parents):
            if (root / "pyproject.toml").is_file() and (root / "assets").is_dir():
                return cls(root=root)
        return None

    @property
    def assets_dir(self) -> Path:
        return self.root / "assets"

    def require(self, relative_path: str | Path) -> Path:
        relative = _validate_relative_asset_path(relative_path)
        path = (self.root / relative).resolve()
        if not path.is_file():
            raise AssetLayoutError(
                f"required repository asset is missing: {relative}. "
                "Restore a complete source clone and run `pip install -e .`."
            )
        return path

    def validate_manifest(self, *, verify_paths: Iterable[str] = ()) -> dict[str, object]:
        return _load_and_validate_manifest(self, verify_paths=verify_paths)


def discover_asset_store(start: Path | None = None) -> RepositoryAssetStore:
    """Find a complete source checkout or fail closed."""
    repo = RepositoryAssetStore.try_discover(start=start)
    if repo is not None:
        return repo
    raise AssetLayoutError(
        "UFACTORY runtime assets were not found. Clone the GitHub source repository "
        "and install it with `pip install -e .`; wheel/sdist installs are not supported."
    )
