"""Repository-only asset discovery for the supported editable installation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, cast


class AssetLayoutError(RuntimeError):
    """Raised when a runtime is detached from the required source assets."""


@dataclass(frozen=True)
class RepositoryAssetStore:
    """Locate tracked assets relative to a cloned source checkout."""

    root: Path

    @classmethod
    def discover(cls, start: Path | None = None) -> RepositoryAssetStore:
        candidate = (start or Path(__file__).resolve()).resolve()
        roots = (candidate, *candidate.parents)
        for root in roots:
            if (root / "pyproject.toml").is_file() and (root / "assets").is_dir():
                return cls(root=root)
        raise AssetLayoutError(
            "UFACTORY runtime assets were not found. Clone the GitHub source repository "
            "and install it with `pip install -e .`; wheel/sdist installs are not supported."
        )

    @property
    def assets_dir(self) -> Path:
        return self.root / "assets"

    def require(self, relative_path: str | Path) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise AssetLayoutError(f"asset path must be repository-relative: {relative}")
        path = (self.root / relative).resolve()
        if not path.is_file():
            raise AssetLayoutError(
                f"required repository asset is missing: {relative}. "
                "Restore a complete source clone and run `pip install -e .`."
            )
        return path

    def validate_manifest(self, *, verify_paths: Iterable[str] = ()) -> dict[str, object]:
        manifest_path = self.require("assets/manifest.json")
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise AssetLayoutError("asset manifest root must be a mapping")
        if data.get("schema_version") != 1:
            raise AssetLayoutError("unsupported assets/manifest.json schema_version")
        declared = data.get("required_paths")
        if not isinstance(declared, list) or not all(isinstance(item, str) for item in declared):
            raise AssetLayoutError("asset manifest required_paths must be a string list")
        for item in (*declared, *verify_paths):
            self.require(item)
        return cast(dict[str, object], data)
