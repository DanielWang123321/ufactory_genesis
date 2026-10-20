"""Asset discovery is restricted to complete source checkouts."""

from __future__ import annotations

from pathlib import Path

import pytest

from ufactory.config.assets import (
    AssetLayoutError,
    RepositoryAssetStore,
    discover_asset_store,
)


def test_discover_asset_store_prefers_repository_checkout() -> None:
    store = discover_asset_store()
    assert isinstance(store, RepositoryAssetStore)
    assert (store.root / "pyproject.toml").is_file()
    assert store.assets_dir.is_dir()
    manifest = store.validate_manifest()
    assert manifest.get("schema_version") == 1


def test_repository_discover_matches_discover_asset_store() -> None:
    via_legacy = RepositoryAssetStore.discover()
    via_shared = discover_asset_store()
    assert isinstance(via_shared, RepositoryAssetStore)
    assert via_legacy.root == via_shared.root


def test_discover_asset_store_fails_closed_without_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RepositoryAssetStore, "try_discover", classmethod(lambda cls, start=None: None))
    with pytest.raises(AssetLayoutError, match="wheel/sdist installs are not supported"):
        discover_asset_store()


def test_require_rejects_parent_and_absolute_paths() -> None:
    store = RepositoryAssetStore.discover()
    with pytest.raises(AssetLayoutError, match="repository-relative"):
        store.require("../pyproject.toml")
    with pytest.raises(AssetLayoutError, match="repository-relative"):
        store.require(Path("/tmp/secret"))
