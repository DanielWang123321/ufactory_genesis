"""AssetStore discovery: repository layout today, package backend as 0.3 hook."""

from __future__ import annotations

from pathlib import Path

import pytest

from ufactory.config.assets import (
    AssetLayoutError,
    PackageAssetStore,
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


def test_package_asset_store_absent_in_source_tree() -> None:
    assert PackageAssetStore.try_discover() is None


def test_package_asset_store_resolves_embedded_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_root = tmp_path / "ufactory"
    assets = package_root / "assets"
    assets.mkdir(parents=True)
    (assets / "manifest.json").write_text(
        '{"schema_version": 1, "required_paths": [], "sha256": {}}',
        encoding="utf-8",
    )
    marker = assets / "configs" / "runtime" / "robots" / "xarm6_1305.yaml"
    marker.parent.mkdir(parents=True)
    marker.write_text("schema_version: 1\n", encoding="utf-8")

    monkeypatch.setattr(
        PackageAssetStore,
        "try_discover",
        classmethod(lambda cls: cls(root=package_root)),
    )
    store = PackageAssetStore.try_discover()
    assert store is not None
    assert store.require("assets/manifest.json").is_file()
    assert store.require("assets/configs/runtime/robots/xarm6_1305.yaml").is_file()
    store.validate_manifest()


def test_discover_asset_store_falls_back_to_package(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    package_root = tmp_path / "ufactory"
    assets = package_root / "assets"
    assets.mkdir(parents=True)
    (assets / "manifest.json").write_text(
        '{"schema_version": 1, "required_paths": ["assets/manifest.json"], "sha256": {}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(RepositoryAssetStore, "try_discover", classmethod(lambda cls, start=None: None))
    monkeypatch.setattr(
        PackageAssetStore,
        "try_discover",
        classmethod(lambda cls: cls(root=package_root)),
    )
    store = discover_asset_store()
    assert isinstance(store, PackageAssetStore)
    store.validate_manifest()


def test_discover_asset_store_fails_closed_without_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RepositoryAssetStore, "try_discover", classmethod(lambda cls, start=None: None))
    monkeypatch.setattr(PackageAssetStore, "try_discover", classmethod(lambda cls: None))
    with pytest.raises(AssetLayoutError, match="wheel/sdist installs are not supported"):
        discover_asset_store()


def test_require_rejects_parent_and_absolute_paths() -> None:
    store = RepositoryAssetStore.discover()
    with pytest.raises(AssetLayoutError, match="repository-relative"):
        store.require("../pyproject.toml")
    with pytest.raises(AssetLayoutError, match="repository-relative"):
        store.require(Path("/tmp/secret"))
