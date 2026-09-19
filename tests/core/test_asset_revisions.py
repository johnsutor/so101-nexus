"""Immutable scanned-asset downloads and separate revision caches."""

import pytest

from so101_nexus import mesh_assets


def test_asset_cache_uses_immutable_default_revision(monkeypatch, tmp_path):
    monkeypatch.setattr(mesh_assets.Path, "home", lambda: tmp_path)
    repo, revision, cache = mesh_assets.asset_cache_config("test", "owner/data", "a" * 40)
    assert repo == "owner/data"
    assert revision == "a" * 40
    assert revision in str(cache)
    monkeypatch.setenv("SO101_TEST_HF_REVISION", "b" * 40)
    assert mesh_assets.asset_cache_config("test", repo, revision)[2] != cache


@pytest.mark.parametrize("revision", ["main", "v1", "", "../cache", "a" * 39])
def test_asset_cache_rejects_mutable_or_invalid_revision(monkeypatch, revision):
    monkeypatch.setenv("SO101_TEST_HF_REVISION", revision)
    with pytest.raises(ValueError, match="40-character commit"):
        mesh_assets.asset_cache_config("test", "owner/data", "a" * 40)


def test_custom_asset_repo_requires_its_own_revision(monkeypatch):
    monkeypatch.setenv("SO101_TEST_HF_REPO", "other/data")
    with pytest.raises(ValueError, match="SO101_TEST_HF_REVISION"):
        mesh_assets.asset_cache_config("test", "owner/data", "a" * 40)
    monkeypatch.setenv("SO101_TEST_HF_REVISION", "b" * 40)
    repo, revision, _ = mesh_assets.asset_cache_config("test", "owner/data", "a" * 40)
    assert (repo, revision) == ("other/data", "b" * 40)
