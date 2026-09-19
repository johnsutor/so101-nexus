"""Run provenance must identify the code checkout, not the caller's directory."""

import subprocess
from pathlib import Path

from so101_nexus import _run_metadata


def test_git_provenance_uses_package_checkout(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    expected = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
    ).strip()
    result = _run_metadata._git_state()
    assert result is not None
    assert result["sha"] == expected


def test_installed_package_does_not_claim_callers_git_revision(monkeypatch, tmp_path):
    monkeypatch.setattr(_run_metadata, "__file__", str(tmp_path / "site" / "pkg" / "metadata.py"))
    assert _run_metadata._git_state() is None
