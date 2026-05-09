from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_SMOKE_WORKFLOW = ROOT / ".github" / "workflows" / "release-smoke.yml"
PUBLISHED_PROJECTS = (
    ROOT / "packages" / "so101-nexus-core" / "pyproject.toml",
    ROOT / "packages" / "so101-nexus-mujoco" / "pyproject.toml",
    ROOT / "packages" / "so101-nexus-maniskill" / "pyproject.toml",
)


def _release_smoke_python_version() -> tuple[int, int]:
    text = RELEASE_SMOKE_WORKFLOW.read_text(encoding="utf-8")
    match = re.search(
        r"- name: Set up Python\n"
        r"\s+uses: actions/setup-python@v5\n"
        r"\s+with:\n"
        r"\s+python-version:\s*[\"']?(\d+\.\d+)[\"']?",
        text,
    )
    assert match is not None, "Release smoke workflow must pin a major.minor Python version."
    return tuple(int(part) for part in match.group(1).split("."))


def _minimum_python_version(pyproject_path: Path) -> tuple[int, int]:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    requires_python = data["project"]["requires-python"]
    match = re.fullmatch(r">=\s*(\d+)\.(\d+)", requires_python)
    assert match is not None, f"Unsupported requires-python format in {pyproject_path}."
    return int(match.group(1)), int(match.group(2))


def test_release_smoke_python_version_satisfies_published_requires_python() -> None:
    smoke_python_version = _release_smoke_python_version()

    for pyproject_path in PUBLISHED_PROJECTS:
        minimum_version = _minimum_python_version(pyproject_path)
        assert smoke_python_version >= minimum_version, (
            "Release smoke tests install from PyPI, so their setup-python version must satisfy "
            f"{pyproject_path.relative_to(ROOT)} requires-python >= {minimum_version[0]}."
            f"{minimum_version[1]}."
        )
