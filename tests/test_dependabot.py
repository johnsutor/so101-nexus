from pathlib import Path


def _python_update_config() -> str:
    config = Path(__file__).resolve().parents[1] / ".github" / "dependabot.yml"
    return next(
        block
        for block in config.read_text().split("  - package-ecosystem: ")
        if block.startswith("uv\n")
    )


def test_python_updates_preserve_runtime_compatibility_bounds() -> None:
    assert "    versioning-strategy: lockfile-only\n" in _python_update_config()


def test_numpy_updates_respect_lerobot_compatibility() -> None:
    assert (
        '      - dependency-name: numpy\n        versions: [">=2.3"]\n' in _python_update_config()
    )
