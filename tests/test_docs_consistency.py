"""Docs-to-code consistency checks.

These tests guard against drift between user-facing documentation and the
public Python API. They are intentionally lightweight: they run without
importing any backend, so they can be executed with the dev dependency group
alone.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "content" / "docs"
TEXT_DOCS = [ROOT / "README.md", ROOT / "examples" / "README.md", *DOCS.rglob("*.mdx")]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_no_em_dashes_or_emoji_in_user_docs() -> None:
    """User-facing docs must not contain em dashes, en dashes, or emoji."""
    emoji = re.compile(r"[\U0001f300-\U0001faff]")
    offenders = []
    for path in TEXT_DOCS:
        text = _read(path)
        if "—" in text or "–" in text or emoji.search(text):  # noqa: RUF001
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == [], f"Found em dashes, en dashes, or emoji in user-facing docs: {offenders}"


def test_environment_nav_lists_all_environment_pages() -> None:
    """The environments sidebar must expose every ``*.mdx`` env page."""
    meta = json.loads((DOCS / "environments" / "meta.json").read_text(encoding="utf-8"))
    pages = {page for page in meta["pages"] if not page.startswith("---")}
    files = {path.stem for path in (DOCS / "environments").glob("*.mdx") if path.stem != "index"}
    missing = sorted(files - pages)
    assert missing == [], f"Environment pages missing from nav: {missing}"


def test_docs_do_not_import_public_objects_from_backend_submodules() -> None:
    """Public configs and object classes live in ``so101_nexus_core``."""
    forbidden = (
        "so101_nexus_mujoco.config",
        "so101_nexus_mujoco.objects",
        "so101_nexus_maniskill.config",
        "so101_nexus_maniskill.objects",
    )
    offenders = []
    for path in DOCS.rglob("*.mdx"):
        text = _read(path)
        for pattern in forbidden:
            if pattern in text:
                offenders.append((str(path.relative_to(ROOT)), pattern))
    assert offenders == [], (
        "Docs import public objects from backend submodules instead of "
        f"so101_nexus_core: {offenders}"
    )


def test_docs_static_search_uses_orama_export() -> None:
    """The static search UI and exported search index must use the same format."""
    layout = _read(ROOT / "docs" / "app" / "layout.tsx")
    route = _read(ROOT / "docs" / "app" / "api" / "search" / "route.ts")

    assert 'type: "static"' in layout
    assert "createFromSource" in route
    assert "flexsearchFromSource" not in route


def test_examples_readme_references_existing_example_scripts() -> None:
    """Every ``python examples/...`` command in the README must resolve."""
    text = _read(ROOT / "examples" / "README.md")
    referenced = re.findall(r"python (examples/[\w./-]+\.py)", text)
    missing = [path for path in referenced if not (ROOT / path).exists()]
    assert missing == [], f"examples/README.md references missing scripts: {missing}"
