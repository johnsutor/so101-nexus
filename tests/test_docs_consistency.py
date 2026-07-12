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
        if "\u2014" in text or "\u2013" in text or emoji.search(text):
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
    """Public configs and object classes live in ``so101_nexus``."""
    forbidden = (
        "so101_nexus.mujoco.config",
        "so101_nexus.mujoco.objects",
    )
    offenders = []
    for path in DOCS.rglob("*.mdx"):
        text = _read(path)
        for pattern in forbidden:
            if pattern in text:
                offenders.append((str(path.relative_to(ROOT)), pattern))
    assert offenders == [], (
        f"Docs import public objects from backend submodules instead of so101_nexus: {offenders}"
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


def test_max_episode_steps_documented_as_make_kwarg_not_config_field() -> None:
    """Episode length is a gym.make/make_vec keyword, never a config field or table row."""
    offenders = []
    for path in TEXT_DOCS:
        for lineno, line in enumerate(_read(path).splitlines(), start=1):
            if "max_episode_steps=" in line and "gym.make" not in line:
                offenders.append((str(path.relative_to(ROOT)), lineno, line.strip()))
            if re.match(r"\|\s*`?max_episode_steps", line):
                offenders.append((str(path.relative_to(ROOT)), lineno, line.strip()))
    assert offenders == [], (
        "max_episode_steps must be documented as a gym.make/make_vec keyword, "
        f"never as a config field: {offenders}"
    )


def test_core_overview_lists_only_real_public_symbols() -> None:
    """Every symbol in core-overview.mdx tables must be a real package export.

    Guards against dead references (e.g. a symbol copied from a sibling
    project) by checking each documented table's first-column identifier
    against the package ``__all__``. Submodule-only functions (the Environment
    Registry section) are excluded on purpose.
    """
    init_src = _read(ROOT / "src" / "so101_nexus" / "__init__.py")
    match = re.search(r"__all__\s*=\s*\[(.*?)\]", init_src, re.DOTALL)
    assert match, "could not locate __all__ in so101_nexus/__init__.py"
    public = set(re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"', match.group(1)))
    assert public, "parsed empty __all__"

    allowed_sections = {
        "Constants",
        "Asset Paths",
        "Color",
        "YCB Asset Management",
        "Reward and observation helpers",
        "Observation Components",
        "Scene Objects",
        "Configuration Classes",
        "Type Aliases",
    }
    text = _read(DOCS / "api" / "core-overview.mdx")
    missing: list[str] = []
    section: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("### "):
            section = stripped[4:].strip()
            continue
        if stripped.startswith("#### "):
            section = stripped[5:].strip()
            continue
        if section not in allowed_sections or not stripped.startswith("|"):
            continue
        if set(stripped) <= {"|", "-", ":", " "}:
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if not cells:
            continue
        name = cells[0].strip("`")
        name = re.sub(r"\(.*\)$", "", name)
        if name.lower() in {
            "name",
            "class",
            "type",
            "definition",
            "function",
            "property",
            "method",
            "argument",
            "parameter",
            "env_id",
            "returns",
            "signature",
            "description",
        }:
            continue
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            continue
        if name not in public:
            missing.append(name)
    assert missing == [], f"core-overview.mdx references non-exported symbols: {missing}"


def test_docs_reference_only_registered_env_ids() -> None:
    """Every ``MuJoCo*``/``Warp*`` env id in docs must be a registered id."""
    registered: set[str] = set()
    for backend in ("mujoco", "warp"):
        src = _read(ROOT / "src" / "so101_nexus" / backend / "__init__.py")
        registered.update(re.findall(r'id="([^"]+)"', src))
    assert registered, "could not parse any registered env ids from backend modules"

    offenders: list[tuple[str, str]] = []
    docs = set(DOCS.rglob("*.mdx")) | {ROOT / "README.md", ROOT / "examples" / "README.md"}
    pattern = re.compile(r"\b(?:MuJoCo|Warp)[A-Za-z]*-v\d+\b")
    for path in docs:
        for found in pattern.finditer(_read(path)):
            if found.group(0) not in registered:
                offenders.append((str(path.relative_to(ROOT)), found.group(0)))
    assert offenders == [], f"docs reference unregistered env ids: {offenders}"


def test_examples_readme_entropy_matches_ppo_warp_defaults() -> None:
    """examples/README.md entropy flags must match ``ppo_warp.py`` Args defaults."""
    ppo = _read(ROOT / "examples" / "ppo_warp.py")
    ent_coef = re.search(r"ent_coef:\s*float\s*=\s*([\d.]+)", ppo)
    ent_coef_final = re.search(r"ent_coef_final:\s*float\s*=\s*([\d.]+)", ppo)
    assert ent_coef, "could not parse ppo_warp.py entropy default ent_coef"
    assert ent_coef_final, "could not parse ppo_warp.py entropy default ent_coef_final"

    readme = _read(ROOT / "examples" / "README.md")
    table_ent = re.search(r"`--ent-coef`\s*\|\s*`([\d.]+)`", readme)
    table_final = re.search(r"`--ent-coef-final`\s*\|\s*`([\d.]+)`", readme)
    assert table_ent, "could not parse examples/README.md --ent-coef table row"
    assert table_final, "could not parse examples/README.md --ent-coef-final table row"
    assert table_ent.group(1) == ent_coef.group(1), (
        f"--ent-coef {table_ent.group(1)} != ppo_warp.py default {ent_coef.group(1)}"
    )
    assert table_final.group(1) == ent_coef_final.group(1), (
        f"--ent-coef-final {table_final.group(1)} != ppo_warp.py default {ent_coef_final.group(1)}"
    )
    # The "Starting commands" bash block repeats the same flags; guard it too.
    bash_ent = re.findall(r"--ent-coef ([\d.]+)", readme)
    bash_final = re.findall(r"--ent-coef-final ([\d.]+)", readme)
    assert bash_ent, "could not parse examples/README.md --ent-coef command flag"
    assert bash_final, "could not parse examples/README.md --ent-coef-final command flag"
    assert all(b == ent_coef.group(1) for b in bash_ent), (
        f"--ent-coef command {bash_ent} != ppo_warp.py default {ent_coef.group(1)}"
    )
    assert all(b == ent_coef_final.group(1) for b in bash_final), (
        f"--ent-coef-final command {bash_final} != ppo_warp.py default {ent_coef_final.group(1)}"
    )
