"""Cross-cutting source guardrails that prevent silent style/API drift.

These tests scan the real package source (and, for the prose check, the test
trees) rather than importing any backend. They are fast, GPU-free, and run in
CI. They encode conventions from CLAUDE.md so that earlier housekeeping cleanup
cannot quietly regress:

- NumPy-style docstrings only (no Google-style ``Args:`` blocks).
- Every top-level package ``__init__`` defines ``__all__``.
- No em dashes, en dashes, or emoji in project-owned source or tests.
- ``max_episode_steps`` is not a config field (episode length is owned by the
  gym registration, not the config object).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from so101_nexus_core.config import (
    EnvironmentConfig,
    LookAtConfig,
    MoveConfig,
    PickAndPlaceConfig,
    PickConfig,
    ReachConfig,
)

# Repo root: this file lives at
# <root>/packages/so101-nexus-core/tests/test_source_guardrails.py
REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGES_DIR = REPO_ROOT / "packages"

PACKAGE_NAMES = (
    "so101-nexus-core",
    "so101-nexus-mujoco",
    "so101-nexus-maniskill",
)
MODULE_NAMES = {
    "so101-nexus-core": "so101_nexus_core",
    "so101-nexus-mujoco": "so101_nexus_mujoco",
    "so101-nexus-maniskill": "so101_nexus_maniskill",
}

# Directory-name fragments that are NOT project-owned and must be excluded from
# prose/style scans. Vendored MuJoCo Menagerie models, mesh/asset blobs, and the
# virtualenv ship third-party text we do not control; docs/superpowers is design
# scratch space outside the package source trees.
EXCLUDED_DIR_NAMES = frozenset(
    {
        "menagerie",
        "mujoco_menagerie",
        "meshes",
        "assets",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".git",
        "superpowers",
    }
)


def _is_excluded(path: Path) -> bool:
    """True when any path component is a vendored/non-project directory name."""
    return any(part in EXCLUDED_DIR_NAMES for part in path.parts)


def _src_dirs() -> list[Path]:
    """Return the three package ``src/<module>`` trees, asserting each exists."""
    dirs: list[Path] = []
    for pkg in PACKAGE_NAMES:
        src = PACKAGES_DIR / pkg / "src" / MODULE_NAMES[pkg]
        assert src.is_dir(), f"missing source tree: {src}"
        dirs.append(src)
    return dirs


def _python_files(root: Path) -> list[Path]:
    """Project-owned ``.py`` files under ``root`` (vendored dirs excluded)."""
    return [p for p in root.rglob("*.py") if not _is_excluded(p)]


# ---------------------------------------------------------------------------
# Guardrail 2: no Google-style ``Args:`` docstring blocks in public source.
# ---------------------------------------------------------------------------

# Matches a docstring line that is exactly an ``Args:`` section header (optional
# indentation), the Google-style block we forbid in favor of NumPy-style
# ``Parameters``.
_ARGS_SECTION = re.compile(r"^[ \t]*Args:[ \t]*$", re.MULTILINE)


def _docstrings(tree: ast.Module):
    """Yield every docstring in module/class/function nodes of ``tree``."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                yield doc


def test_no_google_style_args_blocks_in_source():
    """All public source uses NumPy-style docstrings, never ``Args:`` blocks."""
    offenders: list[str] = []
    for src in _src_dirs():
        for py in _python_files(src):
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
            for doc in _docstrings(tree):
                if _ARGS_SECTION.search(doc):
                    offenders.append(str(py.relative_to(REPO_ROOT)))
                    break
    assert not offenders, (
        "Google-style 'Args:' docstring blocks found (use NumPy-style "
        f"'Parameters' instead): {sorted(set(offenders))}"
    )


# ---------------------------------------------------------------------------
# Guardrail 3: every top-level package __init__ defines __all__.
# ---------------------------------------------------------------------------


def _defines_dunder_all(init_path: Path) -> bool:
    """True when ``init_path`` assigns the name ``__all__`` at module scope."""
    tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                return True
    return False


@pytest.mark.parametrize("pkg", PACKAGE_NAMES)
def test_package_init_defines_dunder_all(pkg):
    """Each top-level package __init__ must define __all__ (may be empty)."""
    init_path = PACKAGES_DIR / pkg / "src" / MODULE_NAMES[pkg] / "__init__.py"
    assert init_path.is_file(), f"missing package __init__: {init_path}"
    assert _defines_dunder_all(init_path), (
        f"{init_path.relative_to(REPO_ROOT)} must define __all__ (an empty list is acceptable)"
    )


# ---------------------------------------------------------------------------
# Guardrail 4: no em dashes, en dashes, or emoji in source or tests.
# ---------------------------------------------------------------------------

# Defined by codepoint so this guardrail file does not itself contain the very
# characters it forbids (which would otherwise make the scan flag itself).
EM_DASH = chr(0x2014)
EN_DASH = chr(0x2013)

# Real emoji blocks only. The arrow block (U+2190..U+21FF, includes the
# legitimate technical arrow used in geometry comments) and general math symbols
# are deliberately NOT treated as emoji.
_EMOJI = re.compile(
    "["
    "\U0001f300-\U0001faff"  # symbols & pictographs, supplemental, extended
    "\U0001f000-\U0001f02f"  # mahjong/dominoes/cards
    "\U00002600-\U000026ff"  # miscellaneous symbols
    "\U00002700-\U000027bf"  # dingbats
    "\U00002b00-\U00002bff"  # arrows-B / stars (e.g. star emoji)
    "\U0000fe0f"  # variation selector-16 (emoji presentation)
    "]"
)


def _prose_scan_roots() -> list[Path]:
    """Package src and tests trees (vendored dirs excluded per file)."""
    roots: list[Path] = []
    for pkg in PACKAGE_NAMES:
        roots.append(PACKAGES_DIR / pkg / "src")
        roots.append(PACKAGES_DIR / pkg / "tests")
    return [r for r in roots if r.is_dir()]


def test_no_em_en_dashes_or_emoji_in_source_and_tests():
    """Project-owned .py files contain no em/en dashes or emoji."""
    offenders: list[str] = []
    for root in _prose_scan_roots():
        for py in _python_files(root):
            text = py.read_text(encoding="utf-8")
            problems = []
            if EM_DASH in text:
                problems.append("em dash")
            if EN_DASH in text:
                problems.append("en dash")
            if _EMOJI.search(text):
                problems.append("emoji")
            if problems:
                rel = py.relative_to(REPO_ROOT)
                offenders.append(f"{rel}: {', '.join(problems)}")
    assert not offenders, "Forbidden characters found:\n" + "\n".join(sorted(offenders))


# ---------------------------------------------------------------------------
# Guardrail 5 (core half): max_episode_steps is not a config field.
# ---------------------------------------------------------------------------

_CONFIG_CLASSES = [
    EnvironmentConfig,
    ReachConfig,
    LookAtConfig,
    MoveConfig,
    PickConfig,
    PickAndPlaceConfig,
]


@pytest.mark.parametrize("config_cls", _CONFIG_CLASSES, ids=[c.__name__ for c in _CONFIG_CLASSES])
def test_max_episode_steps_not_a_config_field(config_cls):
    """Episode length is owned by the gym registration, never the config."""
    cfg = config_cls()
    assert not hasattr(cfg, "max_episode_steps"), (
        f"{config_cls.__name__} must not expose max_episode_steps; episode "
        "length comes from the gym registration"
    )
