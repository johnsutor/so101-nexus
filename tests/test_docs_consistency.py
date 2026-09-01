"""Docs-to-code consistency checks.

These tests guard against drift between user-facing documentation and the
public Python API. They are intentionally lightweight: they run without
importing any backend, so they can be executed with the dev dependency group
alone.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "content" / "docs"
TEXT_DOCS = [
    ROOT / "README.md",
    ROOT / "examples" / "README.md",
    ROOT / "envhub" / "README.md",
    *DOCS.rglob("*.mdx"),
]


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


def test_every_docs_page_is_reachable_from_nav() -> None:
    """Every ``*.mdx`` page must be listed in its section's ``meta.json``.

    An unreferenced page is unreachable from the sidebar, so it is a mistake
    rather than a draft. Root ``index.mdx`` is listed in the root nav; section
    ``index.mdx`` pages are listed in their own section nav.
    """
    orphans: list[str] = []
    for meta_path in DOCS.rglob("meta.json"):
        directory = meta_path.parent
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        listed = {page for page in meta["pages"] if not page.startswith("---")}
        for page in sorted(path.stem for path in directory.glob("*.mdx")):
            if page not in listed:
                orphans.append(str((directory / f"{page}.mdx").relative_to(ROOT)))
        for child in sorted(p.name for p in directory.iterdir() if (p / "meta.json").exists()):
            if child not in listed:
                orphans.append(str((directory / child).relative_to(ROOT)))
    assert orphans == [], f"docs pages missing from nav: {orphans}"


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
            if "max_episode_steps=" in line and not re.search(r"gym(?:nasium)?\.make", line):
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
    docs = set(DOCS.rglob("*.mdx")) | set(TEXT_DOCS)
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


_BRITISH_SPELLINGS = re.compile(
    r"\b("
    r"colour|behaviour|centre|metre|licence|catalogue|grey|fibre|defence|favour|neighbour"
    r"|normalis|optimis|initialis|customis|organis|analyse|analysing|utilis|visualis|recognis"
    r"|serialis|synchronis|prioritis|summaris|minimis|maximis|parameteris"
    r"|labelled|modelling|travelling|cancelled"
    r")\w*",
    re.IGNORECASE,
)


def test_user_docs_use_american_english() -> None:
    """User-facing docs are written in American English."""
    offenders: list[tuple[str, int, str]] = []
    for path in TEXT_DOCS:
        for lineno, line in enumerate(_read(path).splitlines(), start=1):
            for found in _BRITISH_SPELLINGS.finditer(line):
                offenders.append((str(path.relative_to(ROOT)), lineno, found.group(0)))
    assert offenders == [], f"British spellings in user-facing docs: {offenders}"


def _docs_route_exists(route: str) -> bool:
    """Return True if a ``/docs/...`` route resolves to a real page file."""
    relative = route.removeprefix("/docs").strip("/")
    if not relative:
        return (DOCS / "index.mdx").exists()
    return (DOCS / f"{relative}.mdx").exists() or (DOCS / relative / "index.mdx").exists()


_LINK_TARGET = re.compile(r"\]\(([^)\s]+)\)|href=\"([^\"]+)\"")
_SITE_ORIGIN = re.compile(r"^https://so101-nexus\.(?:com|github\.io)")


def test_internal_doc_links_resolve() -> None:
    """Every link into ``/docs`` from user-facing docs must reach a real page.

    Covers both site-relative hrefs and absolute links to the published site,
    so a page rename cannot silently orphan a cross-reference.
    """
    offenders: list[tuple[str, str]] = []
    for path in TEXT_DOCS:
        for found in _LINK_TARGET.finditer(_read(path)):
            target = (found.group(1) or found.group(2)).split("#")[0]
            target = _SITE_ORIGIN.sub("", target)
            if not target.startswith("/docs"):
                continue
            route = target.rstrip("/")
            if not _docs_route_exists(route):
                offenders.append((str(path.relative_to(ROOT)), route))
    assert offenders == [], f"docs links point at pages that do not exist: {sorted(set(offenders))}"


def _public_exports() -> set[str]:
    init_src = _read(ROOT / "src" / "so101_nexus" / "__init__.py")
    match = re.search(r"__all__\s*=\s*\[(.*?)\]", init_src, re.DOTALL)
    assert match, "could not locate __all__ in so101_nexus/__init__.py"
    public = set(re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"', match.group(1)))
    assert public, "parsed empty __all__"
    return public


def test_docs_import_examples_resolve_against_public_api() -> None:
    """Every name in a ``from so101_nexus import ...`` example must be exported."""
    public = _public_exports()
    pattern = re.compile(r"from so101_nexus import \(([^)]*)\)|from so101_nexus import ([^\n(]+)")
    offenders: list[tuple[str, str]] = []
    for path in TEXT_DOCS:
        for found in pattern.finditer(_read(path)):
            body = found.group(1) or found.group(2)
            for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", body):
                if name not in public:
                    offenders.append((str(path.relative_to(ROOT)), name))
    assert offenders == [], f"docs import names missing from so101_nexus.__all__: {offenders}"


def _package_definitions() -> set[str]:
    """Every class and function name defined anywhere in the package."""
    names: set[str] = set()
    for src in (ROOT / "src" / "so101_nexus").rglob("*.py"):
        names.update(re.findall(r"^\s*(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", _read(src), re.M))
    assert names, "parsed no definitions from the package source"
    return names


def test_api_reference_headings_name_real_helpers() -> None:
    """Top-level function headings in the API reference must name real helpers.

    Guards against documenting a helper that does not exist, which is invisible
    to readers until they call it. ``####`` headings are method-level and are
    resolved against their owning class rather than the module namespace.
    """
    known = _public_exports() | _package_definitions()
    pattern = re.compile(r"^#{2,3}\s+`([A-Za-z_][A-Za-z0-9_]*)\(\)`", re.MULTILINE)
    offenders: list[tuple[str, str]] = []
    for path in sorted((DOCS / "api").glob("*.mdx")):
        for found in pattern.finditer(_read(path)):
            if found.group(1) not in known:
                offenders.append((str(path.relative_to(ROOT)), found.group(1)))
    assert offenders == [], f"API reference documents helpers that do not exist: {offenders}"


def test_documented_place_success_predicates_require_release() -> None:
    """PickAndPlace and StackCube success requires releasing the object.

    Both backends gate success on ``is_grasped < 0.5``. Only StackCube also
    requires the arm to be static; for PickAndPlace ``is_robot_static`` is an
    ``info`` diagnostic, so docs must not present it as a success condition.
    """
    for backend, module in (("mujoco", "pick_and_place"), ("warp", "pick_and_place")):
        src = _read(ROOT / "src" / "so101_nexus" / backend / f"{module}.py")
        predicate = re.search(r"success = (.+)", src)
        assert predicate, f"could not parse success predicate from {backend}/{module}.py"
        assert "is_robot_static" not in predicate.group(1), (
            f"{backend}/{module}.py now gates success on the arm; update the docs and this test"
        )

    section = _read(DOCS / "environments" / "index.mdx").split("## Success conditions", 1)[1]
    section = section.split("## Rewards", 1)[0]
    place = section.split("**PickAndPlace**", 1)[1].split("**StackCube**", 1)[0].lower()
    assert "released" in place or "release" in place, (
        "the PickAndPlace success condition must state that the object is released"
    )
    assert "diagnostic" in place, (
        "the PickAndPlace success condition must note that is_robot_static is diagnostic only"
    )


def test_pick_and_place_baseline_matches_bc_ppo_docstring() -> None:
    """Training docs must quote the demo-seeded PickAndPlace result from source.

    ``ppo_warp.py`` has no PickAndPlace baseline but ``bc_ppo_warp.py`` does, and
    the two were previously reported as contradicting each other.
    """
    docstring = _read(ROOT / "examples" / "bc_ppo_warp.py").split('"""')[1]
    mean = re.search(r"mean `([\d.]+)`", docstring)
    assert mean, "could not parse the validated PickAndPlace mean from bc_ppo_warp.py"

    training = _read(DOCS / "workflow" / "training.mdx")
    assert mean.group(1) in training, (
        f"workflow/training.mdx must report the bc_ppo_warp.py PickAndPlace mean {mean.group(1)}"
    )
    assert "excluded until the environment is fixed" not in training, (
        "the PickAndPlace 'excluded' claim is stale: bc_ppo_warp.py solves it"
    )


def test_installation_lists_every_package_extra() -> None:
    """Installation docs must list every supported package extra."""
    extras = tomllib.loads(_read(ROOT / "pyproject.toml"))["project"]["optional-dependencies"]
    installation = _read(DOCS / "getting-started" / "installation.mdx")
    documented = set(re.findall(r"^\| `([^`]+)` \|", installation, re.MULTILINE))

    assert documented == set(extras), (
        "installation docs extras differ from pyproject.toml: "
        f"documented={sorted(documented)}, expected={sorted(extras)}"
    )


def test_policy_adapter_example_configures_default_cameras() -> None:
    """The recorder example must provide each camera that it reads by default."""
    policies = _read(DOCS / "api" / "policies.mdx")
    match = re.search(r"## End-to-End Usage\n\n```python\n(.*?)```", policies, re.DOTALL)

    assert match, "could not locate the policy adapter end-to-end example"
    example = match.group(1)
    assert re.search(
        r"config = PickConfig\(\n"
        r'    obs_mode="visual",\n'
        r"    observations=\[JointPositions\(\), WristCamera\(\), OverheadCamera\(\)\],\n"
        r"\)",
        example,
    )
    assert re.search(
        r"env = gym\.make\(\n"
        r'    "MuJoCoPickLift-v1",\n'
        r"    config=config,",
        example,
    )


def test_lerobot_wrapper_docs_require_a_camera_component() -> None:
    """The LeRobot wrapper docs must state the Dict observation requirement."""
    pages = (
        DOCS / "api" / "lerobot-processors.mdx",
        DOCS / "concepts" / "lerobot.mdx",
    )
    for page in pages:
        text = _read(page)
        assert "at least one camera component" in text
        assert "or another non-default observation" not in text
