"""Packaging checks for the ``rocm`` optional-dependency extra.

Verifies ``uv sync --extra train --extra warp`` keeps resolving torch from the
default index (no accidental breaking change) while ``--extra rocm`` is wired
to the ROCm PyTorch index, and that the accelerator extras cannot silently
combine with dependency sets that pin an incompatible torch range.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
ROCM_INDEX_URL = "https://download.pytorch.org/whl/rocm7.2"


def test_rocm_extra_declared() -> None:
    """The ``rocm`` extra exists and pulls torch plus its Triton companions."""
    extras = PYPROJECT["project"]["optional-dependencies"]
    assert "rocm" in extras
    names = {dep.split(";")[0].split(">")[0].split("=")[0].strip() for dep in extras["rocm"]}
    assert names == {"torch", "pytorch-triton-rocm", "triton-rocm"}


def test_default_extras_do_not_reference_rocm_index() -> None:
    """``train`` and ``warp`` resolve torch from the default package index."""
    extras = PYPROJECT["project"]["optional-dependencies"]
    for extra in ("train", "warp"):
        torch_requirements = [
            dependency for dependency in extras[extra] if dependency.startswith("torch")
        ]
        assert len(torch_requirements) == (2 if extra == "warp" else 1)
        assert all("rocm" not in dependency for dependency in torch_requirements)


def _assert_torch_pairs(dependencies, *, include_codec=False) -> None:
    requirements = [Requirement(dependency) for dependency in dependencies]
    for platform, machine, torch_bound, codec_bound in (
        ("linux", "x86_64", "<2.10", "<0.10"),
        ("darwin", "arm64", "<2.10", "<0.10"),
        ("linux", "aarch64", ">=2.11,<2.12", ">=0.11,<0.12"),
        ("linux", "arm64", ">=2.11,<2.12", ">=0.11,<0.12"),
    ):
        environment = {"sys_platform": platform, "platform_machine": machine}
        for package, bound in (("torch", torch_bound), ("torchcodec", codec_bound)):
            if package == "torchcodec" and not include_codec:
                continue
            active = [
                req
                for req in requirements
                if req.name == package and (req.marker is None or req.marker.evaluate(environment))
            ]
            assert len(active) == 1
            assert active[0].specifier == SpecifierSet(bound)


def test_warp_extra_pins_compatible_runtime_versions() -> None:
    warp = PYPROJECT["project"]["optional-dependencies"]["warp"]
    assert "mujoco-warp>=3.13.0,<4" in warp
    assert "warp-lang>=1.15,<1.16" in warp
    _assert_torch_pairs(warp)


def test_test_group_pins_compatible_torchcodec() -> None:
    _assert_torch_pairs(PYPROJECT["dependency-groups"]["test"], include_codec=True)


def test_torch_sources_route_only_the_rocm_extra_to_the_rocm_index() -> None:
    """``tool.uv.sources`` only redirects torch (and Triton) when rocm is active."""
    sources = PYPROJECT["tool"]["uv"]["sources"]
    for package in ("torch", "pytorch-triton-rocm", "triton-rocm"):
        entries = sources[package]
        assert len(entries) == 1
        assert entries[0]["index"] == "pytorch-rocm"
        assert entries[0]["extra"] == "rocm"


def test_rocm_index_is_explicit_and_points_at_rocm72() -> None:
    """The pytorch-rocm index is explicit so it never leaks into unrelated deps."""
    indexes = {entry["name"]: entry for entry in PYPROJECT["tool"]["uv"]["index"]}
    assert indexes["pytorch-rocm"]["url"] == ROCM_INDEX_URL
    assert indexes["pytorch-rocm"]["explicit"] is True


def test_rocm_conflicts_with_extras_and_groups_pinning_incompatible_torch() -> None:
    """ROCm cannot combine with dependency sets that pin incompatible Torch versions."""
    conflicts = PYPROJECT["tool"]["uv"]["conflicts"]
    rocm_pairs = [{frozenset(m.items()) for m in pair} for pair in conflicts]
    assert frozenset({("extra", "rocm")}) in {p for pair in rocm_pairs for p in pair}
    assert {frozenset({("extra", "rocm")}), frozenset({("extra", "teleop")})} in rocm_pairs
    assert {frozenset({("extra", "rocm")}), frozenset({("group", "test")})} in rocm_pairs
    assert {frozenset({("extra", "rocm")}), frozenset({("group", "dev")})} in rocm_pairs
    assert {frozenset({("extra", "rocm")}), frozenset({("extra", "warp")})} in rocm_pairs
