"""Regression tests for validate_object_rest_poses.py's advisory-only grasp policy.

``validate_one`` gates pose-override acceptance on the settle test alone; the
grasp screen is recorded on the report but never drops an object or blocks an
override. These tests exercise that policy directly against ``validate_one``,
with ``run_settle_test``/``run_grasp_check``/``_settle_from_quat`` monkeypatched
so no real MuJoCo physics or asset downloads are needed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_script():
    """Import scripts/validate_object_rest_poses.py as a module (not on sys.path)."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "validate_object_rest_poses.py"
    spec = importlib.util.spec_from_file_location("validate_object_rest_poses", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v = _load_script()


def _settle(model_id: str, quat: np.ndarray, *, passed: bool) -> v.SettleResult:
    delta_m = 0.001 if passed else 0.05
    delta_deg = 1.0 if passed else 30.0
    return v.SettleResult(
        model_id=model_id,
        predicted_quat=quat,
        predicted_spawn_z=0.01,
        settled_pos=np.array([0.15, 0.0, 0.01]),
        settled_quat=quat,
        translation_delta_m=delta_m,
        rotation_delta_deg=delta_deg,
    )


def _grasp(model_id: str, *, passed: bool) -> v.GraspResult:
    if passed:
        return v.GraspResult(
            model_id=model_id, passed=True, height_fraction=0.5, yaw_deg=0.0, width_mm=20.0
        )
    return v.GraspResult(model_id=model_id, passed=False, detail="no slice fits")


class _FakeObj:
    """Stands in for a SceneObject; validate_one never inspects it directly."""


@pytest.fixture(autouse=True)
def _no_spawn_z_lookup(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(v, "_spawn_z_for_quat", lambda _obj, _quat: 0.01)


def test_settle_pass_grasp_fail_is_not_dropped_and_needs_no_override(
    monkeypatch: pytest.MonkeyPatch,
):
    """The core policy under test: a settle-stable, grasp-advisory-failing object ships as-is."""
    model_id = "test_model"
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(
        v, "run_settle_test", lambda _obj, _mid: _settle(model_id, quat, passed=True)
    )
    monkeypatch.setattr(v, "run_grasp_check", lambda _obj, _mid, _q: _grasp(model_id, passed=False))

    report = v.validate_one(_FakeObj(), model_id)

    assert report.settle.passed is True
    assert report.grasp.passed is False
    assert report.override_quat is None
    assert report.dropped is False


def test_settle_fail_finds_a_settle_only_override_regardless_of_grasp(
    monkeypatch: pytest.MonkeyPatch,
):
    """An unstable heuristic pose gets corrected by the first candidate that settles,
    even when that candidate also fails the (advisory) grasp screen."""
    model_id = "test_model"
    bad_quat = np.array([1.0, 0.0, 0.0, 0.0])
    good_quat = np.array([0.7071068, 0.0, 0.7071068, 0.0])
    monkeypatch.setattr(
        v, "run_settle_test", lambda _obj, _mid: _settle(model_id, bad_quat, passed=False)
    )
    monkeypatch.setattr(v, "run_grasp_check", lambda _obj, _mid, _q: _grasp(model_id, passed=False))

    def _settle_from_quat(_obj, _mid, quat, _spawn_z):
        passed = bool(np.allclose(quat, good_quat))
        return _settle(model_id, quat if not passed else good_quat, passed=passed)

    monkeypatch.setattr(v, "_settle_from_quat", _settle_from_quat)

    report = v.validate_one(_FakeObj(), model_id)

    assert report.override_quat is not None
    np.testing.assert_allclose(report.override_quat, good_quat)
    assert report.settle.passed is True
    assert report.grasp.passed is False
    assert report.dropped is False


def test_no_candidate_settles_drops_the_object(monkeypatch: pytest.MonkeyPatch):
    model_id = "test_model"
    bad_quat = np.array([1.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(
        v, "run_settle_test", lambda _obj, _mid: _settle(model_id, bad_quat, passed=False)
    )
    monkeypatch.setattr(v, "run_grasp_check", lambda _obj, _mid, _q: _grasp(model_id, passed=False))
    monkeypatch.setattr(
        v, "_settle_from_quat", lambda _obj, _mid, quat, _sz: _settle(model_id, quat, passed=False)
    )

    report = v.validate_one(_FakeObj(), model_id)

    assert report.dropped is True
    assert report.override_quat is None
