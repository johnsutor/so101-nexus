"""Settle-test and grasp-screen every supported YCB and GSO object.

``get_mujoco_ycb_rest_pose`` picks a rest orientation from the object's AABB
alone; it is a heuristic, not a physics result. This script validates it
against real MuJoCo physics and records a corrected quaternion for any
object whose heuristic pose does not settle stably (see
``so101_nexus.ycb_geometry.POSE_OVERRIDES``).

It also runs a grasp screen: from the settled pose, measure the object's
collision-mesh cross-section width at several height/yaw slices and compare
the narrowest one to the SO-101 jaw's aperture interval
(``_JAW_MIN_APERTURE_M``..``_JAW_MAX_APERTURE_M``). This is advisory only
(``so101_nexus.ycb_geometry.GRASP_ADVISORY``), not a gate on object support:
it is a deterministic geometric comparison over a finite set of candidate
poses and slices, not a dynamic grasp attempt or a collision-checked
approach path, and the jaw's real aperture is depth-dependent in ways this
check does not model. A full closed-loop dynamic grasp simulation (arm
reaches in, closes the gripper, evaluates contact normals) was attempted
first and abandoned: pass/fail was sensitive to solver warm-start state
carried over between candidates, not a trustworthy per-object signal.

``_JAW_MIN_APERTURE_M``/``_JAW_MAX_APERTURE_M`` come from two measurements
(see GSO_OBJECT_IMPORT.md and the commit that added this script): a
forward-kinematics sweep of the real gripper finger meshes (17-61 mm
closed-to-open), and an isolated cylinder-contact calibration using
``so101_nexus.grasp.opposing_normals_ok`` (reliable up to 40 mm near the
pivot, unreliable at 46 mm+, untested elsewhere along the finger).

Not a pytest suite: this is an audit script whose findings are recorded in
``so101_nexus.ycb_geometry``. Run with ``uv run python
scripts/validate_object_rest_poses.py``.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from so101_nexus import get_so101_mujoco_model_path
from so101_nexus.constants import GSO_OBJECTS, YCB_OBJECTS
from so101_nexus.gso_assets import ensure_gso_assets
from so101_nexus.object_slots import (
    _body_frame_collision_verts,
    _slot_collision_geom_ids,
    build_object_scene_xml,
)
from so101_nexus.objects import GSOObject, SceneObject, YCBObject
from so101_nexus.scene import MUJOCO_SCENE_OPTION_XML
from so101_nexus.ycb_assets import ensure_ycb_assets
from so101_nexus.ycb_geometry import get_mujoco_ycb_rest_pose

_ROBOT_XML = str(get_so101_mujoco_model_path())

_SETTLE_STEPS = 500
_SETTLE_DROP_M = 0.02
_TRANSLATION_GATE_M = 0.01
_ROTATION_GATE_DEG = 10.0

# From the forward-kinematics/cylinder-contact calibration in the module
# docstring.
_JAW_MIN_APERTURE_M = 0.017
_JAW_MAX_APERTURE_M = 0.032

_GRASP_HEIGHT_FRACTIONS = (0.15, 0.3, 0.45, 0.5, 0.6, 0.7, 0.85)
_GRASP_YAWS_DEG = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0)
_SLICE_HALF_THICKNESS_M = 0.003


def _quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    """Geodesic angle in degrees between two wxyz quaternions (double-cover safe)."""
    dot = float(np.clip(abs(np.dot(q1, q2)), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


_ROTATIONALLY_SYMMETRIC_MODEL_IDS = frozenset({"058_golf_ball"})


@dataclass
class SettleResult:
    model_id: str
    predicted_quat: np.ndarray
    predicted_spawn_z: float
    settled_pos: np.ndarray
    settled_quat: np.ndarray
    translation_delta_m: float
    rotation_delta_deg: float

    @property
    def passed(self) -> bool:
        if self.translation_delta_m > _TRANSLATION_GATE_M:
            return False
        if self.model_id in _ROTATIONALLY_SYMMETRIC_MODEL_IDS:
            return True
        return self.rotation_delta_deg <= _ROTATION_GATE_DEG


@dataclass
class GraspResult:
    model_id: str
    passed: bool
    height_fraction: float | None = None
    yaw_deg: float | None = None
    width_mm: float | None = None
    detail: str = ""


@dataclass
class ObjectReport:
    model_id: str
    settle: SettleResult
    grasp: GraspResult
    override_quat: np.ndarray | None = None
    dropped: bool = False
    notes: list[str] = field(default_factory=list)


def _compile(obj: SceneObject) -> mujoco.MjModel:
    xml = build_object_scene_xml(
        [obj],
        ["probe_slot"],
        [0.5, 0.5, 0.5, 1.0],
        option_xml=MUJOCO_SCENE_OPTION_XML,
        robot_xml_path=_ROBOT_XML,
        model_name="pose_grasp_probe",
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", dir=str(get_so101_mujoco_model_path().parent), delete=False
    ) as f:
        f.write(xml)
        path = f.name
    model = mujoco.MjModel.from_xml_path(path)
    Path(path).unlink()
    return model


def _predicted_pose(model: mujoco.MjModel, obj: SceneObject) -> tuple[np.ndarray, float]:
    geom_ids = _slot_collision_geom_ids(model, "probe_slot", obj)
    verts = _body_frame_collision_verts(model, geom_ids)
    return get_mujoco_ycb_rest_pose(verts)


def _settle_from_quat(
    obj: SceneObject, model_id: str, quat: np.ndarray, spawn_z: float
) -> SettleResult:
    """Drop the object from ``quat`` 2 cm up and report where it actually settles."""
    model = _compile(obj)
    data = mujoco.MjData(model)
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "probe_slot_joint")
    qadr = model.jnt_qposadr[joint_id]
    start_pos = np.array([0.15, 0.0, spawn_z + _SETTLE_DROP_M])
    data.qpos[qadr : qadr + 3] = start_pos
    data.qpos[qadr + 3 : qadr + 7] = quat
    mujoco.mj_forward(model, data)
    for _ in range(_SETTLE_STEPS):
        mujoco.mj_step(model, data)

    settled_pos = data.qpos[qadr : qadr + 3].copy()
    settled_quat = data.qpos[qadr + 3 : qadr + 7].copy()
    translation_delta = float(
        max(
            np.linalg.norm(settled_pos[:2] - start_pos[:2]),
            abs(settled_pos[2] - spawn_z),
        )
    )
    return SettleResult(
        model_id=model_id,
        predicted_quat=quat,
        predicted_spawn_z=spawn_z,
        settled_pos=settled_pos,
        settled_quat=settled_quat,
        translation_delta_m=translation_delta,
        rotation_delta_deg=_quat_angle_deg(quat, settled_quat),
    )


def _spawn_z_for_quat(obj: SceneObject, quat: np.ndarray) -> float:
    """Return the floor-clearance spawn height for an arbitrary candidate quaternion."""
    model = _compile(obj)
    geom_ids = _slot_collision_geom_ids(model, "probe_slot", obj)
    verts = _body_frame_collision_verts(model, geom_ids)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    rotated = verts @ rot.reshape(3, 3).T
    return float(-rotated[:, 2].min()) + 0.002


def run_settle_test(obj: SceneObject, model_id: str) -> SettleResult:
    """Settle-test the heuristic's own predicted rest quaternion."""
    model = _compile(obj)
    quat, spawn_z = _predicted_pose(model, obj)
    return _settle_from_quat(obj, model_id, quat, spawn_z)


def run_grasp_check(obj: SceneObject, model_id: str, settled_quat: np.ndarray) -> GraspResult:
    """Find a reachable cross-section within the jaw's validated aperture interval.

    ``settled_quat`` fixes the object's orientation (from the settle test);
    only orientation matters for the shape's cross-section, not position.
    Slices near the top/bottom 15% of the object's height are skipped so the
    result is not a corner or rim, matching the spec's "not the raw AABB".
    """
    model = _compile(obj)
    geom_ids = _slot_collision_geom_ids(model, "probe_slot", obj)
    verts_body = _body_frame_collision_verts(model, geom_ids)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, settled_quat)
    world = verts_body @ rot.reshape(3, 3).T
    z_lo, z_hi = float(world[:, 2].min()), float(world[:, 2].max())
    height = z_hi - z_lo
    if height <= 0.0:
        return GraspResult(model_id=model_id, passed=False, detail="degenerate zero-height mesh")

    best: tuple[float, float, float] | None = None
    for frac in _GRASP_HEIGHT_FRACTIONS:
        slice_z = z_lo + frac * height
        band = world[np.abs(world[:, 2] - slice_z) <= _SLICE_HALF_THICKNESS_M]
        if band.shape[0] < 4:
            continue
        for yaw_deg in _GRASP_YAWS_DEG:
            yaw = np.radians(yaw_deg)
            closing_width = float(np.ptp(band[:, 0] * np.cos(yaw) + band[:, 1] * np.sin(yaw)))
            in_band = _JAW_MIN_APERTURE_M <= closing_width <= _JAW_MAX_APERTURE_M
            if in_band and (best is None or closing_width < best[0]):
                best = (closing_width, frac, yaw_deg)

    if best is None:
        return GraspResult(
            model_id=model_id,
            passed=False,
            detail=(
                "no height/yaw slice within the "
                f"{_JAW_MIN_APERTURE_M * 1000:.0f}-{_JAW_MAX_APERTURE_M * 1000:.0f} mm "
                "jaw aperture interval"
            ),
        )
    width, frac, yaw_deg = best
    return GraspResult(
        model_id=model_id,
        passed=True,
        height_fraction=frac,
        yaw_deg=yaw_deg,
        width_mm=width * 1000,
    )


def _axis_up_seeds() -> list[np.ndarray]:
    """Starting orientations for the override search: each convex-hull axis up.

    These are only starting points for a real settle simulation; the
    candidate is wherever physics actually put the object, which may differ
    from the seed it was dropped from.
    """
    return [
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.7071068, 0.0, 0.7071068, 0.0]),
        np.array([0.7071068, 0.7071068, 0.0, 0.0]),
    ]


def validate_one(obj: SceneObject, model_id: str) -> ObjectReport:
    """Settle-test ``obj``, correcting the pose if unstable; grasp-screen is advisory only."""
    settle = run_settle_test(obj, model_id)
    grasp = run_grasp_check(obj, model_id, settle.settled_quat)
    report = ObjectReport(model_id=model_id, settle=settle, grasp=grasp)

    if settle.passed:
        return report

    report.notes.append(
        f"heuristic settle failed (dx={settle.translation_delta_m * 1000:.1f}mm, "
        f"drot={settle.rotation_delta_deg:.1f}deg)"
    )

    candidate_seeds = [settle.predicted_quat, settle.settled_quat, *_axis_up_seeds()]
    seen: list[np.ndarray] = []
    for seed in candidate_seeds:
        seed_settle = _settle_from_quat(obj, model_id, seed, _spawn_z_for_quat(obj, seed))
        candidate = seed_settle.settled_quat
        if any(_quat_angle_deg(candidate, u) < 5.0 for u in seen):
            continue
        seen.append(candidate)
        # Re-verify the candidate is a genuine fixed point, not a mid-tumble
        # snapshot after only 500 steps.
        cand_settle = _settle_from_quat(obj, model_id, candidate, _spawn_z_for_quat(obj, candidate))
        if not cand_settle.passed:
            continue
        report.settle = cand_settle
        report.grasp = run_grasp_check(obj, model_id, cand_settle.settled_quat)
        report.override_quat = candidate
        report.notes.append(f"override accepted: quat={candidate.tolist()}")
        return report

    report.dropped = True
    report.notes.append("no candidate pose settled stably")
    return report


def _collect_targets(only: list[str] | None) -> list[tuple[str, SceneObject]]:
    targets: list[tuple[str, SceneObject]] = []
    for model_id in YCB_OBJECTS:
        if only and model_id not in only:
            continue
        ensure_ycb_assets(model_id)
        targets.append((model_id, YCBObject(model_id=model_id)))
    for model_id in GSO_OBJECTS:
        if only and model_id not in only:
            continue
        ensure_gso_assets(model_id)
        targets.append((model_id, GSOObject(model_id=model_id)))
    return targets


def _print_report(report: ObjectReport) -> None:
    print(f"=== {report.model_id} ===", flush=True)
    print(
        f"  settle: pass={report.settle.passed} "
        f"dx={report.settle.translation_delta_m * 1000:.2f}mm "
        f"drot={report.settle.rotation_delta_deg:.2f}deg",
        flush=True,
    )
    print(
        f"  grasp (advisory): pass={report.grasp.passed} "
        f"height_frac={report.grasp.height_fraction} yaw={report.grasp.yaw_deg} "
        f"width_mm={report.grasp.width_mm} {report.grasp.detail}",
        flush=True,
    )
    if report.override_quat is not None:
        print(f"  OVERRIDE quat={report.override_quat.tolist()}", flush=True)
    if report.dropped:
        print("  DROPPED: no candidate pose settled stably", flush=True)
    for note in report.notes:
        print(f"  note: {note}", flush=True)


def _print_summary(reports: list[ObjectReport]) -> None:
    print("\n=== summary ===")
    for report in reports:
        if report.dropped:
            status = "DROP"
        elif report.override_quat is not None:
            status = "OVERRIDE"
        else:
            status = "OK"
        print(
            f"{report.model_id:70s} settle={report.settle.passed!s:5s} "
            f"grasp(advisory)={report.grasp.passed!s:5s} -> {status}"
        )


def _write_results(reports: list[ObjectReport], json_out: Path) -> None:
    payload = {
        r.model_id: {
            "settle_pass": r.settle.passed,
            "settle_translation_delta_mm": r.settle.translation_delta_m * 1000,
            "settle_rotation_delta_deg": r.settle.rotation_delta_deg,
            "grasp_pass": r.grasp.passed,
            "grasp_height_fraction": r.grasp.height_fraction,
            "grasp_yaw_deg": r.grasp.yaw_deg,
            "grasp_width_mm": r.grasp.width_mm,
            "override_quat": r.override_quat.tolist() if r.override_quat is not None else None,
            "dropped": r.dropped,
            "notes": r.notes,
        }
        for r in reports
    }
    json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {json_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these model_ids")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "src"
        / "so101_nexus"
        / "gso_pose_validation_results.json",
    )
    args = parser.parse_args()

    reports = []
    for model_id, obj in _collect_targets(args.only):
        report = validate_one(obj, model_id)
        reports.append(report)
        _print_report(report)

    _print_summary(reports)
    if args.json_out:
        _write_results(reports, args.json_out)

    if any(r.dropped for r in reports):
        dropped = [r.model_id for r in reports if r.dropped]
        print(f"\nFAIL: {len(dropped)} object(s) dropped, no pose settled stably: {dropped}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
