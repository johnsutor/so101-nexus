"""Measure both missing and excess distal jaw collision coverage.

Use the visual-ray grid from ``tests/core/test_jaw_collision_geometry.py``.
Signed surface depth measures containment in the collider union, not exact
normal-offset error or jaw aperture. Historical runs load all model assets
from the requested Git revision. Results are deterministic JSON on stdout.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

import mujoco
import numpy as np

from so101_nexus import get_so101_mujoco_model_path

_MODEL_DIRECTORY = Path("src/so101_nexus/assets/SO101_menagerie")
_REPOSITORY = Path(__file__).resolve().parents[1]
_PROBE_RADIUS_M = 1e-7
_GRID_M = 0.0005
_SURFACES = (
    ("gripper", "wrist_roll_follower_so101_v1", -1, (-0.008, 0.008), (-0.104, -0.080)),
    ("moving_jaw_so101_v1", "moving_jaw_so101_v1", 1, (-0.082, -0.060), (0.011, 0.027)),
)


def _compile(path: Path) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(path))
    spec.worldbody.add_geom(
        name="surface_probe",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[_PROBE_RADIUS_M, 0, 0],
        contype=0,
        conaffinity=0,
    )
    return spec.compile()


def _load_model(revision: str | None) -> mujoco.MjModel:
    if revision is None:
        return _compile(get_so101_mujoco_model_path())
    archive = subprocess.check_output(
        ["git", "archive", revision, str(_MODEL_DIRECTORY)], cwd=_REPOSITORY
    )
    with tempfile.TemporaryDirectory() as directory:
        with tarfile.open(fileobj=io.BytesIO(archive)) as contents:
            contents.extractall(directory, filter="data")
        return _compile(Path(directory) / _MODEL_DIRECTORY / "so101.xml")


def _surface_gaps(model, data, body, mesh_name, sign, y_range, z_range) -> np.ndarray:
    body_id = model.body(body).id
    geoms = np.flatnonzero(model.geom_bodyid == body_id)
    visual = next(
        int(g)
        for g in geoms
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        and model.mesh(model.geom_dataid[g]).name == mesh_name
    )
    colliders = [int(g) for g in geoms if model.geom_contype[g]]
    probe = model.geom("surface_probe").id
    rotation = data.xmat[body_id].reshape(3, 3)
    direction = rotation @ np.array([sign, 0, 0])
    gaps = []
    for y in np.arange(*y_range, _GRID_M):
        for z in np.arange(*z_range, _GRID_M):
            origin = data.xpos[body_id] + rotation @ np.array([-sign * 0.06, y, z])
            distance = mujoco.mj_rayMesh(model, data, visual, origin, direction)
            if distance < 0:
                continue
            data.geom_xpos[probe] = origin + distance * direction
            gaps.append(
                min(mujoco.mj_geomDistance(model, data, g, probe, 0.1, None) for g in colliders)
                + _PROBE_RADIUS_M
            )
    if not gaps:
        raise ValueError(f"No visual surface samples found for {body}")
    return np.asarray(gaps)


def main() -> None:
    """Audit the working model or a historical Git revision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", help="Git revision containing the model and mesh assets")
    args = parser.parse_args()
    model = _load_model(args.revision)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    surfaces = {}
    for surface in _SURFACES:
        gaps = _surface_gaps(model, data, *surface)
        surfaces[surface[0]] = {
            "samples": len(gaps),
            "maximum_uncovered_gap_mm": float(max(0, gaps.max()) * 1000),
            "surface_depth_mm": dict(
                zip(
                    ("minimum", "median", "p95", "maximum"),
                    np.percentile(-gaps * 1000, [0, 50, 95, 100]).tolist(),
                    strict=True,
                )
            ),
            "fraction_surface_depth_over_0_3_mm": float(np.mean(gaps < -0.0003)),
        }
    print(
        json.dumps(
            {
                "script": "scripts/grasp_geometry_audit.py",
                "revision": args.revision or "working-tree",
                "mujoco_version": mujoco.__version__,
                "grid_m": _GRID_M,
                "probe_radius_m": _PROBE_RADIUS_M,
                "metric": "signed visible-surface depth in collider union, not aperture error",
                "surfaces": surfaces,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
