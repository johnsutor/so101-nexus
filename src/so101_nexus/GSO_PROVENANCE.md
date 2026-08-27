# Provenance

GSO (Google Scanned Objects) assets are not vendored in this repository. They
are downloaded on first use, the same way YCB assets are (see
`so101_nexus.gso_assets`).

- Original dataset: [Google Scanned Objects](https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/)
  (1,030 objects), published by Google Research.
- License: CC-BY-4.0. Confirmed per-object via the Fuel API, e.g.
  `https://fuel.gazebosim.org/1.0/GoogleResearch/models/<model_id>`
  (`license_name: "Creative Commons Attribution 4.0 International"`).
- Mesh mirror used to build the Hugging Face dataset this library downloads
  from: [kevinzakka/mujoco_scanned_objects](https://github.com/kevinzakka/mujoco_scanned_objects)
  (real-world-scale `.obj` + PNG texture, no rescale needed). The MJCF files
  in that repository are MIT-licensed; the `.obj`/`.png` mesh and texture
  assets themselves stay CC-BY-4.0 from Google.
- Hugging Face mirror: the `SO101_GSO_HF_REPO` environment variable default
  (`so101_nexus.gso_assets._HF_REPO_ID`), following the same directory layout
  as the YCB mirror (`ai-habitat/ycb`).

## Object selection

12 objects were curated from the full 1,030-object GSO corpus (see
`GSO_OBJECT_IMPORT.md` at the repository root for the original selection
criteria and dimensions). Scissors was deliberately excluded from the GSO
selection because it duplicates an existing YCB category
(`037_scissors`); no salt/pepper shaker exists in the GSO corpus.

## Mass estimation

Unlike YCB, GSO ships no benchmark-measured masses. Every `model_id` in
`GSO_MASSES` (`so101_nexus.constants`) uses a hand-estimated mass: convex-hull
volume from the mirrored scan times an assumed effective density for the
filled/packaged object, rounded to the nearest 5 g. See the inline comment on
each entry for the volume and assumed density. Pass `mass_override` to
`GSOObject` to replace an estimate with a measured value.

## Pose and grasp validation

`so101_nexus.ycb_geometry.get_mujoco_ycb_rest_pose()`'s default heuristic
(rotate the mesh's thinnest AABB axis to point up) is a bounding-box guess,
not a physics result. `scripts/validate_object_rest_poses.py` settle-tests
every supported YCB and GSO model against real MuJoCo physics and records a
corrected quaternion in `POSE_OVERRIDES` for any model whose heuristic pose is
dynamically unstable. The same script also runs an advisory (non-gating)
geometric grasp screen; objects it flags are recorded in `GRASP_ADVISORY` with
a short note, not removed from the supported set. See the module docstring of
`so101_nexus.ycb_geometry` for the exact methodology and its limits, and
`src/so101_nexus/gso_pose_validation_results.json` for full per-object
results.
