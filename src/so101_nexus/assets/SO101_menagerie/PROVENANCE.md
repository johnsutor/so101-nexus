# Provenance

This directory vendors the `robotstudio_so101` model from MuJoCo Menagerie
(google-deepmind/mujoco_menagerie), originally the SO-101 arm from The Robot
Studio, developed by I2RT Robotics.

- Upstream: https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101
- Pinned commit: 4c358ef9d9d7f32ca58b40b490884a0c1726a440
- License: Apache 2.0 (see LICENSE)

## Local modifications

- Removed the upstream `so101.png` render preview to keep the wheel small.
- `so101.xml`: removed the wrist camera's physical intrinsics (sensorsize/focal)
  and set an explicit `fovy="48.5"`, so the simulator's fovy-based FOV
  randomization keeps working (MuJoCo ignores `fovy` when physical intrinsics
  are present). A notice comment near the top of `so101.xml` records this.
  The value 48.5 degrees is the vertical FOV equivalent of the original
  intrinsics: `2 * atan(0.00324 / (2 * 0.0036)) ~= 48.5`.
- `so101.xml`: changed the `gripperframe` site quaternion from `1 0 1 0` to
  `0 0 1 0` (upstream used `1 0 1 0`). The library's convention is that the TCP
  site's local z axis is the gripper "forward" direction (wrist toward
  fingertips); TCP-orientation observations depend on it. (The look-at task
  uses the wrist camera optical axis, not this site.)
  The site position is unchanged from upstream.
- `so101.xml`: set `solref="0.005 1"` and `solimp="0.95 0.999 0.001 0.5 2"`
  on both gripper collision classes to reduce penetration while retaining
  tangential compliance. Numerical defaults match the shared `PhysicsConfig`
  preset. Material friction and actuator limits are unchanged.
- `so101.xml`: added six convex pad-edge meshes derived from the existing visual
  jaw triangles in body coordinates, with 64 vertices per hull. Fixed-jaw
  side strips cover `z <= -0.080 m` and `y <= -0.003 m` or `y >= 0.003 m`;
  its tip covers `z <= -0.101 m`. Moving-jaw side strips cover `y <= -0.060 m`
  and `z <= 0.01575 m` or `z >= 0.02175 m`; its tip covers `y <= -0.079 m`.
  These convex strips cover visible edges omitted by the primitive approximation
  while retaining the original central contact surfaces and open grasp aperture.
  `tests/core/test_jaw_collision_geometry.py` checks visible distal surfaces
  against the compiled collider union with `mj_geomDistance` on a 0.5 mm ray grid.

The vendored `scene.xml` / `scene_box.xml` are kept for provenance only; the
runtime synthesizes its own scene wrappers and does not load them.
