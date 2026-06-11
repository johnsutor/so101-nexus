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
  site's local z axis is the gripper "forward"/gaze direction (wrist toward
  fingertips); the look-at task and TCP-orientation observations depend on it.
  The site position is unchanged from upstream.

The vendored `scene.xml` / `scene_box.xml` are kept for provenance only; the
runtime synthesizes its own scene wrappers and does not load them.
