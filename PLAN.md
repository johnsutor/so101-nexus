 Fix ManiSkill Environment Inconsistencies                                                                 

 Context

 The ManiSkill pick-and-place environment has two visible bugs (confirmed via
 utils/episode_frames_maniskill.png):
 1. The SO100/SO101 robot faces the wrong direction — it's not oriented toward the cube/target spawn area
 2. The target disc appears vertical (sticking up from ground) instead of lying flat

 Additionally, the MuJoCo visualization utility references a non-existent environment ID. The pick_cube
 environments share the same robot orientation bug.

 Root Causes

 Bug 1: Robot orientation (ManiSkill pick_and_place.py + pick_cube.py)

 Both SO100 and SO101 agents define a keyframe with pose=sapien.Pose(q=euler2quat(0, 0, np.pi/2)) — a 90°
 Z-rotation so the robot faces the +X direction (where objects spawn at center (0.15, 0)). However, both
 _load_agent() and _initialize_episode() override with sapien.Pose(p=[0, 0, 0]) (identity rotation),
 discarding the keyframe's orientation.

 Files affected:
 - packages/so101-nexus-maniskill/src/so101_nexus_maniskill/pick_and_place.py (lines 213, 318)
 - packages/so101-nexus-maniskill/src/so101_nexus_maniskill/pick_cube.py (lines 202, 300)

 Bug 2: Target disc orientation (ManiSkill pick_and_place.py)

 SAPIEN/PhysX cylinders have their axis along the local Y-axis by default (confirmed by the ground plane
 needing a rotation quaternion [0.7071, 0, -0.7071, 0] to rotate its +Y normal to +Z). The target disc
 cylinder is created via add_cylinder_visual() with no rotation, so the disc stands on edge instead of lying
  flat.

 File affected:
 - packages/so101-nexus-maniskill/src/so101_nexus_maniskill/pick_and_place.py (lines 280-286)

 Bug 3: MuJoCo visualization env name

 utils/visualize_env.py references MuJoCoPickAndPlaceSO100-v1 but the registered name is
 MuJoCoPickAndPlace-v1 (the MuJoCo env only supports SO101).

 File affected:
 - utils/visualize_env.py (line 20)

 Plan (Red/Green TDD)

 Step 1: Write failing tests for robot orientation

 Add TestRobotOrientation class to both ManiSkill test files.

 packages/so101-nexus-maniskill/tests/test_pick_and_place.py — add:
 class TestRobotOrientation:
     @pytest.mark.parametrize("env_id,robot", ENV_IDS)
     def test_robot_base_uses_keyframe_rotation(self, request, env_id, robot):
         """Robot base pose must use the keyframe's Z-rotation so it faces the workspace."""
         env = _get_env(request, env_id)
         env.reset()
         inner = env.unwrapped
         expected_q = inner.agent.keyframes["rest"].pose.q
         actual_q = inner.agent.robot.pose.q[0].cpu().numpy()
         np.testing.assert_allclose(actual_q, expected_q, atol=1e-4)

 packages/so101-nexus-maniskill/tests/test_pick_cube.py — add identical test for all pick_cube variants.

 Step 2: Write failing test for target disc orientation

 packages/so101-nexus-maniskill/tests/test_pick_and_place.py — add:
 class TestTargetDiscOrientation:
     @pytest.mark.parametrize("env_id,robot", ENV_IDS)
     def test_target_disc_lies_flat(self, request, env_id, robot):
         """Target disc cylinder must be rotated so it lies flat on the ground (axis along Z)."""
         env = _get_env(request, env_id)
         env.reset()
         target_q = env.unwrapped.target_site.pose.q[0].cpu().numpy()
         # After rotation, the cylinder's Y-axis should map to Z.
         # Quaternion for 90° around X: [cos(π/4), sin(π/4), 0, 0]
         expected_q = np.array([0.7071068, 0.7071068, 0.0, 0.0])
         np.testing.assert_allclose(np.abs(target_q), np.abs(expected_q), atol=1e-3)

 Step 3: Run tests → confirm RED

 Run tests to confirm they fail with the current code.

 Step 4: Fix robot orientation in pick_and_place.py

 In _load_agent (line 213):
 # Before:
 super()._load_agent(options, sapien.Pose(p=[0, 0, 0]))
 # After:
 super()._load_agent(options, sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, np.pi / 2)))

 In _initialize_episode (line 318):
 # Before:
 self.agent.robot.set_pose(sapien.Pose(p=[0, 0, 0]))
 # After:
 self.agent.robot.set_pose(sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, np.pi / 2)))

 Step 5: Fix robot orientation in pick_cube.py

 Same changes at lines 202 and 300.

 Step 6: Fix target disc orientation in pick_and_place.py

 In _load_scene (lines 280-286), add a pose with 90° X-rotation to the cylinder visual:
 builder.add_cylinder_visual(
     pose=sapien.Pose(q=[0.7071068, 0.7071068, 0, 0]),  # rotate Y-axis to Z-axis
     radius=self.target_disc_radius,
     half_length=0.001,
     material=sapien.render.RenderMaterial(
         base_color=target_rgba,
     ),
 )

 Step 7: Fix MuJoCo visualization env name

 In utils/visualize_env.py (line 20):
 # Before:
 ENV_ID = "MuJoCoPickAndPlaceSO100-v1"
 # After:
 ENV_ID = "MuJoCoPickAndPlace-v1"

 Step 8: Run tests → confirm GREEN

 Run all test suites:
 cd packages/so101-nexus-maniskill && python -m pytest tests/ -v
 cd packages/so101-nexus-mujoco && python -m pytest tests/ -v

 Step 9: Regenerate visualization frames

 Run utils/visualize_env_maniskill.py and utils/visualize_env.py to regenerate the episode frame PNGs and
 visually confirm the robot now faces the workspace and the target disc lies flat.

 Verification

 1. All existing tests continue to pass (no regressions)
 2. New orientation tests pass (GREEN)
 3. Regenerated episode_frames_maniskill.png shows:
   - Robot facing +X toward the cube/target spawn area
   - Target disc lying flat on the ground as a circle (not a vertical line)
 4. MuJoCo visualization utility runs without gymnasium.error.NameNotFound