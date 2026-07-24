"""End-effector control modes on the batched Warp backend.

The load-bearing test here is ``test_ik_matches_the_mujoco_backend``: both
backends hand-roll their own iteration loop around the shared solver in
``so101_nexus.kinematics`` (MuJoCo iterates a scratch ``MjData`` in float64,
Warp iterates the live batched ``Data`` in float32), and nothing at import time
forces those two loops to stay in step. Everything else in this file pins the
Warp-specific parts of that loop: batching, world isolation, and the promise
that running inverse kinematics inside ``step`` leaves the simulation state
alone.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.warp

NUM_ENVS = 4
EE_MODES = ("pd_ee_pose", "pd_ee_delta_pose")

# float32 Warp state against float64 MuJoCo state: what limits agreement is the
# single-precision Jacobian, not the algorithm. Measured max over the sampled
# configurations is ~5e-07 rad, so 1e-4 leaves three orders of headroom without
# being loose enough to hide a real semantic divergence.
CROSS_BACKEND_TOL = 1e-4


def _make(control_mode, num_envs=NUM_ENVS, seed=0, robot=None):
    from so101_nexus.config import PickConfig
    from so101_nexus.warp.pick_env import WarpPickLiftVectorEnv

    env = WarpPickLiftVectorEnv(
        num_envs=num_envs,
        config=PickConfig(robot=robot),
        device="cpu",
        seed=seed,
        control_mode=control_mode,
    )
    env.reset(seed=seed)
    return env


def _refresh(env):
    """Recompute derived kinematics so ``site_xpos`` matches ``qpos`` exactly.

    ``_ee_ik_arm_targets`` restores ``qpos`` but leaves the derived fields at the
    solved configuration, because ``step`` always follows it with ``mjw.step``.
    Tests that call the seam out of band have to close that gap themselves.
    """
    import mujoco_warp as mjw
    import warp as wp

    with wp.ScopedDevice(env._wp_device):
        mjw.forward(env.model, env.data)


def _pin_robot(env, robot_qpos):
    """Put every world in the same six-joint robot configuration at rest.

    The gripper is included: leaving it at its per-world reset value would let
    the isolation test below pass on a technicality, because the gripper clamp
    saturates those differences away.
    """
    import torch

    env.qpos[:, env._qpos_adr] = torch.as_tensor(robot_qpos, dtype=torch.float32)
    env.qvel[:] = 0.0
    _refresh(env)


@pytest.mark.parametrize("mode", EE_MODES)
def test_ee_modes_expose_a_seven_dim_batched_action_space(mode):
    env = _make(mode)
    assert env.single_action_space.shape == (7,)
    assert env.action_space.shape == (NUM_ENVS, 7)


def test_ee_pose_action_space_spans_the_workspace_and_the_gripper_range():
    import numpy as np

    from so101_nexus.warp.base_env import _EE_WORKSPACE_RADIUS

    env = _make("pd_ee_pose")
    space = env.single_action_space
    np.testing.assert_allclose(space.low[:3], -_EE_WORKSPACE_RADIUS)
    np.testing.assert_allclose(space.high[:3], _EE_WORKSPACE_RADIUS)
    np.testing.assert_allclose(space.low[3:6], -np.pi, rtol=1e-6)
    np.testing.assert_allclose(space.high[3:6], np.pi, rtol=1e-6)
    # The gripper channel is a plain joint target, on pd_joint_pos's bounds.
    assert space.low[6] == pytest.approx(float(env._target_low[-1]))
    assert space.high[6] == pytest.approx(float(env._target_high[-1]))


def test_delta_pose_action_space_is_the_normalized_box():
    import numpy as np

    space = _make("pd_ee_delta_pose").single_action_space
    np.testing.assert_array_equal(space.low, -np.ones(7, dtype=np.float32))
    np.testing.assert_array_equal(space.high, np.ones(7, dtype=np.float32))


def test_delta_pose_solves_each_world_in_isolation():
    """Worlds sharing a configuration but not a command must not bleed.

    Every world is pinned to the same robot pose, then given a different command.
    Row ``i`` of the batched solve has to equal the solve where *every* world
    issues world ``i``'s command; a Jacobian sliced or reduced across the world
    axis would fail this and still look plausible per-world.
    """
    import torch

    env = _make("pd_ee_delta_pose")
    _pin_robot(env, [0.1, -0.9, 0.9, 0.4, 0.05, 0.6])

    mixed = torch.zeros((NUM_ENVS, 7))
    mixed[0, 0] = 1.0  # +x
    mixed[1, 0] = -1.0  # -x
    mixed[2, 2] = 1.0  # +z
    mixed[3, 5] = 1.0  # +wz, orientation only
    batched = env._action_to_ctrl(mixed)

    for i in range(NUM_ENVS):
        _refresh(env)
        uniform = env._action_to_ctrl(mixed[i].expand(NUM_ENVS, 7).contiguous())
        assert torch.equal(uniform[i], batched[i]), i
        # Identical configuration plus identical command must give identical rows.
        assert torch.equal(uniform, uniform[0].expand(NUM_ENVS, 6)), i

    # Guard against the assertions above passing because every row is the same.
    assert not torch.equal(batched[0], batched[1])


def test_delta_pose_moves_each_world_along_its_own_commanded_axis():
    """Physics-level direction check, paired against a zero-action baseline.

    The arm is a position-servoed 5-DoF chain under gravity, so the raw TCP
    displacement mixes the commanded motion with free drift. Differencing
    against an identically seeded env stepping zeros isolates the effect of the
    command. Cross-axis motion is deliberately not asserted small: with only
    five joints and orientation de-weighted, a pure single-axis Cartesian path
    is not realizable.
    """
    import torch

    commanded = _make("pd_ee_delta_pose")
    baseline = _make("pd_ee_delta_pose")
    start = commanded._get_tcp_pose7()[:, :3].clone()
    assert torch.equal(start, baseline._get_tcp_pose7()[:, :3])

    axes = torch.tensor([0, 0, 2, 2])
    signs = torch.tensor([1.0, -1.0, 1.0, -1.0])
    action = torch.zeros((NUM_ENVS, 7))
    action[torch.arange(NUM_ENVS), axes] = signs
    zeros = torch.zeros((NUM_ENVS, 7))
    for _ in range(20):
        commanded.step(action)
        baseline.step(zeros)

    drift = baseline._get_tcp_pose7()[:, :3] - start
    moved = commanded._get_tcp_pose7()[:, :3] - start
    relative = moved - drift
    signed = relative[torch.arange(NUM_ENVS), axes] * signs
    assert (signed > 0.02).all(), signed
    # Worlds 0/1 and 2/3 command opposite signs on a shared axis, so their
    # command-attributable displacements must have opposite signs.
    assert relative[0, 0] > 0 > relative[1, 0]
    assert relative[2, 2] > 0 > relative[3, 2]
    assert drift.abs().max() < 0.01


def test_delta_pose_gripper_channel_is_a_clamped_joint_delta():
    """The seventh channel is ``pd_joint_delta_pos``'s gripper, not a pose term.

    It integrates from the measured gripper qpos at ``EE_DELTA_ACTION_SCALE[6]``
    radians per unit and saturates on the joint-target bounds. An in-box action
    can still ask for an out-of-range target, so the clamp is reachable here in
    a way it is not for the absolute mode, whose action space already ends at
    those bounds.
    """
    import torch

    from so101_nexus.kinematics import EE_DELTA_ACTION_SCALE

    env = _make("pd_ee_delta_pose")
    low = float(env._target_low[-1])
    high = float(env._target_high[-1])
    scale = EE_DELTA_ACTION_SCALE[6]
    mid = 0.5 * (low + high)
    start = torch.tensor([high, low, mid, mid])

    arm = torch.tensor([0.1, -0.9, 0.9, 0.4, 0.05])
    env.qpos[:, env._arm_qpos_adr] = arm
    env.qpos[:, env._qpos_adr[-1]] = start
    env.qvel[:] = 0.0
    _refresh(env)

    action = torch.zeros((NUM_ENVS, 7))
    action[:, 6] = torch.tensor([1.0, -1.0, 1.0, -1.0])
    gripper = env._action_to_ctrl(action)[:, -1]

    assert gripper[0] == pytest.approx(high)
    assert gripper[1] == pytest.approx(low)
    assert gripper[2] == pytest.approx(mid + scale, abs=1e-6)
    assert gripper[3] == pytest.approx(mid - scale, abs=1e-6)


@pytest.mark.parametrize("mode", EE_MODES)
def test_ik_leaves_qpos_bit_identical(mode):
    """Solving must not disturb the state it iterates through.

    The Warp IK writes trial configurations into the live ``qpos`` view rather
    than a scratch copy, so this covers the arm block round-tripping *and* the
    object's free-joint entries never being touched at all.
    """
    import torch

    env = _make(mode)
    _refresh(env)
    before = env.qpos.clone()
    if mode == "pd_ee_pose":
        action = _hold_current_pose_action(env)
    else:
        action = torch.full((NUM_ENVS, 7), 0.5)
    env._action_to_ctrl(action)
    assert torch.equal(before, env.qpos)
    assert torch.isfinite(env.qpos).all()


def test_ee_step_matches_the_equivalent_joint_target_step():
    """An EE step is exactly a joint-target step at the solved targets.

    Both envs reset from the same seed to the same state; the EE env resolves
    and applies its own targets, then those same targets are replayed through a
    ``pd_joint_pos`` env. Any residue the IK left in the simulation state (a
    perturbed ``qpos``, a stale velocity) would show up as divergence here.
    """
    import torch

    ee = _make("pd_ee_delta_pose")
    joint = _make("pd_joint_pos")
    assert torch.equal(ee.qpos, joint.qpos)
    assert torch.equal(ee.qvel, joint.qvel)

    action = torch.zeros((NUM_ENVS, 7))
    action[:, 0] = 1.0
    action[:, 4] = -0.5
    _, _, terminated, truncated, _ = ee.step(action)
    assert not bool((terminated | truncated).any())

    joint.step(ee.ctrl[:, ee._act_ids].clone())
    assert torch.equal(ee.qpos, joint.qpos)
    assert torch.equal(ee.qvel, joint.qvel)


def _hold_current_pose_action(env):
    """``pd_ee_pose`` action commanding exactly the pose the arm already holds."""
    import torch

    from so101_nexus.kinematics import quat_to_rotvec

    pose = env._get_tcp_pose7()
    gripper = env._joint_qpos()[:, -1:]
    action = torch.cat([pose[:, :3], quat_to_rotvec(pose[:, 3:]), gripper], dim=1)
    return torch.clamp(action, env._action_low, env._action_high)


def test_ee_pose_commanding_the_current_pose_is_a_fixed_point():
    import torch

    env = _make("pd_ee_pose")
    _refresh(env)
    action = _hold_current_pose_action(env)
    # The commanded position must be a real point of the action space, not a
    # clamped one, or the fixed point would be an artifact of the clamp.
    assert torch.equal(action[:, :6], torch.cat([env._get_tcp_pose7()[:, :3], action[:, 3:6]], 1))

    arm_qpos = env._joint_qpos()[:, :-1]
    ctrl = env._action_to_ctrl(action)
    assert torch.allclose(ctrl[:, :-1], arm_qpos, atol=1e-4)
    assert torch.allclose(ctrl[:, -1], action[:, -1], atol=1e-6)

    start = env._get_tcp_pose7()[:, :3].clone()
    for _ in range(20):
        env.step(action)
    assert torch.linalg.norm(env._get_tcp_pose7()[:, :3] - start, dim=1).max() < 5e-3


@pytest.mark.parametrize("tuned_robot", [False, True], ids=["default_knobs", "tuned_knobs"])
@pytest.mark.parametrize("mode", EE_MODES)
def test_ik_matches_the_mujoco_backend(mode, tuned_robot):
    """Same configuration, same command, same joint targets in both backends.

    Driven through each backend's ``_action_to_ctrl`` seam from matched states
    with physics bypassed, so the comparison is of the two IK loops rather than
    of two integrators that are not expected to agree step for step. Warp runs
    all configurations as a batch, MuJoCo one at a time, which also pins the
    batched Jacobian slice against the single-sample reference.

    Run twice: once on the defaults, once on non-default
    ``ee_orientation_weight`` / ``ee_delta_action_scale``, so the two backends
    cannot read the same knob differently.
    """
    import gymnasium as gym
    import mujoco
    import numpy as np
    import torch

    import so101_nexus.mujoco  # noqa: F401 - registers MuJoCo*-v1
    from so101_nexus.config import PickConfig, RobotConfig
    from so101_nexus.kinematics import quat_to_rotvec

    n = 16
    robot = (
        RobotConfig(
            ee_orientation_weight=0.2,
            ee_delta_action_scale=(0.03, 0.03, 0.03, 0.25, 0.25, 0.25, 0.1),
        )
        if tuned_robot
        else None
    )
    warp_env = _make(mode, num_envs=n, robot=robot)
    mj_env = gym.make("MuJoCoPickLift-v1", control_mode=mode, config=PickConfig(robot=robot))
    mj = mj_env.unwrapped
    mj.reset(seed=0)
    try:
        assert warp_env.mjm.nq == mj.model.nq
        np.testing.assert_array_equal(warp_env._arm_qpos_adr.numpy(), mj._arm_qpos_addrs)

        rng = np.random.default_rng(3)
        low = warp_env._target_low.numpy()
        high = warp_env._target_high.numpy()
        # Arm joints stay off their limits so the comparison exercises the
        # solver rather than the clamp that follows it. The gripper does the
        # opposite: the first four worlds sit on the bounds so both backends
        # have to saturate identically.
        arm = rng.uniform(0.7 * low[:5], 0.7 * high[:5], size=(n, 5))
        gripper = rng.uniform(low[5], high[5], size=n)
        gripper[:4] = [low[5], high[5], low[5], high[5]]
        robot_qpos = np.concatenate([arm, gripper[:, None]], axis=1)
        object_qpos = mj.data.qpos[6:].copy()

        warp_env.qpos[:, warp_env._qpos_adr] = torch.as_tensor(robot_qpos, dtype=torch.float32)
        warp_env.qpos[:, 6:] = torch.as_tensor(object_qpos, dtype=torch.float32)
        warp_env.qvel[:] = 0.0
        _refresh(warp_env)

        if mode == "pd_ee_delta_pose":
            action = rng.uniform(-1.0, 1.0, size=(n, 7))
        else:
            pose = warp_env._get_tcp_pose7()
            action = np.concatenate(
                [
                    pose[:, :3].numpy() + rng.uniform(-0.03, 0.03, size=(n, 3)),
                    quat_to_rotvec(pose[:, 3:]).numpy(),
                    gripper[:, None],
                ],
                axis=1,
            )

        warp_ctrl = warp_env._action_to_ctrl(torch.as_tensor(action, dtype=torch.float32)).numpy()

        mj_ctrl = np.empty((n, 6))
        for i in range(n):
            mj.data.qpos[:6] = robot_qpos[i]
            mj.data.qpos[6:] = object_qpos
            mj.data.qvel[:] = 0.0
            mujoco.mj_forward(mj.model, mj.data)
            mj_ctrl[i] = mj._action_to_ctrl(action[i])

        np.testing.assert_allclose(warp_ctrl, mj_ctrl, atol=CROSS_BACKEND_TOL)
    finally:
        mj_env.close()


def test_ee_orientation_weight_is_a_live_knob():
    """``RobotConfig.ee_orientation_weight`` reaches the batched solve.

    A pure rotation command is the discriminating case: the weight scales both
    sides of the damped least-squares solve, so raising it trades position
    tracking for tool rotation authority. Two envs pinned to the same
    configuration must therefore resolve the same command differently.
    """
    import torch

    from so101_nexus.config import RobotConfig

    qpos = [0.1, -0.9, 0.9, 0.4, 0.05, 0.6]
    action = torch.zeros((NUM_ENVS, 7))
    action[:, 5] = 1.0

    targets = {}
    for weight in (0.01, 0.5):
        env = _make("pd_ee_delta_pose", robot=RobotConfig(ee_orientation_weight=weight))
        _pin_robot(env, qpos)
        targets[weight] = env._action_to_ctrl(action).clone()

    assert (targets[0.01] - targets[0.5]).abs().max() > 1e-3, targets


def test_ee_delta_action_scale_is_a_live_knob():
    """``RobotConfig.ee_delta_action_scale`` sets the physical step of a +/-1 action.

    Checked on the gripper channel, the one element of the seven that maps to a
    joint target directly rather than through the solver, so the assertion is on
    an exact number instead of a tracking tolerance.
    """
    import torch

    from so101_nexus.config import RobotConfig

    scale = (0.04, 0.04, 0.04, 0.25, 0.25, 0.25, 0.05)
    env = _make("pd_ee_delta_pose", robot=RobotConfig(ee_delta_action_scale=scale))
    _pin_robot(env, [0.1, -0.9, 0.9, 0.4, 0.05, 0.0])

    action = torch.zeros((NUM_ENVS, 7))
    action[:, 6] = 1.0
    gripper = env._action_to_ctrl(action)[:, -1]
    assert gripper[0] == pytest.approx(scale[6], abs=1e-6)


def test_cpu_device_falls_back_to_the_direct_solve():
    """CUDA graphs do not exist on the Warp CPU device; the eager solve must stand in."""
    env = _make("pd_ee_delta_pose")
    assert env._ik_graph is None


def _cuda_env(control_mode, num_envs=16, seed=0):
    """Build a CUDA-device env, skipping when no CUDA device is available."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")

    from so101_nexus.config import PickConfig
    from so101_nexus.warp.pick_env import WarpPickLiftVectorEnv

    env = WarpPickLiftVectorEnv(
        num_envs=num_envs,
        config=PickConfig(),
        device="cuda",
        seed=seed,
        control_mode=control_mode,
    )
    env.reset(seed=seed)
    return env


@pytest.mark.parametrize("mode", EE_MODES)
def test_captured_ik_graph_matches_the_direct_solve(mode):
    """The captured graph is the eager loop, bit for bit, and stays replayable.

    Both paths run from an identical pinned configuration with derived kinematics
    refreshed, so any divergence is the graph rather than accumulated simulation
    state. Replaying a second time also has to reproduce the first answer: a
    graph that captured a stale buffer address would pass once and then freeze.
    """
    import numpy as np
    import torch

    env = _cuda_env(mode)
    assert env._ik_graph is not None, "CUDA graph capture of the IK solve failed"

    rng = np.random.default_rng(11)
    n = env.num_envs
    low = env._target_low.cpu().numpy()
    high = env._target_high.cpu().numpy()
    robot_qpos = torch.as_tensor(
        rng.uniform(0.7 * low, 0.7 * high, size=(n, 6)).astype(np.float32), device=env.device
    )
    action = torch.as_tensor(
        rng.uniform(-1.0, 1.0, size=(n, 7)).astype(np.float32), device=env.device
    )
    if mode == "pd_ee_pose":
        action = torch.clamp(action, env._action_low, env._action_high)

    def solve():
        env.qpos[:, env._qpos_adr] = robot_qpos
        env.qvel[:] = 0.0
        _refresh(env)
        return env._action_to_ctrl(action).clone()

    from_graph = solve()
    graph, env._ik_graph = env._ik_graph, None
    from_eager = solve()
    env._ik_graph = graph
    replayed = solve()

    assert torch.equal(from_graph, from_eager)
    assert torch.equal(from_graph, replayed)


def test_captured_ik_graph_leaves_qpos_bit_identical():
    """Replay restores ``qpos`` exactly, as the eager solve does."""
    import torch

    env = _cuda_env("pd_ee_delta_pose")
    _refresh(env)
    before = env.qpos.clone()
    env._action_to_ctrl(torch.full((env.num_envs, 7), 0.5, device=env.device))
    assert torch.equal(before, env.qpos)
    assert torch.isfinite(env.qpos).all()


def test_captured_ik_graph_tracks_a_changing_command():
    """Successive replays must resolve the command written before each one.

    A graph that folded the warmup command into its own nodes instead of reading
    the persistent target buffers would return the same joint targets forever.
    """
    import torch

    env = _cuda_env("pd_ee_delta_pose")
    _refresh(env)
    plus = torch.zeros((env.num_envs, 7), device=env.device)
    plus[:, 0] = 1.0
    first = env._action_to_ctrl(plus).clone()
    _refresh(env)
    second = env._action_to_ctrl(-plus).clone()
    assert (first - second).abs().max() > 1e-3


# torch reports an empty graph because the synthetic failure aborts the capture
# body before a single node is recorded; that is the situation under test.
@pytest.mark.filterwarnings("ignore:The CUDA Graph is empty:UserWarning")
def test_ik_graph_capture_failure_falls_back_to_the_direct_solve(monkeypatch):
    """A failed capture must warn and degrade, never break construction.

    Also checks that the failure does not poison the process: a later env on the
    same device still captures. An aborted capture that left the CUDA caching
    allocator mid-capture would take every subsequent env down with it.
    """
    import torch

    from so101_nexus.warp import base_env as warp_base_env

    original = warp_base_env.SO101NexusWarpVectorEnv._solve_ee_ik
    calls = {"n": 0}

    def flaky(self):
        calls["n"] += 1
        # Call 1 is the pre-capture warmup; call 2 is the one being recorded.
        if calls["n"] == 2:
            raise RuntimeError("synthetic capture failure")
        return original(self)

    monkeypatch.setattr(warp_base_env.SO101NexusWarpVectorEnv, "_solve_ee_ik", flaky)
    with pytest.warns(RuntimeWarning, match="Inverse-kinematics CUDA graph capture failed"):
        env = _cuda_env("pd_ee_delta_pose", num_envs=8)
    monkeypatch.undo()

    assert env._ik_graph is None
    action = torch.zeros((env.num_envs, 7), device=env.device)
    action[:, 0] = 1.0
    assert torch.isfinite(env._action_to_ctrl(action)).all()

    later = _cuda_env("pd_ee_delta_pose", num_envs=8)
    assert later._ik_graph is not None
    assert torch.isfinite(later._action_to_ctrl(action)).all()
