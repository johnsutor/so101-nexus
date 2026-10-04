"""Inactive object slots must affect physics without appearing in camera images."""

import mujoco_warp as mjw
import pytest
import torch
import warp as wp

from so101_nexus import (
    CubeObject,
    MeshObject,
    OverheadCamera,
    PickAndPlaceV2Config,
    PickConfig,
    RenderConfig,
    StackCubeConfig,
)

pytestmark = pytest.mark.warp


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("mode", ["rgb_array", "depth_array"])
def test_inactive_slots_match_removed_geometry_without_mutating_physics(
    env_factory, tmp_path, device, mode
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    mesh_path = tmp_path / "tetrahedron.obj"
    mesh_path.write_text(
        "v -0.02 -0.02 -0.02\nv 0.02 -0.02 -0.02\nv 0 0.02 -0.02\nv 0 0 0.02\n"
        "f 1 3 2\nf 1 2 4\nf 2 3 4\nf 3 1 4\n"
    )
    mesh = MeshObject(str(mesh_path), str(mesh_path), mass=0.01, name="tetrahedron")
    config = PickConfig(
        objects=[CubeObject(color="red"), CubeObject(color="green"), mesh],
        observations=[OverheadCamera(width=64, height=48, modalities=("rgb", "depth"))],
        render=RenderConfig(width=160, height=120, camera="side"),
    )
    env = env_factory(
        backend="warp", task="PickLift", config=config, device=device, render_mode=mode
    )
    targets = [0, 1]
    env.reset(seed=42, options={"target_index": targets})
    visual = env.mjm.geom("pick_slot_2_visual").id
    assert not env._slot_geom_masks[2, visual]
    assert env._hidden_render_geoms[:, visual].all()
    physical = {
        name: wp.to_torch(getattr(env.data, name)).clone()
        for name in ("qpos", "qvel", "ctrl", "geom_xpos", "geom_xmat", "nacon")
    }
    contacts = wp.to_torch(env.data.contact.geom).clone()
    actual = env.render().clone()
    obs = env._compute_obs()
    for name, snapshot in physical.items():
        torch.testing.assert_close(wp.to_torch(getattr(env.data, name)), snapshot)
    torch.testing.assert_close(wp.to_torch(env.data.contact.geom), contacts)

    # Remove inactive reference geoms without recomputing active transforms:
    # forward() would advance render poses to the final integrated qpos.
    positions = wp.to_torch(env.data.geom_xpos)
    for world, target in enumerate(targets):
        for slot, address in enumerate(env._slot_qadr_host):
            if slot != target:
                joint = list(env.mjm.jnt_qposadr).index(address)
                body = env.mjm.jnt_bodyid[joint]
                geoms = torch.as_tensor(env.mjm.geom_bodyid == body, device=env.device)
                positions[world, geoms, 2] = -100.0
    with wp.ScopedDevice(env._wp_device):
        mjw.refit_bvh(env.model, env.data, env._visual_render_ctx)
        mjw.render(env.model, env.data, env._visual_render_ctx)
        expected = env._read_camera_image(
            env._visual_render_ctx, 0, 160, 120, "rgb" if mode == "rgb_array" else "depth"
        )
        mjw.refit_bvh(env.model, env.data, env._render_ctx)
        mjw.render(env.model, env.data, env._render_ctx)
        expected_obs = {
            "overhead_camera"
            if modality == "rgb"
            else "overhead_camera_depth": env._read_camera_image(
                env._render_ctx, 0, 64, 48, modality
            )
            for modality in ("rgb", "depth")
        }
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for key in ("overhead_camera", "overhead_camera_depth"):
        torch.testing.assert_close(obs[key], expected_obs[key], rtol=0, atol=0)


@pytest.mark.parametrize("task", ["PickLift", "PickAndPlace", "StackCube"])
def test_visibility_tracks_active_distractors_and_partial_resets(env_factory, task):
    objects = [CubeObject(color=color) for color in ("red", "green", "blue")]
    distractors = [CubeObject(color=color) for color in ("orange", "purple", "white")]
    common = {"render": RenderConfig(width=32, height=24, camera="side")}
    if task == "StackCube":
        config = StackCubeConfig(
            cube_a_colors=["red", "green"],
            cube_b_colors=["blue", "yellow"],
            distractors=distractors,
            n_distractors=1,
            **common,
        )
        count = 3
    elif task == "PickAndPlace":
        config = PickAndPlaceV2Config(
            objects=objects, distractors=distractors, n_distractors=1, **common
        )
        count = 2
    else:
        config = PickConfig(objects=objects, n_distractors=1, **common)
        count = 2
    env = env_factory(
        backend="warp",
        task=task,
        version=2 if task == "PickAndPlace" else 1,
        config=config,
        render_mode="rgb_array",
    )
    env.reset(seed=3)

    def assert_visibility():
        for world in range(env.num_envs):
            active_count = 0
            for address, geom_mask in zip(env._slot_qadr_host, env._slot_geom_masks, strict=True):
                active = env.qpos[world, address] > -0.1
                active_count += int(active)
                assert torch.all(env._hidden_render_geoms[world, geom_mask] == ~active)
            assert active_count == count

    assert_visibility()
    untouched = env._hidden_render_geoms[1].clone()
    untouched_qpos = env.qpos[1].clone()
    env._write_reset_state(torch.tensor([True, False], device=env.device))
    assert_visibility()
    torch.testing.assert_close(env._hidden_render_geoms[1], untouched)
    torch.testing.assert_close(env.qpos[1], untouched_qpos)


def test_state_only_slots_do_not_allocate_render_masks_or_copies(env_factory):
    env = env_factory(
        backend="warp",
        task="PickLift",
        config=PickConfig(objects=[CubeObject(color="red"), CubeObject(color="green")]),
    )
    env.reset(seed=0)
    assert env._slot_render_geom_masks is None
    assert env._hidden_render_geoms is None
    assert env._render_data_view is None
