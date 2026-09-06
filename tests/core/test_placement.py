"""Analytic union geometry and physical support contracts for placement v2."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from so101_nexus.object_slots import ObjectSlot, pyramid_xml_asset
from so101_nexus.objects import CubeObject, PyramidObject
from so101_nexus.placement import (
    SlotGeometry,
    compiled_slot_geometry,
    footprint_centroid,
    support_state,
    visual_reference,
)


def _rectangle(x0, y0, x1, y1):
    return np.array(
        [[[x0, y0, 0], [x1, y0, 0], [x1, y1, 0]], [[x0, y0, 0], [x1, y1, 0], [x0, y1, 0]]],
        dtype=float,
    )


def _l_shape():
    return np.concatenate((_rectangle(0, 0, 4, 1), _rectangle(0, 0, 1, 4)))


def _yaw(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _compiled(geoms, *, assets="", body_attributes="", children=""):
    model = mujoco.MjModel.from_xml_string(
        '<mujoco><compiler angle="radian"/>'
        f'<asset>{assets}</asset><worldbody><body name="object" {body_attributes}>'
        '<freejoint/><inertial mass="0.02" pos="0 0 0" diaginertia="0.001 0.001 0.001"/>'
        f"{geoms}{children}</body></worldbody></mujoco>"
    )
    slot = ObjectSlot(
        qpos_addr=0,
        dof_addr=0,
        geom_ids=(0,),
        rest_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        spawn_z=0.0,
        bounding_radius=1.0,
        obj=CubeObject(),
    )
    return model, slot


def _l_mesh(offset=(0, 0, 0)):
    outline = np.array([[0, 0], [4, 0], [4, 1], [1, 1], [1, 4], [0, 4]], dtype=float)
    vertices = np.concatenate(
        (np.column_stack((outline, np.full(6, -0.2))), np.column_stack((outline, np.full(6, 0.2))))
    )
    cap = np.array([[0, 1, 3], [1, 2, 3], [0, 3, 5], [3, 4, 5]])
    faces = [*cap[:, ::-1], *(cap + 6)]
    for i in range(6):
        j = (i + 1) % 6
        faces.extend(([i, j, j + 6], [i, j + 6, i + 6]))
    return vertices + offset, np.asarray(faces)


def _mesh_asset(vertices, faces, *, scale="1 1 1"):
    vertex_text = " ".join(map(str, vertices.ravel()))
    face_text = " ".join(map(str, faces.ravel()))
    return f'<mesh name="visual" vertex="{vertex_text}" face="{face_text}" scale="{scale}"/>'


def test_concave_l_uses_union_area_not_hull_or_triangle_average():
    # Two area-4 bars overlap in an area-1 square. Each first moment is 9.5.
    np.testing.assert_allclose(footprint_centroid(_l_shape()), [19 / 14, 19 / 14], atol=1e-12)


def test_ring_keeps_the_offset_hole():
    # Outer 6x4 rectangle minus the 2x2 hole centered at (2, 2).
    ring = np.concatenate(
        (
            _rectangle(0, 0, 1, 4),
            _rectangle(3, 0, 6, 4),
            _rectangle(1, 0, 3, 1),
            _rectangle(1, 3, 3, 4),
        )
    )
    np.testing.assert_allclose(footprint_centroid(ring), [3.2, 2.0], atol=1e-12)


def test_partial_triangle_overlap_counts_once():
    triangles = np.array(
        [[[0, 0, 0], [4, 0, 0], [0, 4, 0]], [[2, 0, 0], [6, 0, 0], [2, 4, 0]]], dtype=float
    )
    # Areas 8 + 8 - 2 give the union's moments (32, 20).
    np.testing.assert_allclose(footprint_centroid(triangles), [16 / 7, 10 / 7], atol=2e-5)


def test_duplicate_contained_and_edge_on_faces_do_not_change_union():
    outer = _rectangle(-4, -3, 2, 1)
    inner = _rectangle(-3, -2, -2, -1)
    edge_on = np.array([[[-3, -2, 0], [-3, 0, 0], [-3, -2, 1]]], dtype=float)
    triangles = np.concatenate((outer, outer[:, ::-1], inner, edge_on))
    np.testing.assert_allclose(footprint_centroid(triangles), [-1.0, -1.0], atol=1e-12)


def test_scanline_vertex_ties_do_not_fill_empty_space():
    # A horizontal edge lies exactly on a midpoint scanline. Empty intervals
    # from remote vertical faces must not extend the real union to the origin.
    triangles = np.concatenate(
        (
            _rectangle(-8, 0, -6, 2),
            np.array([[[-20, 0, 0], [-20, 2, 0], [-20, 1, 1]]], dtype=float),
        )
    )
    np.testing.assert_allclose(footprint_centroid(triangles, scanlines=3), [-7, 1], atol=1e-12)


@pytest.mark.parametrize("angle", [0.37, np.pi / 2, 1.15])
def test_rotated_concave_centroid_has_sub_scanline_error(angle):
    rotation = _yaw(angle)
    expected = (np.array([19 / 14, 19 / 14, 0]) @ rotation.T)[:2]
    rotated = _l_shape() @ rotation.T
    # These canonical polygons stay within 2 mm at unit-metre scale.
    # This is less than 0.15 of a production Y step, not a universal bound.
    np.testing.assert_allclose(footprint_centroid(rotated), expected, atol=0.002)


def test_translation_and_asset_frame_change_preserve_centroid():
    rotation = _yaw(0.37)
    translation = np.array([1000.0, -2000.0, 4.0])
    frame_shift = np.array([17.0, -9.0, 3.0])
    local = _l_shape()
    world = local @ rotation.T + translation
    reframed = (local + frame_shift) @ rotation.T + (translation - frame_shift @ rotation.T)
    np.testing.assert_allclose(footprint_centroid(world), footprint_centroid(reframed), atol=1e-10)
    np.testing.assert_allclose(
        footprint_centroid(world) - translation[:2],
        footprint_centroid(local @ rotation.T),
        atol=1e-10,
    )


def test_elongated_tilted_box_has_its_symmetry_center():
    model, slot = _compiled(
        '<geom type="box" size="3 0.04 0.1" pos="2 -1 0.7" euler="0.7 0.4 0.32"/>'
    )
    triangles = compiled_slot_geometry(model, [slot])[0].triangles
    np.testing.assert_allclose(footprint_centroid(triangles), [2, -1], atol=1e-12)


@pytest.mark.parametrize(
    "geom",
    [
        'type="box" size="3 0.04 0.1"',
        'type="cylinder" size="0.4 0.7"',
        'type="sphere" size="0.4"',
    ],
)
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_symmetric_primitive_reference_transforms_under_tilt(geom, backend):
    model, slot = _compiled(f'<geom {geom} pos="2 -1 0.7" euler="0.7 0.4 0.32"/>')
    geometry = compiled_slot_geometry(model, [slot])[0]
    assert geometry.symmetry_center is not None
    rotation = np.stack(
        [_yaw(0.4), _yaw(-0.7) @ np.array([[1, 0, 0], [0, 0.8, -0.6], [0, 0.6, 0.8]])]
    )
    position = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]])
    center = (rotation @ np.array([2.0, -1.0, 0.7])) + position
    vertices = geometry.triangles.reshape(-1, 3)
    minimum_z = (vertices @ rotation.swapaxes(-1, -2) + position[:, None, :])[..., 2].min(-1)
    if backend == "torch":
        torch = pytest.importorskip("torch")
        geometry = SlotGeometry(
            torch.from_numpy(geometry.triangles), torch.from_numpy(geometry.symmetry_center)
        )
        rotation, position = torch.from_numpy(rotation), torch.from_numpy(position)
    reference = visual_reference(geometry, rotation, position)
    np.testing.assert_allclose(reference[:, :2], center[:, :2], atol=1e-12)
    np.testing.assert_allclose(reference[:, 2], minimum_z, atol=1e-12)


def test_pyramid_side_projection_uses_area_centroid():
    model, slot = _compiled(
        '<geom type="mesh" mesh="primitive_pyramid_0" euler="1.5707963267948966 0 0"/>',
        assets=pyramid_xml_asset(0, PyramidObject(half_size=1.0)),
    )
    triangles = compiled_slot_geometry(model, [slot])[0].triangles
    # The side projection is a triangle, not the vertex or body-origin mean.
    np.testing.assert_allclose(footprint_centroid(triangles), [0, 1 / 3], atol=2e-5)


def test_compiled_visual_union_ignores_collision_hull():
    vertices, faces = _l_mesh()
    model, slot = _compiled(
        '<geom name="collision" type="box" size="4 4 1" group="3"/>'
        '<geom name="object_visual" type="mesh" mesh="visual" group="2" mass="0"/>',
        assets=_mesh_asset(vertices, faces),
    )
    triangles = compiled_slot_geometry(model, [slot])[0].triangles
    np.testing.assert_allclose(footprint_centroid(triangles), [19 / 14, 19 / 14], atol=1e-6)


def test_compiled_mesh_scale_offset_and_geom_rotation_apply_once():
    vertices, faces = _l_mesh(offset=(7, -3, 2))
    rotation = _yaw(0.43)
    scale = np.array([1.7, 0.8, 0.6])
    position = np.array([3, -2, 0.7])
    model, slot = _compiled(
        '<geom type="mesh" mesh="visual" pos="3 -2 0.7" euler="0 0 0.43"/>',
        assets=_mesh_asset(vertices, faces, scale="1.7 0.8 0.6"),
        body_attributes='pos="10 20 30" euler="0.2 0.4 0.6"',
    )
    triangles = compiled_slot_geometry(model, [slot])[0].triangles
    expected = (vertices[faces] * scale) @ rotation.T + position
    # Compare the visible surface rather than the compiler's reordered vertices.
    np.testing.assert_allclose(
        footprint_centroid(triangles), footprint_centroid(expected), atol=3e-6
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    root_rotation = data.xmat[model.body("object").id].reshape(3, 3)
    root_position = data.xpos[model.body("object").id]
    np.testing.assert_allclose(
        footprint_centroid(triangles @ root_rotation.T + root_position),
        footprint_centroid(expected @ root_rotation.T + root_position),
        atol=3e-6,
    )


def test_compiled_asset_frame_translation_preserves_visual_centroid():
    vertices, faces = _l_mesh()
    base, slot = _compiled('<geom type="mesh" mesh="visual"/>', assets=_mesh_asset(vertices, faces))
    moved, moved_slot = _compiled(
        '<geom type="mesh" mesh="visual" pos="-7 3 -2"/>',
        assets=_mesh_asset(vertices + np.array([7, -3, 2]), faces),
    )
    np.testing.assert_allclose(
        footprint_centroid(compiled_slot_geometry(base, [slot])[0].triangles),
        footprint_centroid(compiled_slot_geometry(moved, [moved_slot])[0].triangles),
        atol=1e-6,
    )


def test_rigid_child_geom_uses_root_body_frame():
    model, slot = _compiled(
        '<geom type="box" size="0.1 0.1 0.1" group="3"/>',
        body_attributes='pos="10 20 30" euler="0.2 0.4 0.6"',
        children='<body pos="3 4 0" euler="0 0 1.5707963267948966">'
        '<geom type="box" size="0.2 0.3 0.1" pos="2 0 0" group="2"/></body>',
    )
    np.testing.assert_allclose(
        footprint_centroid(compiled_slot_geometry(model, [slot])[0].triangles), [3, 6], atol=1e-12
    )


@pytest.mark.parametrize(
    "geom",
    ['<geom type="ellipsoid" size="1 2 3"/>', '<geom type="box" size="1 2 3" group="3"/>'],
)
def test_unsupported_or_missing_visual_surface_fails(geom):
    model, slot = _compiled(geom)
    with pytest.raises(ValueError, match="Placement"):
        compiled_slot_geometry(model, [slot])


def test_compiled_degenerate_surface_fails():
    model, slot = _compiled('<geom type="box" size="1 2 3"/>')
    model.geom_size[0] = [0, 0, 0]
    with pytest.raises(ValueError, match="zero-area"):
        compiled_slot_geometry(model, [slot])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_batch_chunks_and_numpy_torch_parity(dtype, monkeypatch):
    from so101_nexus import placement

    torch = pytest.importorskip("torch")
    base = _l_shape().astype(dtype)
    translations = np.arange(18, dtype=dtype).reshape(2, 3, 1, 1, 3)
    triangles = base + translations
    expected = footprint_centroid(triangles)
    # Force independent world and scanline chunks with the same public input.
    monkeypatch.setattr(placement, "_MAX_INTERVAL_VALUES", 17)
    numpy_result = footprint_centroid(triangles)
    tensor_result = footprint_centroid(torch.from_numpy(triangles))
    np.testing.assert_allclose(numpy_result, expected, atol=3e-6)
    np.testing.assert_allclose(tensor_result.numpy(), expected, atol=3e-6)
    assert numpy_result.shape == (2, 3, 2)
    assert numpy_result.dtype == dtype
    assert tensor_result.dtype == torch.from_numpy(triangles).dtype


def test_torch_cuda_retains_device_and_geometry():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    triangles = torch.tensor(_l_shape(), dtype=torch.float32, device="cuda")
    result = footprint_centroid(triangles)
    assert result.device == triangles.device
    np.testing.assert_allclose(result.cpu().numpy(), [19 / 14, 19 / 14], atol=1e-6)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("scanlines", [0, -1, 1.5, True])
def test_invalid_scanline_count_fails(backend, scanlines):
    triangles = _l_shape()
    if backend == "torch":
        triangles = pytest.importorskip("torch").from_numpy(triangles)
    with pytest.raises(ValueError, match="scanlines"):
        footprint_centroid(triangles, scanlines=scanlines)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize(
    "triangles,error",
    [
        (np.empty((0, 3, 3)), ValueError),
        (np.zeros((2, 3, 2)), ValueError),
        (np.zeros((1, 3, 3), dtype=np.int64), TypeError),
        (np.full((1, 3, 3), np.nan), ValueError),
        (np.full((1, 3, 3), np.inf), ValueError),
        (np.zeros((1, 3, 3)), ValueError),
        (np.array([[[0, 0, 0], [1, 1, 0], [2, 2, 0]]], dtype=float), ValueError),
    ],
)
def test_invalid_or_degenerate_projection_fails(backend, triangles, error):
    if backend == "torch":
        triangles = pytest.importorskip("torch").from_numpy(triangles)
    with pytest.raises(error):
        footprint_centroid(triangles)


_SUPPORT = {
    "min_weight_fraction": 0.9,
    "force_tolerance": 0.001,
    "relative_force_tolerance": 0.01,
    "lin_threshold": 0.01,
    "ang_threshold": 0.5,
}


def test_table_support_requires_upward_weight_fraction():
    table = np.array([-1.0, 0.0, 0.899, 0.9, 1.2])
    states = support_state(
        table, np.zeros(5), np.zeros(5), np.ones(5), np.zeros(5), np.zeros(5), **_SUPPORT
    )
    assert states[0].tolist() == [False, False, False, True, True]
    assert states[4].tolist() == [False, False, False, True, True]


def test_absolute_and_weight_relative_force_limits_include_equality():
    weights = np.array([0.05, 0.05, 2.0, 2.0])
    robot = np.array([0.001, 0.00101, 0.02, 0.02001])
    other = robot[::-1].copy()
    states = support_state(weights, robot, other, weights, np.zeros(4), np.zeros(4), **_SUPPORT)
    assert states[1].tolist() == [False, True, False, True]
    assert states[2].tolist() == [True, True, False, False]
    assert states[4].tolist() == [False, False, True, False]


def test_both_motion_thresholds_gate_physical_qualification():
    linear = np.array([0.01, 0.01001, 0.0, 0.0])
    angular = np.array([0.5, 0.0, 0.50001, 0.0])
    states = support_state(
        np.ones(4), np.zeros(4), np.zeros(4), np.ones(4), linear, angular, **_SUPPORT
    )
    assert states[3].tolist() == [True, False, False, True]
    assert states[4].tolist() == [True, False, False, True]


def test_support_numpy_torch_broadcast_parity():
    torch = pytest.importorskip("torch")
    inputs = (
        np.array([[0.0], [1.0], [2.0]]),
        np.array([0.0, 0.1]),
        np.array([0.1, 0.0]),
        np.array([0.05, 2.0]),
        np.array([[0.0], [0.02], [0.0]]),
        np.array([0.0, 0.0]),
    )
    expected = support_state(*inputs, **_SUPPORT)
    got = support_state(*(torch.from_numpy(value) for value in inputs), **_SUPPORT)
    for numpy_state, tensor_state in zip(expected, got, strict=True):
        np.testing.assert_array_equal(tensor_state.numpy(), numpy_state)


def test_tensor_centroid_preserves_translation_gradients():
    torch = pytest.importorskip("torch")
    offset = torch.tensor([0.3, -0.2, 0.5], dtype=torch.float64, requires_grad=True)
    center = footprint_centroid(torch.from_numpy(_l_shape()) + offset)
    center.sum().backward()
    torch.testing.assert_close(offset.grad, torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64))
