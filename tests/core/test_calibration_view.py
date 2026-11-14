"""Reference and measured robot geometry share one interactive coordinate frame."""

import numpy as np
import pytest

from so101_nexus.calibration.view import render_overlay


@pytest.mark.usefixtures("calibration_plotly")
@pytest.mark.parametrize("provisional", [False, True])
def test_overlay_distinguishes_reference_from_measured(provisional):
    figure = render_overlay((0, 0, 0, 0, 0, 0), (15, 0, 0, 0, 0, 0), provisional=provisional)
    reference, measured = figure.data
    assert reference.type == measured.type == "mesh3d"
    assert reference.name == "Reference (ghost)"
    assert measured.name == ("Measured (provisional)" if provisional else "Measured")
    assert reference.opacity < measured.opacity
    assert reference.color != measured.color
    assert reference.showlegend
    assert measured.showlegend
    assert len(reference.x) == len(measured.x) > 100
    assert not np.allclose(reference.x, measured.x)
    assert figure.layout.scene.xaxis.title.text == "X (m)"
    assert figure.layout.scene.yaxis.title.text == "Y (m)"
    assert figure.layout.scene.zaxis.title.text == "Z (m)"
    assert figure.layout.uirevision == "calibration-overlay:front"


@pytest.mark.usefixtures("calibration_plotly")
def test_missing_measurement_has_no_invented_measured_pose():
    figure = render_overlay((0, 0, 0, 0, 0, 0), None)
    assert len(figure.data) == 1
    assert "No live measurement" in figure.layout.annotations[0].text


@pytest.mark.usefixtures("calibration_plotly")
def test_active_joint_axis_uses_forward_kinematics():
    figure = render_overlay((0, 0, 0, 0, 0, 0), None, active_joint="shoulder_lift")
    reference, axis = figure.data
    assert reference.type == "mesh3d"
    assert axis.type == "scatter3d"
    assert axis.name == "Active joint: shoulder lift"
    assert np.linalg.norm(
        np.array([axis.x[-1] - axis.x[0], axis.y[-1] - axis.y[0], axis.z[-1] - axis.z[0]])
    ) == pytest.approx(0.09)


@pytest.mark.usefixtures("calibration_plotly")
@pytest.mark.parametrize("angles", [(0, 0), (0, 0, 0, 0, 0, float("nan"))])
def test_overlay_rejects_invalid_degree_vectors(angles):
    with pytest.raises(ValueError, match="six finite"):
        render_overlay(angles, None)


@pytest.mark.usefixtures("calibration_plotly")
def test_overlay_rejects_unknown_joint():
    with pytest.raises(ValueError, match="Unknown active joint"):
        render_overlay((0, 0, 0, 0, 0, 0), None, active_joint="missing")


@pytest.mark.usefixtures("calibration_plotly")
def test_identical_poses_have_identical_geometry_and_independent_data():
    angles = (20, -30, 40, -20, 15, 30)
    figure = render_overlay(angles, angles, provisional=False)
    np.testing.assert_array_equal(figure.data[0].x, figure.data[1].x)
    before = np.array(figure.data[0].x)
    render_overlay((0, 0, 0, 0, 0, 0), None)
    np.testing.assert_array_equal(figure.data[0].x, before)
    assert len(figure.to_json()) < 700_000


@pytest.mark.usefixtures("calibration_plotly")
@pytest.mark.parametrize("measured", [None, (75, -25, 50, 30, 60, 80)])
def test_default_framing_fits_actual_reference_and_measured_geometry(measured):
    figure = render_overlay((0, 0, 0, 0, 0, 45), measured)
    vertices = np.concatenate(
        [np.column_stack((trace.x, trace.y, trace.z)) for trace in figure.data]
    )
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    for index, name in enumerate(("xaxis", "yaxis", "zaxis")):
        bounds = getattr(figure.layout.scene, name).range
        assert bounds[0] < lower[index]
        assert bounds[1] > upper[index]
        assert (upper[index] - lower[index]) / (bounds[1] - bounds[0]) > 0.8
        assert sum(bounds) / 2 == pytest.approx((lower[index] + upper[index]) / 2, abs=1e-6)
    camera = figure.layout.scene.camera
    assert camera.projection.type == "orthographic"
    assert camera.eye.x == 0
    assert camera.eye.z == 0
    assert camera.up.z == 1


@pytest.mark.usefixtures("calibration_plotly")
@pytest.mark.parametrize("view", ["front", "side", "top"])
def test_named_views_preserve_world_geometry_and_keep_camera_stable_with_live_updates(view):
    reference_deg = (0, -15, 30, 45, 0, 50)
    empty = render_overlay(reference_deg, view=view)
    live = render_overlay(reference_deg, (10, -20, 30, 45, 0, 50), view=view)
    np.testing.assert_array_equal(empty.data[0].x, live.data[0].x)
    np.testing.assert_array_equal(empty.data[0].z, live.data[0].z)
    assert empty.layout.uirevision == live.layout.uirevision
    assert empty.layout.scene.camera == live.layout.scene.camera
    assert live.layout.scene.camera.projection.type == "orthographic"
    if view == "top":
        assert live.layout.scene.camera.up.y == 1
    else:
        assert live.layout.scene.camera.up.z == 1


@pytest.mark.usefixtures("calibration_plotly")
def test_changing_named_view_resets_camera_without_transforming_meshes():
    reference_deg = (0, -15, 30, 45, 0, 50)
    front = render_overlay(reference_deg, view="front")
    side = render_overlay(reference_deg, view="side")
    np.testing.assert_array_equal(front.data[0].x, side.data[0].x)
    assert front.layout.uirevision != side.layout.uirevision
    assert front.layout.scene.camera.eye != side.layout.scene.camera.eye


@pytest.mark.usefixtures("calibration_plotly")
def test_explicit_framing_bounds_stay_fixed_across_monitor_updates():
    reference_deg = (0, -15, 30, 45, 0, 50)
    first = render_overlay(reference_deg)
    bounds = tuple(getattr(first.layout.scene, name).range for name in ("xaxis", "yaxis", "zaxis"))
    update = render_overlay(reference_deg, (5, -20, 30, 45, 0, 50), bounds_m=bounds)
    for name, expected in zip(("xaxis", "yaxis", "zaxis"), bounds, strict=True):
        assert getattr(update.layout.scene, name).range == expected
    assert update.layout.uirevision == first.layout.uirevision


@pytest.mark.usefixtures("calibration_plotly")
@pytest.mark.parametrize("view", ["isometric", "", "FRONT"])
def test_overlay_rejects_unknown_named_view(view):
    with pytest.raises(ValueError, match="view"):
        render_overlay((0, 0, 0, 0, 0, 45), view=view)


@pytest.mark.usefixtures("calibration_plotly")
def test_overlay_bounds_match_rendered_mesh_union_without_mutating_model_coordinates():
    from so101_nexus.calibration.view import overlay_bounds

    reference_deg = (0, 0, 0, 0, 0, 45)
    measured_deg = (75, -25, 50, 30, 60, 80)
    bounds = overlay_bounds(reference_deg, measured_deg)
    figure = render_overlay(reference_deg, measured_deg)
    for name, expected in zip(("xaxis", "yaxis", "zaxis"), bounds, strict=True):
        assert getattr(figure.layout.scene, name).range == expected


@pytest.mark.usefixtures("calibration_plotly")
@pytest.mark.parametrize("viewport_aspect", [0.6, 1.5, 2.0])
def test_framing_preserves_equal_meter_scale_at_portrait_and_wide_viewports(viewport_aspect):
    figure = render_overlay((0, 0, 0, 0, 0, 45), viewport_aspect=viewport_aspect)
    scene = figure.layout.scene
    spans = np.array([np.ptp(getattr(scene, name).range) for name in ("xaxis", "yaxis", "zaxis")])
    ratios = np.array([scene.aspectratio.x, scene.aspectratio.y, scene.aspectratio.z])
    np.testing.assert_allclose(ratios / spans, np.repeat(ratios[0] / spans[0], 3))
    assert np.linalg.norm(ratios) == pytest.approx(1.85 * min(viewport_aspect, 1))


@pytest.mark.usefixtures("calibration_plotly")
@pytest.mark.parametrize("viewport_aspect", [0, -1, float("nan"), True])
def test_overlay_rejects_invalid_viewport_shape(viewport_aspect):
    with pytest.raises(ValueError, match="viewport_aspect"):
        render_overlay((0, 0, 0, 0, 0, 45), viewport_aspect=viewport_aspect)


@pytest.mark.usefixtures("calibration_plotly")
@pytest.mark.parametrize("bounds", [((0, 1),), ((0, 1), (0, 0), (0, 1))])
def test_overlay_rejects_invalid_fixed_meter_ranges(bounds):
    with pytest.raises(ValueError, match="bounds_m"):
        render_overlay((0, 0, 0, 0, 0, 45), bounds_m=bounds)


@pytest.mark.usefixtures("calibration_plotly")
@pytest.mark.parametrize("focus_joint", ["wrist_flex", "wrist_roll"])
def test_joint_focus_preserves_geometry_and_frames_wrist_with_upstream_context(focus_joint):
    from so101_nexus.calibration.view import overlay_bounds

    reference_deg = (0, 0, 0, 0, 0, 45)
    measured_deg = (0, 0, 0, 15, 30, 70)
    whole = render_overlay(reference_deg, measured_deg)
    focused = render_overlay(reference_deg, measured_deg, focus_joint=focus_joint)
    for original, detail in zip(whole.data, focused.data, strict=True):
        for coordinate in ("x", "y", "z"):
            np.testing.assert_array_equal(
                getattr(original, coordinate), getattr(detail, coordinate)
            )
    full_bounds = overlay_bounds(reference_deg, measured_deg)
    focus_bounds = overlay_bounds(reference_deg, measured_deg, focus_joint=focus_joint)
    assert focus_bounds[0][0] > full_bounds[0][0]
    assert focus_bounds[2][0] > full_bounds[2][0]
    assert focus_bounds[0][1] >= full_bounds[0][1] - 0.04
    assert focus_bounds[0][0] < 0.23, "Keep some forearm visible before the wrist hinge"
    assert focused.layout.uirevision != whole.layout.uirevision
    assert focused.layout.scene.camera == whole.layout.scene.camera


def test_focused_bounds_include_measured_jaw_in_both_poses():
    import mujoco

    from so101_nexus import get_so101_mujoco_model_path
    from so101_nexus.calibration.view import overlay_bounds

    model = mujoco.MjModel.from_xml_path(str(get_so101_mujoco_model_path()))
    reference_deg = (0, 0, 0, 0, 0, 45)
    measured_deg = (0, 0, 0, -30, 90, 85)
    bounds = np.array(overlay_bounds(reference_deg, measured_deg, focus_joint="wrist_roll"))
    for pose_deg in (reference_deg, measured_deg):
        data = mujoco.MjData(model)
        for name, angle_deg in zip(
            ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"),
            pose_deg,
            strict=True,
        ):
            data.qpos[model.joint(name).qposadr[0]] = np.deg2rad(angle_deg)
        mujoco.mj_forward(model, data)
        for body_name in ("gripper", "moving_jaw_so101_v1"):
            position = data.xpos[model.body(body_name).id]
            assert np.all(position > bounds[:, 0])
            assert np.all(position < bounds[:, 1])


@pytest.mark.usefixtures("calibration_plotly")
def test_joint_focus_persists_during_timer_updates_and_rejects_unknown_joint():
    reference_deg = (0, 0, 0, 0, 0, 45)
    first = render_overlay(reference_deg, focus_joint="wrist_roll")
    next_frame = render_overlay(reference_deg, (0, 0, 0, 0, 2, 45), focus_joint="wrist_roll")
    assert first.layout.uirevision == next_frame.layout.uirevision
    with pytest.raises(ValueError, match="focus_joint"):
        render_overlay(reference_deg, focus_joint="jaw_position")


@pytest.mark.usefixtures("calibration_plotly")
@pytest.mark.parametrize("view", ["front", "side", "top"])
def test_camera_updates_omit_presets_unless_reset_is_requested(view):
    figure = render_overlay((0, 0, 0, 0, 0, 45), view=view, camera_reset=False)
    assert "camera" not in figure.layout.scene.to_plotly_json()
    camera = {"eye": {"x": 0.7, "y": -0.4, "z": 0.6}, "projection": {"type": "orthographic"}}
    restored = render_overlay((0, 0, 0, 0, 0, 45), view=view, camera=camera, camera_reset=False)
    assert restored.layout.scene.camera.to_plotly_json() == camera
    reset = render_overlay((0, 0, 0, 0, 0, 45), view=view, camera_reset=True)
    assert reset.layout.scene.camera.projection.type == "orthographic"


@pytest.mark.usefixtures("calibration_plotly")
@pytest.mark.parametrize("pose", [(0, 0, 0, 0, 0, 45), (75, -25, 50, 30, 60, 80)])
@pytest.mark.parametrize("viewport_aspect", [0.6, 1.5, 2.0])
def test_arbitrary_orbit_fits_the_entire_bounding_sphere(pose, viewport_aspect):
    figure = render_overlay(pose, viewport_aspect=viewport_aspect)
    scene = figure.layout.scene
    ratios = np.array([scene.aspectratio.x, scene.aspectratio.y, scene.aspectratio.z])
    spans = np.array([np.ptp(getattr(scene, name).range) for name in ("xaxis", "yaxis", "zaxis")])
    center = np.array([np.mean(getattr(scene, name).range) for name in ("xaxis", "yaxis", "zaxis")])
    points = np.column_stack((figure.data[0].x, figure.data[0].y, figure.data[0].z))
    normalized = (points - center) / spans * ratios
    for azimuth in np.linspace(0, 2 * np.pi, 13):
        for elevation in np.linspace(-np.pi / 2, np.pi / 2, 7):
            right = np.array([np.cos(azimuth), np.sin(azimuth), 0])
            up = np.array(
                [
                    -np.sin(azimuth) * np.sin(elevation),
                    np.cos(azimuth) * np.sin(elevation),
                    np.cos(elevation),
                ]
            )
            assert np.max(np.abs(normalized @ right)) < viewport_aspect
            assert np.max(np.abs(normalized @ up)) < 1


@pytest.mark.usefixtures("calibration_plotly")
def test_expand_bounds_keeps_encoder_jitter_fixed_and_encloses_large_pose_changes():
    from so101_nexus.calibration.view import expand_bounds, overlay_bounds

    reference = (0, 0, 0, 0, 0, 45)
    initial = overlay_bounds(reference)
    assert expand_bounds(initial, reference, (0.01, 0, 0, 0, 0, 45)) == initial
    measured = (75, -25, 50, 30, 60, 80)
    expanded = np.array(expand_bounds(initial, reference, measured))
    assert np.all(expanded[:, 0] <= np.array(initial)[:, 0])
    assert np.all(expanded[:, 1] >= np.array(initial)[:, 1])
    assert np.any(expanded != np.array(initial))
    figure = render_overlay(reference, measured)
    for trace in figure.data:
        points = np.column_stack((trace.x, trace.y, trace.z))
        assert np.all(points > expanded[:, 0])
        assert np.all(points < expanded[:, 1])
    assert expand_bounds(expanded.tolist(), reference, reference) == tuple(map(tuple, expanded))


def test_expand_bounds_honors_joint_focus_and_validates_existing_ranges():
    from so101_nexus.calibration.view import expand_bounds, overlay_bounds

    reference = (0, 0, 0, 0, 0, 45)
    initial = overlay_bounds(reference, focus_joint="wrist_roll")
    assert expand_bounds(initial, reference, reference, focus_joint="wrist_roll") == initial
    with pytest.raises(ValueError, match="bounds_m"):
        expand_bounds(((0, 0),), reference, reference)
