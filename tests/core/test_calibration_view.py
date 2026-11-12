"""Reference and measured robot geometry share one interactive coordinate frame."""

import numpy as np
import pytest

from so101_nexus.calibration.view import render_overlay


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
    assert figure.layout.uirevision == "calibration-overlay"


def test_missing_measurement_has_no_invented_measured_pose():
    figure = render_overlay((0, 0, 0, 0, 0, 0), None)
    assert len(figure.data) == 1
    assert "No live measurement" in figure.layout.annotations[0].text


def test_active_joint_axis_uses_forward_kinematics():
    figure = render_overlay((0, 0, 0, 0, 0, 0), None, active_joint="shoulder_lift")
    reference, axis = figure.data
    assert reference.type == "mesh3d"
    assert axis.type == "scatter3d"
    assert axis.name == "Active joint: shoulder lift"
    assert np.linalg.norm(
        np.array([axis.x[-1] - axis.x[0], axis.y[-1] - axis.y[0], axis.z[-1] - axis.z[0]])
    ) == pytest.approx(0.09)


@pytest.mark.parametrize("angles", [(0, 0), (0, 0, 0, 0, 0, float("nan"))])
def test_overlay_rejects_invalid_degree_vectors(angles):
    with pytest.raises(ValueError, match="six finite"):
        render_overlay(angles, None)


def test_overlay_rejects_unknown_joint():
    with pytest.raises(ValueError, match="Unknown active joint"):
        render_overlay((0, 0, 0, 0, 0, 0), None, active_joint="missing")


def test_identical_poses_have_identical_geometry_and_independent_data():
    angles = (20, -30, 40, -20, 15, 30)
    figure = render_overlay(angles, angles, provisional=False)
    np.testing.assert_array_equal(figure.data[0].x, figure.data[1].x)
    before = np.array(figure.data[0].x)
    render_overlay((0, 0, 0, 0, 0, 0), None)
    np.testing.assert_array_equal(figure.data[0].x, before)
    assert len(figure.to_json()) < 700_000
