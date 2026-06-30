"""Tests for the lerobot-free dataset row to sim qpos conversions."""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest

from so101_nexus import (
    SO101_GRIPPER_LIMITS_RAD,
    dataset_row_to_sim_qpos,
    sim_qpos_to_dataset_row,
)


def test_dataset_row_gripper_decodes_as_range_0_100_not_degrees() -> None:
    low, high = SO101_GRIPPER_LIMITS_RAD
    row = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 15.0])
    decoded = dataset_row_to_sim_qpos(row)

    assert decoded[5] == pytest.approx(low + (15.0 / 100.0) * (high - low))
    # The naive deg2rad-the-whole-vector decode is the bug; it is far off here.
    assert abs(decoded[5] - math.radians(15.0)) > 0.1


def test_sim_qpos_to_dataset_row_round_trips_batched() -> None:
    rng = np.random.default_rng(0)
    qpos = rng.uniform(-0.5, 0.5, size=(4, 6))
    low, high = SO101_GRIPPER_LIMITS_RAD
    qpos[:, 5] = rng.uniform(low, high, size=4)

    row = sim_qpos_to_dataset_row(qpos)
    np.testing.assert_allclose(dataset_row_to_sim_qpos(row), qpos, atol=1e-12)
    np.testing.assert_allclose(row[:, :5], np.rad2deg(qpos[:, :5]), atol=1e-12)
    assert np.all((row[:, 5] >= 0.0) & (row[:, 5] <= 100.0))


def test_dataset_row_to_sim_qpos_respects_custom_gripper_limits() -> None:
    row = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 50.0])
    decoded = dataset_row_to_sim_qpos(row, gripper_limits_rad=(0.0, 2.0))
    assert decoded[5] == pytest.approx(1.0)


def test_dataset_row_to_sim_qpos_supports_torch_tensor() -> None:
    torch = pytest.importorskip("torch")

    row_np = np.array([[10.0, -20.0, 30.0, -5.0, 1.0, 15.0], [0.0, 0.0, 0.0, 0.0, 0.0, 100.0]])
    decoded_t = dataset_row_to_sim_qpos(torch.tensor(row_np))

    assert isinstance(decoded_t, torch.Tensor)
    np.testing.assert_allclose(decoded_t.numpy(), dataset_row_to_sim_qpos(row_np), atol=1e-12)


def test_dataset_row_to_sim_qpos_rejects_wrong_width() -> None:
    with pytest.raises(ValueError, match="6"):
        dataset_row_to_sim_qpos(np.zeros(5))


def test_decode_helpers_import_without_lerobot() -> None:
    """The decode path must work without the teleop extra (LeRobot)."""
    code = (
        "import sys; sys.modules['lerobot'] = None;"  # make `import lerobot` raise
        "import numpy as np;"
        "from so101_nexus import dataset_row_to_sim_qpos, SO101_GRIPPER_LIMITS_RAD;"
        "low, high = SO101_GRIPPER_LIMITS_RAD;"
        "row = np.zeros(6); row[5] = 50.0;"
        "q = dataset_row_to_sim_qpos(row);"
        "assert q.shape == (6,);"
        "assert abs(q[5] - (low + 0.5 * (high - low))) < 1e-9;"
        "print('ok')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("ok")
