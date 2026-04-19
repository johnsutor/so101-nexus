"""Unit tests for so101_nexus_core.visualization.

To run locally, try
VISUAL_TEST_MODEL="openai/unsloth/Qwen3_5-35B-A3B-UD-Q4_K_M"
  VISUAL_TEST_API_BASE="http://127.0.0.1:1337/v1" make test-visual-mujoco
"""

from __future__ import annotations

import numpy as np
import pytest

from so101_nexus_core import visualization
from so101_nexus_core.visualization import (
    INFO_BAR_H,
    CameraView,
    compose_frame,
    make_info_bar,
    resize,
    save_frame_grid,
    save_video,
    scalar,
    to_uint8,
)


class TestToUint8:
    class _FakeTensor:
        def __init__(self, arr: np.ndarray):
            self._arr = arr

        def cpu(self):
            return self

        def numpy(self):
            return self._arr

    def test_float32_converts(self):
        img = np.random.rand(4, 4, 3).astype(np.float32)
        result = to_uint8(img)
        assert result.dtype == np.uint8
        assert result.shape == (4, 4, 3)

    def test_uint8_passthrough(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        result = to_uint8(img)
        assert result.dtype == np.uint8
        assert np.array_equal(result, img)

    def test_4d_batch_squeeze(self):
        img = np.random.rand(1, 4, 4, 3).astype(np.float32)
        result = to_uint8(img)
        assert result.ndim == 3
        assert result.shape == (4, 4, 3)

    def test_clips_out_of_range(self):
        img = np.array([[[1.5, -0.5, 0.5]]], dtype=np.float32)
        result = to_uint8(img)
        assert result[0, 0, 0] == 255
        assert result[0, 0, 1] == 0
        assert result[0, 0, 2] == 127

    def test_tensor_like_input_uses_cpu_numpy(self):
        img = self._FakeTensor(np.ones((2, 2, 3), dtype=np.float32))
        result = to_uint8(img)
        assert result.dtype == np.uint8
        assert result.shape == (2, 2, 3)


class TestResize:
    def test_output_shape(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = resize(img, 50, 25)
        assert result.shape == (25, 50, 3)

    def test_upscale(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        result = resize(img, 100, 100)
        assert result.shape == (100, 100, 3)

    def test_nearest_neighbor_fallback_without_pillow(self, monkeypatch):
        monkeypatch.setattr(visualization, "_PIL_AVAILABLE", False)
        img = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
        result = visualization.resize(img, 2, 2)
        assert result.shape == (2, 2, 3)


class TestComposeFrame:
    def _make_view(self, name: str = "cam") -> CameraView:
        return CameraView(name=name, image=np.zeros((100, 100, 3), dtype=np.uint8))

    def test_single_view(self):
        frame = compose_frame([self._make_view()])
        assert frame.ndim == 3
        assert frame.shape[0] == 240
        assert frame.shape[1] == 320
        assert frame.dtype == np.uint8

    def test_multiple_views(self):
        frame = compose_frame([self._make_view("a"), self._make_view("b")])
        expected_w = 320 * 2 + 2
        assert frame.shape == (240, expected_w, 3)

    def test_with_info_bar(self):
        frame = compose_frame([self._make_view()], step=0, reward=0.0, success=False)
        assert frame.shape[0] == 240 + INFO_BAR_H

    def test_without_info_bar(self):
        frame = compose_frame([self._make_view()])
        assert frame.shape[0] == 240


class TestMakeInfoBar:
    def test_shape(self):
        bar = make_info_bar(640, step=1, reward=0.5, success=True)
        assert bar.shape == (INFO_BAR_H, 640, 3)
        assert bar.dtype == np.uint8

    def test_fallback_without_pillow(self, monkeypatch):
        monkeypatch.setattr(visualization, "_PIL_AVAILABLE", False)
        bar = visualization.make_info_bar(100, step=1, reward=0.0, success=False)
        assert bar.shape == (INFO_BAR_H, 100, 3)


class TestAddLabel:
    def test_fallback_without_pillow(self, monkeypatch):
        monkeypatch.setattr(visualization, "_PIL_AVAILABLE", False)
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        out = visualization.add_label(img, "x")
        assert np.array_equal(out, img)


class TestSaveFrameGrid:
    def test_produces_png(self, tmp_path):
        pytest.importorskip("imageio")
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(8)]
        path = str(tmp_path / "grid.png")
        save_frame_grid(frames, path)
        import imageio.v3 as iio

        grid = iio.imread(path)
        assert grid.ndim == 3
        assert grid.shape[0] == 200
        assert grid.shape[1] == 400


class TestSaveVideo:
    def test_produces_mp4(self, tmp_path):
        pytest.importorskip("imageio")
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(10)]
        path = str(tmp_path / "video.mp4")
        save_video(frames, path, fps=10)
        assert (tmp_path / "video.mp4").exists()
        assert (tmp_path / "video.mp4").stat().st_size > 0


class TestScalar:
    def test_from_numpy(self):
        assert scalar(np.float64(3.14)) == pytest.approx(3.14)

    def test_from_array(self):
        assert scalar(np.array([42.0])) == pytest.approx(42.0)

    def test_plain_float(self):
        assert scalar(1.5) == pytest.approx(1.5)
