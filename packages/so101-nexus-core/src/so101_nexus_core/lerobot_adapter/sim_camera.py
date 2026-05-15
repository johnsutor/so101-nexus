"""LeRobot camera adapter for simulator-rendered frames."""

from typing import Any, cast

import numpy as np
from lerobot.cameras import Camera

from so101_nexus_core.lerobot_adapter.sim_camera_config import SimCameraConfig


class SimCamera(Camera):
    """Camera that reads RGB arrays from the active simulator environment."""

    def __init__(self, config: SimCameraConfig) -> None:
        super().__init__(config)
        self.config = config
        self._env: object | None = None
        self._connected = False

    def bind_env(self, env: object) -> None:
        """Bind the simulator env that will produce this camera's frames."""
        self._env = env

    @property
    def is_connected(self) -> bool:
        """Return whether the camera is bound to an active simulator env."""
        return self._connected and self._env is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """Return no hardware cameras because simulator cameras are configured by source."""
        return []

    def connect(self, warmup: bool = True) -> None:
        """Mark the camera connected after the simulator env has been bound."""
        if self._env is None:
            raise RuntimeError("SimCamera.bind_env(env) must be called before connect()")
        self._connected = True
        if warmup:
            self.read()

    def read(self) -> np.ndarray:
        """Read one RGB frame from the simulator."""
        if not self.is_connected:
            raise RuntimeError("SimCamera.bind_env(env) must be called before connect()")
        frame = self._read_frame()
        return self._validate_shape(self._to_uint8_hwc(frame))

    def async_read(self, timeout_ms: float = 200) -> np.ndarray:
        """Read one frame synchronously for LeRobot's async camera API."""
        return self.read()

    def read_latest(self, max_age_ms: int = 500) -> np.ndarray:
        """Return the latest simulator frame."""
        return self.read()

    def disconnect(self) -> None:
        """Disconnect from the currently bound simulator env."""
        self._connected = False
        self._env = None

    def _read_frame(self) -> Any:
        assert self._env is not None
        env = getattr(self._env, "unwrapped", self._env)
        get_obs = getattr(env, "_get_obs", None)
        if callable(get_obs):
            obs = get_obs()
            frame = self._extract_from_obs(obs)
            if frame is None:
                raise KeyError(f"Camera source {self.config.source!r} not found in simulator obs")
            return frame
        render = getattr(env, "render", None)
        if callable(render):
            return render()
        raise TypeError("Simulator env must expose _get_obs() or render().")

    def _extract_from_obs(self, obs: object) -> Any | None:
        if not isinstance(obs, dict):
            return None
        obs_dict = cast("dict[str, Any]", obs)
        if self.config.source in obs_dict:
            return obs_dict[self.config.source]

        sensor_data = obs_dict.get("sensor_data")
        if isinstance(sensor_data, dict):
            camera_data = sensor_data.get(self.config.source)
            if isinstance(camera_data, dict):
                return camera_data.get("rgb", camera_data.get("Color"))
        return None

    def _to_uint8_hwc(self, frame: Any) -> np.ndarray:
        if hasattr(frame, "detach") and callable(frame.detach):
            frame = frame.detach().cpu().numpy()
        arr = np.asarray(frame)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 3 or arr.shape[-1] < 3:
            raise ValueError(f"Sim camera frame shape {arr.shape} is not HWC RGB")
        if arr.shape[-1] > 3:
            arr = arr[..., :3]
        if arr.dtype == np.uint8:
            return arr

        from so101_nexus_core.visualization import to_uint8

        return to_uint8(arr)

    def _validate_shape(self, frame: np.ndarray) -> np.ndarray:
        expected = (self.height, self.width, 3)
        if self.height is not None and self.width is not None and frame.shape != expected:
            raise ValueError(f"Sim camera frame shape {frame.shape} != expected {expected}")
        return frame
