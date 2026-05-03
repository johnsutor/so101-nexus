"""Robot-specific camera and mounting presets."""

from __future__ import annotations

import numpy as np

# sqrt(2)/2 — used for 90-degree rotation quaternions in camera presets.
SQRT_HALF = float(np.sqrt(0.5))


class RobotCameraPreset:
    """Robot-specific camera and mounting parameters.

    Args:
        base_quat: Base quaternion (w, x, y, z).
        sensor_cam_eye_pos: Sensor camera eye position.
        sensor_cam_target_pos: Sensor camera target position.
        human_cam_eye_pos: Human camera eye position.
        human_cam_target_pos: Human camera target position.
        wrist_camera_mount_link: Link name for wrist camera mounting.
        wrist_cam_pos_center: Center position for wrist camera.
        wrist_cam_pos_noise: Position noise for wrist camera.
        wrist_cam_euler_center_deg: Center Euler angles in degrees.
        wrist_cam_euler_noise_deg: Euler angle noise in degrees.
    """

    def __init__(
        self,
        base_quat: tuple[float, float, float, float],
        sensor_cam_eye_pos: tuple[float, float, float],
        sensor_cam_target_pos: tuple[float, float, float],
        human_cam_eye_pos: tuple[float, float, float],
        human_cam_target_pos: tuple[float, float, float],
        wrist_camera_mount_link: str,
        wrist_cam_pos_center: tuple[float, float, float],
        wrist_cam_pos_noise: tuple[float, float, float],
        wrist_cam_euler_center_deg: tuple[float, float, float],
        wrist_cam_euler_noise_deg: tuple[float, float, float],
    ) -> None:
        self.base_quat = base_quat
        self.sensor_cam_eye_pos = sensor_cam_eye_pos
        self.sensor_cam_target_pos = sensor_cam_target_pos
        self.human_cam_eye_pos = human_cam_eye_pos
        self.human_cam_target_pos = human_cam_target_pos
        self.wrist_camera_mount_link = wrist_camera_mount_link
        self.wrist_cam_pos_center = wrist_cam_pos_center
        self.wrist_cam_pos_noise = wrist_cam_pos_noise
        self.wrist_cam_euler_center_deg = wrist_cam_euler_center_deg
        self.wrist_cam_euler_noise_deg = wrist_cam_euler_noise_deg

    @property
    def wrist_cam_euler_center_rad(self) -> tuple[float, float, float]:
        """Wrist camera Euler center angles converted to radians."""
        x, y, z = self.wrist_cam_euler_center_deg
        return (float(np.radians(x)), float(np.radians(y)), float(np.radians(z)))

    @property
    def wrist_cam_euler_noise_rad(self) -> tuple[float, float, float]:
        """Wrist camera Euler noise angles converted to radians."""
        x, y, z = self.wrist_cam_euler_noise_deg
        return (float(np.radians(x)), float(np.radians(y)), float(np.radians(z)))

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"RobotCameraPreset(wrist_camera_mount_link={self.wrist_camera_mount_link!r}, "
            f"wrist_cam_euler_center_deg={self.wrist_cam_euler_center_deg})"
        )


ROBOT_CAMERA_PRESETS: dict[str, RobotCameraPreset] = {
    # SO-100: base rotated 90° around Z (faces +X). Wrist cam on Fixed_Jaw link.
    "so100": RobotCameraPreset(
        base_quat=(SQRT_HALF, 0.0, 0.0, SQRT_HALF),
        sensor_cam_eye_pos=(0.0, 0.3, 0.3),
        sensor_cam_target_pos=(0.15, 0.0, 0.02),
        human_cam_eye_pos=(0.0, 0.4, 0.4),
        human_cam_target_pos=(0.15, 0.0, 0.05),
        wrist_camera_mount_link="Fixed_Jaw",
        wrist_cam_pos_center=(0.0, -0.045, -0.045),
        wrist_cam_pos_noise=(0.0, 0.015, 0.015),
        wrist_cam_euler_center_deg=(-180.0, -37.5, -90.0),
        wrist_cam_euler_noise_deg=(0.0, 7.5, 0.0),
    ),
    # SO-101: base identity quaternion (faces +X natively). Wrist cam on gripper_link.
    # Euler noise 11.459° ≈ 0.2 rad — larger than SO-100 due to different gripper geometry.
    "so101": RobotCameraPreset(
        base_quat=(1.0, 0.0, 0.0, 0.0),
        sensor_cam_eye_pos=(0.0, 0.3, 0.3),
        sensor_cam_target_pos=(0.15, 0.0, 0.02),
        human_cam_eye_pos=(0.0, 0.4, 0.4),
        human_cam_target_pos=(0.15, 0.0, 0.05),
        wrist_camera_mount_link="gripper_link",
        wrist_cam_pos_center=(0.0, 0.04, -0.04),
        wrist_cam_pos_noise=(0.005, 0.01, 0.01),
        wrist_cam_euler_center_deg=(-180.0, 37.5, -90.0),
        wrist_cam_euler_noise_deg=(0.0, 11.4591559026, 0.0),
    ),
}
