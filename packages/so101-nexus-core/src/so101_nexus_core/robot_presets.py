"""Build ManiSkill robot config dicts from canonical presets."""

from __future__ import annotations

from typing import Any

from so101_nexus_core.config import (
    ROBOT_CAMERA_PRESETS,
    EnvironmentConfig,
    PickAndPlaceConfig,
    PickCubeConfig,
    PickYCBConfig,
)


def build_maniskill_robot_configs(
    config: EnvironmentConfig = EnvironmentConfig(),
) -> dict[str, dict[str, Any]]:
    """Build ManiSkill robot config dicts from canonical presets."""
    configs: dict[str, dict[str, Any]] = {}
    fov_range = list(config.camera.wrist_fov_rad_range)

    for uid, preset in ROBOT_CAMERA_PRESETS.items():
        cfg: dict[str, Any] = {
            "goal_thresh": config.goal_thresh,
            "cube_spawn_half_size": config.spawn_half_size,
            "cube_spawn_center": config.spawn_center,
            "base_quat": tuple(preset.base_quat),
            "sensor_cam_eye_pos": list(preset.sensor_cam_eye_pos),
            "sensor_cam_target_pos": list(preset.sensor_cam_target_pos),
            "human_cam_eye_pos": list(preset.human_cam_eye_pos),
            "human_cam_target_pos": list(preset.human_cam_target_pos),
            "wrist_camera_mount_link": preset.wrist_camera_mount_link,
            "wrist_cam_pos_center": list(preset.wrist_cam_pos_center),
            "wrist_cam_pos_noise": list(preset.wrist_cam_pos_noise),
            "wrist_cam_euler_center": list(preset.wrist_cam_euler_center),
            "wrist_cam_euler_noise": list(preset.wrist_cam_euler_noise),
            "wrist_cam_fov_range": fov_range,
        }
        if isinstance(config, (PickCubeConfig, PickAndPlaceConfig)):
            cfg["cube_half_size"] = config.cube_half_size
        if isinstance(config, (PickCubeConfig, PickYCBConfig)):
            cfg["max_goal_height"] = config.max_goal_height
        configs[uid] = cfg

    return configs
