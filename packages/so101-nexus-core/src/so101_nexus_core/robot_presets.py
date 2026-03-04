from __future__ import annotations

from typing import Any

from so101_nexus_core.config import DEFAULT_ENV_CONFIG, ROBOT_CAMERA_PRESETS


def build_maniskill_robot_configs(
    *, include_cube_half_size: bool, include_max_goal_height: bool
) -> dict[str, dict[str, Any]]:
    """Build ManiSkill robot config dicts in legacy shape from canonical presets."""
    configs: dict[str, dict[str, Any]] = {}
    fov_range = list(DEFAULT_ENV_CONFIG.camera.wrist_fov_rad_range)

    for uid, preset in ROBOT_CAMERA_PRESETS.items():
        cfg: dict[str, Any] = {
            "goal_thresh": DEFAULT_ENV_CONFIG.task.goal_thresh,
            "cube_spawn_half_size": DEFAULT_ENV_CONFIG.task.cube_spawn_half_size,
            "cube_spawn_center": DEFAULT_ENV_CONFIG.task.cube_spawn_center,
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
        if include_cube_half_size:
            cfg["cube_half_size"] = DEFAULT_ENV_CONFIG.task.cube_half_size
        if include_max_goal_height:
            cfg["max_goal_height"] = DEFAULT_ENV_CONFIG.task.max_goal_height
        configs[uid] = cfg

    return configs
