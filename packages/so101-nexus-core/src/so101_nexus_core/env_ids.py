"""Helpers for SO101-Nexus Gymnasium environment IDs."""

from __future__ import annotations

from so101_nexus_core.config import YCB_ENV_NAME_MAP


def all_registered_env_ids() -> list[str]:
    env_ids = [
        "MuJoCoPickCubeLift-v1",
        "MuJoCoPickAndPlace-v1",
        "MuJoCoPickCubeMultipleLift-v1",
        "MuJoCoPickYCBMultipleLift-v1",
        "MuJoCoPickYCBLift-v1",
        "ManiSkillPickCubeLift-v1",
        "ManiSkillPickCubeLiftSO100-v1",
        "ManiSkillPickCubeLiftSO101-v1",
        "ManiSkillPickAndPlace-v1",
        "ManiSkillPickAndPlaceSO100-v1",
        "ManiSkillPickAndPlaceSO101-v1",
        "ManiSkillPickCubeMultipleLift-v1",
        "ManiSkillPickCubeMultipleLiftSO100-v1",
        "ManiSkillPickCubeMultipleLiftSO101-v1",
        "ManiSkillPickYCBLift-v1",
        "ManiSkillPickYCBMultipleLift-v1",
    ]

    for env_name in YCB_ENV_NAME_MAP.values():
        env_ids.extend(
            [
                f"MuJoCoPick{env_name}Lift-v1",
                f"MuJoCoPick{env_name}LiftSO101-v1",
                f"MuJoCoPick{env_name}MultipleLift-v1",
                f"ManiSkillPick{env_name}LiftSO100-v1",
                f"ManiSkillPick{env_name}LiftSO101-v1",
                f"ManiSkillPick{env_name}MultipleLiftSO100-v1",
                f"ManiSkillPick{env_name}MultipleLiftSO101-v1",
            ]
        )
    return env_ids
