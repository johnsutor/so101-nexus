"""Helpers for SO101-Nexus Gymnasium environment IDs."""

from __future__ import annotations


def all_registered_env_ids() -> list[str]:
    return [
        "MuJoCoPickCubeLift-v1",
        "MuJoCoPickAndPlace-v1",
        "MuJoCoPickCubeMultipleLift-v1",
        "MuJoCoPickYCBLift-v1",
        "MuJoCoPickYCBMultipleLift-v1",
        "ManiSkillPickCubeLiftSO100-v1",
        "ManiSkillPickAndPlaceSO100-v1",
        "ManiSkillPickCubeMultipleLiftSO100-v1",
        "ManiSkillPickYCBLiftSO100-v1",
        "ManiSkillPickYCBMultipleLiftSO100-v1",
        "ManiSkillPickCubeLiftSO101-v1",
        "ManiSkillPickAndPlaceSO101-v1",
        "ManiSkillPickCubeMultipleLiftSO101-v1",
        "ManiSkillPickYCBLiftSO101-v1",
        "ManiSkillPickYCBMultipleLiftSO101-v1",
    ]
