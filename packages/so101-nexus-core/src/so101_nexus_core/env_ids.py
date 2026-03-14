"""Helpers for SO101-Nexus Gymnasium environment IDs."""

from __future__ import annotations


def all_registered_env_ids() -> list[str]:
    return [
        "MuJoCoPickLift-v1",
        "MuJoCoPickAndPlace-v1",
        "MuJoCoReach-v1",
        "MuJoCoLookAt-v1",
        "MuJoCoMove-v1",
    ]
