"""Verify the config package re-exports the same names as the old config.py."""

from __future__ import annotations

# Names captured from config.py via AST before the split.
_EXPECTED_PUBLIC_NAMES = sorted(
    [
        "ControlMode",
        "DIRECTION_VECTORS",
        "EXTENDED_POSE",
        "EnvironmentConfig",
        "JointSpec",
        "LookAtConfig",
        "MoveConfig",
        "MoveDirection",
        "ObsMode",
        "PickAndPlaceConfig",
        "PickConfig",
        "Pose",
        "POSES",
        "REST_POSE",
        "ROBOT_CAMERA_PRESETS",
        "ReachConfig",
        "RenderConfig",
        "RewardConfig",
        "RobotCameraPreset",
        "RobotConfig",
        "SO101_JOINT_NAMES",
        "SQRT_HALF",
        "YcbModelId",
    ]
)


def test_config_package_exports_all_expected_names() -> None:
    """Every public name from the old config.py is re-exported by config/__init__.py."""
    import so101_nexus_core.config

    actual = sorted(
        name
        for name in dir(so101_nexus_core.config)
        if not name.startswith("_") and name.isidentifier()
    )
    missing = set(_EXPECTED_PUBLIC_NAMES) - set(actual)
    extra = set(actual) - set(_EXPECTED_PUBLIC_NAMES)

    # The package transitions to a dir, so 'config' the module is gone.
    # The following are normal subpackage machinery items.
    extra.discard("base")
    extra.discard("cameras")
    extra.discard("pose")
    extra.discard("render")
    extra.discard("reward")
    extra.discard("robot")
    extra.discard("tasks")

    assert not missing, f"Missing re-exports: {sorted(missing)}"
    assert not extra, f"Unexpected public names: {sorted(extra)}"


def test_every_config_subclass_is_an_environment_config() -> None:
    """All task config types should be subclasses of EnvironmentConfig."""
    import so101_nexus_core.config

    task_configs = [
        so101_nexus_core.config.PickConfig,
        so101_nexus_core.config.PickAndPlaceConfig,
        so101_nexus_core.config.ReachConfig,
        so101_nexus_core.config.LookAtConfig,
        so101_nexus_core.config.MoveConfig,
    ]
    for cls in task_configs:
        assert issubclass(cls, so101_nexus_core.config.EnvironmentConfig)
