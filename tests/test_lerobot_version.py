"""Dependency contracts for the supported LeRobot release and policy extras."""

import tomllib
from pathlib import Path


def test_lerobot_061_includes_dataset_dependencies_and_opt_in_molmoact2():
    project = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    extras = project["project"]["optional-dependencies"]
    requirement = "lerobot[dataset,feetech]>=0.6.1,<0.7"
    assert requirement in extras["teleop"]
    assert "lerobot[dataset,feetech,molmoact2]>=0.6.1,<0.7" in project["dependency-groups"]["test"]
    assert "lerobot[molmoact2]>=0.6.1,<0.7" in extras["molmoact2"]
    assert not any("git+" in dependency for dependency in extras["molmoact2"])
