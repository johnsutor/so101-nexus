"""Create a robot profile, collect automatic excitation, and fit candidate dynamics."""

from pathlib import Path
from tempfile import TemporaryDirectory

import tyro

from so101_nexus import (
    ExcitationConfig,
    RecordedTrajectory,
    RobotProfile,
    collect_excitation,
    fit_robot_profile,
)


def create(
    calibration: Path,
    robot_id: str,
    output: Path,
    hardware_body_deg: tuple[float, float, float, float, float],
    model_body_deg: tuple[float, float, float, float, float],
    gripper_limits_deg: tuple[float, float] = (-10, 100),
    servo_gains: tuple[int, int, int] = (16, 0, 32),
) -> None:
    """Import LeRobot calibration and align a measured physical reference pose."""
    if output.exists():
        raise FileExistsError(f"Profile already exists: {output}")
    profile = RobotProfile.from_lerobot(
        calibration,
        robot_id=robot_id,
        gripper_limits_deg=gripper_limits_deg,
        servo_gains=servo_gains,
    ).with_reference_pose(hardware_body_deg=hardware_body_deg, model_body_deg=model_body_deg)
    profile.save(output)


def collect(
    profile_path: Path,
    port: str,
    output_dir: Path,
    center_qpos_deg: tuple[float, float, float, float, float, float],
    duration_s: float = 20,
    table_z_m: float = 0,
    seed: int = 0,
) -> None:
    """Check generated motion, then collect three episodes from a physical arm."""
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    profile = RobotProfile.load(profile_path)
    with TemporaryDirectory(prefix="nexus-calibration-") as directory:
        calibration_dir = Path(directory)
        profile.export_lerobot(calibration_dir / f"{profile.robot_id}.json")
        robot = SO101Follower(
            SO101FollowerConfig(
                port=port,
                id=profile.robot_id,
                calibration_dir=calibration_dir,
                use_degrees=True,
                position_p_coefficient=profile.servo_gains[0],
                position_i_coefficient=profile.servo_gains[1],
                position_d_coefficient=profile.servo_gains[2],
            )
        )
        collect_excitation(
            robot,
            profile,
            ExcitationConfig(
                center_qpos_deg=center_qpos_deg,
                duration_s=duration_s,
                table_z_m=table_z_m,
                seed=seed,
            ),
            output_dir=output_dir,
        )


def fit(profile_path: Path, recordings_dir: Path, output_dir: Path, max_iters: int = 100) -> None:
    """Fit candidate dynamics and report held-out prediction errors."""
    result = fit_robot_profile(
        RobotProfile.load(profile_path),
        [RecordedTrajectory.load(path) for path in sorted(recordings_dir.glob("*.npz"))],
        output_dir=output_dir,
        max_iters=max_iters,
    )
    print(f"Held-out prediction improved: {result.validation_improved}")
    for episode in result.episodes:
        print(f"{episode.name} ({episode.split}): {episode.identified_rmse_deg} deg RMSE")
    print(f"Review {output_dir / 'metrics.json'} before selecting candidate_profile.json.")


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"create": create, "collect": collect, "fit": fit})
