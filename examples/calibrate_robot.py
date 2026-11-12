"""Create a robot profile, collect automatic excitation, and fit candidate dynamics."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import tyro

from so101_nexus import (
    ExcitationConfig,
    RecordedTrajectory,
    RobotProfile,
    collect_excitation,
    fit_robot_profile,
    generate_excitation,
    preflight_excitation,
)


def create(
    calibration: Path,
    robot_id: str,
    output: Path,
    hardware_body_deg: tuple[float, float, float, float, float],
    model_body_deg: tuple[float, float, float, float, float],
    gripper_limits_deg: tuple[float, float] = (-10, 100),
    servo_gains: tuple[int, int, int] = (16, 0, 32),
    control_period_s: float = 0.02,
) -> None:
    """Import LeRobot calibration and align a measured physical reference pose.

    Parameters
    ----------
    calibration : Path
        Existing LeRobot follower calibration JSON.
    robot_id : str
        Follower ID from the calibration filename.
    output : Path
        New profile JSON file.
    hardware_body_deg : tuple of float
        Measured LeRobot degrees: shoulder pan, shoulder lift, elbow flex,
        wrist flex, wrist roll.
    model_body_deg : tuple of float
        Model angles of the same physical pose, in the same joint order.
    gripper_limits_deg : tuple of float
        Model gripper angles at hardware 0 and 100 percent.
    servo_gains : tuple of int
        Hardware P/I/D register coefficients used during collection.
    control_period_s : float
        Seconds between commands, 0.02 for 50 Hz or 0.03333333333333333 for 30 Hz.
    """
    if output.exists():
        raise FileExistsError(f"Profile already exists: {output}")
    profile = RobotProfile.from_lerobot(
        calibration,
        robot_id=robot_id,
        gripper_limits_deg=gripper_limits_deg,
        servo_gains=servo_gains,
    ).with_reference_pose(hardware_body_deg=hardware_body_deg, model_body_deg=model_body_deg)
    replace(profile, control_period_s=control_period_s).save(output)
    print(f"Saved profile: {output}")


def _excitation_config(profile, center_qpos_deg, duration_s, table_z_m, seed):
    config = ExcitationConfig(
        center_qpos_deg=center_qpos_deg,
        duration_s=duration_s,
        ramp_s=min(2.0, duration_s / 5),
        table_z_m=table_z_m,
        seed=seed,
    )
    duration_s = round(duration_s / profile.control_period_s) * profile.control_period_s
    return replace(config, duration_s=duration_s, ramp_s=min(2.0, duration_s / 5))


def check(
    profile_path: Path,
    center_qpos_deg: tuple[float, float, float, float, float, float],
    duration_s: float = 20,
    table_z_m: float = 0,
    seed: int = 0,
) -> None:
    """Check the excitation in simulation without a hardware connection.

    Parameters
    ----------
    profile_path : Path
        Saved robot profile JSON.
    center_qpos_deg : tuple of float
        Model degrees: shoulder pan, shoulder lift, elbow flex, wrist flex,
        wrist roll, gripper. Use a raised pose with room for motion.
    duration_s : float
        Seconds per episode, rounded to the nearest whole command period.
    table_z_m : float
        Table height in meters relative to the robot base.
    seed : int
        Seed for repeatable excitation.
    """
    profile = RobotProfile.load(profile_path)
    config = _excitation_config(profile, center_qpos_deg, duration_s, table_z_m, seed)
    trajectories = generate_excitation(config, profile)
    preflight_excitation(trajectories, profile, config)
    print(
        f"Verified {len(trajectories)} trajectories, {config.duration_s:g} s each, "
        f"at {1 / profile.control_period_s:g} Hz."
    )


def collect(
    profile_path: Path,
    port: str,
    output_dir: Path,
    center_qpos_deg: tuple[float, float, float, float, float, float],
    duration_s: float = 20,
    table_z_m: float = 0,
    seed: int = 0,
) -> None:
    """Check generated motion, then collect three episodes from a physical arm.

    Parameters
    ----------
    profile_path : Path
        Saved robot profile JSON.
    port : str
        Follower serial port, for example /dev/ttyACM1.
    output_dir : Path
        New directory for the three recordings.
    center_qpos_deg : tuple of float
        Model degrees: shoulder pan, shoulder lift, elbow flex, wrist flex,
        wrist roll, gripper. Place and support the arm at this pose first.
    duration_s : float
        Seconds per episode, rounded to the nearest whole command period.
    table_z_m : float
        Table height in meters relative to the robot base.
    seed : int
        Seed for repeatable excitation.
    """
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
            _excitation_config(profile, center_qpos_deg, duration_s, table_z_m, seed),
            output_dir=output_dir,
        )
    print(f"Saved recordings: {output_dir}")


def fit(profile_path: Path, recordings_dir: Path, output_dir: Path, max_iters: int = 100) -> None:
    """Fit candidate dynamics and report held-out prediction errors."""
    if not recordings_dir.is_dir():
        raise FileNotFoundError(f"Recordings directory does not exist: {recordings_dir}")
    paths = sorted(recordings_dir.glob("*.npz"))
    if not paths:
        raise ValueError(f"No .npz recordings in {recordings_dir}. Run collect first.")
    result = fit_robot_profile(
        RobotProfile.load(profile_path),
        [RecordedTrajectory.load(path) for path in paths],
        output_dir=output_dir,
        max_iters=max_iters,
    )
    print(f"Held-out prediction improved: {result.validation_improved}")
    for episode in result.episodes:
        print(f"{episode.name} ({episode.split}): {episode.identified_rmse_deg} deg RMSE")
    print(f"Review {output_dir / 'metrics.json'} before selecting candidate_profile.json.")


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict(
        {"create": create, "check": check, "collect": collect, "fit": fit}
    )
