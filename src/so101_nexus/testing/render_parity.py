"""Cross-backend render parity measurement for camera observations.

The MuJoCo and Warp backends build the same MJCF and place their cameras
identically, but they rasterize it with different renderers. mujoco_warp's
rasterizer ignores per-light ``diffuse`` and applies every active light at unit
intensity, so the Warp image is systematically brighter than MuJoCo's and clips
highlights (see ``so101_nexus.warp.base_env``'s module docstring). Nothing in the
observation-space contract exposes this, so a consumer can pass the shipped
contract suite on both backends while a vision policy trained on one sees a
substantially different image distribution from the other.

These helpers make the gap measurable. ``measure_render_parity`` pins both
backends to bit-identical simulator state and camera pose so the only remaining
variable is the rasterizer, and reports per-camera pixel statistics.
``assert_render_parity`` turns those statistics into a test assertion against an
explicit, caller-supplied tolerance. There is deliberately no default tolerance:
the honest value depends on the task, the camera, and what the consumer's policy
is sensitive to, and baking in a lenient constant would recreate the silence this
module exists to break.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from so101_nexus.config import EnvironmentConfig


class CameraParity(NamedTuple):
    """Pixel-level comparison of one camera observation across the two backends.

    Attributes
    ----------
    name : str
        Observation key of the compared camera (e.g. ``"wrist_camera"``).
    mean_abs_diff : float
        Mean absolute per-channel difference in 0-255 units.
    frac_pixels_differing : float
        Fraction of pixels whose largest per-channel difference exceeds the
        measurement's ``diff_threshold``.
    mujoco_mean : float
        Mean pixel value of the MuJoCo image, in 0-255 units.
    warp_mean : float
        Mean pixel value of the Warp image, in 0-255 units.
    """

    name: str
    mean_abs_diff: float
    frac_pixels_differing: float
    mujoco_mean: float
    warp_mean: float


class RenderParityReport(NamedTuple):
    """Per-camera parity results plus the state-match residual that produced them.

    Attributes
    ----------
    cameras : tuple[CameraParity, ...]
        One entry per camera observation, in configured order.
    max_qpos_diff : float
        Largest absolute difference between the two backends' ``qpos`` after
        state matching. Non-zero means the comparison is confounded by state,
        not just by shading, and the parity numbers should not be trusted.
    """

    cameras: tuple[CameraParity, ...]
    max_qpos_diff: float


def _camera_images(obs: Mapping[str, Any], names: tuple[str, ...], index: int | None) -> list:
    """Extract named camera images from an observation dict as numpy arrays."""
    images = []
    for name in names:
        image = obs[name]
        if index is not None:
            image = image[index]
        images.append(np.asarray(image.cpu() if hasattr(image, "cpu") else image))
    return images


def measure_render_parity(
    task: str,
    config_factory: Callable[[], EnvironmentConfig],
    *,
    seed: int = 0,
    diff_threshold: int = 25,
    device: str = "cpu",
) -> RenderParityReport:
    """Compare MuJoCo and Warp camera images at bit-identical simulator state.

    Both backends are reset with the same seed, then the MuJoCo world's ``qpos``
    and ``qvel`` are overwritten with the Warp world's and the wrist camera's
    per-world pose and field of view are copied across, so geometry, camera
    intrinsics and camera extrinsics are eliminated as sources of difference and
    only the rasterizer remains. Caller-supplied configs should collapse the
    wrist camera's domain-randomization ranges to single points: the pose copy
    already makes both backends render the same camera, but with wide ranges the
    sampled pose, and so the measured gap, becomes an arbitrary function of
    ``seed``, and a tolerance pinned against one seed will not hold on another.

    Parameters
    ----------
    task : str
        Task name without the backend prefix or version suffix, e.g.
        ``"PickAndPlace"`` for ``MuJoCoPickAndPlace-v1`` / ``WarpPickAndPlace-v1``.
    config_factory : callable
        Zero-argument callable returning a fresh config with ``obs_mode="visual"``
        and at least one camera observation. Called once per backend, since a
        config instance must not be shared between environments.
    seed : int
        Reset seed used for both backends.
    diff_threshold : int
        Per-channel difference in 0-255 units above which a pixel counts as
        differing, for ``frac_pixels_differing``.
    device : str
        Warp device for the batched backend. Defaults to ``"cpu"`` so the
        measurement runs without a CUDA GPU, matching the rest of the suite; pass
        ``"cuda"`` to measure the device a consumer actually trains on.

    Returns
    -------
    RenderParityReport
        Per-camera statistics and the post-match ``qpos`` residual.
    """
    import gymnasium as gym
    import mujoco

    import so101_nexus.mujoco
    import so101_nexus.warp  # noqa: F401
    from so101_nexus.observations import CameraObservation, WristCamera

    m_config = config_factory()
    camera_names = tuple(
        c.name for c in (m_config.observations or []) if isinstance(c, CameraObservation)
    )
    if not camera_names:
        raise ValueError("config_factory must produce a config with a camera observation")

    m_env = None
    w_env = None
    try:
        m_env = gym.make(f"MuJoCo{task}-v1", config=m_config)
        w_env = gym.make_vec(f"Warp{task}-v1", num_envs=1, config=config_factory(), device=device)
        m = cast("Any", m_env.unwrapped)
        w = cast("Any", w_env.unwrapped)
        m.reset(seed=seed)
        w_obs, _ = w.reset(seed=seed)

        m.data.qpos[:] = w.qpos[0].detach().cpu().numpy().astype(np.float64)
        m.data.qvel[:] = w.qvel[0].detach().cpu().numpy().astype(np.float64)
        has_wrist = any(isinstance(c, WristCamera) for c in (m_config.observations or []))
        if has_wrist:
            # The Warp backend randomizes the wrist camera into per-world model
            # arrays; copy world 0's realization onto the MuJoCo model so both
            # renderers see one camera, not two independently sampled ones.
            m_id = mujoco.mj_name2id(m.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
            w_id = mujoco.mj_name2id(w.mjm, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
            m.model.cam_pos[m_id] = w._cam_pos[0, w_id].detach().cpu().numpy()
            m.model.cam_quat[m_id] = w._cam_quat[0, w_id].detach().cpu().numpy()
            m.model.cam_fovy[m_id] = float(w._cam_fovy[0, w_id])
        mujoco.mj_forward(m.model, m.data)

        max_qpos_diff = float(
            np.abs(m.data.qpos - w.qpos[0].detach().cpu().numpy()).max(initial=0.0)
        )
        m_images = _camera_images(m._get_obs(), camera_names, None)
        w_images = _camera_images(w_obs, camera_names, 0)

        cameras = []
        for name, m_image, w_image in zip(camera_names, m_images, w_images, strict=True):
            diff = np.abs(m_image.astype(np.int16) - w_image.astype(np.int16))
            cameras.append(
                CameraParity(
                    name=name,
                    mean_abs_diff=float(diff.mean()),
                    frac_pixels_differing=float((diff.max(axis=-1) > diff_threshold).mean()),
                    mujoco_mean=float(m_image.mean()),
                    warp_mean=float(w_image.mean()),
                )
            )
        return RenderParityReport(cameras=tuple(cameras), max_qpos_diff=max_qpos_diff)
    finally:
        # Close both even if one raises, and never let a close-time error replace
        # the in-flight exception that actually explains the failure.
        for env in (w_env, m_env):
            if env is not None:
                with contextlib.suppress(Exception):
                    env.close()


def assert_render_parity(
    task: str,
    config_factory: Callable[[], EnvironmentConfig],
    *,
    max_mean_abs_diff: float,
    max_frac_pixels_differing: float = 1.0,
    seed: int = 0,
    diff_threshold: int = 25,
    device: str = "cpu",
) -> RenderParityReport:
    """Assert every camera's cross-backend image difference is within tolerance.

    Wraps ``measure_render_parity``. Tolerances are required rather than
    defaulted, because the shipped backends are NOT pixel-interchangeable and any
    default this function chose would be a claim about a consumer's sensitivity
    that it is in no position to make. Measure first, then pin the tolerance you
    measured so a regression is visible.

    Parameters
    ----------
    task : str
        Task name without the backend prefix or version suffix.
    config_factory : callable
        Zero-argument callable returning a fresh visual config; see
        ``measure_render_parity``.
    max_mean_abs_diff : float
        Maximum allowed mean absolute per-channel difference, in 0-255 units.
    max_frac_pixels_differing : float
        Maximum allowed fraction of pixels differing by more than
        ``diff_threshold``. Defaults to ``1.0`` (unconstrained).
    seed : int
        Reset seed used for both backends.
    diff_threshold : int
        Per-channel difference above which a pixel counts as differing.
    device : str
        Warp device for the batched backend; see ``measure_render_parity``.

    Returns
    -------
    RenderParityReport
        The measurement, so callers can log or further inspect it.
    """
    report = measure_render_parity(
        task, config_factory, seed=seed, diff_threshold=diff_threshold, device=device
    )
    assert report.max_qpos_diff == 0.0, (
        f"backends were not pinned to identical state (max|qpos diff| = "
        f"{report.max_qpos_diff}); the parity numbers would measure state, not shading"
    )
    for camera in report.cameras:
        assert camera.mean_abs_diff <= max_mean_abs_diff, (
            f"{task} {camera.name}: mean abs pixel difference "
            f"{camera.mean_abs_diff:.1f} > {max_mean_abs_diff} "
            f"(MuJoCo mean {camera.mujoco_mean:.1f}, Warp mean {camera.warp_mean:.1f})"
        )
        assert camera.frac_pixels_differing <= max_frac_pixels_differing, (
            f"{task} {camera.name}: {camera.frac_pixels_differing:.3f} of pixels differ by "
            f"more than {diff_threshold} > {max_frac_pixels_differing}"
        )
    return report
