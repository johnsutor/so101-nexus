"""NumPy/torch parity and correctness for the shared tool-frame kinematics."""

from __future__ import annotations

import numpy as np
import pytest

from so101_nexus.kinematics import (
    EE_IK_DAMPING,
    EE_ORIENTATION_WEIGHT,
    damped_least_squares,
    ee_ik_delta_q,
    orientation_error,
    pose_error,
    quat_conjugate,
    quat_multiply,
    quat_to_rotvec,
    rotvec_to_quat,
)

BACKENDS = ("numpy", "torch")
SEED = 20260724


@pytest.fixture(params=BACKENDS)
def backend(request) -> str:
    if request.param == "torch":
        pytest.importorskip("torch")
    return request.param


def to_backend(array: np.ndarray, backend: str):
    """Move a float64 NumPy array onto the backend under test."""
    if backend == "numpy":
        return np.asarray(array, dtype=np.float64)
    import torch

    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float64))


def to_numpy(value) -> np.ndarray:
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def random_rotvecs(rng: np.random.Generator, count: int, low: float, high: float) -> np.ndarray:
    """Uniform random axes with angles drawn from ``[low, high]``."""
    axes = rng.normal(size=(count, 3))
    axes /= np.linalg.norm(axes, axis=-1, keepdims=True)
    return axes * rng.uniform(low, high, size=(count, 1))


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Reference ``[w, x, y, z]`` quaternion to rotation matrix, NumPy only."""
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def rank_deficient_jacobian() -> np.ndarray:
    """A 6x5 tool Jacobian with two dependent columns and an unreachable twist."""
    rng = np.random.default_rng(SEED)
    jac = rng.normal(size=(6, 5))
    jac[:, 4] = jac[:, 3]
    jac[5, :] = 0.0
    return jac


def test_rotvec_quat_round_trip(backend: str) -> None:
    rng = np.random.default_rng(SEED)
    rotvecs = np.vstack(
        [
            np.zeros((1, 3)),
            random_rotvecs(rng, 64, 1e-6, np.pi - 1e-6),
        ]
    )
    out = to_numpy(quat_to_rotvec(rotvec_to_quat(to_backend(rotvecs, backend))))
    np.testing.assert_allclose(out, rotvecs, rtol=0, atol=1e-12)


def test_rotvec_to_quat_produces_unit_quaternions(backend: str) -> None:
    rng = np.random.default_rng(SEED + 1)
    rotvecs = random_rotvecs(rng, 64, 0.0, np.pi)
    quats = to_numpy(rotvec_to_quat(to_backend(rotvecs, backend)))
    np.testing.assert_allclose(np.linalg.norm(quats, axis=-1), 1.0, rtol=0, atol=1e-12)


def test_small_angle_round_trip_is_finite(backend: str) -> None:
    """A 1e-12 rotation must survive both conversions without a 0/0 blow-up."""
    rotvec = np.array([[1e-12, 0.0, 0.0], [0.0, -1e-12, 0.0], [0.0, 0.0, 0.0]])
    quat = to_numpy(rotvec_to_quat(to_backend(rotvec, backend)))
    out = to_numpy(quat_to_rotvec(rotvec_to_quat(to_backend(rotvec, backend))))
    assert np.isfinite(quat).all()
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, rotvec, rtol=0, atol=1e-18)


def test_quat_to_rotvec_resolves_the_double_cover(backend: str) -> None:
    rng = np.random.default_rng(SEED + 2)
    quats = to_numpy(rotvec_to_quat(random_rotvecs(rng, 64, 1e-6, np.pi - 1e-6)))
    positive = to_numpy(quat_to_rotvec(to_backend(quats, backend)))
    negative = to_numpy(quat_to_rotvec(to_backend(-quats, backend)))
    np.testing.assert_array_equal(positive, negative)


def test_quat_multiply_matches_rotation_matrix_product(backend: str) -> None:
    rng = np.random.default_rng(SEED + 3)
    a = to_numpy(rotvec_to_quat(random_rotvecs(rng, 32, 0.05, np.pi - 0.05)))
    b = to_numpy(rotvec_to_quat(random_rotvecs(rng, 32, 0.05, np.pi - 0.05)))
    product = to_numpy(quat_multiply(to_backend(a, backend), to_backend(b, backend)))
    for qa, qb, qp in zip(a, b, product, strict=True):
        np.testing.assert_allclose(
            quat_to_matrix(qp), quat_to_matrix(qa) @ quat_to_matrix(qb), rtol=0, atol=1e-12
        )


def test_quat_conjugate_inverts_the_rotation(backend: str) -> None:
    rng = np.random.default_rng(SEED + 4)
    quats = to_numpy(rotvec_to_quat(random_rotvecs(rng, 32, 0.05, np.pi - 0.05)))
    tensor = to_backend(quats, backend)
    identity = to_numpy(quat_multiply(tensor, quat_conjugate(tensor)))
    np.testing.assert_allclose(identity, np.tile([1.0, 0.0, 0.0, 0.0], (32, 1)), atol=1e-12)


def test_orientation_error_rotates_current_onto_target(backend: str) -> None:
    rng = np.random.default_rng(SEED + 5)
    current = to_numpy(rotvec_to_quat(random_rotvecs(rng, 64, 0.05, np.pi - 0.05)))
    target = to_numpy(rotvec_to_quat(random_rotvecs(rng, 64, 0.05, np.pi - 0.05)))
    cur_t = to_backend(current, backend)
    tgt_t = to_backend(target, backend)

    applied = to_numpy(quat_multiply(rotvec_to_quat(orientation_error(cur_t, tgt_t)), cur_t))
    # Quaternions double cover SO(3), so equality holds only up to an overall sign.
    signed = applied * np.sign(np.sum(applied * target, axis=-1, keepdims=True))
    np.testing.assert_allclose(signed, target, rtol=0, atol=1e-12)


def test_orientation_error_is_zero_for_identical_orientations(backend: str) -> None:
    rng = np.random.default_rng(SEED + 6)
    quats = to_backend(to_numpy(rotvec_to_quat(random_rotvecs(rng, 16, 0.05, 3.0))), backend)
    np.testing.assert_allclose(to_numpy(orientation_error(quats, quats)), 0.0, atol=1e-12)


def test_pose_error_weights_only_the_rotational_block(backend: str) -> None:
    rng = np.random.default_rng(SEED + 7)
    cur_pos = rng.normal(size=(8, 3))
    tgt_pos = rng.normal(size=(8, 3))
    cur_quat = to_numpy(rotvec_to_quat(random_rotvecs(rng, 8, 0.05, 3.0)))
    tgt_quat = to_numpy(rotvec_to_quat(random_rotvecs(rng, 8, 0.05, 3.0)))
    args = [to_backend(a, backend) for a in (cur_pos, cur_quat, tgt_pos, tgt_quat)]

    weight = 0.25
    err = to_numpy(pose_error(*args, orientation_weight=weight))
    np.testing.assert_allclose(err[..., :3], tgt_pos - cur_pos, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        err[..., 3:], weight * to_numpy(orientation_error(args[1], args[3])), rtol=0, atol=1e-12
    )


def test_damped_least_squares_stays_finite_and_reduces_the_residual(backend: str) -> None:
    """The SO-101 tool Jacobian is always rank deficient, so the undamped normal
    equations are singular; damping must keep the solve finite and still descend."""
    jac = rank_deficient_jacobian()
    rng = np.random.default_rng(SEED + 8)
    # A reachable component plus an unreachable one, which is the realistic case.
    error = jac @ rng.normal(size=5) + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    assert np.linalg.matrix_rank(jac) < jac.shape[1]
    dq = to_numpy(
        damped_least_squares(to_backend(jac, backend), to_backend(error, backend)),
    )
    assert np.isfinite(dq).all()
    assert np.linalg.norm(jac @ dq - error) < 0.999 * np.linalg.norm(error)


def test_damped_least_squares_matches_the_closed_form(backend: str) -> None:
    jac = rank_deficient_jacobian()
    rng = np.random.default_rng(SEED + 9)
    error = rng.normal(size=6)
    expected = jac.T @ np.linalg.solve(
        jac @ jac.T + EE_IK_DAMPING**2 * np.eye(6),
        error,
    )
    dq = to_numpy(damped_least_squares(to_backend(jac, backend), to_backend(error, backend)))
    np.testing.assert_allclose(dq, expected, rtol=0, atol=1e-12)


def test_ee_ik_delta_q_weights_the_rotational_jacobian_rows(backend: str) -> None:
    """Weighted least squares scales both sides of the solve.

    De-weighting only the rotational error while passing a full-weight rotational
    Jacobian asks the solver to hold tool orientation exactly, which the five-joint
    SO-101 can satisfy only by barely moving, collapsing position tracking. Backends
    therefore pass the raw ``mj_jacSite`` Jacobian and this function weights it.
    """
    rng = np.random.default_rng(SEED + 12)
    jac = rng.normal(size=(6, 5))
    cur_pos = rng.normal(size=3)
    tgt_pos = rng.normal(size=3)
    cur_quat = to_numpy(rotvec_to_quat(random_rotvecs(rng, 1, 0.05, 3.0)))[0]
    tgt_quat = to_numpy(rotvec_to_quat(random_rotvecs(rng, 1, 0.05, 3.0)))[0]

    error = pose_error(cur_pos, cur_quat, tgt_pos, tgt_quat)
    weighted_jac = np.vstack([jac[:3], EE_ORIENTATION_WEIGHT * jac[3:]])
    expected = damped_least_squares(weighted_jac, error)
    unweighted = damped_least_squares(jac, error)

    dq = to_numpy(
        ee_ik_delta_q(
            *(to_backend(a, backend) for a in (jac, cur_pos, cur_quat, tgt_pos, tgt_quat))
        )
    )
    np.testing.assert_allclose(dq, expected, rtol=0, atol=1e-12)
    assert np.linalg.norm(dq - unweighted) > 1e-6, (
        "ee_ik_delta_q returned the unweighted solve, so orientation is no longer "
        "de-weighted and position tracking will collapse."
    )


def test_batched_leading_dims_match_the_unbatched_call(backend: str) -> None:
    rng = np.random.default_rng(SEED + 10)
    shape = (2, 3)
    count = int(np.prod(shape))
    jac = rng.normal(size=(*shape, 6, 5))
    cur_pos = rng.normal(size=(*shape, 3))
    tgt_pos = rng.normal(size=(*shape, 3))
    cur_quat = to_numpy(rotvec_to_quat(random_rotvecs(rng, count, 0.05, 3.0))).reshape(*shape, 4)
    tgt_quat = to_numpy(rotvec_to_quat(random_rotvecs(rng, count, 0.05, 3.0))).reshape(*shape, 4)

    batched = to_numpy(
        ee_ik_delta_q(
            *(to_backend(a, backend) for a in (jac, cur_pos, cur_quat, tgt_pos, tgt_quat))
        )
    )
    assert batched.shape == (*shape, 5)
    for i in range(shape[0]):
        for j in range(shape[1]):
            single = to_numpy(
                ee_ik_delta_q(
                    *(
                        to_backend(a[i, j], backend)
                        for a in (jac, cur_pos, cur_quat, tgt_pos, tgt_quat)
                    )
                )
            )
            np.testing.assert_allclose(batched[i, j], single, rtol=0, atol=1e-12)


def test_numpy_and_torch_agree_in_float64() -> None:
    pytest.importorskip("torch")
    rng = np.random.default_rng(SEED + 11)
    jac = rng.normal(size=(5, 6, 5))
    cur_pos = rng.normal(size=(5, 3))
    tgt_pos = rng.normal(size=(5, 3))
    cur_quat = to_numpy(rotvec_to_quat(random_rotvecs(rng, 5, 0.0, np.pi)))
    tgt_quat = to_numpy(rotvec_to_quat(random_rotvecs(rng, 5, 0.0, np.pi)))
    arrays = (jac, cur_pos, cur_quat, tgt_pos, tgt_quat)

    for name, fn, args in (
        ("rotvec_to_quat", rotvec_to_quat, (cur_pos,)),
        ("quat_to_rotvec", quat_to_rotvec, (cur_quat,)),
        ("quat_conjugate", quat_conjugate, (cur_quat,)),
        ("quat_multiply", quat_multiply, (cur_quat, tgt_quat)),
        ("orientation_error", orientation_error, (cur_quat, tgt_quat)),
        ("pose_error", pose_error, (cur_pos, cur_quat, tgt_pos, tgt_quat)),
        ("ee_ik_delta_q", ee_ik_delta_q, arrays),
    ):
        out_np = to_numpy(fn(*(to_backend(a, "numpy") for a in args)))
        out_torch = to_numpy(fn(*(to_backend(a, "torch") for a in args)))
        np.testing.assert_allclose(out_torch, out_np, rtol=0, atol=1e-12, err_msg=name)


def test_torch_preserves_dtype_and_type() -> None:
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(SEED)
    rotvec = torch.rand(4, 3, generator=generator, dtype=torch.float64)
    quat = rotvec_to_quat(rotvec)
    assert isinstance(quat, torch.Tensor)
    assert quat.dtype is torch.float64
    assert quat_to_rotvec(quat).dtype is torch.float64
