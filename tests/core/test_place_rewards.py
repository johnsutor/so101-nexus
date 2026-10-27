"""Independent placement potential contracts across scalar and tensor backends."""

import math

import numpy as np
import pytest

from so101_nexus.rewards import (
    place_grasp_potential,
    place_reach_potential,
    place_task_potential,
)


@pytest.mark.parametrize("backend", ["scalar", "numpy", "torch"])
def test_place_potentials_preserve_transport_release_and_settling(backend):
    # TCP distance, horizontal error, height gap, arm speed, grasp score, placed.
    cases = [
        (0.13, 0.17, 0.04, 0.9, 0.51, False),
        (0.23, 0.02, 0.19, 0.7, 0.51, False),
        (0.31, 0.08, 0.08, 0.4, 0.5, False),
        (0.11, 0.06, 0.03, 0.2, 0.49, False),
        (1.4, 0.01, 0.0, 0.6, 0.0, True),
        (0.0, 0.0, 0.0, 0.0, 1.0, True),
        (-0.2, -0.03, -0.09, -0.4, 0.0, True),
        (0.4, -0.05, 0.07, 1.2, 0.8, True),
    ]
    scale, velocity_scale = 3.7, 1.3
    expected = []
    for tcp, xy, height, speed, grasp, placed in cases:
        reaching = 1.0 - math.tanh(scale * max(tcp, 0.0))
        transport = 1.0 - math.tanh(scale * max(xy, height, 0.0))
        stillness = 1.0 - math.tanh(velocity_scale * max(speed, 0.0))
        held_or_delivered = grasp > 0.5 or placed
        expected.append(
            (
                1.0 if placed else reaching,
                float(held_or_delivered),
                (transport + (stillness if placed else 0.0)) / 2.0 if held_or_delivered else 0.0,
            )
        )

    def evaluate(tcp, xy, height, speed, grasp, placed):
        return (
            place_reach_potential(tcp, placed, scale=scale),
            place_grasp_potential(grasp, placed),
            place_task_potential(
                xy, height, speed, grasp, placed, scale=scale, velocity_scale=velocity_scale
            ),
        )

    if backend == "scalar":
        for case, rewards in zip(cases, expected, strict=True):
            actual = evaluate(*case)
            assert all(isinstance(value, float) for value in actual)
            assert actual == pytest.approx(rewards, rel=1e-12, abs=1e-12)
        return

    columns = list(zip(*cases, strict=True))
    if backend == "numpy":
        inputs = [np.array(column) for column in columns]
    else:
        torch = pytest.importorskip("torch")
        inputs = [
            torch.tensor(column, dtype=torch.bool if index == 5 else torch.float64)
            for index, column in enumerate(columns)
        ]
    for index, (actual, rewards) in enumerate(
        zip(evaluate(*inputs), zip(*expected, strict=True), strict=True)
    ):
        assert actual.shape == (len(cases),)
        if backend == "numpy":
            assert isinstance(actual, np.ndarray)
            assert actual.dtype == np.float64
        else:
            assert isinstance(actual, torch.Tensor)
            assert actual.dtype == (torch.get_default_dtype() if index == 1 else torch.float64)
        values = actual if backend == "numpy" else actual.numpy()
        np.testing.assert_allclose(values, rewards, rtol=1e-12, atol=1e-12)
