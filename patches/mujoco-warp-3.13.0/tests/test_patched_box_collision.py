"""Portable regression: run pytest on this file with the patched mujoco-warp installed.

Requires mujoco==3.13.0, mujoco-warp==3.13.0, warp-lang==1.15.0, numpy and pytest.
No SO101 imports are needed. Set BOX_TEST_DEVICE=cuda:0 to include CUDA graphs.
"""

import os

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import warp as wp

FINGER_XML = """<mujoco><option cone="elliptic" impratio="10" iterations="100" ls_iterations="100"/>
<worldbody><body><geom type="box" size=".001 .004 .004" condim="6" priority="1"
friction="1 .005 .0005" solref=".01 1" solimp=".9 .95 .001 .5 2"/></body>
<body pos=".010328006418783002 -.007704534125544985 -.00031456452612223733"
quat=".97536242709329224 -.01294553231378316 .0013112067980120207 .22022449851124004">
<freejoint/><geom type="box" size=".01 .01 .01" mass=".01" condim="4" priority="0"
friction="1 .05 .001" solref=".01 1" solimp=".95 .98 .001 .5 2"/></body></worldbody></mujoco>"""


def box_xml(euler, position):
    return f'''<mujoco><option cone="elliptic" impratio="10" iterations="100" ls_iterations="100"/>
    <worldbody><geom type="box" size=".01 .01 .01"/>
    <body pos="{position}" euler="{euler}"><freejoint/>
    <geom type="box" size=".01 .01 .01" mass=".01"/></body></worldbody></mujoco>'''


@pytest.mark.parametrize(
    "xml,count",
    [
        (FINGER_XML, 4),
        (box_xml("0 0 0", "0 0 .019"), 4),
        (box_xml("45 45 30", ".015 .015 .015"), 1),
    ],
    ids=["thin-finger", "aligned-faces", "edge"],
)
@pytest.mark.parametrize("execution", ["direct", "graph"])
def test_cpu_warp_box_manifold(xml, count, execution):
    device = os.environ.get("BOX_TEST_DEVICE", "cpu")
    if execution == "graph" and device == "cpu":
        pytest.skip("CUDA graph variant requires BOX_TEST_DEVICE=cuda:0")
    model = mujoco.MjModel.from_xml_string(xml)
    model.opt.tolerance = float(np.float32(1e-6))
    cpu = mujoco.MjData(model)
    mujoco.mj_forward(model, cpu)
    with wp.ScopedDevice(device):
        wm = mjw.put_model(model)
        wd = mjw.put_data(model, mujoco.MjData(model), nworld=1, nconmax=32, njmax=128)
        mjw.forward(wm, wd)
        if execution == "graph":
            wd = mjw.put_data(model, mujoco.MjData(model), nworld=1, nconmax=32, njmax=128)
            with wp.ScopedCapture() as capture:
                mjw.forward(wm, wd)
            wp.capture_launch(capture.graph)
        assert int(wd.nacon.numpy()[0]) == count
        assert not wd.overflow.numpy().any()
        gpu = mujoco.MjData(model)
        mjw.get_data_into(gpu, model, wd)
    assert cpu.ncon == gpu.ncon == count
    left = np.array([[c.dist, *c.pos, *c.frame[:3]] for c in cpu.contact[: cpu.ncon]])
    right = np.array([[c.dist, *c.pos, *c.frame[:3]] for c in gpu.contact[: gpu.ncon]])
    distances = np.linalg.norm(left[:, None, 1:4] - right[None, :, 1:4], axis=-1)
    match = distances.argmin(axis=1)
    assert len(set(match)) == count
    np.testing.assert_allclose(left[:, :4], right[match, :4], atol=1e-6, rtol=0)
    np.testing.assert_allclose(left[:, 4:], right[match, 4:], atol=1e-4, rtol=0)
    totals = []
    for data in (cpu, gpu):
        normal = 0.0
        for i in range(data.ncon):
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            normal += force[0]
        totals.append(normal)
    np.testing.assert_allclose(totals[0], totals[1], atol=1e-4, rtol=0)
