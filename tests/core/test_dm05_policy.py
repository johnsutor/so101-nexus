"""OpenDM inference protocol and teleop action replay tests."""

from __future__ import annotations

import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from threading import Thread

import numpy as np
import pytest
from PIL import Image

from so101_nexus.policy_adapters.collection import load_collection_policy


@pytest.fixture
def opendm_server():
    requests = []
    response = {"actions": [[10.0] * 6, [20.0] * 6]}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(
                (self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/infer", requests, response
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def dm05_batch():
    return {
        "observation.state": np.array([10, -20, 30, 40, 50, 60], dtype=np.float32),
        "observation.images.overhead": np.full((3, 4, 3), 255, dtype=np.uint8),
        "observation.images.wrist": np.zeros((3, 4, 3), dtype=np.uint8),
        "task": "Pick the cube",
    }


@pytest.mark.parametrize("camera_map", [None, {}])
def test_dm05_loader_sends_opendm_observation_and_replays_absolute_targets(
    opendm_server, dm05_batch, camera_map
):
    url, requests, _ = opendm_server
    policy = load_collection_policy(path=url, policy_type="dm05", fps=30, camera_map=camera_map)
    assert policy.fps == 30
    np.testing.assert_array_equal(policy.select_action(dm05_batch), [10] * 6)
    np.testing.assert_array_equal(policy.select_action(dm05_batch), [20] * 6)
    assert len(requests) == 1
    path, body = requests[0]
    assert path == "/v1/infer"
    observation = body["observation"]
    assert observation["state"] == [10, -20, 30, 40, 50, 60]
    assert observation["robot_type"] == "SO101"
    assert observation["prompt"] == "Pick the cube"
    assert set(observation["images"]) == {"1", "2"}
    for slot, value in [("1", 255), ("2", 0)]:
        decoded = Image.open(BytesIO(base64.b64decode(observation["images"][slot])))
        np.testing.assert_array_equal(np.asarray(decoded), np.full((3, 4, 3), value))
    policy.select_action(dm05_batch)
    assert len(requests) == 2
    policy.reset()
    np.testing.assert_array_equal(policy.select_action(dm05_batch), [10] * 6)
    assert len(requests) == 3


def test_dm05_camera_mapping_selects_ordered_views(opendm_server, dm05_batch):
    url, requests, _ = opendm_server
    policy = load_collection_policy(
        path=url,
        policy_type="dm05",
        fps=30,
        camera_map={
            "observation.images.wrist": "observation.images.images_1",
            "observation.images.overhead": "observation.images.images_2",
        },
    )
    policy.select_action(dm05_batch)
    first = requests[0][1]["observation"]["images"]["1"]
    assert np.asarray(Image.open(BytesIO(base64.b64decode(first)))).max() == 0


@pytest.mark.parametrize("actions", [[], [1] * 6, [[1] * 7], [[float("nan")] * 6], None])
def test_dm05_rejects_invalid_action_chunks(opendm_server, dm05_batch, actions):
    url, _, response = opendm_server
    response["actions"] = actions
    policy = load_collection_policy(path=url, policy_type="dm05", fps=30)
    with pytest.raises(ValueError, match=r"finite.*six"):
        policy.select_action(dm05_batch)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"path": "http://localhost:7891/v1/infer"}, "policy-fps"),
        ({"path": "http://localhost:7891/v1/infer", "fps": 0}, "policy-fps"),
        (
            {
                "path": "http://localhost:7891/v1/infer",
                "fps": 30,
                "camera_map": {"observation.images.side": "observation.images.front"},
            },
            "images_1.*images_2",
        ),
    ],
)
def test_dm05_configuration_fails_before_connecting(kwargs, message):
    with pytest.raises(ValueError, match=message):
        load_collection_policy(policy_type="dm05", **kwargs)


def test_opendm_server_adapter_requires_url():
    from so101_nexus.policy_adapters.dm05 import DM05PolicyAdapter

    with pytest.raises(ValueError, match=r"OpenDM.*URL"):
        DM05PolicyAdapter("Dexmal/DM05-SO101-Pick-Cube", fps=30)


def test_dm05_missing_camera_fails_before_request(opendm_server, dm05_batch):
    url, requests, _ = opendm_server
    del dm05_batch["observation.images.wrist"]
    policy = load_collection_policy(path=url, policy_type="dm05", fps=30)
    with pytest.raises(ValueError, match="wrist"):
        policy.select_action(dm05_batch)
    assert requests == []


@pytest.mark.parametrize("state", [[0] * 5, [float("inf")] * 6])
def test_dm05_invalid_state_fails_before_request(opendm_server, dm05_batch, state):
    url, requests, _ = opendm_server
    dm05_batch["observation.state"] = state
    policy = load_collection_policy(path=url, policy_type="dm05", fps=30)
    with pytest.raises(ValueError, match="six-joint SO101 state"):
        policy.select_action(dm05_batch)
    assert requests == []


@pytest.mark.parametrize("image", [np.zeros((3, 4, 3)), np.zeros((3, 4), dtype=np.uint8)])
def test_dm05_invalid_image_fails_before_request(opendm_server, dm05_batch, image):
    url, requests, _ = opendm_server
    dm05_batch["observation.images.wrist"] = image
    policy = load_collection_policy(path=url, policy_type="dm05", fps=30)
    with pytest.raises(ValueError, match="HWC uint8 RGB"):
        policy.select_action(dm05_batch)
    assert requests == []
