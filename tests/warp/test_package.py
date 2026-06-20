import pytest

pytestmark = pytest.mark.warp


def test_mujoco_warp_imports():
    import mujoco_warp as mjw

    assert hasattr(mjw, "put_model")
    assert hasattr(mjw, "put_data")
    assert hasattr(mjw, "step")


def test_warp_backend_module_imports():
    import so101_nexus.warp  # noqa: F401
