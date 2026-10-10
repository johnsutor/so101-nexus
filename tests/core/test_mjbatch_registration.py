"""Discovery of the optional CPU batch backend without importing its dependency."""

import subprocess
import sys


def test_mjbatch_registration_is_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import gymnasium as gym; import so101_nexus; "
            "assert 'mjbatch' not in sys.modules; "
            "import so101_nexus.mjbatch; "
            "from so101_nexus.env_ids import env_ids_for_backend, backend_for_env_id; "
            "ids = env_ids_for_backend('mjbatch'); "
            "assert len(ids) == 8; "
            "assert all(gym.spec(i).vector_entry_point for i in ids); "
            "assert backend_for_env_id('MJBatchTouch-v1') == 'mjbatch'; "
            "assert 'mjbatch' not in sys.modules; assert 'mujoco' not in sys.modules; "
            "assert 'torch' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_missing_extra_has_installation_guidance():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import gymnasium as gym; import so101_nexus.mjbatch\n"
            "sys.modules['mjbatch'] = None\n"
            "try:\n"
            "    gym.make_vec('MJBatchTouch-v1')\n"
            "except ImportError as exc:\n"
            "    assert 'so101-nexus[mjbatch]' in str(exc)\n"
            "else:\n"
            "    raise AssertionError('missing extra must fail')\n",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
