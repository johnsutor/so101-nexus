"""Test configuration for headless MuJoCo rendering."""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
