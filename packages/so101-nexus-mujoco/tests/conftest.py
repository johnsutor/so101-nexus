import os

# Use EGL for headless rendering so MuJoCo tests run without a display.
os.environ.setdefault("MUJOCO_GL", "egl")
