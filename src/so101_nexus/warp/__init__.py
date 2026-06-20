"""GPU-batched MuJoCo Warp backend for SO101-Nexus.

Importing this module registers the ``Warp*-v1`` environments. Registration uses
a string ``vector_entry_point`` (Task 7), so this module stays import-light: it
imports neither torch nor mujoco_warp, which are pulled only when the env class is
instantiated. The top-level ``so101_nexus`` package never imports this module
(mirroring ``so101_nexus.mujoco``).
"""

from __future__ import annotations
