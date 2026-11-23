"""Collision-checked PickLift reset poses for MuJoCo models."""

import mujoco
import numpy as np

from so101_nexus.config import SO101_JOINT_NAMES, SO101_TCP_SITE_NAME
from so101_nexus.kinematics import ee_ik_delta_q
from so101_nexus.object_slots import ObjectSlot


class PregraspSolver:
    """Solve downward-facing tool poses in a scratch scene."""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.site = model.site(SO101_TCP_SITE_NAME).id
        joints = [model.joint(name).id for name in SO101_JOINT_NAMES]
        actuators = [model.actuator(name).id for name in SO101_JOINT_NAMES]
        self.base_body = model.jnt_bodyid[joints[0]]
        self.wrist_flex = model.joint("wrist_flex").id
        self.qpos = model.jnt_qposadr[joints]
        self.dofs = model.jnt_dofadr[joints[:-1]]
        self.low = np.maximum(model.jnt_range[joints, 0], model.actuator_ctrlrange[actuators, 0])
        self.high = np.minimum(model.jnt_range[joints, 1], model.actuator_ctrlrange[actuators, 1])
        robot_root = model.body_rootid[self.base_body]
        self.robot_geom = model.body_rootid[model.geom_bodyid] == robot_root

    def solve(
        self, scene_qpos: np.ndarray, slot: ObjectSlot, draws: np.ndarray
    ) -> np.ndarray | None:
        """Return six open-gripper joint targets, or None when all candidates fail."""
        model, data = self.model, self.data
        data.qpos[:] = scene_qpos
        mujoco.mj_kinematics(model, data)
        geoms = list(slot.geom_ids)
        # Project each compiled local bounding box onto world Z.
        z_axis = data.geom_xmat[geoms].reshape(-1, 3, 3)[:, 2]
        bounds = model.geom_aabb[geoms]
        top = max(
            data.geom_xpos[geoms, 2]
            + (z_axis * bounds[:, :3] + abs(z_axis) * bounds[:, 3:]).sum(axis=1)
        )
        target_xy = scene_qpos[slot.qpos_addr : slot.qpos_addr + 2]
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros_like(jacp)
        quat = np.zeros(4)
        desired_quat = np.zeros(4)
        for x, y, z, roll, pitch in draws:
            goal = np.array(
                [*(target_xy + (np.array([x, y]) - 0.5) * 0.02), top + 0.025 + z * 0.02]
            )
            delta = goal[:2] - data.xpos[self.base_body, :2]
            q = np.clip(
                [
                    -np.arctan2(delta[1], delta[0]),
                    -1,
                    1,
                    np.pi / 2,
                    np.deg2rad(20 * roll - 10),
                    1.3,
                ],
                self.low,
                self.high,
            )
            for iteration in range(45):
                data.qpos[self.qpos] = q
                mujoco.mj_kinematics(model, data)
                mujoco.mj_comPos(model, data)
                mujoco.mju_mat2Quat(quat, data.site_xmat[self.site])
                if iteration == 0:
                    # Tilt around the wrist-flex axis, a reachable pointing error for this arm.
                    tilt = np.empty(4)
                    wrist_axis = data.xaxis[self.wrist_flex]
                    mujoco.mju_axisAngle2Quat(tilt, wrist_axis, np.deg2rad(20 * pitch - 10))
                    mujoco.mju_mulQuat(desired_quat, tilt, quat)
                error = np.linalg.norm(data.site_xpos[self.site] - goal)
                if error < 0.003 and data.site_xmat[self.site, 8] < -0.65:
                    mujoco.mj_forward(model, data)
                    if not self.robot_geom[data.contact.geom].any():
                        return q
                    break
                mujoco.mj_jacSite(model, data, jacp, jacr, self.site)
                dq = ee_ik_delta_q(
                    np.vstack((jacp[:, self.dofs], jacr[:, self.dofs])),
                    data.site_xpos[self.site],
                    quat,
                    goal,
                    desired_quat,
                    # Relax orientation when joint limits prevent the requested pitch.
                    orientation_weight=0.035 if iteration < 15 else 0.012,
                    damping=0.02,
                )
                q[:-1] = np.clip(q[:-1] + dq, self.low[:-1], self.high[:-1])
        return None
