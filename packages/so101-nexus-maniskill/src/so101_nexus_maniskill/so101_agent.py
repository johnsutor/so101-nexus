"""ManiSkill agent definition for the SO101 robot arm."""

import copy

import numpy as np
import sapien
import sapien.render
import torch
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import PDJointPosControllerConfig
from mani_skill.agents.registration import register_agent
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat

from so101_nexus_core import get_so101_mujoco_model_path
from so101_nexus_core.config import RobotConfig
from so101_nexus_maniskill import menagerie_constants as mc


@register_agent()
class SO101(BaseAgent):
    """ManiSkill agent wrapping the SO101 robot arm MJCF (MuJoCo Menagerie)."""

    uid = "so101"
    mjcf_path = str(get_so101_mujoco_model_path())

    keyframes = {
        "rest": Keyframe(
            qpos=np.radians(np.array(RobotConfig().rest_qpos_deg, dtype=np.float64)),
            pose=sapien.Pose(q=euler2quat(0, 0, 0)),
        ),
        "extended": Keyframe(
            qpos=np.array([0, np.radians(-30.0), np.radians(20.0), np.radians(10.0), 0, -1.1]),
            pose=sapien.Pose(q=euler2quat(0, 0, 0)),
        ),
        "zero": Keyframe(
            qpos=np.array([0.0] * 6),
            pose=sapien.Pose(q=euler2quat(0, 0, 0)),
        ),
    }

    arm_joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    ]
    gripper_joint_names = [
        "gripper",
    ]

    @property
    def _controller_configs(self) -> dict:
        """Return a dict of available controller configurations for the SO101.

        Includes absolute joint-position control, delta joint-position control,
        and target-based delta joint-position control.
        """
        pd_joint_pos = PDJointPosControllerConfig(
            [joint.name for joint in self.robot.active_joints],
            lower=None,
            upper=None,
            stiffness=[1e3] * 6,
            damping=[1e2] * 6,
            force_limit=100,
            normalize_action=False,
        )

        pd_joint_delta_pos = PDJointPosControllerConfig(
            [joint.name for joint in self.robot.active_joints],
            [-0.05, -0.05, -0.05, -0.05, -0.05, -0.2],
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.2],
            stiffness=[1e3] * 6,
            damping=[1e2] * 6,
            force_limit=100,
            use_delta=True,
            use_target=False,
        )

        pd_joint_target_delta_pos = copy.deepcopy(pd_joint_delta_pos)
        pd_joint_target_delta_pos.use_target = True

        controller_configs = {
            "pd_joint_delta_pos": pd_joint_delta_pos,
            "pd_joint_pos": pd_joint_pos,
            "pd_joint_target_delta_pos": pd_joint_target_delta_pos,
        }
        return copy.deepcopy(controller_configs)

    def _after_loading_articulation(self) -> None:
        """Cache frequently-used link references after the articulation is loaded.

        Menagerie fidelity patches (inertials, gripper friction, armature) are
        applied here once implemented; see ``_apply_menagerie_patches``.
        """
        super()._after_loading_articulation()
        self.finger1_link = self.robot.links_map["gripper"]
        self.finger2_link = self.robot.links_map["moving_jaw_so101_v1"]
        self._tcp_offset = Pose.create_from_pq(
            p=torch.tensor(mc.TCP_OFFSET_POS, dtype=torch.float32),
            q=torch.tensor(mc.TCP_OFFSET_QUAT, dtype=torch.float32),
        )
        self._apply_menagerie_patches()

    def _apply_menagerie_patches(self) -> None:
        """Restore fidelity the ManiSkill MJCF importer drops.

        Applies per-link inertials, gripper-link friction, and joint armature
        via SAPIEN PhysX setters. Iterates every per-scene PhysX object
        (``link._objs`` / ``joint._objs``) so vectorized envs are fully patched;
        ManiSkill's wrapper structs do not forward these setters. See
        ``menagerie_constants`` for provenance.
        """
        # Inertials: mass, principal inertia, and center-of-mass pose.
        for name, inertial in mc.LINK_INERTIALS.items():
            link = self.robot.links_map[name]
            cmass = sapien.Pose(p=inertial.com_pos, q=inertial.principal_quat)
            for obj in link._objs:
                obj.set_mass(inertial.mass)
                obj.set_inertia(list(inertial.principal_moments))
                obj.set_cmass_local_pose(cmass)

        # Gripper friction: link-level static/dynamic friction + patch radii.
        # Patching is link-level because built PhysX shapes do not retain MJCF
        # geom names/classes, so per-class (condim-6 only) selection is not
        # possible post-build. Lossless for sliding friction (every collision
        # geom on these bodies has MuJoCo mu = 1.0). The one approximation: the
        # plain condim-3 wrist-roll-follower box on `gripper` also receives the
        # patch radii, giving it torsional resistance it lacks in MuJoCo.
        # Accepted per the design.
        for name in mc.GRIPPER_FRICTION_LINKS:
            link = self.robot.links_map[name]
            for obj in link._objs:
                for shape in obj.get_collision_shapes():
                    mat = shape.get_physical_material()
                    mat.set_static_friction(mc.GRIPPER_STATIC_FRICTION)
                    mat.set_dynamic_friction(mc.GRIPPER_DYNAMIC_FRICTION)
                    shape.set_physical_material(mat)
                    shape.set_patch_radius(mc.GRIPPER_PATCH_RADIUS)
                    shape.set_min_patch_radius(mc.GRIPPER_MIN_PATCH_RADIUS)

        # Armature: float32 array shaped (dof,) per 1-DoF joint object.
        armature = np.array([mc.JOINT_ARMATURE], dtype=np.float32)
        for joint in self.robot.active_joints:
            for jobj in joint._objs:
                jobj.set_armature(armature)

    @property
    def tcp_pose(self) -> Pose:
        """World-space pose of the tool-centre point (TCP).

        The menagerie model has no TCP link; the TCP is the ``gripperframe``
        site, composed here as a fixed offset on the ``gripper`` link
        (``finger1_link``, cached in ``_after_loading_articulation``).
        """
        return self.finger1_link.pose * self._tcp_offset

    @property
    def tcp_pos(self) -> torch.Tensor:
        """World-space position of the tool-centre point (TCP)."""
        return self.tcp_pose.p

    def is_grasping(
        self, object: Actor | None = None, min_force: float = 0.5, max_angle: float = 110
    ) -> torch.Tensor:
        """Check if the robot is grasping an object.

        Parameters
        ----------
        object : Actor
            The object to check if the robot is grasping.
        min_force : float
            Minimum contact force (N) required for a finger to be considered in contact.
        max_angle : float
            Maximum angle (degrees) between the finger's closing direction and the contact
            force vector.

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape ``(num_envs,)`` indicating whether each
            environment's robot is grasping *object*.
        """
        if object is None:
            return super().is_grasping(object)

        l_contact_forces = self.scene.get_pairwise_contact_forces(self.finger1_link, object)
        r_contact_forces = self.scene.get_pairwise_contact_forces(self.finger2_link, object)
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)

        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(lforce >= min_force, torch.rad2deg(langle) <= max_angle)
        rflag = torch.logical_and(rforce >= min_force, torch.rad2deg(rangle) <= max_angle)
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold: float = 0.2) -> torch.Tensor:
        """Return whether the arm joints are effectively stationary.

        Parameters
        ----------
        threshold : float
            Maximum absolute joint velocity (rad/s) below which the robot is considered static.

        Returns
        -------
        torch.Tensor
            Boolean tensor of shape ``(num_envs,)`` that is ``True`` when all
            arm joints are within *threshold*.
        """
        qvel = self.robot.get_qvel()[:, :-1]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @staticmethod
    def build_grasp_pose(
        approaching: np.ndarray,
        closing: np.ndarray,
        center: np.ndarray,
    ) -> sapien.Pose:
        """Construct a 6-DOF grasp pose from three orthogonal unit vectors.

        The resulting pose has its Z-axis aligned with *approaching*, its
        Y-axis aligned with *closing*, and its X-axis equal to
        ``cross(closing, approaching)``.

        Parameters
        ----------
        approaching : np.ndarray
            Unit vector pointing from the TCP toward the object along the approach direction.
        closing : np.ndarray
            Unit vector pointing along the gripper-closing direction (from one finger toward
            the other).
        center : np.ndarray
            3-D position of the grasp centre in world coordinates.

        Returns
        -------
        sapien.Pose
            A :class:`sapien.Pose` representing the desired grasp frame.
        """
        assert np.abs(1 - np.linalg.norm(approaching)) < 1e-3
        assert np.abs(1 - np.linalg.norm(closing)) < 1e-3
        assert np.abs(approaching @ closing) <= 1e-3
        ortho = np.cross(closing, approaching)
        T = np.eye(4)
        T[:3, :3] = np.stack([ortho, closing, approaching], axis=1)
        T[:3, 3] = center
        return sapien.Pose(T)
