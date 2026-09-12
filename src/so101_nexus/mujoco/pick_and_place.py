"""MuJoCo pick-and-place environment.

The carried object is chosen per episode from a compiled object pool (shared with
the unified pick env via ``so101_nexus.object_slots``); the goal is a visible,
non-colliding disc whose colour is randomized per episode. Supported carried
objects: ``CubeObject``, ``ScannedMeshObject`` (``YCBObject``, ``GSOObject``),
``MeshObject``.

When ``PickAndPlaceConfig.n_distractors > 0``, one freejoint slot per entry in
``PickAndPlaceConfig.distractors`` is compiled after the carried pool; each reset
activates ``n_distractors`` of them and parks the remainder off-world with
collisions disabled, matching ``StackCubeEnv``'s slot machinery. As in stack-cube
the distractor pool is separate from the task objects, so unlike ``PickEnv`` (whose
distractors are drawn from the same pool as its target) a distractor here can never
become the carried object.
"""

from __future__ import annotations

import tempfile
from typing import ClassVar

import mujoco
import numpy as np

from so101_nexus import (
    get_so101_mujoco_model_dir,
    get_so101_mujoco_model_path,
)
from so101_nexus.config import (
    ControlMode,
    PickAndPlaceConfig,
    PickAndPlaceV2Config,
    describe_place_target,
)
from so101_nexus.constants import COLOR_MAP, sample_color_name
from so101_nexus.mujoco.base_env import SO101NexusMuJoCoBaseEnv, _observation_scoped
from so101_nexus.mujoco.spawn_utils import (
    activate_distractor_slots,
    hide_freejoint_slot,
    place_freejoint_slot,
    set_slot_contacts,
)
from so101_nexus.object_slots import (
    ObjectSlot,
    build_object_scene_xml,
    ensure_scanned_mesh_assets,
    extract_object_slots,
)
from so101_nexus.objects import CubeObject, ScannedMeshObject, SceneObject
from so101_nexus.placement import (
    compiled_slot_geometry,
    resolved_placement_contract,
    support_state,
    visual_reference,
)
from so101_nexus.rewards import (
    object_static_ok,
    place_grasp_potential,
    place_reach_potential,
    place_task_potential,
    potential_shaping,
)
from so101_nexus.scene import MUJOCO_SCENE_OPTION_XML

_SO101_DIR = get_so101_mujoco_model_dir()
_SO101_XML = get_so101_mujoco_model_path()

# Object placed (not lifted) when within this vertical slack of its rest height.
_PLACE_Z_SLACK = 0.01
_TARGET_Z = 0.001


def _target_disc_body(target_disc_radius: float, target_rgba: list[float]) -> str:
    tr, tg, tb, ta = target_rgba
    return (
        f'    <body name="target" pos="0.15 0 {_TARGET_Z}">\n'
        f'      <geom name="target_disc" type="cylinder" size="{target_disc_radius} 0.001"\n'
        f'            rgba="{tr} {tg} {tb} {ta}" contype="0" conaffinity="0"/>\n'
        f"    </body>\n"
    )


class PickAndPlaceEnv(SO101NexusMuJoCoBaseEnv):
    """Pick-and-place environment: carry a pooled object onto a goal disc."""

    config: PickAndPlaceConfig
    default_config_cls: ClassVar[type[PickAndPlaceConfig]] = PickAndPlaceConfig

    def __init__(
        self,
        config: PickAndPlaceConfig | None = None,
        render_mode: str | None = None,
        control_mode: ControlMode = "pd_joint_pos",
        robot_init_qpos_noise: float = 0.02,
    ):
        if config is None:
            config = self.default_config_cls()
        expects_v2 = issubclass(self.default_config_cls, PickAndPlaceV2Config)
        if not isinstance(config, PickAndPlaceConfig) or (
            isinstance(config, PickAndPlaceV2Config) != expects_v2
        ):
            raise TypeError(f"{type(self).__name__} requires {self.default_config_cls.__name__}.")
        self._init_common(
            config=config,
            render_mode=render_mode,
            control_mode=control_mode,
            robot_init_qpos_noise=robot_init_qpos_noise,
        )

        scene_objects: list[SceneObject] = config.object_pool()
        n_carried = len(scene_objects)
        # Distractor slots are only compiled when requested, so the default
        # single-object scene stays identical.
        distractor_pool = list(config.distractors) if config.n_distractors else []
        scene_objects += distractor_pool
        for obj in scene_objects:
            if isinstance(obj, ScannedMeshObject):
                ensure_scanned_mesh_assets(obj)
        slot_names = [f"pick_slot_{i}" for i in range(n_carried)] + [
            f"distractor_slot_{i}" for i in range(len(distractor_pool))
        ]

        self.cube_half_size = config.cube_half_size
        self.target_disc_radius = config.target_disc_radius
        # First configured colours seed the compiled model; the disc colour is
        # re-sampled per episode (geom_rgba is per-geom in the scalar backend).
        first = scene_objects[0]
        self.cube_color_name = first.color if isinstance(first, CubeObject) else ""
        self.target_color_name = (
            config.target_colors
            if isinstance(config.target_colors, str)
            else config.target_colors[0]
        )

        ground_name = (
            config.ground_colors
            if isinstance(config.ground_colors, str)
            else config.ground_colors[0]
        )
        xml_string = build_object_scene_xml(
            scene_objects,
            slot_names,
            COLOR_MAP[ground_name],
            option_xml=MUJOCO_SCENE_OPTION_XML,
            robot_xml_path=str(_SO101_XML),
            model_name="pick_and_place_scene",
            extra_bodies=_target_disc_body(
                config.target_disc_radius, COLOR_MAP[self.target_color_name]
            ),
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=_SO101_DIR, delete=True) as f:
            f.write(xml_string)
            f.flush()
            self.model = mujoco.MjModel.from_xml_path(f.name)
        self.data = mujoco.MjData(self.model)

        slots = extract_object_slots(self.model, slot_names, scene_objects)
        # ``_slots`` stays the carried pool alone, so pool indices (and
        # ``info["target_index"]``) are unaffected by the distractor slots.
        self._slots: list[ObjectSlot] = slots[:n_carried]
        self._distractor_slots: list[ObjectSlot] = slots[n_carried:]
        self._n_distractors = config.n_distractors
        self._target_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_disc"
        )
        self._target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")

        self._target_slot_idx: int = 0
        self._initial_obj_z: float = 0.0
        self._prev_task_potential: float = 0.0
        self._prev_reach_progress: float = 0.0
        self._prev_grasp_progress: float = 0.0
        self._set_target_geoms(self._slots[0].geom_ids)
        self.task_description = config.task_description

        self._finish_model_setup()

    def _get_object_pose(self) -> np.ndarray:
        addr = self._slots[self._target_slot_idx].qpos_addr
        return self.data.qpos[addr : addr + 7].copy()

    def _get_object_vel(self) -> np.ndarray:
        """Return the carried object's free-joint velocity ``[lin(3), ang(3)]``."""
        addr = self._slots[self._target_slot_idx].dof_addr
        return self.data.qvel[addr : addr + 6].copy()

    def _get_target_pos(self) -> np.ndarray:
        return self.data.xpos[self._target_body_id].copy()

    def _gaze_target_pos(self) -> np.ndarray:
        return self._get_object_pose()[:3]

    def _get_component_data(self, component: object) -> np.ndarray:
        from so101_nexus.observations import (
            ObjectOffset as _ObjectOffset,
        )
        from so101_nexus.observations import (
            ObjectPose as _ObjectPose,
        )
        from so101_nexus.observations import (
            ObjectVelocity as _ObjectVelocity,
        )
        from so101_nexus.observations import (
            TargetOffset as _TargetOffset,
        )
        from so101_nexus.observations import (
            TargetPosition as _TargetPosition,
        )

        if isinstance(component, _ObjectPose):
            return self._get_object_pose()
        if isinstance(component, _ObjectVelocity):
            return self._get_object_vel()
        if isinstance(component, _ObjectOffset):
            return self._get_object_pose()[:3] - self._get_tcp_pose()[:3]
        if isinstance(component, _TargetPosition):
            return self._get_target_pos()
        if isinstance(component, _TargetOffset):
            return self._get_target_pos() - self._get_object_pose()[:3]
        return super()._get_component_data(component)

    def _placement_reference_pos(self) -> np.ndarray:
        """Return the object reference for placement and reward calculations."""
        return self._get_object_pose()[:3]

    def _placement_success(
        self, is_obj_placed: bool, is_obj_static: bool, is_grasped: float
    ) -> bool:
        return is_obj_placed and is_obj_static and is_grasped < 0.5

    def _describe_target(self, obj: SceneObject, target_name: str) -> str:
        return describe_place_target(obj, target_name)

    def _obj_placement_state(
        self, obj_pos: np.ndarray, target_pos: np.ndarray
    ) -> tuple[float, bool]:
        """Return ``(obj_to_target_xy_dist, is_obj_placed)`` for the given pose."""
        obj_to_target_dist = float(np.linalg.norm(obj_pos[:2] - target_pos[:2]))
        # bool(): the height term compares a numpy.float64, so without this the
        # declared -> bool return would actually be a numpy.bool_ whenever that
        # term decides the result, leaking into info and breaking JSON encoding.
        is_obj_placed = bool(
            obj_to_target_dist <= self.config.goal_thresh
            and obj_pos[2] < self._initial_obj_z + _PLACE_Z_SLACK
        )
        return obj_to_target_dist, is_obj_placed

    def _is_obj_static(self) -> bool:
        """Return True if the carried object's speeds are below the static thresholds."""
        vel = self._get_object_vel()
        return bool(
            object_static_ok(
                float(np.linalg.norm(vel[:3])),
                float(np.linalg.norm(vel[3:])),
                lin_threshold=self.config.object_static_lin_threshold,
                ang_threshold=self.config.object_static_ang_threshold,
            )
        )

    def _task_potential(
        self, obj_pos: np.ndarray, target_pos: np.ndarray, is_grasped: float, is_obj_placed: bool
    ) -> float:
        """``Phi_place(s)``: staged transport-then-settle progress toward completion.

        Thin physics-query wrapper around ``rewards.place_task_potential``,
        which is monotone non-decreasing along the ideal grasp-lift-carry-
        lower-settle trajectory so ``rewards.potential_shaping`` pays a
        positive delta for forward progress and ~0 for dwelling at any fixed
        state.
        """
        obj_to_target_dist, _ = self._obj_placement_state(obj_pos, target_pos)
        height_gap = max(0.0, float(obj_pos[2]) - self._initial_obj_z)
        arm_speed = float(np.linalg.norm(self.data.qvel[self._arm_qvel_addrs]))
        return place_task_potential(
            obj_to_target_dist,
            height_gap,
            arm_speed,
            is_grasped,
            is_obj_placed,
            scale=self.config.reward.tanh_shaping_scale,
            velocity_scale=self.config.reward.velocity_shaping_scale,
        )

    def _get_info(self) -> dict:
        tcp_pos = self._get_tcp_pose()[:3]
        obj_pos = self._placement_reference_pos()
        target_pos = self._get_target_pos()
        is_grasped = self._is_grasping()

        obj_to_target_dist, is_obj_placed = self._obj_placement_state(obj_pos, target_pos)
        is_robot_static = self._is_robot_static()
        is_obj_static = self._is_obj_static()
        lift_height = float(obj_pos[2] - self._initial_obj_z)
        # Completion is measured on the OBJECT, not the arm: the goal is a disc on
        # the table, so the intended terminal behaviour is release-and-retreat and
        # an arm-velocity gate would score the retreat itself as failure. Requiring
        # the object to be settled and released is strictly stronger (it rejects
        # flung placements the arm-static check never inspects, and the still-held
        # placement the arm-static check accepts) and keeps the predicate
        # perceivable from vision. ``is_robot_static`` stays in ``info`` as a
        # diagnostic. Mirrors WarpPickAndPlaceVectorEnv._compute_reward_terminated.
        success = self._placement_success(is_obj_placed, is_obj_static, is_grasped)

        info = {
            "obj_to_target_dist": obj_to_target_dist,
            "is_obj_placed": is_obj_placed,
            "is_grasped": is_grasped,
            "is_robot_static": is_robot_static,
            "is_obj_static": is_obj_static,
            "lift_height": lift_height,
            "success": success,
            "tcp_to_obj_dist": float(np.linalg.norm(obj_pos - tcp_pos)),
            "target_index": self._target_slot_idx,
            "target_object": repr(self._slots[self._target_slot_idx].obj),
            "task_potential": self._task_potential(obj_pos, target_pos, is_grasped, is_obj_placed),
        }
        if self._privileged_state is not None:
            info["privileged_state"] = self._privileged_state
        return info

    def _compute_reward(self, info: dict) -> float:
        scale = self.config.reward.tanh_shaping_scale
        # Dwelling while grasped must pay ~0/step. Placement holds both potentials
        # so releasing on the goal and retreating do not incur negative deltas.
        reach_now = place_reach_potential(
            info["tcp_to_obj_dist"], info["is_obj_placed"], scale=scale
        )
        grasp_now = place_grasp_potential(info["is_grasped"], info["is_obj_placed"])
        reach_delta = potential_shaping(reach_now, self._prev_reach_progress)
        grasp_delta = potential_shaping(grasp_now, self._prev_grasp_progress)
        self._prev_reach_progress = reach_now
        self._prev_grasp_progress = grasp_now
        # task_progress is a potential-based delta (Ng, Harada & Russell, ICML
        # 1999; see _task_potential), not the raw potential -- dwelling at any
        # fixed state pays ~0 per step instead of the potential's full value.
        task_potential = info["task_potential"]
        task_progress = potential_shaping(task_potential, self._prev_task_potential)
        self._prev_task_potential = task_potential
        components = self.config.reward.compute_components(
            reach_progress=reach_delta,
            is_grasped=grasp_delta,
            task_progress=task_progress,
            is_complete=info["success"],
            action_delta_norm=info.get("action_delta_norm", 0.0),
            energy_norm=info.get("energy_norm", 0.0),
        )
        info["reward_components"] = components
        return sum(components.values())

    def _refresh_reset_reference_state(self) -> None:
        """Refresh the placement, reach, and grasp baselines from the post-settle pose."""
        obj_pos = self._placement_reference_pos()
        self._initial_obj_z = float(obj_pos[2])
        target_pos = self._get_target_pos()
        is_grasped = self._is_grasping()
        _, is_obj_placed = self._obj_placement_state(obj_pos, target_pos)
        self._prev_task_potential = self._task_potential(
            obj_pos, target_pos, is_grasped, is_obj_placed
        )
        scale = self.config.reward.tanh_shaping_scale
        tcp_to_obj_dist = float(np.linalg.norm(obj_pos - self._get_tcp_pose()[:3]))
        self._prev_reach_progress = place_reach_potential(
            tcp_to_obj_dist, is_obj_placed, scale=scale
        )
        self._prev_grasp_progress = place_grasp_potential(is_grasped, is_obj_placed)

    def _choose_target_index(self, rng: np.random.Generator, n_pool: int) -> int:
        """Return the carried object's pool index, honouring ``target_index``.

        The draw is consumed even when a pin overrides it, so a pinned reset
        leaves the rest of the episode's RNG stream (disc colour, disc pose,
        object pose, yaw) exactly where an unpinned reset on the same seed would
        leave it. Skipping the draw would shift every later consumer and move
        the whole scene, defeating the counterfactual pair the pin exists for.
        """
        drawn = int(rng.choice(n_pool))
        override = self._resolve_target_index(n_pool)
        return drawn if override is None else override

    def _task_reset(self) -> None:
        rng = self.np_random
        n_pool = len(self._slots)

        # Restore collisions on every slot; the chosen target collides, the rest
        # are hidden below the floor with their contact bits zeroed.
        for slot in self._slots:
            set_slot_contacts(self.model, slot, True)

        target_idx = self._choose_target_index(rng, n_pool)
        target_slot = self._slots[target_idx]
        target_obj = target_slot.obj
        self._target_slot_idx = target_idx
        self._set_target_geoms(target_slot.geom_ids)
        self.cube_color_name = target_obj.color if isinstance(target_obj, CubeObject) else ""

        # The disc colour is sampled per episode (reproducible under reset(seed=...)).
        self.target_color_name = sample_color_name(self.config.target_colors, rng)
        self.model.geom_rgba[self._target_geom_id] = COLOR_MAP[self.target_color_name]
        self.task_description = self._describe_target(target_obj, self.target_color_name)

        min_r = self.config.spawn_min_radius
        max_r = self.config.spawn_max_radius
        angle_half = float(np.radians(self.config.spawn_angle_half_range_deg))
        cx, cy = self.config.spawn_center

        r_t = rng.uniform(min_r, max_r)
        theta_t = rng.uniform(-angle_half, angle_half)
        target_x = cx + r_t * np.cos(theta_t)
        target_y = cy + r_t * np.sin(theta_t)
        self.model.body_pos[self._target_body_id] = [target_x, target_y, _TARGET_Z]

        distractors = activate_distractor_slots(
            self.model, self.data, self._distractor_slots, self._n_distractors, rng
        )
        active_slots = [target_slot, *distractors]
        positions = self._sample_object_positions(rng, active_slots, (target_x, target_y))
        for slot, xy in zip(active_slots, positions, strict=True):
            place_freejoint_slot(self.model, self.data, slot, rng, xy)
        for idx, slot in enumerate(self._slots):
            if idx != target_idx:
                hide_freejoint_slot(self.model, self.data, slot)

        self._initial_obj_z = target_slot.spawn_z

    def _sample_object_positions(
        self,
        rng: np.random.Generator,
        slots: list[ObjectSlot],
        disc_xy: tuple[float, float],
    ) -> list[tuple[float, float]]:
        """Sample one XY per active slot, clear of the goal disc and of each other.

        Two clearances apply: every object stays ``min_object_target_separation``
        plus its bounding radius from the disc center, and any two objects stay
        ``min_object_separation`` plus their bounding radii apart. The disc
        clearance is measured to the disc center, not its rim, so with a
        ``target_disc_radius`` wider than that margin an object footprint may
        still overlap the disc's outer annulus. Falls back to the last candidate
        once the attempt budget runs out, matching
        ``spawn_utils.sample_separated_positions``.
        """
        cfg = self.config
        min_r, max_r = cfg.spawn_min_radius, cfg.spawn_max_radius
        angle_half = float(np.radians(cfg.spawn_angle_half_range_deg))
        cx, cy = cfg.spawn_center
        disc_x, disc_y = disc_xy
        placed: list[tuple[float, float, float]] = []  # x, y, bounding radius
        for slot in slots:
            radius = slot.bounding_radius
            disc_sep = cfg.min_object_target_separation + radius
            for _ in range(100):
                r = rng.uniform(min_r, max_r)
                theta = rng.uniform(-angle_half, angle_half)
                x = cx + r * np.cos(theta)
                y = cy + r * np.sin(theta)
                if np.hypot(x - disc_x, y - disc_y) < disc_sep:
                    continue
                if all(
                    np.hypot(x - px, y - py) >= radius + pr + cfg.min_object_separation
                    for px, py, pr in placed
                ):
                    break
            placed.append((x, y, radius))
        return [(x, y) for x, y, _ in placed]


class PickAndPlaceV2Env(PickAndPlaceEnv):
    """Place the visual footprint center on the disc with sustained table support."""

    config: PickAndPlaceV2Config
    default_config_cls: ClassVar[type[PickAndPlaceConfig]] = PickAndPlaceV2Config

    def __init__(
        self,
        config: PickAndPlaceV2Config | None = None,
        render_mode: str | None = None,
        control_mode: ControlMode = "pd_joint_pos",
        robot_init_qpos_noise: float = 0.02,
    ):
        super().__init__(config, render_mode, control_mode, robot_init_qpos_noise)
        self._placement_geometry = compiled_slot_geometry(self.model, self._slots)
        self._placement_body_ids = [
            int(self.model.body_rootid[self.model.geom_bodyid[slot.geom_id]])
            for slot in self._slots
        ]
        self._placement_weights = [
            float(self.model.body_subtreemass[body] * np.linalg.norm(self.model.opt.gravity))
            for body in self._placement_body_ids
        ]
        self._floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        robot_root = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        robot_bodies = np.zeros(self.model.nbody, dtype=bool)
        robot_bodies[robot_root] = True
        for body in range(robot_root + 1, self.model.nbody):
            robot_bodies[body] = robot_bodies[self.model.body_parentid[body]]
        self._placement_robot_geoms = robot_bodies[self.model.geom_bodyid]
        self._placement_force_buffer = np.zeros(6)
        self._placement_velocity_buffer = np.zeros(6)
        self._placement_dwell = 0.0
        self._placement_objects = [slot.obj for slot in self._slots]

    @_observation_scoped
    def _placement_reference_pos(self) -> np.ndarray:
        """Use the projected visual union center and its lowest world-space point."""
        index = self._target_slot_idx
        body = self._placement_body_ids[index]
        rotation = self.data.xmat[body].reshape(3, 3)
        return visual_reference(
            self._placement_geometry[index],
            rotation,
            self.data.xpos[body],
            scanlines=self.config.footprint_scanlines,
        )

    @_observation_scoped
    def _support_info(self) -> dict:
        """Read solved contact forces and COM motion without advancing the dwell."""
        table_force = robot_force = other_force = 0.0
        contacts = self.data.contact
        geoms = contacts.geom[: self.data.ncon]
        is_object = self._obj_geom_mask[geoms]
        rows = np.flatnonzero(is_object[:, 0] != is_object[:, 1])
        force = self._placement_force_buffer
        for row in rows:
            object_first = bool(is_object[row, 0])
            other = int(geoms[row, 1] if object_first else geoms[row, 0])
            mujoco.mj_contactForce(self.model, self.data, row, force)
            if other == self._floor_geom_id:
                # The contact frame maps positive force onto geom2.
                upward = float(contacts.frame[row].reshape(3, 3)[:, 2] @ force[:3])
                table_force += -upward if object_first else upward
            elif self._placement_robot_geoms[other]:
                robot_force += float(np.linalg.norm(force[:3]))
            elif other != self._target_geom_id:
                other_force += float(np.linalg.norm(force[:3]))

        velocity = self._placement_velocity_buffer
        mujoco.mj_objectVelocity(
            self.model,
            self.data,
            mujoco.mjtObj.mjOBJ_BODY,
            self._placement_body_ids[self._target_slot_idx],
            velocity,
            0,
        )
        # mjOBJ_BODY measures motion at the inertial-frame center, not the freejoint origin.
        linear_speed = float(np.linalg.norm(velocity[3:]))
        angular_speed = float(np.linalg.norm(velocity[:3]))
        table_ok, robot_supported, other_supported, static, qualified = support_state(
            table_force,
            robot_force,
            other_force,
            self._placement_weights[self._target_slot_idx],
            linear_speed,
            angular_speed,
            min_weight_fraction=self.config.support_min_weight_fraction,
            force_tolerance=self.config.support_force_tolerance,
            relative_force_tolerance=self.config.support_relative_force_tolerance,
            lin_threshold=self.config.object_static_lin_threshold,
            ang_threshold=self.config.object_static_ang_threshold,
        )
        return {
            "supported_by_table": bool(table_ok),
            "supported_by_robot": bool(robot_supported),
            "supported_by_other": bool(other_supported),
            "table_support_force": table_force,
            "robot_support_force": robot_force,
            "other_support_force": other_force,
            "object_linear_speed": linear_speed,
            "object_angular_speed": angular_speed,
            "is_obj_static": bool(static),
            "qualified": bool(qualified),
        }

    @_observation_scoped
    def _placement_info(self) -> dict:
        """Return current placement diagnostics without changing the physics clock."""
        support = self._support_info().copy()
        qualified = support.pop("qualified")
        reference = self._placement_reference_pos()
        distance = float(np.linalg.norm(reference[:2] - self._get_target_pos()[:2]))
        geometric_ok = distance <= self.config.target_disc_radius
        placed = geometric_ok and qualified
        return {
            **support,
            "placement_contract": resolved_placement_contract(
                self.config,
                self._placement_objects,
                self._placement_weights,
                float(self.model.opt.timestep),
                self._target_slot_idx,
            ),
            "placement_centroid_xy": reference[:2].tolist(),
            "obj_to_target_dist": distance,
            "geometric_placement_ok": geometric_ok,
            "is_obj_placed": placed,
            "placement_dwell": self._placement_dwell,
            "success": placed and self._placement_dwell >= self.config.placement_dwell_time,
        }

    def _obj_placement_state(
        self, obj_pos: np.ndarray, target_pos: np.ndarray
    ) -> tuple[float, bool]:
        info = self._placement_info()
        return info["obj_to_target_dist"], info["is_obj_placed"]

    def _is_obj_static(self) -> bool:
        return self._support_info()["is_obj_static"]

    def _placement_success(
        self, is_obj_placed: bool, is_obj_static: bool, is_grasped: float
    ) -> bool:
        return is_obj_placed and self._placement_dwell >= self.config.placement_dwell_time

    def _describe_target(self, obj: SceneObject, target_name: str) -> str:
        return self.config.describe_target(obj, target_name)

    def _get_info(self) -> dict:
        # Direct info queries need the same local memo window as _observe().
        previous_cache = self._read_cache
        if previous_cache is None:
            self._read_cache = {}
        try:
            info = super()._get_info()
            info.update(self._placement_info())
            return info
        finally:
            self._read_cache = previous_cache

    def _task_reset(self) -> None:
        self._placement_dwell = 0.0
        super()._task_reset()

    def _advance_physics(self) -> None:
        for _ in range(self._N_SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
            if self._support_info()["qualified"]:
                self._placement_dwell += float(self.model.opt.timestep)
            else:
                self._placement_dwell = 0.0
        # Match geometry and observations to the final integrated state.
        mujoco.mj_forward(self.model, self.data)
