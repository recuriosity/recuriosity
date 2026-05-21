from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional

import numpy as np

from modules.eval.scene_filtering import SceneFilterState, infer_scene_filter_from_sweep
from . import env as base_env


# Re-export commonly used constants/helpers for drop-in compatibility.
ACT_LIST = base_env.ACT_LIST
STEP_METERS = base_env.STEP_METERS
YAW_DEG = base_env.YAW_DEG
PITCH_DEG = base_env.PITCH_DEG
PoseProcess = base_env.PoseProcess
list_scene_glbs = base_env.list_scene_glbs
intrinsics_from_hfov = base_env.intrinsics_from_hfov
reward_from_mse_threshold = base_env.reward_from_mse_threshold
mse_lowfreq_blur_then_downsample = base_env.mse_lowfreq_blur_then_downsample


class HabitatMP3DEnv(base_env.HabitatMP3DEnv):
    """HM3D safe env variant for image-goal navigation."""

    def __init__(
        self,
        scene_list: List[Dict[str, Optional[str]]],
        max_steps: int = 128,
        render_mode: Optional[str] = None,
        gpu_id: int = 0,
        check_holes: bool = False,
        *,
        num_apples: int = 0,
        apple_collect_radius_m: float = 1.5,
        goal_success_reward: float = 0.5,
        goal_step_penalty: float = -2e-4,
        goal_visibility_threshold: float = 0.50,
        goal_distance_threshold_m: float = 1.5,
        goal_visibility_depth_margin_m: float = 0.08,
        goal_point_stride: int = 6,
        goal_depth_min_m: float = 0.05,
        goal_depth_max_m: float = 8.0,
        goal_sample_min_start_distance_m: float = 1.0,
        goal_sample_max_tries: int = 192,
        goal_terminate_on_success: bool = True,
        goal_boundary_margin_m: float = 0.60,
        goal_geometry_clearance_m: float = 0.35,
        goal_max_height_delta_m: float = 0.03,
        goal_reachability_projection_tolerance_m: float = 0.15,
        goal_require_navmesh_path: bool = True,
        goal_scene_retry_limit: int = 8,
    ):
        # Compatibility shim: apple-based trainers pass apple-specific kwargs into
        # HabitatMP3DEnv construction. Image-goal env does not use them, but accepts
        # them to stay drop-in safe.
        del num_apples
        del apple_collect_radius_m

        goal_visibility_threshold_env = os.environ.get(
            "IMAGE_GOAL_VISIBILITY_THRESHOLD", ""
        ).strip()
        if goal_visibility_threshold_env:
            goal_visibility_threshold = float(goal_visibility_threshold_env)
        goal_distance_threshold_env = os.environ.get(
            "IMAGE_GOAL_DISTANCE_THRESHOLD_M", ""
        ).strip()
        if goal_distance_threshold_env:
            goal_distance_threshold_m = float(goal_distance_threshold_env)
        goal_sample_min_start_distance_env = os.environ.get(
            "IMAGE_GOAL_SAMPLE_MIN_START_DISTANCE_M", ""
        ).strip()
        if goal_sample_min_start_distance_env:
            goal_sample_min_start_distance_m = float(goal_sample_min_start_distance_env)
        goal_terminate_on_success_env = os.environ.get(
            "IMAGE_GOAL_TERMINATE_ON_SUCCESS", ""
        ).strip()
        if goal_terminate_on_success_env:
            goal_terminate_on_success = (
                goal_terminate_on_success_env.lower() in {"1", "true", "yes", "on"}
            )

        super().__init__(
            scene_list=scene_list,
            max_steps=max_steps,
            render_mode=render_mode,
            gpu_id=gpu_id,
            check_holes=check_holes,
        )

        self.goal_success_reward = float(goal_success_reward)
        self.goal_step_penalty = float(goal_step_penalty)
        self.goal_visibility_threshold = float(goal_visibility_threshold)
        self.goal_distance_threshold_m = float(goal_distance_threshold_m)
        self.goal_visibility_depth_margin_m = float(goal_visibility_depth_margin_m)
        self.goal_point_stride = max(1, int(goal_point_stride))
        self.goal_depth_min_m = float(goal_depth_min_m)
        self.goal_depth_max_m = float(goal_depth_max_m)
        self.goal_sample_min_start_distance_m = float(goal_sample_min_start_distance_m)
        self.goal_sample_max_tries = max(1, int(goal_sample_max_tries))
        self.goal_terminate_on_success = bool(goal_terminate_on_success)
        self.goal_boundary_margin_m = max(0.0, float(goal_boundary_margin_m))
        self.goal_geometry_clearance_m = max(0.0, float(goal_geometry_clearance_m))
        self.goal_max_height_delta_m = max(0.0, float(goal_max_height_delta_m))
        self.goal_reachability_projection_tolerance_m = max(
            0.0, float(goal_reachability_projection_tolerance_m)
        )
        self.goal_require_navmesh_path = bool(goal_require_navmesh_path)
        self.goal_scene_retry_limit = max(1, int(goal_scene_retry_limit))

        self._goal_position: np.ndarray | None = None
        self._goal_rotation: np.ndarray | None = None
        self._goal_rgb: np.ndarray | None = None
        self._goal_depth: np.ndarray | None = None
        self._goal_fxfycxcy: np.ndarray | None = None
        self._goal_c2w: np.ndarray | None = None
        self._goal_points_world = np.empty((0, 3), dtype=np.float32)
        self._goal_scene_filter: SceneFilterState | None = None
        self._goal_scene_filter_diag: dict[str, Any] = {}
        self._goal_interior_mask: np.ndarray | None = None
        self._goal_boundary_margin_used_m: float = 0.0
        self._goal_interior_cell_count: int = 0

        self._goal_sample_attempts = 0
        self._last_goal_visible_ratio = 0.0
        self._last_goal_visible_points = 0
        self._last_goal_total_points = 0
        self._last_goal_distance_m = math.inf
        self._last_goal_success = False
        self._last_goal_reward = 0.0
        self._episode_reward_total = 0.0
        self._disable_curiosity_reward_panel = True
        self._disable_curiosity_reward_panel_reason = (
            "Image-goal task (no curiosity RGB-difference reward)"
        )
        self._prefer_aux_status_panel = True
        self._aux_status_panel_title = "Image-Goal Status"
        self._disable_bottom_metric_strip = True
        self._render_layout_goal_order = True
        self._policy_viz_labels = ["forward", "left", "right", "stop"]

        big = 1e9
        obs_spaces = dict(self.observation_space.spaces)
        obs_spaces.update(
            {
                "goal_rgb": base_env.spaces.Box(
                    low=0,
                    high=255,
                    shape=(base_env.HEIGHT, base_env.WIDTH, 3),
                    dtype=np.uint8,
                ),
                "goal_depth": base_env.spaces.Box(
                    low=0,
                    high=big,
                    shape=(base_env.HEIGHT, base_env.WIDTH),
                    dtype=np.float32,
                ),
                "goal_fxfycxcy": base_env.spaces.Box(
                    low=-big,
                    high=big,
                    shape=(4,),
                    dtype=np.float32,
                ),
                "goal_c2w": base_env.spaces.Box(
                    low=-big,
                    high=big,
                    shape=(4, 4),
                    dtype=np.float32,
                ),
                "goal_agent_position": base_env.spaces.Box(
                    low=-big,
                    high=big,
                    shape=(3,),
                    dtype=np.float32,
                ),
            }
        )
        self.observation_space = base_env.spaces.Dict(obs_spaces)

    def reset(self, seed=None, options=None):
        options_dict = dict(options or {}) if isinstance(options, dict) else {}
        explicit_goal_pose_initial = self._extract_goal_pose(options_dict)
        has_forced_scene = options_dict.get("scene") is not None
        retry_enabled = (
            explicit_goal_pose_initial is None
            and not has_forced_scene
        )
        scene_attempt_limit = self.goal_scene_retry_limit if retry_enabled else 1
        fixed_start_pose = bool(options_dict.get("fixed_start_pose", False))

        last_goal_error: RuntimeError | None = None
        for scene_attempt_idx in range(scene_attempt_limit):
            reset_options = dict(options_dict)
            if scene_attempt_idx > 0:
                reset_options.pop("restore_state", None)
                candidate_scenes = [
                    scene
                    for scene in self.scene_list
                    if scene.get("scene_name") != self.scene_name
                ]
                if candidate_scenes:
                    reset_options["scene"] = candidate_scenes[
                        int(np.random.randint(len(candidate_scenes)))
                    ]
                else:
                    reset_options.pop("scene", None)

            reset_seed = seed if scene_attempt_idx == 0 else None
            obs, info = super().reset(seed=reset_seed, options=reset_options)
            start_position, start_rotation = self._capture_agent_pose()
            self._goal_scene_filter, self._goal_scene_filter_diag = self._infer_goal_scene_filter(
                start_position
            )
            self._goal_interior_mask, self._goal_boundary_margin_used_m = (
                self._compute_goal_interior_mask(self._goal_scene_filter)
            )
            self._goal_interior_cell_count = int(
                np.count_nonzero(self._goal_interior_mask)
            ) if self._goal_interior_mask is not None else 0

            explicit_goal_pose = self._extract_goal_pose(reset_options)
            try:
                if explicit_goal_pose is None:
                    goal_position, goal_rotation = self._sample_goal_pose(
                        start_position=start_position,
                        fixed_start_pose=fixed_start_pose,
                        scene_filter=self._goal_scene_filter,
                    )
                else:
                    goal_position, goal_rotation = explicit_goal_pose
                    self._goal_sample_attempts = 0
                    if not self._is_goal_reachable_from_start(
                        start_position=start_position,
                        goal_position=goal_position,
                        scene_filter=self._goal_scene_filter,
                    ):
                        raise RuntimeError(
                            "Explicit image-goal pose is not reachable from the sampled start pose"
                        )
            except RuntimeError as exc:
                msg = str(exc)
                if "Failed to sample a valid image-goal camera pose" not in msg:
                    raise
                last_goal_error = exc
                if scene_attempt_idx + 1 < scene_attempt_limit:
                    print(
                        "[image_goal] skipping scene after goal sampling failure: "
                        f"scene={self.scene_name} "
                        f"(attempt {scene_attempt_idx + 1}/{scene_attempt_limit})",
                        flush=True,
                    )
                    continue
                raise RuntimeError(
                    "Failed to sample a valid image-goal camera pose "
                    f"after {scene_attempt_limit} scene attempts"
                ) from exc

            self._set_agent_pose(goal_position, goal_rotation)
            goal_obs = self._observe()
            self._cache_goal(goal_obs, goal_position, goal_rotation)

            self._set_agent_pose(start_position, start_rotation)
            obs = self._observe()

            (
                self._last_goal_visible_ratio,
                self._last_goal_visible_points,
                self._last_goal_total_points,
                self._last_goal_distance_m,
                self._last_goal_success,
            ) = self._evaluate_goal_success(obs)
            self._last_goal_reward = float(self.goal_step_penalty)
            self._episode_reward_total = 0.0
            self._update_render_panels(obs)

            obs = self._augment_obs(obs)
            info = dict(info)
            info.update(self._goal_info_payload())
            return obs, info

        if last_goal_error is not None:
            raise RuntimeError(
                "Failed to sample a valid image-goal camera pose after scene retries"
            ) from last_goal_error
        raise RuntimeError("Failed to initialize image-goal reset")

    def step(self, action: int):
        obs, _reward_unused, terminated, truncated, info = super().step(action)

        (
            visible_ratio,
            visible_points,
            total_points,
            goal_distance_m,
            goal_success,
        ) = self._evaluate_goal_success(obs)

        reward = float(self.goal_success_reward if goal_success else self.goal_step_penalty)
        self._episode_reward_total += reward
        self._last_goal_visible_ratio = float(visible_ratio)
        self._last_goal_visible_points = int(visible_points)
        self._last_goal_total_points = int(total_points)
        self._last_goal_distance_m = float(goal_distance_m)
        self._last_goal_success = bool(goal_success)
        self._last_goal_reward = float(reward)
        self._update_render_panels(obs)

        terminated = bool(
            terminated
            or (self.goal_terminate_on_success and goal_success)
        )

        obs = self._augment_obs(obs)
        info = dict(info)
        info.update(self._goal_info_payload())
        info["episode_reward_total"] = float(self._episode_reward_total)
        info["goal_step_reward"] = float(reward)
        return obs, reward, terminated, truncated, info

    def get_scene_and_agent_state(self):
        state = super().get_scene_and_agent_state()
        if self._goal_position is not None and self._goal_rotation is not None:
            state["goal_position"] = self._goal_position.astype(np.float32).copy()
            state["goal_rotation"] = self._goal_rotation.astype(np.float32).copy()
        return state

    def get_topdown_overlay_world_points(self) -> list[np.ndarray]:
        """Render-log hook: draw goal camera position as a red dot on topdown panels."""
        if self._goal_position is None:
            return []
        return [self._goal_position]

    def get_aux_status_panel_lines(self) -> list[str]:
        lines: list[str] = []
        lines.append(f"Reward sum: {float(self._episode_reward_total):.3f}")
        lines.append(
            "Step reward: "
            f"{float(self._last_goal_reward):.3f} "
            f"(success={self._last_goal_success})"
        )
        lines.append(
            f"Visible ratio: {float(self._last_goal_visible_ratio):.3f} "
            f"({int(self._last_goal_visible_points)}/{int(self._last_goal_total_points)})"
        )
        lines.append(
            f"Goal distance: {float(self._last_goal_distance_m):.2f} m "
            f"(th={float(self.goal_distance_threshold_m):.2f})"
        )
        lines.append(
            f"Success thresholds: vis>={float(self.goal_visibility_threshold):.2f}, "
            f"dist<{float(self.goal_distance_threshold_m):.2f}m"
        )
        return lines

    def _update_render_panels(self, obs: dict[str, Any]) -> None:
        rgb_obs = obs.get("rgb")
        if rgb_obs is not None:
            self._gt_rgb = np.asarray(rgb_obs, dtype=np.uint8).copy()
        self._pred_rgb = self._render_goal_target_panel_rgb()

    def _render_goal_target_panel_rgb(self) -> np.ndarray:
        if self._goal_rgb is None:
            return np.full((base_env.HEIGHT, base_env.WIDTH, 3), 255, dtype=np.uint8)
        return np.asarray(self._goal_rgb, dtype=np.uint8).copy()

    def _capture_agent_pose(self) -> tuple[np.ndarray, np.ndarray]:
        agent = self.sim.get_agent(0)
        st = agent.get_state()
        position = np.asarray(st.position, dtype=np.float32).reshape(3)
        rotation = np.asarray(base_env.quat_to_coeffs(st.rotation), dtype=np.float32).reshape(4)
        return position, rotation

    def _set_agent_pose(self, position: np.ndarray, rotation: np.ndarray) -> None:
        agent = self.sim.get_agent(0)
        st = agent.get_state()
        st.position = np.asarray(position, dtype=np.float32).reshape(3)
        st.rotation = np.asarray(rotation, dtype=np.float32).reshape(4)
        agent.set_state(st)

    def _extract_goal_pose(self, options_dict: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
        goal_position = options_dict.get("goal_position")
        goal_rotation = options_dict.get("goal_rotation")
        restore_state = options_dict.get("restore_state")
        if isinstance(restore_state, dict):
            if goal_position is None:
                goal_position = restore_state.get("goal_position")
            if goal_rotation is None:
                goal_rotation = restore_state.get("goal_rotation")
        if goal_position is None or goal_rotation is None:
            return None
        pos = np.asarray(goal_position, dtype=np.float64).reshape(-1)
        rot = np.asarray(goal_rotation, dtype=np.float64).reshape(-1)
        if pos.size != 3 or rot.size != 4:
            raise ValueError("goal_position must be xyz and goal_rotation must be quaternion coeffs")
        if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(rot)):
            raise ValueError("goal_position/goal_rotation must be finite")
        return pos.astype(np.float32), rot.astype(np.float32)

    def _sample_goal_pose(
        self,
        *,
        start_position: np.ndarray,
        fixed_start_pose: bool,
        scene_filter: SceneFilterState | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        best_pose: tuple[np.ndarray, np.ndarray, float] | None = None
        best_dist = -1.0

        start_position = np.asarray(start_position, dtype=np.float32).reshape(3)
        candidate_rows = np.empty((0,), dtype=np.int32)
        candidate_cols = np.empty((0,), dtype=np.int32)
        if scene_filter is not None:
            candidate_mask = self._goal_interior_mask
            if candidate_mask is None or not np.any(candidate_mask):
                candidate_mask = np.asarray(scene_filter.island_mask, dtype=bool)
            candidate_rows, candidate_cols = np.nonzero(candidate_mask)

        if len(candidate_rows) > 0:
            order = np.arange(len(candidate_rows), dtype=np.int64)
            np.random.shuffle(order)
            max_candidate_checks = min(len(order), int(self.goal_sample_max_tries) * 4)
            start_y = float(start_position[1])
            checks_done = 0

            for idx in order[:max_candidate_checks]:
                row = int(candidate_rows[idx])
                col = int(candidate_cols[idx])
                world_xz = scene_filter.grid_to_world(np.asarray([row]), np.asarray([col]))[0]
                goal_position = np.asarray(
                    [float(world_xz[0]), start_y, float(world_xz[1])],
                    dtype=np.float32,
                )
                yaw = 0.0 if fixed_start_pose else float(np.random.uniform(-math.pi, math.pi))
                goal_rotation_quat = base_env.quat_from_angle_axis(
                    yaw, np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
                )
                goal_rotation = np.asarray(
                    base_env.quat_to_coeffs(goal_rotation_quat),
                    dtype=np.float32,
                ).reshape(4)

                if not self._all_directions_clear(
                    goal_position,
                    goal_rotation_quat,
                    radius=base_env.COLLISION_RADIUS,
                    uuid="depth",
                    check_holes=self.check_holes,
                    debug_context={"source": "_sample_goal_pose_from_scene_filter"},
                ):
                    checks_done += 1
                    continue
                if not self._has_goal_geometry_clearance(goal_position):
                    checks_done += 1
                    continue
                if not self._is_goal_reachable_from_start(
                    start_position=start_position,
                    goal_position=goal_position,
                    scene_filter=scene_filter,
                ):
                    checks_done += 1
                    continue

                dist = float(
                    np.linalg.norm(
                        goal_position.astype(np.float64) - start_position.astype(np.float64)
                    )
                )
                if dist > best_dist:
                    best_dist = dist
                    best_pose = (goal_position.copy(), goal_rotation.copy(), dist)
                checks_done += 1
                if dist >= float(self.goal_sample_min_start_distance_m):
                    self._goal_sample_attempts = int(checks_done)
                    return goal_position, goal_rotation

            if best_pose is not None:
                self._goal_sample_attempts = int(max(1, checks_done))
                return best_pose[0], best_pose[1]

        # Fallback: random-sample valid start poses and keep the furthest reachable.
        for attempt_idx in range(self.goal_sample_max_tries):
            sampled_state = self._randomize_start(max_tries=1, fixed_start_pose=fixed_start_pose)
            if sampled_state is None:
                continue
            goal_position = np.asarray(sampled_state.position, dtype=np.float32).reshape(3)
            goal_rotation = np.asarray(
                base_env.quat_to_coeffs(sampled_state.rotation),
                dtype=np.float32,
            ).reshape(4)
            if not self._has_goal_geometry_clearance(goal_position):
                continue
            if not self._is_goal_reachable_from_start(
                start_position=start_position,
                goal_position=goal_position,
                scene_filter=scene_filter,
            ):
                continue
            dist = float(
                np.linalg.norm(
                    goal_position.astype(np.float64) - start_position.astype(np.float64)
                )
            )
            if dist > best_dist:
                best_dist = dist
                best_pose = (goal_position.copy(), goal_rotation.copy(), dist)
            if dist >= float(self.goal_sample_min_start_distance_m):
                self._goal_sample_attempts = int(attempt_idx + 1)
                return goal_position, goal_rotation

        if best_pose is not None:
            self._goal_sample_attempts = int(self.goal_sample_max_tries)
            return best_pose[0], best_pose[1]
        raise RuntimeError("Failed to sample a valid image-goal camera pose")

    def _cache_goal(
        self,
        goal_obs: dict[str, Any],
        goal_position: np.ndarray,
        goal_rotation: np.ndarray,
    ) -> None:
        self._goal_position = np.asarray(goal_position, dtype=np.float32).reshape(3).copy()
        self._goal_rotation = np.asarray(goal_rotation, dtype=np.float32).reshape(4).copy()
        self._goal_rgb = np.asarray(goal_obs["rgb"], dtype=np.uint8).copy()
        self._goal_depth = np.asarray(goal_obs["depth"], dtype=np.float32).copy()
        self._goal_fxfycxcy = np.asarray(goal_obs["fxfycxcy"], dtype=np.float32).reshape(4).copy()
        self._goal_c2w = np.asarray(goal_obs["c2w"], dtype=np.float32).reshape(4, 4).copy()
        self._goal_points_world = self._backproject_goal_points(
            depth=self._goal_depth,
            fxfycxcy=self._goal_fxfycxcy,
            c2w=self._goal_c2w,
        )

    def _augment_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        out = dict(obs)
        if self._goal_rgb is None:
            out["goal_rgb"] = np.zeros((base_env.HEIGHT, base_env.WIDTH, 3), dtype=np.uint8)
            out["goal_depth"] = np.zeros((base_env.HEIGHT, base_env.WIDTH), dtype=np.float32)
            out["goal_fxfycxcy"] = np.zeros((4,), dtype=np.float32)
            out["goal_c2w"] = np.eye(4, dtype=np.float32)
            out["goal_agent_position"] = np.zeros((3,), dtype=np.float32)
            return out

        out["goal_rgb"] = self._goal_rgb
        out["goal_depth"] = self._goal_depth
        out["goal_fxfycxcy"] = self._goal_fxfycxcy
        out["goal_c2w"] = self._goal_c2w
        out["goal_agent_position"] = self._goal_position
        return out

    def _backproject_goal_points(
        self,
        *,
        depth: np.ndarray,
        fxfycxcy: np.ndarray,
        c2w: np.ndarray,
    ) -> np.ndarray:
        depth_map = np.asarray(depth, dtype=np.float64)
        h, w = depth_map.shape[:2]
        ys = np.arange(0, h, self.goal_point_stride, dtype=np.int32)
        xs = np.arange(0, w, self.goal_point_stride, dtype=np.int32)
        if ys.size == 0 or xs.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        uu, vv = np.meshgrid(xs, ys)
        zz = depth_map[vv, uu]
        valid = np.isfinite(zz) & (zz > float(self.goal_depth_min_m))
        if self.goal_depth_max_m > 0.0:
            valid &= zz <= float(self.goal_depth_max_m)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32)

        fx, fy, cx, cy = np.asarray(fxfycxcy, dtype=np.float64).reshape(4)
        uu = uu[valid].astype(np.float64)
        vv = vv[valid].astype(np.float64)
        zz = zz[valid].astype(np.float64)

        xx = (uu - cx) * zz / max(float(fx), 1e-9)
        yy = (vv - cy) * zz / max(float(fy), 1e-9)
        pts_cam = np.stack([xx, yy, zz, np.ones_like(zz)], axis=1)
        c2w64 = np.asarray(c2w, dtype=np.float64).reshape(4, 4)
        pts_world = (c2w64 @ pts_cam.T).T[:, :3]
        finite = np.all(np.isfinite(pts_world), axis=1)
        return pts_world[finite].astype(np.float32, copy=False)

    def _infer_goal_scene_filter(
        self,
        start_position: np.ndarray,
    ) -> tuple[SceneFilterState, dict[str, Any]]:
        if self.sim is None:
            raise RuntimeError("sim is not initialized")
        start = np.asarray(start_position, dtype=np.float64).reshape(3)
        state, diagnostics = infer_scene_filter_from_sweep(
            self.sim,
            self.sim.pathfinder,
            start_position=start,
            camera_height_m=float(base_env.AGENT_HEIGHT),
            agent_radius_m=float(base_env.AGENT_RADIUS),
        )
        return state, diagnostics

    def _compute_goal_interior_mask(
        self,
        scene_filter: SceneFilterState | None,
    ) -> tuple[np.ndarray | None, float]:
        if scene_filter is None:
            return None, 0.0
        island = np.asarray(scene_filter.island_mask, dtype=bool)
        if not np.any(island):
            return island, 0.0

        target_margin_m = float(self.goal_boundary_margin_m)
        if target_margin_m <= 1e-6:
            return island, 0.0

        try:
            from scipy import ndimage
        except Exception:
            return island, 0.0

        dist_cells = ndimage.distance_transform_edt(island)
        dist_m = dist_cells * float(scene_filter.map_scale)
        for scale in (1.0, 0.75, 0.50, 0.25, 0.0):
            margin_m = float(target_margin_m * scale)
            interior = island & (dist_m >= margin_m)
            if np.any(interior):
                return interior, margin_m
        return island, 0.0

    def _has_goal_geometry_clearance(self, goal_position: np.ndarray) -> bool:
        clearance_m = float(self.goal_geometry_clearance_m)
        if clearance_m <= 1e-6:
            return True
        if self.sim is None:
            return False

        pos = np.asarray(goal_position, dtype=np.float32).reshape(3)
        if not np.all(np.isfinite(pos)):
            return False
        if self._sensor_pos_inside_geometry(pos):
            return False

        def _ring_clear(
            center_xyz: np.ndarray,
            *,
            ring_clearance_m: float,
            ring_samples: int,
        ) -> bool:
            if ring_clearance_m <= 1e-6:
                return True
            origin = base_env.mn.Vector3(
                float(center_xyz[0]),
                float(center_xyz[1]),
                float(center_xyz[2]),
            )
            for i in range(max(8, int(ring_samples))):
                yaw = (2.0 * math.pi * float(i)) / float(max(8, int(ring_samples)))
                direction = base_env.mn.Vector3(math.cos(yaw), 0.0, math.sin(yaw))
                ray = base_env.habitat_sim.geo.Ray(origin, direction)
                hit = self.sim.cast_ray(ray, max_distance=float(ring_clearance_m))
                if hit.has_hits():
                    return False
            return True

        camera_center = pos + np.asarray([0.0, float(base_env.AGENT_HEIGHT), 0.0], dtype=np.float32)
        if not _ring_clear(camera_center, ring_clearance_m=clearance_m, ring_samples=24):
            return False

        torso_center = pos + np.asarray([0.0, float(base_env.AGENT_HEIGHT) * 0.5, 0.0], dtype=np.float32)
        torso_clearance = max(float(base_env.COLLISION_RADIUS), clearance_m * 0.75)
        if not _ring_clear(torso_center, ring_clearance_m=torso_clearance, ring_samples=16):
            return False

        return True

    def _point_projects_to_scene_island(
        self,
        point_world: np.ndarray,
        scene_filter: SceneFilterState | None,
    ) -> bool:
        if scene_filter is None:
            return True

        point = np.asarray(point_world, dtype=np.float64).reshape(3)
        scale = max(float(scene_filter.map_scale), 1e-9)
        col = int(np.floor((float(point[0]) - float(scene_filter.min_x)) / scale))
        row = int(np.floor((float(point[2]) - float(scene_filter.min_z)) / scale))

        mask = self._goal_interior_mask
        if mask is None or mask.shape != scene_filter.island_mask.shape:
            mask = np.asarray(scene_filter.island_mask, dtype=bool)
        h, w = mask.shape
        if not (0 <= row < h and 0 <= col < w):
            return False

        if self.goal_boundary_margin_m > 0.0:
            return bool(mask[row, col])

        tol_cells = int(
            math.ceil(float(self.goal_reachability_projection_tolerance_m) / scale)
        )
        if tol_cells <= 0:
            return bool(mask[row, col])

        r0 = max(0, row - tol_cells)
        r1 = min(h, row + tol_cells + 1)
        c0 = max(0, col - tol_cells)
        c1 = min(w, col + tol_cells + 1)
        if r0 >= r1 or c0 >= c1:
            return False
        return bool(np.any(mask[r0:r1, c0:c1]))

    def _navmesh_path_exists(
        self,
        start_position: np.ndarray,
        goal_position: np.ndarray,
    ) -> bool:
        if not self.goal_require_navmesh_path:
            return True
        if self.sim is None:
            return False
        try:
            shortest_path = base_env.habitat_sim.ShortestPath()
            shortest_path.requested_start = (
                np.asarray(start_position, dtype=np.float32).reshape(3)
            )
            shortest_path.requested_end = (
                np.asarray(goal_position, dtype=np.float32).reshape(3)
            )
            found = bool(self.sim.pathfinder.find_path(shortest_path))
            if not found:
                return False
            geodesic = float(getattr(shortest_path, "geodesic_distance", math.inf))
            return bool(np.isfinite(geodesic) and geodesic >= 0.0)
        except Exception:
            # Some habitat-sim builds expose differing path APIs; fall back to island filter only.
            return True

    def _is_goal_reachable_from_start(
        self,
        *,
        start_position: np.ndarray,
        goal_position: np.ndarray,
        scene_filter: SceneFilterState | None,
    ) -> bool:
        start_position = np.asarray(start_position, dtype=np.float64).reshape(3)
        goal_position = np.asarray(goal_position, dtype=np.float64).reshape(3)
        height_delta = abs(float(goal_position[1]) - float(start_position[1]))
        if height_delta > float(self.goal_max_height_delta_m):
            return False
        if not self._point_projects_to_scene_island(goal_position, scene_filter):
            return False
        return self._navmesh_path_exists(start_position, goal_position)

    def _evaluate_goal_success(
        self, obs: dict[str, Any]
    ) -> tuple[float, int, int, float, bool]:
        total_points = int(self._goal_points_world.shape[0])
        if total_points <= 0 or self._goal_position is None:
            return 0.0, 0, total_points, math.inf, False

        current_c2w = np.asarray(obs["c2w"], dtype=np.float64).reshape(4, 4)
        w2c = np.linalg.inv(current_c2w)
        points_world = self._goal_points_world.astype(np.float64, copy=False)

        points_cam = (w2c[:3, :3] @ points_world.T).T + w2c[:3, 3]
        z_proj = points_cam[:, 2]
        valid_z = np.isfinite(z_proj) & (z_proj > 1e-6)

        fx, fy, cx, cy = np.asarray(obs["fxfycxcy"], dtype=np.float64).reshape(4)
        u_proj = np.full_like(z_proj, np.nan, dtype=np.float64)
        v_proj = np.full_like(z_proj, np.nan, dtype=np.float64)
        if np.any(valid_z):
            u_proj[valid_z] = fx * (points_cam[valid_z, 0] / z_proj[valid_z]) + cx
            v_proj[valid_z] = fy * (points_cam[valid_z, 1] / z_proj[valid_z]) + cy

        depth_now = np.asarray(obs["depth"], dtype=np.float64)
        h, w = depth_now.shape[:2]
        in_frame = (
            valid_z
            & (u_proj >= 0.0)
            & (u_proj < float(w))
            & (v_proj >= 0.0)
            & (v_proj < float(h))
        )

        visible_points = 0
        if np.any(in_frame):
            ui = np.clip(np.rint(u_proj[in_frame]).astype(np.int64), 0, w - 1)
            vi = np.clip(np.rint(v_proj[in_frame]).astype(np.int64), 0, h - 1)
            observed_depth = depth_now[vi, ui]
            valid_depth = np.isfinite(observed_depth) & (observed_depth > 1e-6)
            z_visible = z_proj[in_frame]
            visible_mask = valid_depth & (
                z_visible <= observed_depth + float(self.goal_visibility_depth_margin_m)
            )
            visible_points = int(np.count_nonzero(visible_mask))

        visible_ratio = float(visible_points / max(total_points, 1))
        current_agent_pos, _ = self._capture_agent_pose()
        goal_distance_m = float(
            np.linalg.norm(
                current_agent_pos.astype(np.float64) - self._goal_position.astype(np.float64)
            )
        )
        success = bool(
            (visible_ratio >= float(self.goal_visibility_threshold))
            and (goal_distance_m < float(self.goal_distance_threshold_m))
        )
        return visible_ratio, visible_points, total_points, goal_distance_m, success

    def _goal_info_payload(self) -> dict[str, Any]:
        return {
            "image_goal_visible_ratio": float(self._last_goal_visible_ratio),
            "image_goal_visible_points": int(self._last_goal_visible_points),
            "image_goal_total_points": int(self._last_goal_total_points),
            "image_goal_distance_m": float(self._last_goal_distance_m),
            "image_goal_success": bool(self._last_goal_success),
            "image_goal_success_threshold": float(self.goal_visibility_threshold),
            "image_goal_distance_threshold_m": float(self.goal_distance_threshold_m),
            "image_goal_step_penalty": float(self.goal_step_penalty),
            "image_goal_success_reward": float(self.goal_success_reward),
            "image_goal_sample_attempts": int(self._goal_sample_attempts),
            "image_goal_reachability_mode": (
                "scene_sweep_island_and_navmesh_path"
                if self.goal_require_navmesh_path
                else "scene_sweep_island_only"
            ),
            "image_goal_reachability_projection_tolerance_m": float(
                self.goal_reachability_projection_tolerance_m
            ),
            "image_goal_boundary_margin_m": float(self.goal_boundary_margin_m),
            "image_goal_boundary_margin_used_m": float(self._goal_boundary_margin_used_m),
            "image_goal_geometry_clearance_m": float(self.goal_geometry_clearance_m),
            "image_goal_interior_cell_count": int(self._goal_interior_cell_count),
            "image_goal_max_height_delta_m": float(self.goal_max_height_delta_m),
            "image_goal_scene_component_label": (
                None
                if self._goal_scene_filter is None
                else int(self._goal_scene_filter.component_label)
            ),
            "image_goal_scene_filter_mode": self._goal_scene_filter_diag.get("mode"),
            "image_goal_scene_island_cell_count": self._goal_scene_filter_diag.get(
                "island_cell_count"
            ),
            "episode_reward_total": float(self._episode_reward_total),
        }


__all__ = [
    "ACT_LIST",
    "STEP_METERS",
    "YAW_DEG",
    "PITCH_DEG",
    "PoseProcess",
    "HabitatMP3DEnv",
    "intrinsics_from_hfov",
    "list_scene_glbs",
    "mse_lowfreq_blur_then_downsample",
    "reward_from_mse_threshold",
]
