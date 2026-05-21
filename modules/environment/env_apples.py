from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from . import env as base_env
from modules.eval.scene_filtering import SceneFilterState, infer_scene_filter_from_sweep


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


@dataclass
class AppleState:
    object_id: int | None
    position_world: np.ndarray
    active: bool = True
    object_ref: Any | None = None


class HabitatMP3DEnv(base_env.HabitatMP3DEnv):
    """HM3D safe env variant with collectible apples.

    Behavior:
    - On reset: spawn `num_apples` apples on the eval sweep-derived reachable plane.
    - On step: if an apple is visible and within `apple_collect_radius_m`, collect it.
    - Reward: number of apples collected in the current step plus a per-step penalty.

    Notes:
    - Spawns are at "camera-height-like" y = floor_y + AGENT_HEIGHT + apple_height_offset_m.
    - Visibility uses projection + depth-occlusion in the current view.
    """

    def __init__(
        self,
        scene_list: List[Dict[str, Optional[str]]],
        max_steps: int = 128,
        render_mode: Optional[str] = None,
        gpu_id: int = 0,
        check_holes: bool = False,
        *,
        num_apples: int = 5,
        apple_asset_path: str = "data/Apple.glb",
        apple_collect_radius_m: float = 1.5,
        apple_diameter_m: float = 0.40,  # 3x larger than previous default; override via ctor/env
        apple_spawn_min_separation_m: float = 0.8,
        apple_spawn_boundary_margin_m: float = 0.15,
        apple_spawn_clearance_m: float = 0.10,
        apple_height_offset_m: float = -0.15,
        apple_step_penalty: float = -2e-4,
        apple_visibility_depth_margin_m: float = 0.08,
        apple_terminate_on_completion: bool = False,
        apple_seed: int | None = None,
    ):
        super().__init__(
            scene_list=scene_list,
            max_steps=max_steps,
            render_mode=render_mode,
            gpu_id=gpu_id,
            check_holes=check_holes,
        )

        self.num_apples = max(0, int(num_apples))
        self.apple_asset_path = os.path.abspath(apple_asset_path)
        self.apple_collect_radius_m = float(apple_collect_radius_m)
        self.apple_diameter_m = max(1e-3, float(apple_diameter_m))
        self.apple_spawn_min_separation_m = max(0.0, float(apple_spawn_min_separation_m))
        self.apple_spawn_boundary_margin_m = max(0.0, float(apple_spawn_boundary_margin_m))
        self.apple_spawn_clearance_m = max(0.0, float(apple_spawn_clearance_m))
        self.apple_height_offset_m = float(apple_height_offset_m)
        self.apple_step_penalty = float(apple_step_penalty)
        self.apple_visibility_depth_margin_m = max(0.0, float(apple_visibility_depth_margin_m))
        self.apple_terminate_on_completion = bool(apple_terminate_on_completion)
        self.apple_seed = apple_seed

        self._apple_template_handle: str | None = None
        self._apple_template_id: int | None = None
        self._apple_states: list[AppleState] = []
        self._last_apple_visible: bool = False
        self._last_apples_collected: int = 0
        self._episode_reward_total: float = 0.0
        self._last_apple_status_rows: list[dict[str, Any]] = []
        self._apple_uniform_scale = self._estimate_apple_uniform_scale()
        self._apple_scene_filter: SceneFilterState | None = None
        self._apple_scene_filter_diag: dict[str, Any] = {}
        self._apple_floor_y_ref: float | None = None
        self._apple_spawn_mode: str = "scene_sweep_island_filter"
        # Render-log styling for apple overlay dots on floorplan/progress panels.
        self._topdown_overlay_point_radius_px = 6
        self._progress_overlay_point_radius_px = 6

    def _make_sim(self, glb_path: str, navmesh_path: Optional[str] = None):
        sim = super()._make_sim(glb_path=glb_path, navmesh_path=navmesh_path)
        self._apple_template_handle = self._register_apple_template(sim)
        return sim

    @staticmethod
    def _scene_identifier(scene: dict[str, Any] | None) -> str:
        if not isinstance(scene, dict):
            return ""
        scene_name = scene.get("scene_name")
        if scene_name:
            return str(scene_name)
        glb_path = scene.get("glb_path")
        if glb_path:
            return str(glb_path)
        return ""

    @staticmethod
    def _is_retryable_apple_spawn_error(exc: Exception) -> bool:
        msg = str(exc)
        return (
            "Unable to sample enough sweep-plane apple points" in msg
            or "scene sweep island is empty; cannot place apples" in msg
        )

    def _choose_retry_scene(self, *, excluded_scene_ids: set[str], seed, attempt_idx: int) -> dict[str, Any] | None:
        candidates: list[dict[str, Any]] = []
        for scene in self.scene_list:
            scene_id = self._scene_identifier(scene)
            if scene_id and scene_id in excluded_scene_ids:
                continue
            candidates.append(scene)
        if not candidates:
            return None
        if seed is None:
            idx = int(np.random.randint(0, len(candidates)))
        else:
            rng = np.random.default_rng((int(seed) % (2**32)) + int(attempt_idx) + 1)
            idx = int(rng.integers(0, len(candidates)))
        return candidates[idx]

    def _extract_explicit_apple_positions(
        self,
        options_dict: dict[str, Any],
    ) -> list[np.ndarray] | None:
        raw_positions = options_dict.get("apple_positions")
        restore_state = options_dict.get("restore_state")
        if raw_positions is None and isinstance(restore_state, dict):
            raw_positions = restore_state.get("apple_positions")
        if raw_positions is None:
            return None
        if not isinstance(raw_positions, (list, tuple)):
            raise ValueError("apple_positions must be a list of [x, y, z] points")
        normalized_positions: list[np.ndarray] = []
        for idx, point in enumerate(raw_positions):
            arr = np.asarray(point, dtype=np.float64).reshape(-1)
            if arr.size != 3 or not np.all(np.isfinite(arr)):
                raise ValueError(
                    f"apple_positions[{idx}] must be finite xyz, got {point!r}"
                )
            normalized_positions.append(arr.astype(np.float32))
        return normalized_positions

    def reset(self, seed=None, options=None):
        # Ensure previous-episode apples never affect start-state sampling.
        self._clear_all_apples()
        if options is not None and not isinstance(options, dict):
            # Preserve base-env behavior for non-dict options.
            obs, info = super().reset(seed=seed, options=options)
            self._clear_all_apples()
            self._apple_scene_filter, self._apple_scene_filter_diag = self._infer_spawn_scene_filter()
            self._spawn_apples(seed=seed, scene_filter=self._apple_scene_filter)
            self._episode_reward_total = 0.0
            self._last_apple_status_rows = self._compute_apple_status_rows(obs)
            self._last_apple_visible = any(bool(row.get("visible", False)) for row in self._last_apple_status_rows)
            self._last_apples_collected = 0
            info = dict(info)
            info.update(self._apple_info_payload())
            return obs, info

        options_dict = dict(options or {})
        explicit_apple_positions = self._extract_explicit_apple_positions(options_dict)
        restore_state = options_dict.get("restore_state")
        restore_scene = restore_state.get("scene") if isinstance(restore_state, dict) else None
        explicit_scene = bool(
            options_dict.get("scene") is not None
            or restore_scene is not None
            or explicit_apple_positions is not None
        )
        max_scene_attempts = 1 if explicit_scene else max(1, int(len(self.scene_list)))
        excluded_scene_ids: set[str] = set()
        last_spawn_exc: RuntimeError | None = None

        for attempt_idx in range(max_scene_attempts):
            options_try = dict(options_dict)
            if not explicit_scene and attempt_idx > 0:
                retry_scene = self._choose_retry_scene(
                    excluded_scene_ids=excluded_scene_ids,
                    seed=seed,
                    attempt_idx=attempt_idx,
                )
                if retry_scene is None:
                    break
                options_try["scene"] = retry_scene

            obs, info = super().reset(seed=seed, options=(options_try or None))
            self._clear_all_apples()
            try:
                self._apple_scene_filter, self._apple_scene_filter_diag = self._infer_spawn_scene_filter()
                if explicit_apple_positions is None:
                    self._spawn_apples(seed=seed, scene_filter=self._apple_scene_filter)
                else:
                    self._spawn_apples_from_positions(explicit_apple_positions)
            except RuntimeError as exc:
                if not self._is_retryable_apple_spawn_error(exc):
                    raise
                if explicit_scene:
                    raise
                scene_id = self._scene_identifier(getattr(self, "_scene_dict", None))
                if scene_id:
                    excluded_scene_ids.add(scene_id)
                last_spawn_exc = exc
                continue

            self._episode_reward_total = 0.0
            self._last_apple_status_rows = self._compute_apple_status_rows(obs)
            self._last_apple_visible = any(bool(row.get("visible", False)) for row in self._last_apple_status_rows)
            self._last_apples_collected = 0

            info = dict(info)
            info.update(self._apple_info_payload())
            if attempt_idx > 0:
                info["apple_spawn_scene_retry_count"] = int(attempt_idx)
            return obs, info

        if last_spawn_exc is not None:
            raise RuntimeError(
                f"{last_spawn_exc}. Retried {max_scene_attempts} scene(s) "
                f"and skipped {len(excluded_scene_ids)} scene(s) due to apple placement constraints."
            ) from last_spawn_exc
        raise RuntimeError("Failed to reset apple environment: no valid scene available for apple spawning.")

    def step(self, action: int):
        obs, _reward_unused, terminated, truncated, info = super().step(action)

        rgb_obs = obs.get("rgb")
        if rgb_obs is not None:
            self._gt_rgb = np.asarray(rgb_obs, dtype=np.uint8).copy()

        apples_collected = self._collect_visible_apples(obs)
        reward = float(apples_collected) + float(self.apple_step_penalty)
            
        self._episode_reward_total += float(reward)
        self._last_apples_collected = apples_collected
        self._last_apple_status_rows = self._compute_apple_status_rows(obs)
        self._last_apple_visible = any(bool(row.get("visible", False)) for row in self._last_apple_status_rows)
        apples_remaining = sum(1 for apple in self._apple_states if apple.active)
        apples_completed = bool(self.num_apples > 0 and apples_remaining <= 0)
        terminated = bool(
            terminated
            or (self.apple_terminate_on_completion and apples_completed)
        )

        info = dict(info)
        info.update(self._apple_info_payload())
        info["apples_collected_step"] = int(apples_collected)
        info["apples_completed"] = bool(apples_completed)
        info["episode_reward_total"] = float(self._episode_reward_total)

        return obs, reward, terminated, truncated, info

    def close(self):
        self._clear_all_apples()
        super().close()

    def get_topdown_overlay_world_points(self) -> list[np.ndarray]:
        """Render-log hook: draw active apples as dots on floor map panels."""
        return [apple.position_world for apple in self._apple_states if apple.active]

    def _estimate_apple_uniform_scale(self) -> float:
        """Estimate uniform scale so the largest apple extent ~= bowling-ball diameter."""
        if not os.path.isfile(self.apple_asset_path):
            return 1.0

        try:
            import trimesh

            mesh = trimesh.load(self.apple_asset_path, force="mesh")
            bounds = np.asarray(mesh.bounds, dtype=np.float64)
            if bounds.shape != (2, 3):
                return 1.0
            extents = bounds[1] - bounds[0]
            max_extent = float(np.max(extents))
            if not np.isfinite(max_extent) or max_extent <= 1e-6:
                return 1.0
            raw_scale = float(self.apple_diameter_m / max_extent)
            # Keep a lower bound for tiny assets and a higher cap to avoid exploding
            # scales while still allowing visibly larger apples.
            return float(np.clip(raw_scale, 0.01, 3.0))
        except Exception:
            return 1.0

    def _apple_info_payload(self) -> dict[str, Any]:
        active = sum(1 for apple in self._apple_states if apple.active)
        return {
            "apple_visible": bool(self._last_apple_visible),
            "apples_remaining": int(active),
            "num_apples_total": int(len(self._apple_states)),
            "apple_spawn_mode": str(self._apple_spawn_mode),
            "apple_coordinate_frame": "habitat_scene_world",
            "apple_template_handle": self._apple_template_handle,
            "apple_floor_y_ref": self._apple_floor_y_ref,
            "episode_reward_total": float(self._episode_reward_total),
        }

    def _compute_apple_status_rows(self, obs: dict[str, Any]) -> list[dict[str, Any]]:
        camera_pos = self._camera_position_from_obs(obs)
        rows: list[dict[str, Any]] = []
        for idx, apple in enumerate(self._apple_states):
            row: dict[str, Any] = {
                "index": int(idx),
                "active": bool(apple.active),
            }
            if not apple.active:
                row.update(
                    {
                        "visible": False,
                        "close": False,
                        "collectable": False,
                        "distance_m": None,
                    }
                )
                rows.append(row)
                continue

            dist = float(np.linalg.norm(apple.position_world.astype(np.float64) - camera_pos))
            visible = bool(self._is_apple_visible(apple, obs))
            close = bool(dist <= float(self.apple_collect_radius_m))
            row.update(
                {
                    "visible": visible,
                    "close": close,
                    "collectable": bool(visible and close),
                    "distance_m": dist,
                }
            )
            rows.append(row)
        return rows

    def get_aux_status_panel_lines(self) -> list[str]:
        lines: list[str] = []
        lines.append(f"Reward sum: {float(self._episode_reward_total):.3f}")
        lines.append(
            "Step reward: "
            f"{float(self._last_apples_collected) + float(self.apple_step_penalty):.3f} "
            f"(apples={int(self._last_apples_collected)}, penalty={float(self.apple_step_penalty):.4f})"
        )
        lines.append(
            f"Collect radius: {float(self.apple_collect_radius_m):.2f} m"
        )
        lines.append(
            f"Remaining: {sum(1 for a in self._apple_states if a.active)}/{len(self._apple_states)}"
        )
        if not self._last_apple_status_rows:
            lines.append("No apples")
            return lines

        for row in self._last_apple_status_rows:
            idx = int(row.get("index", -1))
            active = bool(row.get("active", False))
            if not active:
                lines.append(f"A{idx:02d}: collected")
                continue

            vis = "T" if bool(row.get("visible", False)) else "F"
            close = "T" if bool(row.get("close", False)) else "F"
            collectable = "T" if bool(row.get("collectable", False)) else "F"
            dist = row.get("distance_m", None)
            dist_txt = "n/a" if dist is None else f"{float(dist):.2f}m"
            lines.append(
                f"A{idx:02d}: vis={vis} close={close} collect={collectable} d={dist_txt}"
            )
        return lines

    @staticmethod
    def _extract_object_id(obj: Any) -> int | None:
        for name in ("object_id", "id"):
            if hasattr(obj, name):
                try:
                    return int(getattr(obj, name))
                except Exception:
                    pass
        return None

    def _register_apple_template(self, sim) -> str | None:
        if not os.path.isfile(self.apple_asset_path):
            raise FileNotFoundError(
                f"apple asset not found: {self.apple_asset_path}"
            )

        otm_getter = getattr(sim, "get_object_template_manager", None)
        if otm_getter is None:
            raise RuntimeError("habitat_sim build missing object template manager")
        otm = otm_getter()

        # Try both full path and directory loading to handle habitat-sim API differences.
        candidate_paths = [self.apple_asset_path, os.path.dirname(self.apple_asset_path)]
        for candidate in candidate_paths:
            for loader_name in ("load_configs", "load_object_configs"):
                loader = getattr(otm, loader_name, None)
                if loader is None:
                    continue
                try:
                    _ = loader(candidate)
                except TypeError:
                    try:
                        _ = loader(candidate, False)
                    except Exception:
                        pass
                except Exception:
                    pass

        basename = os.path.basename(self.apple_asset_path)
        stem = os.path.splitext(basename)[0]
        object_cfg_basename = f"{stem}.object_config.json"
        object_cfg_path = os.path.join(
            os.path.dirname(self.apple_asset_path),
            object_cfg_basename,
        )

        # Retry loads with explicit object-config path when present.
        if os.path.isfile(object_cfg_path):
            for loader_name in ("load_configs", "load_object_configs"):
                loader = getattr(otm, loader_name, None)
                if loader is None:
                    continue
                try:
                    _ = loader(object_cfg_path)
                except Exception:
                    pass

        handles: list[str] = []
        get_handles = getattr(otm, "get_template_handles", None)
        if callable(get_handles):
            # Do not query empty-string handles here: those include primitive defaults
            # (cube/capsule/etc), which can silently mask a failed Apple asset load.
            for query in (object_cfg_basename, basename, stem):
                try:
                    out = list(get_handles(query))
                except TypeError:
                    continue
                except Exception:
                    continue
                for h in out:
                    hs = str(h)
                    if hs not in handles:
                        handles.append(hs)

        if not handles:
            all_handles: list[str] = []
            if callable(get_handles):
                try:
                    all_handles = [str(h) for h in list(get_handles(""))[:20]]
                except Exception:
                    all_handles = []
            autogenerated = self._register_apple_template_from_render_asset(
                otm=otm,
                stem=stem,
            )
            if autogenerated is None:
                raise RuntimeError(
                    "Unable to resolve an apple object template for "
                    f"{self.apple_asset_path}. "
                    "This Habitat-Sim build likely requires object template metadata "
                    "(object config) for non-primitive objects. "
                    f"Sample available template handles: {all_handles}"
                )
            handles = [autogenerated]

        stem_l = stem.lower()
        basename_l = basename.lower()
        matched = [
            h for h in handles
            if stem_l in h.lower() or basename_l in h.lower()
        ]
        if not matched:
            autogenerated = self._register_apple_template_from_render_asset(
                otm=otm,
                stem=stem,
            )
            if autogenerated is None:
                raise RuntimeError(
                    f"Apple template match failed for {basename}. "
                    f"Matching handles were empty; candidates={handles}"
                )
            matched = [autogenerated]
        handles = matched

        def _score(handle: str) -> tuple[int, int]:
            h = handle.lower()
            return (
                int(stem.lower() in h or basename.lower() in h),
                -len(h),
            )

        chosen = sorted(handles, key=_score, reverse=True)[0]

        # Apply uniform scale if supported by this habitat-sim build.
        getter = getattr(otm, "get_template_by_handle", None)
        register = getattr(otm, "register_template", None)
        if callable(getter) and callable(register):
            try:
                attrs = getter(chosen)
                if hasattr(attrs, "scale"):
                    try:
                        import magnum as mn

                        attrs.scale = mn.Vector3(
                            float(self._apple_uniform_scale),
                            float(self._apple_uniform_scale),
                            float(self._apple_uniform_scale),
                        )
                    except Exception:
                        attrs.scale = np.asarray(
                            [
                                float(self._apple_uniform_scale),
                                float(self._apple_uniform_scale),
                                float(self._apple_uniform_scale),
                            ],
                            dtype=np.float32,
                        )
                scaled_handle = f"{chosen}::apple_scaled"
                registered = register(attrs, scaled_handle)
                if isinstance(registered, str):
                    chosen = registered
                else:
                    chosen = scaled_handle
            except Exception:
                pass

        self._apple_template_id = None
        get_template_id = getattr(otm, "get_template_id_by_handle", None)
        if callable(get_template_id):
            try:
                self._apple_template_id = int(get_template_id(chosen))
            except Exception:
                self._apple_template_id = None

        return chosen

    def _register_apple_template_from_render_asset(self, *, otm, stem: str) -> str | None:
        """Fallback for Habitat builds that don't auto-register raw GLBs as object templates."""
        getter = getattr(otm, "get_template_by_handle", None)
        register = getattr(otm, "register_template", None)
        if not callable(getter) or not callable(register):
            return None

        base_handles = [
            "uvSphereSolid_rings_8_segments_16_useTexCoords_false_useTangents_false",
            "icosphereSolid_subdivs_1",
            "cubeSolid",
        ]
        for base_handle in base_handles:
            try:
                attrs = getter(base_handle)
            except Exception:
                continue
            if attrs is None:
                continue

            wrote_render_asset = False
            for field in ("render_asset_handle", "render_asset_fullpath", "render_asset"):
                if hasattr(attrs, field):
                    try:
                        setattr(attrs, field, self.apple_asset_path)
                        wrote_render_asset = True
                    except Exception:
                        pass
            if not wrote_render_asset:
                continue

            for field in ("collision_asset_handle", "collision_asset_fullpath", "collision_asset"):
                if hasattr(attrs, field):
                    try:
                        setattr(attrs, field, self.apple_asset_path)
                    except Exception:
                        pass
            if hasattr(attrs, "use_bounding_box_for_collision"):
                try:
                    setattr(attrs, "use_bounding_box_for_collision", True)
                except Exception:
                    pass
            if hasattr(attrs, "join_collision_meshes"):
                try:
                    setattr(attrs, "join_collision_meshes", True)
                except Exception:
                    pass
            if hasattr(attrs, "mass"):
                try:
                    setattr(attrs, "mass", 0.02)
                except Exception:
                    pass
            if hasattr(attrs, "scale"):
                try:
                    import magnum as mn

                    attrs.scale = mn.Vector3(
                        float(self._apple_uniform_scale),
                        float(self._apple_uniform_scale),
                        float(self._apple_uniform_scale),
                    )
                except Exception:
                    try:
                        attrs.scale = np.asarray(
                            [
                                float(self._apple_uniform_scale),
                                float(self._apple_uniform_scale),
                                float(self._apple_uniform_scale),
                            ],
                            dtype=np.float32,
                        )
                    except Exception:
                        pass

            handle = f"{stem}::autogen_template"
            try:
                registered = register(attrs, handle)
                if isinstance(registered, str):
                    return registered
                return handle
            except Exception:
                continue

        return None

    def _get_rigid_object_manager(self):
        if self.sim is None:
            return None
        getter = getattr(self.sim, "get_rigid_object_manager", None)
        if getter is None:
            return None
        return getter()

    def _clear_all_apples(self):
        manager = self._get_rigid_object_manager()
        if manager is None:
            self._apple_states = []
            return

        for apple in self._apple_states:
            if apple.active:
                self._remove_apple_object(apple, manager)
        self._apple_states = []
        self._last_apple_visible = False
        self._last_apples_collected = 0
        self._episode_reward_total = 0.0
        self._last_apple_status_rows = []
        self._apple_spawn_mode = "scene_sweep_island_filter"

    def _spawn_apples(self, seed=None, scene_filter: SceneFilterState | None = None):
        self._apple_states = []
        if self.sim is None or self.num_apples <= 0:
            return

        if self._apple_template_handle is None:
            self._apple_template_handle = self._register_apple_template(self.sim)
        if not self._apple_template_handle:
            return

        manager = self._get_rigid_object_manager()
        if manager is None:
            raise RuntimeError("habitat_sim build missing rigid object manager")
        if scene_filter is None:
            raise RuntimeError("apple spawning requires scene sweep filter state")

        rng_seed = self.apple_seed if self.apple_seed is not None else seed
        rng = np.random.default_rng(0 if rng_seed is None else int(rng_seed) % (2**32))
        floor_y = float(self._apple_floor_y_ref) if self._apple_floor_y_ref is not None else 0.0
        spawn_y = float(floor_y + base_env.AGENT_HEIGHT + self.apple_height_offset_m)
        points = self._sample_sweep_plane_points(
            scene_filter=scene_filter,
            count=self.num_apples,
            min_separation=self.apple_spawn_min_separation_m,
            spawn_y=spawn_y,
            rng=rng,
        )
        self._apple_spawn_mode = "scene_sweep_island_filter"

        for floor_point in points:
            center = np.array(
                [
                    float(floor_point[0]),
                    float(spawn_y),
                    float(floor_point[2]),
                ],
                dtype=np.float32,
            )
            obj = self._add_apple_object(center, manager)
            self._apple_states.append(
                AppleState(
                    object_id=self._extract_object_id(obj),
                    position_world=center,
                    active=True,
                    object_ref=obj,
                )
            )

    def _spawn_apples_from_positions(self, apple_positions_world: list[np.ndarray]):
        self._apple_states = []
        if self.sim is None or self.num_apples <= 0:
            return
        if self._apple_template_handle is None:
            self._apple_template_handle = self._register_apple_template(self.sim)
        if not self._apple_template_handle:
            return
        if not apple_positions_world:
            raise RuntimeError("Explicit apple_positions are empty")
        if len(apple_positions_world) < int(self.num_apples):
            raise RuntimeError(
                "Explicit apple_positions has fewer points than requested apples: "
                f"requested={int(self.num_apples)} provided={len(apple_positions_world)}"
            )

        manager = self._get_rigid_object_manager()
        if manager is None:
            raise RuntimeError("habitat_sim build missing rigid object manager")
        self._apple_spawn_mode = "fixed_positions"
        for position_world in apple_positions_world[: int(self.num_apples)]:
            center = np.asarray(position_world, dtype=np.float32).reshape(3)
            obj = self._add_apple_object(center, manager)
            self._apple_states.append(
                AppleState(
                    object_id=self._extract_object_id(obj),
                    position_world=center,
                    active=True,
                    object_ref=obj,
                )
            )

    def _infer_spawn_scene_filter(self) -> tuple[SceneFilterState, dict[str, Any]]:
        if self.sim is None:
            raise RuntimeError("sim not initialised")

        start_pos = np.asarray(self.sim.get_agent(0).get_state().position, dtype=np.float64).reshape(3)
        self._apple_floor_y_ref = float(start_pos[1])
        state, diagnostics = infer_scene_filter_from_sweep(
            self.sim,
            self.sim.pathfinder,
            start_position=start_pos,
            camera_height_m=float(base_env.AGENT_HEIGHT),
            agent_radius_m=float(base_env.AGENT_RADIUS),
        )
        return state, diagnostics

    def _sample_sweep_plane_points(
        self,
        *,
        scene_filter: SceneFilterState,
        count: int,
        min_separation: float,
        spawn_y: float,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        full_mask = np.asarray(scene_filter.island_mask, dtype=bool)
        if not np.any(full_mask):
            raise RuntimeError("scene sweep island is empty; cannot place apples")

        chosen: list[np.ndarray] = []

        edge_margin_m = float(self.apple_spawn_boundary_margin_m) + 0.5 * float(self.apple_diameter_m)
        margin_cells = edge_margin_m / max(float(scene_filter.map_scale), 1e-9)
        strict_mask = self._erode_mask_by_margin(full_mask, margin_cells=margin_cells)
        masks = [strict_mask, full_mask]

        clearance_radius_m = 0.5 * float(self.apple_diameter_m) + float(self.apple_spawn_clearance_m)
        clearance_radii = [clearance_radius_m, 0.5 * float(self.apple_diameter_m), 0.0]

        for mask in masks:
            rows, cols = np.nonzero(mask)
            if len(rows) == 0:
                continue
            order = np.arange(len(rows), dtype=np.int64)
            rng.shuffle(order)

            for radius_m in clearance_radii:
                for idx in order:
                    row = int(rows[idx])
                    col = int(cols[idx])

                    world_xz = scene_filter.grid_to_world(np.asarray([row]), np.asarray([col]))[0]
                    x = float(world_xz[0])
                    z = float(world_xz[1])

                    if min_separation > 0 and chosen:
                        candidate_xz = np.asarray([x, z], dtype=np.float32)
                        deltas = np.asarray(chosen, dtype=np.float32)[:, [0, 2]] - candidate_xz[None, :]
                        dists = np.linalg.norm(deltas, axis=1)
                        if np.any(dists < float(min_separation)):
                            continue

                    center = np.asarray([x, float(spawn_y), z], dtype=np.float32)
                    if radius_m > 0 and not self._apple_position_has_clearance(center, clearance_radius_m=radius_m):
                        continue

                    chosen.append(np.asarray([x, 0.0, z], dtype=np.float32))
                    if len(chosen) >= count:
                        break
                if len(chosen) >= count:
                    break
            if len(chosen) >= count:
                break

        if len(chosen) < count:
            rows, cols = np.nonzero(full_mask)
            order = np.arange(len(rows), dtype=np.int64)
            rng.shuffle(order)
            for idx in order:
                row = int(rows[idx])
                col = int(cols[idx])
                world_xz = scene_filter.grid_to_world(np.asarray([row]), np.asarray([col]))[0]
                x = float(world_xz[0])
                z = float(world_xz[1])
                if min_separation > 0 and chosen:
                    candidate_xz = np.asarray([x, z], dtype=np.float32)
                    deltas = np.asarray(chosen, dtype=np.float32)[:, [0, 2]] - candidate_xz[None, :]
                    dists = np.linalg.norm(deltas, axis=1)
                    if np.any(dists < float(min_separation)):
                        continue
                chosen.append(np.asarray([x, 0.0, z], dtype=np.float32))
                if len(chosen) >= count:
                    break

        if len(chosen) < count:
            raise RuntimeError(
                "Unable to sample enough sweep-plane apple points: "
                f"requested={count}, got={len(chosen)}"
            )

        return chosen[:count]

    @staticmethod
    def _erode_mask_by_margin(mask: np.ndarray, *, margin_cells: float) -> np.ndarray:
        if margin_cells <= 1e-6:
            return np.asarray(mask, dtype=bool)
        try:
            from scipy import ndimage
        except Exception:
            return np.asarray(mask, dtype=bool)

        base = np.asarray(mask, dtype=bool)
        if not np.any(base):
            return base
        dist = ndimage.distance_transform_edt(base)
        eroded = base & (dist >= float(margin_cells))
        if np.any(eroded):
            return eroded
        return base

    def _apple_position_has_clearance(self, center: np.ndarray, *, clearance_radius_m: float) -> bool:
        if self.sim is None or clearance_radius_m <= 1e-6:
            return True
        try:
            import habitat_sim
            import magnum as mn
        except Exception:
            return True

        origin = mn.Vector3(float(center[0]), float(center[1]), float(center[2]))
        dirs = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, -1.0],
                [-1.0, 0.0, 1.0],
                [-1.0, 0.0, -1.0],
                [1.0, 1.0, 0.0],
                [1.0, -1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [-1.0, -1.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, -1.0],
                [0.0, -1.0, 1.0],
                [0.0, -1.0, -1.0],
            ],
            dtype=np.float64,
        )
        for d in dirs:
            n = float(np.linalg.norm(d))
            if n <= 1e-6:
                continue
            d_unit = d / n
            direction = mn.Vector3(float(d_unit[0]), float(d_unit[1]), float(d_unit[2]))
            ray = habitat_sim.geo.Ray(origin, direction)
            hit = self.sim.cast_ray(ray, max_distance=float(clearance_radius_m))
            if hit.has_hits():
                return False
        return True

    def _add_apple_object(self, center: np.ndarray, manager):
        obj = None
        adder_by_handle = getattr(manager, "add_object_by_template_handle", None)
        if callable(adder_by_handle) and self._apple_template_handle is not None:
            try:
                obj = adder_by_handle(self._apple_template_handle)
            except Exception:
                obj = None

        if obj is None and self._apple_template_id is not None:
            adder_by_id = getattr(manager, "add_object_by_template_id", None)
            if callable(adder_by_id):
                try:
                    obj = adder_by_id(int(self._apple_template_id))
                except Exception:
                    obj = None

        if obj is None:
            raise RuntimeError("Failed to add apple object to simulator")

        # Non-blocking collectible. We do not want apples changing nav collision behavior.
        for attr_name in ("collidable", "is_collidable"):
            if hasattr(obj, attr_name):
                try:
                    setattr(obj, attr_name, False)
                except Exception:
                    pass

        if hasattr(obj, "motion_type"):
            try:
                import habitat_sim

                obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
            except Exception:
                pass

        self._set_object_translation(obj, center)
        return obj

    @staticmethod
    def _set_object_translation(obj: Any, position: np.ndarray):
        vec = np.asarray(position, dtype=np.float32).reshape(3)
        try:
            import magnum as mn

            obj.translation = mn.Vector3(float(vec[0]), float(vec[1]), float(vec[2]))
            return
        except Exception:
            pass

        try:
            obj.translation = vec
        except Exception:
            pass

    def _remove_apple_object(self, apple: AppleState, manager):
        removed = False
        if apple.object_ref is not None:
            for remover_name in ("remove_object",):
                remover = getattr(manager, remover_name, None)
                if remover is None:
                    continue
                try:
                    remover(apple.object_ref)
                    removed = True
                    break
                except Exception:
                    pass

        if not removed and apple.object_id is not None:
            for remover_name in ("remove_object_by_id", "remove_object"):
                remover = getattr(manager, remover_name, None)
                if remover is None:
                    continue
                try:
                    remover(int(apple.object_id))
                    removed = True
                    break
                except Exception:
                    pass

        # Fallback: hide far below floor if remove API differs on this build.
        if not removed and apple.object_ref is not None:
            self._set_object_translation(
                apple.object_ref,
                np.asarray([0.0, -10_000.0, 0.0], dtype=np.float32),
            )

        apple.active = False

    def _camera_position_from_obs(self, obs: dict[str, Any]) -> np.ndarray:
        c2w = np.asarray(obs["c2w"], dtype=np.float64).reshape(4, 4)
        return c2w[:3, 3].astype(np.float64)

    def _world_to_pixel(self, point_world: np.ndarray, obs: dict[str, Any]) -> tuple[bool, float, float, float]:
        c2w = np.asarray(obs["c2w"], dtype=np.float64).reshape(4, 4)
        w2c = np.linalg.inv(c2w)

        p = np.ones((4,), dtype=np.float64)
        p[:3] = np.asarray(point_world, dtype=np.float64).reshape(3)
        pc = w2c @ p

        z = float(pc[2])
        if not np.isfinite(z) or z <= 1e-6:
            return False, math.nan, math.nan, z

        fx, fy, cx, cy = np.asarray(obs["fxfycxcy"], dtype=np.float64).reshape(4)
        u = float(fx * (pc[0] / z) + cx)
        v = float(fy * (pc[1] / z) + cy)

        h, w = np.asarray(obs["depth"]).shape[:2]
        in_frame = 0.0 <= u < float(w) and 0.0 <= v < float(h)
        return bool(in_frame), u, v, z

    def _is_apple_visible(self, apple: AppleState, obs: dict[str, Any]) -> bool:
        if not apple.active:
            return False

        _in_frame, center_u, center_v, center_z = self._world_to_pixel(apple.position_world, obs)
        if not np.isfinite(center_z) or center_z <= 1e-6:
            return False

        depth = np.asarray(obs["depth"], dtype=np.float64)
        h, w = depth.shape
        if h <= 0 or w <= 0:
            return False

        if not (0.0 <= center_u < float(w) and 0.0 <= center_v < float(h)):
            return False

        ui = int(np.clip(round(center_u), 0, w - 1))
        vi = int(np.clip(round(center_v), 0, h - 1))
        window = depth[max(0, vi - 1) : min(h, vi + 2), max(0, ui - 1) : min(w, ui + 2)]
        valid = np.isfinite(window) & (window > 1e-4)
        if not np.any(valid):
            return False

        observed_depth = float(np.min(window[valid]))
        apple_radius_m = 0.5 * float(self.apple_diameter_m)
        expected_surface_depth = max(0.0, float(center_z - apple_radius_m))
        return observed_depth + float(self.apple_visibility_depth_margin_m) >= expected_surface_depth

    def _any_visible_apple(self, obs: dict[str, Any]) -> bool:
        for apple in self._apple_states:
            if self._is_apple_visible(apple, obs):
                return True
        return False

    def _collect_visible_apples(self, obs: dict[str, Any]) -> int:
        manager = self._get_rigid_object_manager()
        if manager is None:
            return 0

        camera_pos = self._camera_position_from_obs(obs)
        collected = 0

        for apple in self._apple_states:
            if not apple.active:
                continue
            if not self._is_apple_visible(apple, obs):
                continue

            dist = float(np.linalg.norm(apple.position_world.astype(np.float64) - camera_pos))
            if dist <= float(self.apple_collect_radius_m):
                self._remove_apple_object(apple, manager)
                collected += 1

        return collected


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
