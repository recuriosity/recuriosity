#!/usr/bin/env python3
"""Generate HM3D val episodes with fixed apple placements.

This script is intended to be run after `generate_hm3d_split.py --test` from the
new split pipeline. It reads fixed-start val episodes and writes a companion file
with `apple_positions` added to each episode.

make generate-val-apples-split APPLE_SPLIT_SCRIPT=scripts/generate_val_apples_split.py ARGS="--apple-spread-selection-mode distance_weighted --apple-spread-weight-power 0.5 --apple-min-separation-m 2.0 --apple-black-fraction-required-pass-ratio 0.5 --apple-max-black-fraction 0.5"
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modules.environment.env import (
    AGENT_HEIGHT,
    AGENT_RADIUS,
    HEIGHT,
    HFOV_DEG,
    WIDTH,
)
from modules.eval.eval_utils import derive_navmesh_path, load_episode_specs, resolve_hm3d_stage_path
from modules.eval.scene_filtering import infer_scene_filter_from_sweep
from scripts.dl_hm3d_data import resolve_hm3d_root


def _warn(message: str) -> None:
    print(f"[generate_val_apples_split] warning: {message}", file=sys.stderr, flush=True)


def _build_simulator(glb_path: str, navmesh_path: str | None):
    import habitat_sim

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = glb_path
    sim_cfg.enable_physics = True

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [HEIGHT, WIDTH]
    depth_spec.position = [0.0, AGENT_HEIGHT, 0.0]
    depth_spec.hfov = HFOV_DEG

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [HEIGHT, WIDTH]
    rgb_spec.position = [0.0, AGENT_HEIGHT, 0.0]
    rgb_spec.hfov = HFOV_DEG

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [depth_spec, rgb_spec]
    agent_cfg.height = AGENT_HEIGHT
    agent_cfg.radius = AGENT_RADIUS

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)

    pathfinder = sim.pathfinder
    if navmesh_path and os.path.isfile(navmesh_path):
        if not pathfinder.load_nav_mesh(navmesh_path):
            sim.close()
            return None
    else:
        settings = habitat_sim.nav.NavMeshSettings()
        settings.cell_size = 0.05
        settings.cell_height = 0.2
        settings.agent_radius = AGENT_RADIUS
        settings.agent_height = AGENT_HEIGHT
        settings.agent_max_climb = 0.2
        settings.agent_max_slope = 45.0
        if not sim.recompute_navmesh(pathfinder, settings, include_static_objects=True):
            sim.close()
            return None

    if not pathfinder.is_loaded:
        sim.close()
        return None
    return sim


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


def _candidate_world_xz_points(
    scene_filter_state,
    *,
    apple_diameter_m: float,
    boundary_margin_m: float,
) -> np.ndarray:
    full_mask = np.asarray(scene_filter_state.island_mask, dtype=bool)
    if not np.any(full_mask):
        raise RuntimeError("scene sweep island is empty")

    edge_margin_m = float(boundary_margin_m) + 0.5 * float(apple_diameter_m)
    margin_cells = edge_margin_m / max(float(scene_filter_state.map_scale), 1e-9)
    strict_mask = _erode_mask_by_margin(full_mask, margin_cells=float(margin_cells))
    use_mask = strict_mask if np.any(strict_mask) else full_mask

    rows, cols = np.nonzero(use_mask)
    if len(rows) == 0:
        raise RuntimeError("scene sweep island contains no candidate cells")
    return np.asarray(scene_filter_state.grid_to_world(rows, cols), dtype=np.float32)


def _is_enclosed_by_rays(
    *,
    sim,
    x: float,
    spawn_y: float,
    z: float,
    max_distance_m: float,
    min_distance_m: float,
    required_hit_ratio: float,
) -> bool:
    import habitat_sim
    import magnum as mn

    if float(max_distance_m) <= 0.0:
        return False

    origin = mn.Vector3(float(x), float(spawn_y), float(z))
    min_distance_m = float(min_distance_m)
    max_distance_m = float(max_distance_m)

    # 10-ray enclosure check around apple center:
    # +/-X, +/-Z, +/-Y, and 4 horizontal diagonals.
    ray_dirs = np.asarray(
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
        ],
        dtype=np.float64,
    )

    norms = np.linalg.norm(ray_dirs, axis=1, keepdims=True)
    ray_dirs = ray_dirs / np.maximum(norms, 1e-9)

    ratio = float(np.clip(float(required_hit_ratio), 0.0, 1.0))
    required_hits = max(1, int(np.ceil(ratio * float(len(ray_dirs)))))
    hit_count = 0

    for ray_dir in ray_dirs:
        direction = mn.Vector3(float(ray_dir[0]), float(ray_dir[1]), float(ray_dir[2]))
        hit = sim.cast_ray(
            habitat_sim.geo.Ray(origin, direction),
            max_distance=max_distance_m,
        )
        if not hit.has_hits():
            continue
        distance = float(hit.hits[0].ray_distance)
        if not np.isfinite(distance):
            continue
        if min_distance_m <= distance <= max_distance_m:
            hit_count += 1
            if hit_count >= required_hits:
                return True

    return False


def _snap_point(pathfinder, point_xyz: np.ndarray) -> np.ndarray | None:
    point = np.asarray(point_xyz, dtype=np.float32).reshape(3)
    try:
        snapped = pathfinder.snap_point(point)
    except Exception:
        try:
            import magnum as mn

            snapped = pathfinder.snap_point(
                mn.Vector3(float(point[0]), float(point[1]), float(point[2]))
            )
        except Exception:
            return None

    snapped_np = np.asarray([float(snapped[0]), float(snapped[1]), float(snapped[2])], dtype=np.float32)
    if not np.all(np.isfinite(snapped_np)):
        return None
    return snapped_np


def _get_island_index(pathfinder, point_xyz: np.ndarray) -> int | None:
    point = np.asarray(point_xyz, dtype=np.float32).reshape(3)
    try:
        return int(pathfinder.get_island(point))
    except Exception:
        try:
            import magnum as mn

            return int(pathfinder.get_island(mn.Vector3(float(point[0]), float(point[1]), float(point[2]))))
        except Exception:
            return None


def _has_navmesh_path(
    *,
    sim,
    start_position: np.ndarray,
    end_position: np.ndarray,
    require_navmesh_path: bool,
) -> bool:
    if not bool(require_navmesh_path):
        return True
    try:
        import habitat_sim
    except Exception:
        return True

    try:
        shortest_path = habitat_sim.ShortestPath()
        shortest_path.requested_start = np.asarray(start_position, dtype=np.float32).reshape(3)
        shortest_path.requested_end = np.asarray(end_position, dtype=np.float32).reshape(3)
        found = bool(sim.pathfinder.find_path(shortest_path))
        if not found:
            return False
        geodesic = float(getattr(shortest_path, "geodesic_distance", math.inf))
        return bool(np.isfinite(geodesic) and geodesic >= 0.0)
    except Exception:
        return True


def _invalid_depth_fraction_from_depth_frame(depth_frame: np.ndarray) -> float:
    depth = np.asarray(depth_frame, dtype=np.float32)
    if depth.size == 0:
        return 1.0
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    elif depth.ndim > 2:
        depth = np.squeeze(depth)
    if depth.ndim != 2:
        return 1.0
    valid_mask = np.isfinite(depth) & (depth > 0.0)
    return float(1.0 - (float(np.count_nonzero(valid_mask)) / float(depth.size)))


def _candidate_view_passes_depth_validity(
    *,
    sim,
    position: np.ndarray,
    max_invalid_depth_fraction: float,
    yaw_samples: int,
    required_pass_ratio: float,
) -> bool:
    if float(max_invalid_depth_fraction) >= 1.0:
        return True
    yaw_samples = max(1, int(yaw_samples))
    ratio = float(np.clip(float(required_pass_ratio), 0.0, 1.0))
    required_passes = max(1, int(math.ceil(ratio * float(yaw_samples))))

    try:
        from habitat_sim.utils.common import quat_from_angle_axis, quat_to_coeffs
    except Exception:
        return True

    agent = sim.get_agent(0)
    original_state = agent.get_state()
    pass_count = 0

    try:
        for yaw_idx in range(yaw_samples):
            yaw = (2.0 * math.pi * float(yaw_idx)) / float(yaw_samples)
            q = quat_from_angle_axis(yaw, np.asarray([0.0, 1.0, 0.0], dtype=np.float32))

            state = agent.get_state()
            state.position = np.asarray(position, dtype=np.float32).reshape(3)
            state.rotation = np.asarray(quat_to_coeffs(q), dtype=np.float32)
            agent.set_state(state)

            try:
                observations = sim.get_sensor_observations()
            except Exception:
                return False
            depth = observations.get("depth")
            if depth is None:
                return False
            invalid_depth_fraction = _invalid_depth_fraction_from_depth_frame(np.asarray(depth))
            if invalid_depth_fraction <= float(max_invalid_depth_fraction):
                pass_count += 1
                if pass_count >= required_passes:
                    return True

            remaining = yaw_samples - (yaw_idx + 1)
            if pass_count + remaining < required_passes:
                return False
    finally:
        agent.set_state(original_state)

    return pass_count >= required_passes


def _random_sample_with_ceiling(
    *,
    sim,
    world_xz: np.ndarray,
    start_position: np.ndarray,
    start_xz: np.ndarray,
    count: int,
    rng: np.random.Generator,
    min_separation_m: float,
    min_start_distance_m: float,
    spawn_y: float,
    ceiling_max_distance_m: float,
    ceiling_min_distance_m: float,
    enclosed_required_hit_ratio: float,
    max_nav_snap_distance_m: float,
    max_height_delta_m: float,
    require_navmesh_path: bool,
    max_black_fraction: float,
    black_pixel_threshold: float,
    black_fraction_yaw_samples: int,
    black_fraction_required_pass_ratio: float,
    spread_weight_power: float,
    spread_selection_mode: str,
) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 2), dtype=np.float32)
    if len(world_xz) == 0:
        raise RuntimeError("candidate points are empty")

    points = np.asarray(world_xz, dtype=np.float32)

    order = np.arange(len(points), dtype=np.int64)
    rng.shuffle(order)
    selected_indices: list[int] = []
    selected_xz: list[np.ndarray] = []
    selected_set: set[int] = set()

    nav_ok_cache: dict[int, bool] = {}
    snapped_point_cache: dict[int, np.ndarray] = {}
    snapped_xz_cache: dict[int, np.ndarray] = {}
    depth_ok_cache: dict[int, bool] = {}
    enclosed_ok_cache: dict[int, bool] = {}
    start_dist_ok_cache: dict[int, bool] = {}

    pathfinder = sim.pathfinder
    start_pos = np.asarray(start_position, dtype=np.float32).reshape(3)
    floor_y = float(start_pos[1])
    start_island = _get_island_index(pathfinder, start_pos)

    max_nav_snap_distance_m = max(0.0, float(max_nav_snap_distance_m))
    max_height_delta_m = max(0.0, float(max_height_delta_m))
    max_invalid_depth_fraction = float(np.clip(float(max_black_fraction), 0.0, 1.0))
    spread_weight_power = max(float(spread_weight_power), 1e-6)
    spread_selection_mode = str(spread_selection_mode)

    def _candidate_xz(idx: int) -> np.ndarray:
        snapped = snapped_xz_cache.get(idx)
        if snapped is not None:
            return snapped
        return np.asarray(points[idx], dtype=np.float32)

    def _nav_ok(idx: int) -> bool:
        cached = nav_ok_cache.get(idx)
        if cached is not None:
            return cached

        x = float(points[idx, 0])
        z = float(points[idx, 1])
        candidate_floor = np.asarray([x, floor_y, z], dtype=np.float32)
        snapped = _snap_point(pathfinder, candidate_floor)
        if snapped is None:
            nav_ok_cache[idx] = False
            return False

        snapped_planar_dist = float(np.linalg.norm(snapped[[0, 2]] - candidate_floor[[0, 2]]))
        if max_nav_snap_distance_m > 0.0 and snapped_planar_dist > max_nav_snap_distance_m:
            nav_ok_cache[idx] = False
            return False

        if max_height_delta_m > 0.0 and abs(float(snapped[1]) - floor_y) > max_height_delta_m:
            nav_ok_cache[idx] = False
            return False

        candidate_island = _get_island_index(pathfinder, snapped)
        if start_island is not None and candidate_island is not None:
            if int(candidate_island) != int(start_island):
                nav_ok_cache[idx] = False
                return False

        if not _has_navmesh_path(
            sim=sim,
            start_position=start_pos,
            end_position=snapped,
            require_navmesh_path=bool(require_navmesh_path),
        ):
            nav_ok_cache[idx] = False
            return False

        snapped_point_cache[idx] = np.asarray(snapped, dtype=np.float32)
        snapped_xz_cache[idx] = np.asarray([float(snapped[0]), float(snapped[2])], dtype=np.float32)
        nav_ok_cache[idx] = True
        return True

    def _enclosed_ok(idx: int) -> bool:
        cached = enclosed_ok_cache.get(idx)
        if cached is not None:
            return cached
        if not _nav_ok(idx):
            enclosed_ok_cache[idx] = False
            return False
        xz = _candidate_xz(idx)
        x = float(xz[0])
        z = float(xz[1])
        ok = _is_enclosed_by_rays(
            sim=sim,
            x=x,
            spawn_y=float(spawn_y),
            z=z,
            max_distance_m=float(ceiling_max_distance_m),
            min_distance_m=float(ceiling_min_distance_m),
            required_hit_ratio=float(enclosed_required_hit_ratio),
        )
        enclosed_ok_cache[idx] = ok
        return ok

    def _depth_ok(idx: int) -> bool:
        cached = depth_ok_cache.get(idx)
        if cached is not None:
            return cached
        if max_invalid_depth_fraction >= 1.0:
            depth_ok_cache[idx] = True
            return True
        if not _nav_ok(idx):
            depth_ok_cache[idx] = False
            return False
        snapped_position = snapped_point_cache.get(idx)
        if snapped_position is None:
            depth_ok_cache[idx] = False
            return False

        ok = _candidate_view_passes_depth_validity(
            sim=sim,
            position=snapped_position,
            max_invalid_depth_fraction=float(max_invalid_depth_fraction),
            yaw_samples=int(black_fraction_yaw_samples),
            required_pass_ratio=float(black_fraction_required_pass_ratio),
        )
        depth_ok_cache[idx] = ok
        return ok

    def _start_distance_ok(idx: int) -> bool:
        cached = start_dist_ok_cache.get(idx)
        if cached is not None:
            return cached
        if float(min_start_distance_m) <= 0.0:
            start_dist_ok_cache[idx] = True
            return True
        d = float(np.linalg.norm(_candidate_xz(idx) - start_xz))
        ok = bool(d >= float(min_start_distance_m))
        start_dist_ok_cache[idx] = ok
        return ok

    min_separation_m = max(0.0, float(min_separation_m))

    candidate_indices: list[int] = []
    for idx64 in order:
        idx = int(idx64)
        if not _start_distance_ok(idx):
            continue
        candidate_indices.append(idx)

    candidate_points = np.asarray([_candidate_xz(idx) for idx in candidate_indices], dtype=np.float32)

    while len(selected_indices) < int(count) and candidate_indices:
        if not selected_xz:
            pick_local = int(rng.integers(len(candidate_indices)))
        else:
            selected_points = np.asarray(selected_xz, dtype=np.float32)
            dists = np.linalg.norm(
                candidate_points[:, None, :] - selected_points[None, :, :],
                axis=2,
            )
            min_dists = np.min(dists, axis=1).astype(np.float64)

            if min_separation_m > 0.0:
                feasible = np.nonzero(min_dists >= min_separation_m)[0]
            else:
                feasible = np.arange(len(candidate_indices), dtype=np.int64)

            if spread_selection_mode == "uniform_valid":
                if feasible.size == 0:
                    break
                pick_local = int(rng.choice(feasible))
            else:
                if feasible.size > 0:
                    pool = feasible
                    # Softer than max-min: weighted random among candidates that satisfy separation.
                    base_weights = np.maximum(min_dists[pool] - min_separation_m, 0.0) + 1e-6
                else:
                    # No candidate satisfies strict separation; fallback to best-spread weighted sampling.
                    pool = np.arange(len(candidate_indices), dtype=np.int64)
                    base_weights = np.maximum(min_dists, 0.0) + 1e-6

                # Relaxed weighting: flatten extremes compared to linear distance weighting.
                weights = np.power(base_weights, spread_weight_power)

                weight_sum = float(np.sum(weights))
                if not np.isfinite(weight_sum) or weight_sum <= 0.0:
                    pick_local = int(rng.integers(len(pool)))
                    pick_local = int(pool[pick_local])
                else:
                    probs = weights / weight_sum
                    pick_local = int(rng.choice(pool, p=probs))

        idx = int(candidate_indices.pop(pick_local))
        candidate = np.asarray(candidate_points[pick_local], dtype=np.float32)
        if len(candidate_points) <= 1:
            candidate_points = np.empty((0, 2), dtype=np.float32)
        else:
            candidate_points = np.delete(candidate_points, pick_local, axis=0)

        # Enclosure filtering disabled: rely on nav/path + depth-validity checks.
        # if not _enclosed_ok(idx):
        #     continue
        if not _depth_ok(idx):
            continue

        selected_indices.append(idx)
        selected_xz.append(np.asarray(candidate, dtype=np.float32))
        selected_set.add(idx)

    if len(selected_indices) < int(count):
        num_with_start_clearance = int(sum(1 for ok in start_dist_ok_cache.values() if ok))
        num_nav_ok = int(sum(1 for ok in nav_ok_cache.values() if ok))
        num_enclosed = int(sum(1 for ok in enclosed_ok_cache.values() if ok))
        num_depth_ok = int(sum(1 for ok in depth_ok_cache.values() if ok))
        raise RuntimeError(
            "failed to sample enough random apple points with enclosure/navigation/depth-validity constraints: "
            f"requested={count} selected={len(selected_indices)} "
            f"total_candidates={len(points)} "
            f"candidates_with_start_clearance={num_with_start_clearance} "
            f"candidates_nav_ok={num_nav_ok} "
            f"candidates_enclosed={num_enclosed} "
            f"candidates_depth_valid={num_depth_ok}"
        )

    return np.asarray(selected_xz[:count], dtype=np.float32)


def _episode_apple_positions(
    *,
    sim,
    episode: dict[str, Any],
    num_apples: int,
    apple_diameter_m: float,
    boundary_margin_m: float,
    min_separation_m: float,
    min_start_distance_m: float,
    apple_height_offset_m: float,
    seed: int,
    ceiling_max_distance_m: float,
    ceiling_min_distance_m: float,
    enclosed_required_hit_ratio: float,
    apple_max_nav_snap_distance_m: float,
    apple_max_height_delta_m: float,
    apple_require_navmesh_path: bool,
    apple_max_black_fraction: float,
    apple_black_pixel_threshold: float,
    apple_black_fraction_yaw_samples: int,
    apple_black_fraction_required_pass_ratio: float,
    spread_weight_power: float,
    spread_selection_mode: str,
) -> list[list[float]]:
    start_position = np.asarray(episode["start_position"], dtype=np.float64).reshape(3)
    start_xz = np.asarray([float(start_position[0]), float(start_position[2])], dtype=np.float32)
    scene_filter_state, _scene_filter_diagnostics = infer_scene_filter_from_sweep(
        sim,
        sim.pathfinder,
        start_position=start_position,
        camera_height_m=float(AGENT_HEIGHT),
        agent_radius_m=float(AGENT_RADIUS),
    )

    candidate_world_xz = _candidate_world_xz_points(
        scene_filter_state,
        apple_diameter_m=float(apple_diameter_m),
        boundary_margin_m=float(boundary_margin_m),
    )
    rng = np.random.default_rng(int(seed) % (2**32))
    spawn_y = float(start_position[1] + AGENT_HEIGHT + float(apple_height_offset_m))
    spread_world_xz = _random_sample_with_ceiling(
        sim=sim,
        world_xz=candidate_world_xz,
        start_position=start_position.astype(np.float32),
        start_xz=start_xz,
        count=int(num_apples),
        rng=rng,
        min_separation_m=float(min_separation_m),
        min_start_distance_m=float(min_start_distance_m),
        spawn_y=float(spawn_y),
        ceiling_max_distance_m=float(ceiling_max_distance_m),
        ceiling_min_distance_m=float(ceiling_min_distance_m),
        enclosed_required_hit_ratio=float(enclosed_required_hit_ratio),
        max_nav_snap_distance_m=float(apple_max_nav_snap_distance_m),
        max_height_delta_m=float(apple_max_height_delta_m),
        require_navmesh_path=bool(apple_require_navmesh_path),
        max_black_fraction=float(apple_max_black_fraction),
        black_pixel_threshold=float(apple_black_pixel_threshold),
        black_fraction_yaw_samples=int(apple_black_fraction_yaw_samples),
        black_fraction_required_pass_ratio=float(apple_black_fraction_required_pass_ratio),
        spread_weight_power=float(spread_weight_power),
        spread_selection_mode=str(spread_selection_mode),
    )
    return [[float(x), float(spawn_y), float(z)] for x, z in spread_world_xz]


def _load_json(path: str) -> dict[str, Any]:
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _per_episode_seed(base_seed: int, episode_id: str, episode_idx: int) -> int:
    mix = int(base_seed) * 1_000_003 + int(episode_idx) * 97 + sum(ord(c) for c in episode_id)
    return int(mix % (2**32))


def _validate_episode_fields(episodes: Sequence[dict[str, Any]]) -> None:
    required = ("episode_id", "scene_id", "scene_name", "start_position", "start_rotation")
    for idx, episode in enumerate(episodes):
        for key in required:
            if key not in episode:
                raise ValueError(f"episodes[{idx}] missing required key '{key}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed apple placements for HM3D val episodes.")
    parser.add_argument(
        "--episodes-json",
        type=str,
        default=os.path.join(_ROOT, "data", "splits", "hm3d", "val", "val.json.gz"),
        help="Input episode file (json or json.gz) containing fixed start poses",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=os.path.join(_ROOT, "data", "splits", "hm3d", "val", "val_apples.json"),
        help="Output file path (json or json.gz)",
    )
    parser.add_argument("--num-apples", type=int, default=5, help="Apples per episode")
    parser.add_argument("--seed", type=int, default=42, help="Global seed")
    parser.add_argument("--apple-diameter-m", type=float, default=0.40)
    parser.add_argument("--apple-boundary-margin-m", type=float, default=0.15)
    parser.add_argument(
        "--apple-min-separation-m",
        type=float,
        default=3.0,
        help="Try to keep apples at least this far apart (meters); relaxes automatically if impossible",
    )
    parser.add_argument(
        "--apple-spread-weight-power",
        type=float,
        default=0.5,
        help=(
            "Distance-weight exponent for spread sampling (0.5 = relaxed spread, "
            "1.0 = original linear weighting)"
        ),
    )
    parser.add_argument(
        "--apple-spread-selection-mode",
        type=str,
        default="distance_weighted",
        choices=("distance_weighted", "uniform_valid"),
        help=(
            "Spread selection strategy: "
            "'distance_weighted' uses relaxed distance weighting; "
            "'uniform_valid' samples uniformly among candidates that satisfy min-separation."
        ),
    )
    parser.add_argument(
        "--apple-min-start-distance-m",
        type=float,
        default=0.60,
        help="Reject candidate apples whose center is closer than this to episode start position",
    )
    parser.add_argument("--apple-height-offset-m", type=float, default=-0.15)
    parser.add_argument(
        "--ceiling-max-distance-m",
        type=float,
        default=30.0,
        help="Max distance for enclosure rays to count as valid hits",
    )
    parser.add_argument(
        "--ceiling-min-distance-m",
        type=float,
        default=0.10,
        help="Min distance for enclosure rays to avoid selecting points intersecting nearby geometry",
    )
    parser.add_argument(
        "--enclosed-required-hit-ratio",
        type=float,
        default=0.50,
        help="Required fraction of enclosure rays that must hit geometry (default 0.5 = 5/10)",
    )
    parser.add_argument(
        "--apple-max-nav-snap-distance-m",
        type=float,
        default=0.40,
        help="Reject candidate points whose navmesh snap drifts farther than this in XZ",
    )
    parser.add_argument(
        "--apple-max-height-delta-m",
        type=float,
        default=0.35,
        help="Reject candidate points whose snapped Y differs from start-floor Y by more than this",
    )
    parser.add_argument(
        "--apple-require-navmesh-path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require a valid navmesh path from episode start to sampled apple point",
    )
    parser.add_argument(
        "--apple-max-black-fraction",
        type=float,
        default=0.30,
        help=(
            "Legacy name: max INVALID depth fraction per frame (0.30 = 30% invalid depth pixels). "
            "Used with depth-validity filtering, not RGB blackness."
        ),
    )
    parser.add_argument(
        "--apple-black-pixel-threshold",
        type=float,
        default=8.0,
        help="Deprecated legacy arg; ignored by depth-validity filtering",
    )
    parser.add_argument(
        "--apple-black-fraction-yaw-samples",
        type=int,
        default=4,
        help="How many evenly spaced yaw views to check for black-fraction filtering",
    )
    parser.add_argument(
        "--apple-black-fraction-required-pass-ratio",
        type=float,
        default=0.75,
        help=(
            "Fraction of yaw samples that must satisfy max invalid-depth-fraction "
            "(legacy arg name retained for compatibility)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of episodes to process (for quick sanity checks)",
    )
    args = parser.parse_args()

    if int(args.num_apples) <= 0:
        raise ValueError("--num-apples must be > 0")
    if float(args.apple_spread_weight_power) <= 0.0:
        raise ValueError("--apple-spread-weight-power must be > 0")
    if float(args.apple_max_nav_snap_distance_m) < 0.0:
        raise ValueError("--apple-max-nav-snap-distance-m must be >= 0")
    if float(args.apple_max_height_delta_m) < 0.0:
        raise ValueError("--apple-max-height-delta-m must be >= 0")
    if not (0.0 <= float(args.apple_max_black_fraction) <= 1.0):
        raise ValueError("--apple-max-black-fraction must be in [0, 1]")
    if not (0.0 <= float(args.apple_black_pixel_threshold) <= 255.0):
        raise ValueError("--apple-black-pixel-threshold must be in [0, 255]")
    if int(args.apple_black_fraction_yaw_samples) <= 0:
        raise ValueError("--apple-black-fraction-yaw-samples must be > 0")
    if not (0.0 <= float(args.apple_black_fraction_required_pass_ratio) <= 1.0):
        raise ValueError("--apple-black-fraction-required-pass-ratio must be in [0, 1]")

    resolve_hm3d_root()

    episodes = load_episode_specs(args.episodes_json)
    normalized_episode_count = len(episodes)
    _validate_episode_fields(episodes)
    if args.limit is not None:
        episodes = episodes[: max(0, int(args.limit))]
    if not episodes:
        raise RuntimeError(f"No episodes loaded from {args.episodes_json}")

    input_payload = _load_json(args.episodes_json)
    input_episodes = input_payload.get("episodes")
    if not isinstance(input_episodes, list) or len(input_episodes) != int(normalized_episode_count):
        _warn("Input JSON payload episode count differs from normalized list; output will use normalized episodes")
        input_episodes = episodes
    if args.limit is not None:
        input_episodes = input_episodes[: max(0, int(args.limit))]

    output_episodes: list[dict[str, Any]] = []
    active_scene_key: tuple[str, str | None] | None = None
    active_sim = None

    min_pairwise_distances: list[float] = []
    try:
        for episode_idx, episode in enumerate(episodes):
            raw_episode = dict(input_episodes[episode_idx])
            scene_path = os.path.abspath(str(episode["scene_id"]))
            navmesh_path = derive_navmesh_path(scene_path)
            stage_glb_path = resolve_hm3d_stage_path(scene_path, navmesh_path)
            scene_key = (stage_glb_path, navmesh_path)

            if active_scene_key != scene_key:
                if active_sim is not None:
                    active_sim.close()
                    active_sim = None
                active_sim = _build_simulator(stage_glb_path, navmesh_path)
                if active_sim is None:
                    raise RuntimeError(
                        f"Failed to initialize simulator for scene {episode['scene_name']} at {stage_glb_path}"
                    )
                active_scene_key = scene_key

            apple_positions = _episode_apple_positions(
                sim=active_sim,
                episode=episode,
                num_apples=int(args.num_apples),
                apple_diameter_m=float(args.apple_diameter_m),
                boundary_margin_m=float(args.apple_boundary_margin_m),
                min_separation_m=float(args.apple_min_separation_m),
                min_start_distance_m=float(args.apple_min_start_distance_m),
                apple_height_offset_m=float(args.apple_height_offset_m),
                seed=_per_episode_seed(int(args.seed), str(episode["episode_id"]), episode_idx),
                ceiling_max_distance_m=float(args.ceiling_max_distance_m),
                ceiling_min_distance_m=float(args.ceiling_min_distance_m),
                enclosed_required_hit_ratio=float(args.enclosed_required_hit_ratio),
                apple_max_nav_snap_distance_m=float(args.apple_max_nav_snap_distance_m),
                apple_max_height_delta_m=float(args.apple_max_height_delta_m),
                apple_require_navmesh_path=bool(args.apple_require_navmesh_path),
                apple_max_black_fraction=float(args.apple_max_black_fraction),
                apple_black_pixel_threshold=float(args.apple_black_pixel_threshold),
                apple_black_fraction_yaw_samples=int(args.apple_black_fraction_yaw_samples),
                apple_black_fraction_required_pass_ratio=float(args.apple_black_fraction_required_pass_ratio),
                spread_weight_power=float(args.apple_spread_weight_power),
                spread_selection_mode=str(args.apple_spread_selection_mode),
            )
            raw_episode["apple_positions"] = apple_positions
            output_episodes.append(raw_episode)

            xyz = np.asarray(apple_positions, dtype=np.float32)
            if len(xyz) >= 2:
                xz = xyz[:, [0, 2]]
                deltas = xz[:, None, :] - xz[None, :, :]
                dists = np.sqrt(np.sum(deltas**2, axis=2))
                dists += np.eye(len(xz), dtype=np.float32) * 1e9
                min_pairwise_distances.append(float(np.min(dists)))
    finally:
        if active_sim is not None:
            active_sim.close()

    output_payload = dict(input_payload)
    output_payload["episodes"] = output_episodes
    output_payload["apple_generation"] = {
        "source_episodes_json": os.path.abspath(args.episodes_json),
        "num_apples": int(args.num_apples),
        "seed": int(args.seed),
        "apple_diameter_m": float(args.apple_diameter_m),
        "apple_boundary_margin_m": float(args.apple_boundary_margin_m),
        "apple_min_separation_m": float(args.apple_min_separation_m),
        "apple_min_start_distance_m": float(args.apple_min_start_distance_m),
        "apple_height_offset_m": float(args.apple_height_offset_m),
        "sampling_mode": "random_with_euclidean_spread",
        "apple_spread_weight_power": float(args.apple_spread_weight_power),
        "apple_spread_selection_mode": str(args.apple_spread_selection_mode),
        "ceiling_max_distance_m": float(args.ceiling_max_distance_m),
        "ceiling_min_distance_m": float(args.ceiling_min_distance_m),
        "enclosed_required_hit_ratio": float(args.enclosed_required_hit_ratio),
        "apple_max_nav_snap_distance_m": float(args.apple_max_nav_snap_distance_m),
        "apple_max_height_delta_m": float(args.apple_max_height_delta_m),
        "apple_require_navmesh_path": bool(args.apple_require_navmesh_path),
        "apple_visibility_filter_mode": "depth_invalid_fraction",
        "apple_max_invalid_depth_fraction": float(args.apple_max_black_fraction),
        "apple_invalid_depth_fraction_required_pass_ratio": float(
            args.apple_black_fraction_required_pass_ratio
        ),
        "apple_invalid_depth_yaw_samples": int(args.apple_black_fraction_yaw_samples),
        "apple_black_pixel_threshold_legacy_ignored": float(args.apple_black_pixel_threshold),
        "apple_max_black_fraction": float(args.apple_max_black_fraction),
        "apple_black_pixel_threshold": float(args.apple_black_pixel_threshold),
        "apple_black_fraction_yaw_samples": int(args.apple_black_fraction_yaw_samples),
        "apple_black_fraction_required_pass_ratio": float(args.apple_black_fraction_required_pass_ratio),
        "episode_count": int(len(output_episodes)),
    }

    _write_json(args.output_json, output_payload)
    print(f"Wrote {len(output_episodes)} episodes with apple positions to {args.output_json}")
    if min_pairwise_distances:
        print(
            "Pairwise spread stats (min apple distance per episode): "
            f"mean={float(np.mean(min_pairwise_distances)):.3f}m "
            f"min={float(np.min(min_pairwise_distances)):.3f}m "
            f"max={float(np.max(min_pairwise_distances)):.3f}m",
        )


if __name__ == "__main__":
    main()
