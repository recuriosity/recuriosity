#!/usr/bin/env python3
"""
Generate HM3D train and eval splits.

Train: one episode per scene with placeholder start (env does random reset).
Eval: 2 fixed navmesh start positions per scene for reproducible evaluation across all baselines and methods.

Eval episodes include:
- scene_name: human-readable scene identifier
- start_position, start_rotation: fixed spawn for reproducibility
- island_index: navmesh island ID for reachability (used by completeness metric to filter GT mesh to
  only the reachable subset — rooms/floor reachable from start; excludes disconnected floors, locked rooms)
"""

from __future__ import annotations

import gzip
import json
import math
import os
import random
import sys

import numpy as np

# Add curious_camera to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_sys_path = os.path.dirname(_script_dir)
if _sys_path not in sys.path:
    sys.path.insert(0, _sys_path)

from modules.environment.env import (
    AGENT_RADIUS,
    HEIGHT,
    HFOV_DEG,
    WIDTH,
    list_scene_glbs,
)

AGENT_HEIGHT = 1.25
# Match HabitatMP3DEnv's default headroom filter so generated eval starts
# are valid under the same ceiling-clearance rule used at runtime.
MIN_CEILING_CLEARANCE = 1.0  # metres above camera; reject spawns too close to ceiling
MIN_ISLAND_RADIUS = 1.0  # reject tiny disconnected regions (e.g. small landing)
MIN_START_SEPARATION_M = 2.0  # metres between fixed eval starts in the same scene
MIN_START_VALID_DEPTH_FRAC = 0.6  # mirror env._all_directions_clear
MIN_START_CLEAR_DEPTH_M = 0.08  # mirror env.COLLISION_RADIUS
MIN_LOCAL_SAME_FLOOR_COVERAGE = 0.55  # reject starts with weak same-floor support nearby
MIN_START_ENCLOSURE_FRAC = 0.35  # reject starts with too many open horizontal rays
LOCAL_SAME_FLOOR_Y_TOL_M = 0.35  # nearby samples must stay near the start floor height
LOCAL_SAME_FLOOR_MAX_PLANAR_SNAP_DIST_M = 0.40  # reject offsets that only snap back from far away
LOCAL_SAME_FLOOR_RADII_M = (0.75, 1.5, 2.25)
LOCAL_SAME_FLOOR_NUM_ANGLES = 16
START_ENCLOSURE_NUM_RAYS = 16


def _warn(message: str) -> None:
    print(f"[generate_hm3d_split] warning: {message}", file=sys.stderr, flush=True)


def _is_far_enough(
    position: np.ndarray,
    starts: list[dict],
    min_start_separation_m: float,
) -> bool:
    if min_start_separation_m <= 0:
        return True
    for start in starts:
        existing = np.asarray(start["start_position"], dtype=np.float64)
        if np.linalg.norm(position - existing) < float(min_start_separation_m):
            return False
    return True


def _select_start_candidates(
    candidates: list[dict],
    *,
    n_starts: int,
    min_start_separation_m: float,
    prefer_distinct_islands: bool,
) -> list[dict]:
    # Prefer starts with stronger local floor support, then globally cleaner placements.
    ordered_candidates = sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate.get("local_same_floor_coverage", 0.0)),
            -float(candidate.get("start_enclosure_fraction", 0.0)),
            -float(candidate.get("island_radius", 0.0)),
            -float(candidate.get("ceiling_clearance", 0.0)),
            -float(candidate.get("start_min_clear_depth", 0.0)),
            -float(candidate.get("valid_depth_fraction", 0.0)),
            float(np.asarray(candidate["start_position"], dtype=np.float64)[1]),
        ),
    )
    selected: list[dict] = []
    if prefer_distinct_islands:
        seen_islands: set[int] = set()
        # Optional first pass: diversify islands while keeping starts well separated.
        for candidate in ordered_candidates:
            if len(selected) >= n_starts:
                break
            if candidate["island_index"] in seen_islands:
                continue
            pos = np.asarray(candidate["start_position"], dtype=np.float64)
            if not _is_far_enough(pos, selected, min_start_separation_m):
                continue
            selected.append(candidate)
            seen_islands.add(int(candidate["island_index"]))

    for candidate in ordered_candidates:
        if len(selected) >= n_starts:
            break
        pos = np.asarray(candidate["start_position"], dtype=np.float64)
        if not _is_far_enough(pos, selected, min_start_separation_m):
            continue
        if any(
            np.allclose(
                np.asarray(existing["start_position"], dtype=np.float64),
                pos,
            )
            for existing in selected
        ):
            continue
        selected.append(candidate)

    return selected


def _measure_ceiling_clearance(sim, agent_pos: np.ndarray) -> float:
    """Return distance in metres from the camera to the nearest hit above it."""
    import habitat_sim
    import magnum as mn

    cam_pos = agent_pos.astype(np.float32) + np.array([0, AGENT_HEIGHT, 0], dtype=np.float32)
    origin = mn.Vector3(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))
    up = mn.Vector3(0, 1, 0)
    ray = habitat_sim.geo.Ray(origin, up)
    hit = sim.cast_ray(ray, max_distance=10.0)
    if not hit.has_hits():
        return float("inf")
    return float(min(h.ray_distance for h in hit.hits))


def _sensor_pos_inside_geometry(sim, agent_pos: np.ndarray) -> bool:
    """Mirror the safe env's quick reject for starts embedded in geometry."""
    import habitat_sim
    import magnum as mn

    sensor_pos = agent_pos.astype(np.float32) + np.array([0, AGENT_HEIGHT, 0], dtype=np.float32)
    if np.any(np.isnan(sensor_pos)) or np.any(np.isinf(sensor_pos)):
        return True

    origin = mn.Vector3(float(sensor_pos[0]), float(sensor_pos[1]), float(sensor_pos[2]))
    min_clearance = 0.02
    axes = (
        mn.Vector3(1, 0, 0),
        mn.Vector3(-1, 0, 0),
        mn.Vector3(0, 1, 0),
        mn.Vector3(0, -1, 0),
        mn.Vector3(0, 0, 1),
        mn.Vector3(0, 0, -1),
    )
    for direction in axes:
        ray = habitat_sim.geo.Ray(origin, direction)
        hit = sim.cast_ray(ray, max_distance=min_clearance)
        if hit.has_hits():
            return True
    return False


def _measure_start_view_clearance(
    sim,
    *,
    position: np.ndarray,
    rotation_coeffs: list[float],
) -> tuple[float, float]:
    """Return safe-env-style (valid_depth_fraction, min_clear_depth_m)."""
    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = position.astype(np.float32)
    state.rotation = np.asarray(rotation_coeffs, dtype=np.float32)
    agent.set_state(state)

    observations = sim.get_sensor_observations()
    depth = np.asarray(observations["depth"], dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    valid_depth_fraction = float(valid.mean())
    min_clear_depth = float(depth[valid].min()) if np.any(valid) else 0.0
    return valid_depth_fraction, min_clear_depth


def _measure_start_enclosure_fraction(
    sim,
    *,
    position: np.ndarray,
    num_rays: int = START_ENCLOSURE_NUM_RAYS,
) -> float:
    """Approximate enclosure using binary forward-depth hits over multiple headings.

    HM3D stage raycasts are not reliable enough for indoor/outdoor classification:
    horizontal cast_ray() calls often miss room walls. Instead, rotate the agent
    through evenly spaced yaws and treat each heading as "enclosed" if the
    central forward-looking depth ROI contains any finite depth.
    """
    from habitat_sim.utils.common import quat_from_angle_axis, quat_to_coeffs

    if num_rays <= 0:
        return 1.0

    agent = sim.get_agent(0)
    state = agent.get_state()
    state.position = position.astype(np.float32)
    hits = 0
    for ray_idx in range(int(num_rays)):
        theta = (2.0 * math.pi * ray_idx) / float(num_rays)
        q = quat_from_angle_axis(theta, np.array([0, 1, 0], np.float32))
        state.rotation = np.asarray(quat_to_coeffs(q), dtype=np.float32)
        agent.set_state(state)

        observations = sim.get_sensor_observations()
        depth = np.asarray(observations["depth"], dtype=np.float32)
        h, w = depth.shape
        h0 = int(round(0.40 * h))
        h1 = int(round(0.60 * h))
        w0 = int(round(0.45 * w))
        w1 = int(round(0.55 * w))
        center_roi = depth[h0:h1, w0:w1]
        valid = np.isfinite(center_roi) & (center_roi > 0)
        if np.any(valid):
            hits += 1
    return float(hits) / float(num_rays)


def _measure_local_same_floor_coverage(
    pathfinder,
    *,
    position: np.ndarray,
    island_index: int,
    y_tol_m: float = LOCAL_SAME_FLOOR_Y_TOL_M,
    max_planar_snap_dist_m: float = LOCAL_SAME_FLOOR_MAX_PLANAR_SNAP_DIST_M,
    radii_m: tuple[float, ...] = LOCAL_SAME_FLOOR_RADII_M,
    num_angles: int = LOCAL_SAME_FLOOR_NUM_ANGLES,
) -> float:
    """Approximate how much nearby space stays on the same floor and island.

    Offsets are sampled on deterministic rings around the candidate. A nearby
    offset only counts if it snaps close to the requested XZ location, remains
    on the same navmesh island, and stays within a small Y band of the start.
    """
    import magnum as mn

    pos = np.asarray(position, dtype=np.float64).reshape(3)
    if num_angles <= 0 or not radii_m:
        return 0.0

    accepted = 0
    total = 0
    for radius in radii_m:
        radius = float(radius)
        if radius <= 0:
            continue
        # Nearby offsets only count if they stay close in XZ, same height band, same island.
        for angle_idx in range(num_angles):
            theta = (2.0 * math.pi * angle_idx) / float(num_angles)
            target = np.array(
                [
                    pos[0] + radius * math.cos(theta),
                    pos[1],
                    pos[2] + radius * math.sin(theta),
                ],
                dtype=np.float64,
            )
            snapped = pathfinder.snap_point(
                mn.Vector3(float(target[0]), float(target[1]), float(target[2]))
            )
            snapped_np = np.asarray(
                [float(snapped[0]), float(snapped[1]), float(snapped[2])],
                dtype=np.float64,
            )
            total += 1
            if not np.all(np.isfinite(snapped_np)):
                continue

            planar_snap_dist = float(
                np.linalg.norm(snapped_np[[0, 2]] - target[[0, 2]])
            )
            if planar_snap_dist > float(max_planar_snap_dist_m):
                continue
            if abs(float(snapped_np[1]) - float(pos[1])) > float(y_tol_m):
                continue

            snapped_island = int(
                pathfinder.get_island(
                    mn.Vector3(
                        float(snapped_np[0]),
                        float(snapped_np[1]),
                        float(snapped_np[2]),
                    )
                )
            )
            if snapped_island != int(island_index):
                continue
            accepted += 1

    if total <= 0:
        return 0.0
    return float(accepted) / float(total)


def _sample_navmesh_starts(
    glb_path: str,
    navmesh_path: str | None,
    n_starts: int,
    seed: int,
    *,
    min_ceiling_clearance: float = MIN_CEILING_CLEARANCE,
    min_island_radius: float = MIN_ISLAND_RADIUS,
    min_start_separation_m: float = MIN_START_SEPARATION_M,
    min_start_valid_depth_frac: float = MIN_START_VALID_DEPTH_FRAC,
    min_start_clear_depth_m: float = MIN_START_CLEAR_DEPTH_M,
    min_local_same_floor_coverage: float = MIN_LOCAL_SAME_FLOOR_COVERAGE,
    min_start_enclosure_frac: float = MIN_START_ENCLOSURE_FRAC,
    local_same_floor_y_tol_m: float = LOCAL_SAME_FLOOR_Y_TOL_M,
    prefer_distinct_islands: bool = False,
) -> list[dict]:
    """Sample n_starts navigable positions + yaws per scene. Deterministic given seed.

    Filters out: low ceiling, points on tiny disconnected islands, starts with
    weak same-floor support nearby, starts with too many open horizontal rays,
    and starts that fail the safe env's initial depth-clearance checks.
    Each returned dict includes island_index for reachability / completeness metric.
    """
    import magnum as mn
    import habitat_sim
    from habitat_sim.utils.common import quat_from_angle_axis, quat_to_coeffs

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = glb_path
    sim_cfg.enable_physics = False
    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [HEIGHT, WIDTH]
    depth_spec.position = [0.0, AGENT_HEIGHT, 0.0]
    depth_spec.hfov = HFOV_DEG

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [depth_spec]
    agent_cfg.height = AGENT_HEIGHT
    agent_cfg.radius = AGENT_RADIUS
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    pf = sim.pathfinder
    if navmesh_path and os.path.isfile(navmesh_path):
        if not pf.load_nav_mesh(navmesh_path):
            sim.close()
            return []
    else:
        settings = habitat_sim.nav.NavMeshSettings()
        settings.cell_size = 0.05
        settings.cell_height = 0.2
        settings.agent_radius = AGENT_RADIUS
        settings.agent_height = AGENT_HEIGHT
        settings.agent_max_climb = 0.2
        settings.agent_max_slope = 45.0
        if not sim.recompute_navmesh(sim.pathfinder, settings, include_static_objects=True):
            sim.close()
            return []
    if not pf.is_loaded:
        sim.close()
        return []

    pf.seed(seed)
    rng = random.Random(seed)
    candidates: list[dict] = []
    max_attempts = max(128, n_starts * 192)
    # Over-sample raw navmesh points, then keep only the robust ones.
    for _ in range(max_attempts):
        p = pf.get_random_navigable_point()
        if np.any(np.isnan(p)) or np.any(np.isinf(p)):
            continue
        snapped = pf.snap_point(p)
        if np.any(np.isnan(snapped)):
            continue
        pos = np.array([float(snapped[0]), float(snapped[1]), float(snapped[2])])

        ceiling_clearance = _measure_ceiling_clearance(sim, pos)
        if ceiling_clearance < float(min_ceiling_clearance):
            continue

        pt_vec = mn.Vector3(float(pos[0]), float(pos[1]), float(pos[2]))
        island_idx = pf.get_island(pt_vec)
        island_radius = 0.0
        if min_island_radius > 0:
            try:
                island_radius = float(pf.island_radius(island_idx))
                if island_radius < min_island_radius:
                    continue
            except Exception:
                pass

        # Use local same-floor support to reject edges, stair landings, and awkward ledges.
        local_same_floor_coverage = _measure_local_same_floor_coverage(
            pf,
            position=pos,
            island_index=int(island_idx),
            y_tol_m=local_same_floor_y_tol_m,
        )
        if local_same_floor_coverage < float(min_local_same_floor_coverage):
            continue

        start_enclosure_fraction = _measure_start_enclosure_fraction(
            sim,
            position=pos,
        )
        if start_enclosure_fraction < float(min_start_enclosure_frac):
            continue

        # Depth validity is last because it depends on the sampled heading.
        yaw = rng.uniform(-math.pi, math.pi)
        q = quat_from_angle_axis(yaw, np.array([0, 1, 0], np.float32))
        rot = list(quat_to_coeffs(q))
        if _sensor_pos_inside_geometry(sim, pos):
            continue
        valid_depth_frac, start_min_clear_depth = _measure_start_view_clearance(
            sim,
            position=pos,
            rotation_coeffs=rot,
        )
        if valid_depth_frac < float(min_start_valid_depth_frac):
            continue
        if start_min_clear_depth < float(min_start_clear_depth_m):
            continue
        candidates.append({
            "start_position": pos.tolist(),
            "start_rotation": rot,
            "island_index": int(island_idx),
            "island_radius": float(island_radius),
            "local_same_floor_coverage": float(local_same_floor_coverage),
            "start_enclosure_fraction": float(start_enclosure_fraction),
            "ceiling_clearance": float(ceiling_clearance),
            "start_min_clear_depth": float(start_min_clear_depth),
            "valid_depth_fraction": float(valid_depth_frac),
        })
        if len(candidates) >= max(16, n_starts * 8):
            break

    starts = _select_start_candidates(
        candidates,
        n_starts=n_starts,
        min_start_separation_m=min_start_separation_m,
        prefer_distinct_islands=prefer_distinct_islands,
    )
    sim.close()
    return starts


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: data/splits/hm3d)",
    )
    parser.add_argument("--test", action="store_true", help="Generate eval split (hm3d-val scenes)")
    parser.add_argument(
        "--starts-per-scene",
        type=int,
        default=2,
        help="For eval: fixed start positions per scene (default 2). Train always uses 1 placeholder.",
    )
    parser.add_argument("--val-seed", type=int, default=42, help="Seed for eval navmesh sampling")
    parser.add_argument(
        "--min-ceiling-clearance",
        type=float,
        default=MIN_CEILING_CLEARANCE,
        help=f"Min metres above camera; reject spawns too close to ceiling (default {MIN_CEILING_CLEARANCE})",
    )
    parser.add_argument(
        "--min-island-radius",
        type=float,
        default=MIN_ISLAND_RADIUS,
        help=f"Min island radius (m); reject tiny disconnected regions (default {MIN_ISLAND_RADIUS})",
    )
    parser.add_argument(
        "--min-start-separation",
        type=float,
        default=MIN_START_SEPARATION_M,
        help=(
            "Min Euclidean separation (m) between fixed eval start positions in the same scene "
            f"(default {MIN_START_SEPARATION_M})"
        ),
    )
    parser.add_argument(
        "--min-start-valid-depth-frac",
        type=float,
        default=MIN_START_VALID_DEPTH_FRAC,
        help=(
            "Min fraction of valid pixels in the initial rendered depth frame for an eval start "
            f"(default {MIN_START_VALID_DEPTH_FRAC})"
        ),
    )
    parser.add_argument(
        "--min-start-clear-depth",
        type=float,
        default=MIN_START_CLEAR_DEPTH_M,
        help=(
            "Min closest valid depth (m) in the initial rendered depth frame; mirrors the safe env's "
            f"clearance radius check (default {MIN_START_CLEAR_DEPTH_M})"
        ),
    )
    parser.add_argument(
        "--min-local-same-floor-coverage",
        type=float,
        default=MIN_LOCAL_SAME_FLOOR_COVERAGE,
        help=(
            "Min fraction of nearby ring samples that stay on the same island and floor-height band "
            f"(default {MIN_LOCAL_SAME_FLOOR_COVERAGE})"
        ),
    )
    parser.add_argument(
        "--min-start-enclosure-frac",
        type=float,
        default=MIN_START_ENCLOSURE_FRAC,
        help=(
            "Min fraction of horizontal rays from the start sensor position that hit any geometry; "
            "rays with no hit are treated as open/outside "
            f"(default {MIN_START_ENCLOSURE_FRAC})"
        ),
    )
    parser.add_argument(
        "--local-same-floor-y-tol",
        type=float,
        default=LOCAL_SAME_FLOOR_Y_TOL_M,
        help=(
            "Max Y difference (m) for nearby ring samples to count as the same floor "
            f"(default {LOCAL_SAME_FLOOR_Y_TOL_M})"
        ),
    )
    parser.add_argument(
        "--prefer-distinct-islands",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For eval, prefer choosing fixed starts from different navmesh islands when available. "
            "Disabled by default so selection biases toward the largest/cleanest islands."
        ),
    )
    args = parser.parse_args()

    scene_list = list_scene_glbs(test=args.test)
    split = "val" if args.test else "train"
    out_dir_base = args.output_dir
    if out_dir_base is None:
        out_dir_base = os.path.join(_script_dir, "..", "data", "splits", "hm3d")
    episodes = []

    if args.test:
        # Eval: fixed starts per scene for reproducible eval across all methods
        n_starts = max(1, min(args.starts_per_scene, 5))
        skipped_scenes = 0
        for i, sc in enumerate(scene_list):
            starts = _sample_navmesh_starts(
                sc["glb_path"],
                sc.get("navmesh"),
                n_starts,
                args.val_seed + i,
                min_ceiling_clearance=args.min_ceiling_clearance,
                min_island_radius=args.min_island_radius,
                min_start_separation_m=args.min_start_separation,
                min_start_valid_depth_frac=args.min_start_valid_depth_frac,
                min_start_clear_depth_m=args.min_start_clear_depth,
                min_local_same_floor_coverage=args.min_local_same_floor_coverage,
                min_start_enclosure_frac=args.min_start_enclosure_frac,
                local_same_floor_y_tol_m=args.local_same_floor_y_tol,
                prefer_distinct_islands=args.prefer_distinct_islands,
            )
            if len(starts) < n_starts:
                skipped_scenes += 1
                _warn(
                    f"Skipping eval scene {sc['scene_name']}: found {len(starts)}/{n_starts} "
                    "valid start positions after sampling"
                )
                continue
            for j, s in enumerate(starts):
                episodes.append({
                    "episode_id": f"{split}_{i}_{j}",
                    "scene_id": sc["glb_path"],
                    "scene_name": sc["scene_name"],
                    "start_position": s["start_position"],
                    "start_rotation": s["start_rotation"],
                    "island_index": s["island_index"],
                })
        if skipped_scenes:
            _warn(f"Skipped {skipped_scenes} eval scenes due to insufficient valid start positions")
    else:
        # Train: placeholder (env does random reset)
        for i, sc in enumerate(scene_list):
            episodes.append({
                "episode_id": f"{split}_{i}",
                "scene_id": sc["glb_path"],
                "scene_name": sc["scene_name"],
                "start_position": [0.0, 0.0, 0.0],
                "start_rotation": [0.0, 0.0, 0.0, 1.0],
            })

    out_dir = os.path.join(out_dir_base, split)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{split}.json.gz")
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump({"episodes": episodes}, f, indent=2)
    print(f"Wrote {len(episodes)} episodes to {out_path}")


if __name__ == "__main__":
    main()