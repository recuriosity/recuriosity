"""
Completeness metric utilities for HM3D exploration eval.

Completeness = backward chamfer (GT -> observed): how well the observed point cloud
covers the ground-truth mesh. We sample GT surface points from the full transformed mesh,
keep only the episode's visible floor-height band, and then keep only samples whose
floor-projected navmesh snap lands on the chosen reachable island.

Coordinate systems:
  - GLB/trimesh: raw mesh coords as loaded by trimesh (no Habitat transform)
  - Habitat scene: pathfinder and agent positions; scene = (glb_x, glb_z, -glb_y)

All filtering utilities in this module operate in Habitat scene coordinates.
Convert GLB meshes to Habitat scene once at the boundary, then keep all
subsequent filtering, sampling, and metric computation in that frame.

Legacy eval filtering algorithm retained for `scene_filtering_old.py`:
  1. Sample a visibility-checked disk around the initial camera position.
  2. Cast those disk points downward and cluster the first-hit heights.
  3. Pick the dominant floor cluster and keep its hits as `good_samples`.
  4. Project each `good_sample` to `(x, floor_y, z)`, snap to navmesh, and majority-vote
     the reachable island.
  5. Sample GT points from the full Habitat-frame mesh and keep only those inside the
     visible floor/ceiling height band.
  6. Project each surviving GT point to `(x, floor_y, z)`, snap to navmesh, reject large
     snap distances, and keep only points on the chosen island.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

# Expand the mesh band slightly below the walkable navmesh height so the actual floor slab
# of the current level is not cut away by a purely walkable-surface-based bound.
FLOOR_MESH_FLOOR_MARGIN_M = 0.05

MESH_CEILING_KEEP_MARGIN_M = 0.05

# Robust height-band estimation from mesh surface support near the start island.
HEIGHT_BAND_MAX_FLOOR_SUPPORT_SAMPLES = 50000
HEIGHT_BAND_MAX_RAY_ORIGINS = 2048
HEIGHT_BAND_FLOOR_SUPPORT_PROBE_RADIUS_M = 1.50
HEIGHT_BAND_FLOOR_SUPPORT_Y_TOL_M = 0.30
HEIGHT_BAND_MODE_BIN_M = 0.05
HEIGHT_BAND_FLOOR_MODE_Y_TOL_M = 0.10
HEIGHT_BAND_RAY_EPS_M = 0.05
HEIGHT_BAND_CEILING_MIN_CLEARANCE_M = 1.80
HEIGHT_BAND_CEILING_NORMAL_Y_MAX = -0.60

# When projecting a GT mesh point to the inferred floor and snapping it to the
# navmesh, reject points whose nearest navmesh location is still too far away.
# This prevents distant non-walkable geometry from being accepted just because
# the scene's navmesh has only one nearby island to snap to.
ISLAND_FILTER_MAX_SNAP_DISTANCE_M = 1.5

# Infer the local floor and navmesh island from visible downward ray hits around the
# camera rather than trusting the raw start pose height / metadata island directly.
LOCAL_FLOOR_RAYCAST_RADIUS_M = 4.0
LOCAL_FLOOR_RAYCAST_HEIGHT_BIN_M = 0.10
LOCAL_FLOOR_RAYCAST_DOWN_MAX_DIST_M = 8.0
LOCAL_FLOOR_RAYCAST_VISIBILITY_EPS_M = 0.05
LOCAL_FLOOR_RAYCAST_FLOOR_NORMAL_Y_MIN = 0.5

_LOCAL_FLOOR_RAYCAST_RING_RADII_M = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
_LOCAL_FLOOR_RAYCAST_RING_COUNTS = (1, 8, 16, 24, 32, 40, 48, 56, 64)

DEFAULT_TORCH_NN_QUERY_CHUNK_SIZE = 2048
DEFAULT_TORCH_NN_REFERENCE_CHUNK_SIZE = 8192
_SUPPORTED_NN_BACKENDS = {
    "auto",
    "scipy_ckdtree",
    "torch_cuda_exact",
    "numpy_bruteforce",
}

# The standard HM3D/MP3D GLB-to-Habitat-scene rotation.
# GLB files use Z-up convention; Habitat uses Y-up.
# This rotation converts: (glb_x, glb_y, glb_z) -> (scene_x, scene_y, scene_z)
# Specifically: scene = (glb_x, glb_z, -glb_y) for Z-up to Y-up conversion
_FIXED_GLB_TO_SCENE_ROTATION = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def glb_vertices_to_scene(vertices: np.ndarray) -> np.ndarray:
    """Convert GLB/trimesh vertices to Habitat scene coords using the fixed axis swap."""
    return apply_vertex_transform(vertices, _fixed_glb_to_scene_transform())


def scene_vertices_to_glb(vertices: np.ndarray) -> np.ndarray:
    """Convert Habitat scene coords back to raw GLB/trimesh coords."""
    return apply_vertex_transform(vertices, np.linalg.inv(_fixed_glb_to_scene_transform()))


def _fixed_glb_to_scene_transform() -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _FIXED_GLB_TO_SCENE_ROTATION
    return transform


def apply_vertex_transform(vertices: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a row-vector rigid/affine transform to vertices."""
    vertices = np.asarray(vertices, dtype=np.float64)
    return vertices @ transform[:3, :3].T + transform[:3, 3]


def _vector3_to_numpy(vec3) -> np.ndarray:
    if vec3 is None:
        raise ValueError("vec3 is None")
    try:
        return np.asarray([float(vec3[0]), float(vec3[1]), float(vec3[2])], dtype=np.float64)
    except Exception:
        return np.asarray([float(vec3.x), float(vec3.y), float(vec3.z)], dtype=np.float64)


def _range3d_to_numpy_bounds(range3d) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(range3d, "min") and hasattr(range3d, "max"):
        return _vector3_to_numpy(range3d.min), _vector3_to_numpy(range3d.max)

    arr = np.asarray(range3d, dtype=np.float64)
    if arr.shape == (2, 3):
        return arr[0], arr[1]
    raise ValueError(f"Unsupported Range3D representation: shape={arr.shape}")


def _sample_points_deterministic(points: np.ndarray, max_points: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if len(points) <= max_points:
        return points
    sample_idx = np.linspace(0, len(points) - 1, num=max_points, dtype=np.int64)
    return points[sample_idx]


def _get_stage_scale(stage_template) -> np.ndarray:
    scale = np.ones(3, dtype=np.float64)
    if stage_template is not None:
        try:
            scale = _vector3_to_numpy(stage_template.scale)
        except Exception:
            pass
        try:
            scale = scale * float(stage_template.units_to_meters)
        except Exception:
            pass

    scale = np.asarray(scale, dtype=np.float64)
    if not np.all(np.isfinite(scale)) or np.any(np.abs(scale) < 1e-9):
        return np.ones(3, dtype=np.float64)
    return scale


def infer_habitat_stage_transform(
    mesh_vertices: np.ndarray,
    sim,
    *,
    pathfinder=None,
    island_index: int = -1,
    max_score_points: int = 4096,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build the raw-GLB -> Habitat-scene transform used for completeness evaluation."""
    vertices = np.asarray(mesh_vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Expected mesh vertices of shape (N, 3), got {vertices.shape}")
    del pathfinder, max_score_points

    try:
        stage_template = sim.get_stage_initialization_template()
    except Exception:
        stage_template = None

    scene_min, scene_max = _range3d_to_numpy_bounds(sim.scene_aabb)
    scene_center = 0.5 * (scene_min + scene_max)
    scene_extent = scene_max - scene_min

    scale = _get_stage_scale(stage_template)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _FIXED_GLB_TO_SCENE_ROTATION @ np.diag(scale)

    transformed = apply_vertex_transform(vertices, transform)
    transformed_min = np.min(transformed, axis=0)
    transformed_max = np.max(transformed, axis=0)
    extent_error = float(np.linalg.norm((transformed_max - transformed_min) - scene_extent))
    center_error = float(
        np.linalg.norm(0.5 * (transformed_max + transformed_min) - scene_center)
    )

    metadata = {
        "mode": "habitat_fixed_scale",
        "rotation_name": "fixed_glb_to_scene",
        "rotation": _FIXED_GLB_TO_SCENE_ROTATION.copy(),
        "scale": scale.copy(),
        "translation": np.zeros(3, dtype=np.float64),
        "extent_error": extent_error,
        "center_error": center_error,
        "floor_alignment_error": None,
        "reachable_hits": None,
        "scene_aabb_min": scene_min.copy(),
        "scene_aabb_max": scene_max.copy(),
        "alignment_island_index": int(island_index),
        "navmesh_vertex_count": None,
        "navmesh_all_vertex_count": None,
        "stage_asset_fullpath": getattr(stage_template, "render_asset_fullpath", None)
        if stage_template is not None
        else None,
        "stage_asset_handle": getattr(stage_template, "render_asset_handle", None)
        if stage_template is not None
        else None,
        "stage_origin": _vector3_to_numpy(stage_template.origin)
        if stage_template is not None and hasattr(stage_template, "origin")
        else None,
        "stage_orient_up": _vector3_to_numpy(stage_template.orient_up)
        if stage_template is not None and hasattr(stage_template, "orient_up")
        else None,
        "stage_orient_front": _vector3_to_numpy(stage_template.orient_front)
        if stage_template is not None and hasattr(stage_template, "orient_front")
        else None,
        "transformed_bounds_min": transformed_min.copy(),
        "transformed_bounds_max": transformed_max.copy(),
        "navmesh_bounds_min": None,
        "navmesh_bounds_max": None,
        "navmesh_all_bounds_min": None,
        "navmesh_all_bounds_max": None,
    }
    return transform, metadata


def validate_mesh_observation_alignment(
    mesh_vertices_scene: np.ndarray,
    observed_points: np.ndarray,
    *,
    tolerance_m: float = 0.5,
) -> dict[str, Any]:
    """Validate that mesh vertices and observed points are in the same coordinate system.
    
    This is useful for debugging alignment issues.
    """
    mesh = np.asarray(mesh_vertices_scene, dtype=np.float64)
    obs = np.asarray(observed_points, dtype=np.float64)
    
    if len(mesh) == 0 or len(obs) == 0:
        return {
            "valid": False,
            "reason": "empty_input",
            "mesh_count": len(mesh),
            "observed_count": len(obs),
        }
    
    # Check bounding box overlap
    mesh_min = np.min(mesh, axis=0)
    mesh_max = np.max(mesh, axis=0)
    obs_min = np.min(obs, axis=0)
    obs_max = np.max(obs, axis=0)
    
    # Compute overlap
    overlap_min = np.maximum(mesh_min, obs_min)
    overlap_max = np.minimum(mesh_max, obs_max)
    overlap_extent = np.maximum(0, overlap_max - overlap_min)
    overlap_volume = float(np.prod(overlap_extent))
    
    mesh_extent = mesh_max - mesh_min
    obs_extent = obs_max - obs_min
    mesh_volume = float(np.prod(mesh_extent))
    obs_volume = float(np.prod(obs_extent))
    
    # Sample observed points and check distance to mesh
    sample_count = min(1000, len(obs))
    if len(obs) > sample_count:
        sample_idx = np.random.choice(len(obs), sample_count, replace=False)
        obs_sample = obs[sample_idx]
    else:
        obs_sample = obs
    
    # For each observed point, find minimum distance to mesh vertices
    mesh_sample_count = min(10000, len(mesh))
    if len(mesh) > mesh_sample_count:
        mesh_idx = np.linspace(0, len(mesh) - 1, mesh_sample_count, dtype=np.int64)
        mesh_sample = mesh[mesh_idx]
    else:
        mesh_sample = mesh
    
    # Compute distances
    from scipy.spatial import cKDTree
    tree = cKDTree(mesh_sample)
    distances, _ = tree.query(obs_sample, k=1)
    
    mean_distance = float(np.mean(distances))
    median_distance = float(np.median(distances))
    within_tolerance = float(np.mean(distances < tolerance_m))
    
    return {
        "valid": within_tolerance > 0.5,
        "mean_distance_m": mean_distance,
        "median_distance_m": median_distance,
        "within_tolerance_ratio": within_tolerance,
        "tolerance_m": tolerance_m,
        "mesh_bounds": {"min": mesh_min.tolist(), "max": mesh_max.tolist()},
        "obs_bounds": {"min": obs_min.tolist(), "max": obs_max.tolist()},
        "overlap_volume": overlap_volume,
        "mesh_volume": mesh_volume,
        "obs_volume": obs_volume,
        "overlap_ratio_of_obs": overlap_volume / max(obs_volume, 1e-9),
    }


def mesh_to_habitat_scene(
    mesh,
    *,
    sim=None,
    pathfinder=None,
    island_index: int = -1,
    return_metadata: bool = False,
):
    """Return a trimesh with vertices expressed in Habitat scene coordinates.

    If ``sim`` is provided, use Habitat's instantiated stage metadata/AABB to infer the
    transform actually used by the simulator. Otherwise use the fixed
    GLB -> scene axis swap.
    
    IMPORTANT: When using with depth observations from Habitat, always pass ``sim``
    to ensure the mesh is in the same coordinate system as the observations.
    """
    import trimesh

    vertices = np.asarray(mesh.vertices)
    if sim is not None:
        if pathfinder is None:
            pathfinder = sim.pathfinder
        transform, metadata = infer_habitat_stage_transform(
            vertices,
            sim,
            pathfinder=pathfinder,
            island_index=island_index,
        )
        vertices_scene = apply_vertex_transform(vertices, transform)
    else:
        transform = _fixed_glb_to_scene_transform()
        metadata = {
            "mode": "fixed_fallback",
            "rotation_name": "fixed_glb_to_scene",
            "rotation": _FIXED_GLB_TO_SCENE_ROTATION.copy(),
            "scale": np.ones(3, dtype=np.float64),
            "translation": np.zeros(3, dtype=np.float64),
            "extent_error": None,
            "center_error": None,
            "floor_alignment_error": None,
            "reachable_hits": None,
            "scene_aabb_min": None,
            "scene_aabb_max": None,
            "navmesh_vertex_count": None,
            "stage_asset_fullpath": None,
            "stage_asset_handle": None,
            "stage_origin": None,
            "stage_orient_up": None,
            "stage_orient_front": None,
        }
        vertices_scene = apply_vertex_transform(vertices, transform)

    faces = getattr(mesh, "faces", None)
    if faces is None:
        faces = getattr(mesh, "triangles", None)
    if faces is None:
        raise ValueError("mesh needs .faces or .triangles")

    kwargs = {"vertices": vertices_scene, "faces": np.asarray(faces), "process": False}
    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        kwargs["vertex_colors"] = np.asarray(mesh.visual.vertex_colors).copy()
    mesh_scene = trimesh.Trimesh(**kwargs)
    if return_metadata:
        metadata = dict(metadata)
        metadata["transform"] = transform.copy()
        return mesh_scene, metadata
    return mesh_scene


def _classify_mesh_points_to_island(
    mesh_points: np.ndarray,
    pathfinder,
    island_index: int,
    y_lo: float,
    y_hi: float,
    projection_y: float,
    max_snap_distance_m: float = ISLAND_FILTER_MAX_SNAP_DISTANCE_M,
) -> np.ndarray:
    """Keep points whose floor-height navmesh snap lands on ``island_index``."""
    import magnum as mn

    points = np.asarray(mesh_points, dtype=np.float64)
    keep_mask = np.zeros(len(points), dtype=bool)
    if island_index < 0:
        return keep_mask

    height_mask = (points[:, 1] >= y_lo) & (points[:, 1] < y_hi)
    for i in np.where(height_mask)[0]:
        point = points[i]
        projected_point = np.asarray(
            [float(point[0]), float(projection_y), float(point[2])],
            dtype=np.float64,
        )
        snapped = pathfinder.snap_point(
            mn.Vector3(
                float(projected_point[0]),
                float(projected_point[1]),
                float(projected_point[2]),
            )
        )
        snapped_np = np.asarray(snapped, dtype=np.float64)
        if snapped_np.shape != (3,) or not np.all(np.isfinite(snapped_np)):
            continue
        if float(np.linalg.norm(snapped_np - projected_point)) > float(max_snap_distance_m):
            continue
        keep_mask[i] = int(pathfinder.get_island(snapped)) == int(island_index)
    return keep_mask


def _mask_mesh_points_by_height_band(
    mesh_points: np.ndarray,
    *,
    y_lo: float,
    y_hi: float,
) -> np.ndarray:
    mesh_points = np.asarray(mesh_points, dtype=np.float64)
    if len(mesh_points) == 0:
        return np.zeros(0, dtype=bool)
    return (mesh_points[:, 1] >= float(y_lo)) & (mesh_points[:, 1] < float(y_hi))


def _compute_island_filter_band(
    reference_vertices: np.ndarray,
    island_index: int,
    *,
    start_y: float | None = None,
    return_diagnostics: bool = False,
) -> tuple[float, float, float] | tuple[tuple[float, float, float], dict[str, Any]]:
    """Derive a simple mesh height band from start_y and mesh extent.

    start_y comes from raycasting (infer_local_floor_island_from_raycast)
    and is already the correct floor height. The band is simply:
      y_lo = start_y - FLOOR_MESH_FLOOR_MARGIN_M
      y_hi = start_y + HEIGHT_BAND_CEILING_MIN_CLEARANCE_M  (conservative default)
    clamped to the actual mesh extents.

    The main eval path usually passes an explicit visible floor/ceiling band, but
    standalone helpers can still use this simpler start_y-based band.
    """
    diagnostics: dict[str, Any] = {
        "input_island_index": int(island_index),
        "input_start_y": None if start_y is None else float(start_y),
    }

    def _ret(result: tuple[float, float, float]) -> tuple[float, float, float] | tuple[tuple[float, float, float], dict[str, Any]]:
        diagnostics["final_band"] = {
            "y_lo": float(result[0]),
            "y_hi": float(result[1]),
            "y_mid": float(result[2]),
        }
        return (result, diagnostics) if return_diagnostics else result

    reference_vertices = np.asarray(reference_vertices, dtype=np.float64)
    if len(reference_vertices) == 0:
        diagnostics["mode"] = "empty_reference_vertices"
        return _ret((0.0, 0.0, 0.0))

    mesh_max_y = float(np.max(reference_vertices[:, 1]))
    mesh_min_y = float(np.min(reference_vertices[:, 1]))
    diagnostics["mesh_min_y"] = mesh_min_y
    diagnostics["mesh_max_y"] = mesh_max_y

    if island_index < 0 and start_y is None:
        diagnostics["mode"] = "no_island_no_start_y"
        return _ret((mesh_min_y, mesh_max_y, 0.5 * (mesh_min_y + mesh_max_y)))

    if start_y is None or not np.isfinite(float(start_y)):
        diagnostics["mode"] = "no_start_y_full_mesh"
        return _ret((mesh_min_y, mesh_max_y, 0.5 * (mesh_min_y + mesh_max_y)))

    floor_y = float(start_y)
    y_lo = max(mesh_min_y, floor_y - float(FLOOR_MESH_FLOOR_MARGIN_M))
    y_hi = min(mesh_max_y, floor_y + float(HEIGHT_BAND_CEILING_MIN_CLEARANCE_M))
    if y_hi <= y_lo:
        y_hi = mesh_max_y
    y_mid = float(np.clip(
        floor_y,
        y_lo + HEIGHT_BAND_RAY_EPS_M,
        max(y_lo + HEIGHT_BAND_RAY_EPS_M, y_hi - HEIGHT_BAND_RAY_EPS_M),
    ))
    diagnostics["mode"] = "start_y_direct"
    return _ret((y_lo, y_hi, y_mid))


def _compute_visible_floor_height_band(
    reference_vertices: np.ndarray,
    sim,
    *,
    start_position: np.ndarray | Sequence[float],
    camera_height_m: float,
    sample_radius_m: float = LOCAL_FLOOR_RAYCAST_RADIUS_M,
    downward_max_distance_m: float = LOCAL_FLOOR_RAYCAST_DOWN_MAX_DIST_M,
    visibility_epsilon_m: float = LOCAL_FLOOR_RAYCAST_VISIBILITY_EPS_M,
    floor_normal_y_min: float = LOCAL_FLOOR_RAYCAST_FLOOR_NORMAL_Y_MIN,
    return_diagnostics: bool = False,
) -> tuple[float, float, float] | tuple[tuple[float, float, float], dict[str, Any]]:
    """Estimate a floor/ceiling band using visible scene geometry only.

    This path intentionally avoids navmesh snapping and island metadata. It:
      1. samples a deterministic disk around the camera,
      2. rejects origins occluded from the agent,
      3. casts downward rays and clusters first-hit heights,
      4. uses the dominant floor-height mode,
      5. casts upward rays from floor-mode points and uses the dominant ceiling mode.
    """
    diagnostics: dict[str, Any] = {
        "input_start_position": np.asarray(start_position, dtype=np.float64).reshape(3).tolist(),
        "camera_height_m": float(camera_height_m),
    }

    def _ret(result: tuple[float, float, float]) -> tuple[float, float, float] | tuple[tuple[float, float, float], dict[str, Any]]:
        diagnostics["final_band"] = {
            "y_lo": float(result[0]),
            "y_hi": float(result[1]),
            "y_mid": float(result[2]),
        }
        return (result, diagnostics) if return_diagnostics else result

    reference_vertices = np.asarray(reference_vertices, dtype=np.float64)
    if len(reference_vertices) == 0:
        diagnostics["mode"] = "empty_reference_vertices"
        return _ret((0.0, 0.0, 0.0))

    import habitat_sim
    import magnum as mn

    start_position = np.asarray(start_position, dtype=np.float64).reshape(3)
    camera_position = start_position.copy()
    camera_position[1] += float(camera_height_m)
    mesh_min_y = float(np.min(reference_vertices[:, 1]))
    mesh_max_y = float(np.max(reference_vertices[:, 1]))
    diagnostics["mesh_min_y"] = mesh_min_y
    diagnostics["mesh_max_y"] = mesh_max_y

    origins = _iter_local_floor_raycast_origins(
        camera_position,
        sample_radius_m=float(sample_radius_m),
    )
    visible_candidate_count = 0
    rejected_blocked_count = 0
    rejected_downward_miss_count = 0
    rejected_non_floor_count = 0
    downward_hits: list[np.ndarray] = []

    camera_origin = mn.Vector3(
        float(camera_position[0]),
        float(camera_position[1]),
        float(camera_position[2]),
    )
    for origin in origins:
        horizontal_delta = np.asarray(origin - camera_position, dtype=np.float64)
        horizontal_delta[1] = 0.0
        horizontal_dist = float(np.linalg.norm(horizontal_delta))
        if horizontal_dist > float(visibility_epsilon_m):
            direction = horizontal_delta / horizontal_dist
            vis_ray = habitat_sim.geo.Ray(
                camera_origin,
                mn.Vector3(float(direction[0]), 0.0, float(direction[2])),
            )
            max_distance = max(0.0, horizontal_dist - float(visibility_epsilon_m))
            vis_hit = sim.cast_ray(vis_ray, max_distance=max_distance)
            if vis_hit.has_hits():
                nearest = min(float(hit.ray_distance) for hit in vis_hit.hits)
                if nearest < max_distance:
                    rejected_blocked_count += 1
                    continue

        visible_candidate_count += 1
        down_ray = habitat_sim.geo.Ray(
            mn.Vector3(float(origin[0]), float(origin[1]), float(origin[2])),
            mn.Vector3(0.0, -1.0, 0.0),
        )
        down_hit = sim.cast_ray(down_ray, max_distance=float(downward_max_distance_m))
        if not down_hit.has_hits():
            rejected_downward_miss_count += 1
            continue

        best_hit = min(down_hit.hits, key=lambda hit: float(hit.ray_distance))
        hit_point = _vector3_to_numpy(best_hit.point)
        if not np.all(np.isfinite(hit_point)):
            rejected_downward_miss_count += 1
            continue

        normal = getattr(best_hit, "normal", None)
        if normal is not None:
            normal_np = _vector3_to_numpy(normal)
            normal_norm = float(np.linalg.norm(normal_np))
            if normal_norm > 1e-12 and float(normal_np[1]) / normal_norm < float(floor_normal_y_min):
                rejected_non_floor_count += 1
                continue

        downward_hits.append(hit_point)

    diagnostics["candidate_count"] = len(origins)
    diagnostics["visible_candidate_count"] = int(visible_candidate_count)
    diagnostics["accepted_candidate_count"] = int(len(downward_hits))
    diagnostics["rejected_blocked_count"] = int(rejected_blocked_count)
    diagnostics["rejected_downward_miss_count"] = int(rejected_downward_miss_count)
    diagnostics["rejected_non_floor_count"] = int(rejected_non_floor_count)

    if not downward_hits:
        raise RuntimeError("No visible downward floor hits found for height-band inference")

    downward_hits_arr = np.asarray(downward_hits, dtype=np.float64)
    downward_hit_y = downward_hits_arr[:, 1]
    bins = np.arange(
        float(np.min(downward_hit_y)),
        float(np.max(downward_hit_y)) + 1.5 * HEIGHT_BAND_MODE_BIN_M,
        HEIGHT_BAND_MODE_BIN_M,
        dtype=np.float64,
    )
    if len(bins) < 2:
        floor_y = float(np.median(downward_hit_y))
    else:
        hist, edges = np.histogram(downward_hit_y, bins=bins)
        floor_idx = int(np.argmax(hist))
        floor_y = float(0.5 * (edges[floor_idx] + edges[floor_idx + 1]))
    diagnostics["estimated_floor_y"] = float(floor_y)

    y_lo = max(mesh_min_y, floor_y - float(FLOOR_MESH_FLOOR_MARGIN_M))
    y_hi = mesh_max_y
    floor_mode_mask = np.abs(downward_hit_y - floor_y) <= HEIGHT_BAND_FLOOR_MODE_Y_TOL_M
    floor_mode_points = downward_hits_arr[floor_mode_mask]
    if len(floor_mode_points) < 16:
        floor_mode_points = downward_hits_arr
    diagnostics["floor_mode_point_count"] = int(len(floor_mode_points))

    ceiling_hits: list[float] = []
    ray_origins = _sample_points_deterministic(
        floor_mode_points,
        HEIGHT_BAND_MAX_RAY_ORIGINS,
    )
    upward_max_distance = max(0.0, mesh_max_y - floor_y + 1.0)
    for point in ray_origins:
        up_ray = habitat_sim.geo.Ray(
            mn.Vector3(float(point[0]), float(floor_y) + HEIGHT_BAND_RAY_EPS_M, float(point[2])),
            mn.Vector3(0.0, 1.0, 0.0),
        )
        up_hit = sim.cast_ray(up_ray, max_distance=float(upward_max_distance))
        if not up_hit.has_hits():
            continue

        nearest_y: float | None = None
        nearest_distance: float | None = None
        for hit in up_hit.hits:
            hit_point = _vector3_to_numpy(hit.point)
            if not np.all(np.isfinite(hit_point)):
                continue
            hit_y = float(hit_point[1])
            if hit_y <= floor_y + HEIGHT_BAND_CEILING_MIN_CLEARANCE_M:
                continue

            normal = getattr(hit, "normal", None)
            if normal is not None:
                normal_np = _vector3_to_numpy(normal)
                normal_norm = float(np.linalg.norm(normal_np))
                if (
                    normal_norm > 1e-12
                    and float(normal_np[1]) / normal_norm > HEIGHT_BAND_CEILING_NORMAL_Y_MAX
                ):
                    continue

            distance = float(hit.ray_distance)
            if nearest_distance is None or distance < nearest_distance:
                nearest_distance = distance
                nearest_y = hit_y

        if nearest_y is not None:
            ceiling_hits.append(float(nearest_y))

    diagnostics["ceiling_hit_count"] = int(len(ceiling_hits))
    if ceiling_hits:
        ceiling_hits_arr = np.asarray(ceiling_hits, dtype=np.float64)
        bins = np.arange(
            float(np.min(ceiling_hits_arr)),
            float(np.max(ceiling_hits_arr)) + 1.5 * HEIGHT_BAND_MODE_BIN_M,
            HEIGHT_BAND_MODE_BIN_M,
            dtype=np.float64,
        )
        if len(bins) < 2:
            ceiling_y = float(np.median(ceiling_hits_arr))
        else:
            hist, edges = np.histogram(ceiling_hits_arr, bins=bins)
            ceil_idx = int(np.argmax(hist))
            ceiling_y = float(0.5 * (edges[ceil_idx] + edges[ceil_idx + 1]))
        diagnostics["estimated_ceiling_y"] = float(ceiling_y)
        y_hi = min(mesh_max_y, ceiling_y + float(MESH_CEILING_KEEP_MARGIN_M))

    if not np.isfinite(y_hi) or y_hi <= y_lo:
        y_hi = mesh_max_y

    if y_hi <= y_lo:
        diagnostics["mode"] = "degenerate_final_band"
        return _ret((float(y_lo), float(y_hi), float(floor_y)))

    y_mid = float(
        np.clip(
            float(floor_y),
            float(y_lo) + HEIGHT_BAND_RAY_EPS_M,
            float(y_hi) - HEIGHT_BAND_RAY_EPS_M,
        )
    )
    diagnostics["mode"] = "visible_floor_plus_upward_rays"
    return _ret((float(y_lo), float(y_hi), y_mid))


def filter_gt_mesh_points_by_island(
    mesh_vertices: np.ndarray,
    pathfinder,
    island_index: int,
    start_y: float | None = None,
) -> np.ndarray:
    """Keep only points inside the inferred band that project to the start island."""
    if island_index < 0:
        return np.ones(len(mesh_vertices), dtype=bool)

    y_lo, y_hi, y_mid = _compute_island_filter_band(
        mesh_vertices,
        island_index,
        start_y=start_y,
    )
    height_mask = _mask_mesh_points_by_height_band(mesh_vertices, y_lo=y_lo, y_hi=y_hi)
    if not np.any(height_mask):
        return height_mask

    keep_mask = np.zeros(len(mesh_vertices), dtype=bool)
    band_points = np.asarray(mesh_vertices, dtype=np.float64)[height_mask]
    projection_y = (
        float(start_y)
        if start_y is not None and np.isfinite(float(start_y))
        else float(y_mid)
    )
    keep_mask[height_mask] = _classify_mesh_points_to_island(
        band_points,
        pathfinder,
        island_index,
        y_lo,
        y_hi,
        projection_y,
    )
    return keep_mask


def sample_mesh_and_filter(
    mesh,
    pathfinder,
    island_index: int,
    n_samples: int = 200000,
    start_y: float | None = None,
    filter_by_island: bool = True,
    height_band: tuple[float, float, float] | None = None,
    seed: int | None = None,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    """Sample mesh surface points then apply two independent filtering phases.

    Phase 1 — height band filter (always, no navmesh):
        Keep only points whose Y falls inside the inferred floor/ceiling band.
        Band is derived from either an explicit floor/ceiling raycast band or a
        simpler start_y-based band; no snap_point calls.

    Phase 2 — island membership filter (only when filter_by_island=True):
        Project surviving points onto the inferred local floor height, snap them
        onto the navmesh, and keep those that land on island_index.

    Assumes mesh is already in Habitat scene coordinates.

    Args:
        mesh: trimesh.Trimesh (or any object with .vertices and .faces/.triangles)
        pathfinder: habitat_sim pathfinder (used only in Phase 2)
        island_index: target navmesh island (-1 skips both band estimation and island filter)
        n_samples: target number of accepted surface samples
        start_y: agent floor height hint for band estimation
        filter_by_island: whether to run Phase 2
        height_band: explicit (y_lo, y_hi, y_mid) — skips band estimation entirely
        seed: deterministic seed for surface sampling; None keeps trimesh defaults

    Returns:
        (M, 3) array of accepted GT surface samples in Habitat scene coordinates.
        If return_diagnostics=True, also returns a diagnostics dict with ordered
        stage counts for the sampling/filtering pipeline.
    """
    import trimesh

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = getattr(mesh, "faces", None)
    if faces is None:
        faces = getattr(mesh, "triangles", None)
    faces = (
        np.empty((0, 3), dtype=np.int64)
        if faces is None
        else np.asarray(faces, dtype=np.int64)
    )

    def _stage(stage: str, count: int, **extra: Any) -> dict[str, Any]:
        entry: dict[str, Any] = {"stage": stage, "count": int(count)}
        for key, value in extra.items():
            if value is not None:
                entry[key] = value
        return entry

    diagnostics: dict[str, Any] = {
        "filter_mode": "unfiltered" if island_index < 0 and height_band is None and not filter_by_island else "filtered",
        "requested_points": int(n_samples),
        "mesh_vertex_count": int(len(vertices)),
        "mesh_face_count": int(len(faces)),
        "surface_sample_rounds": [],
        "sampled_surface_points_count": 0,
        "height_filtered_points_before_cap": 0,
        "height_filtered_points_count": 0,
        "island_filtered_points_count": 0,
        "final_points_count": 0,
        "height_band": None,
        "stage_counts": [
            _stage("mesh_vertices", int(len(vertices))),
            _stage("mesh_faces", int(len(faces))),
            _stage("requested_surface_samples", int(n_samples)),
        ],
    }

    def _return(points: np.ndarray) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        diagnostics["final_points_count"] = int(len(points))
        if return_diagnostics:
            return points, diagnostics
        return points

    if n_samples <= 0:
        diagnostics["filter_mode"] = "empty_request"
        diagnostics["stage_counts"].append(_stage("final_gt_points", 0))
        return _return(np.empty((0, 3), dtype=np.float64))

    if len(faces) == 0 or len(vertices) == 0:
        diagnostics["filter_mode"] = "empty_mesh"
        diagnostics["stage_counts"].append(_stage("final_gt_points", 0))
        return _return(np.empty((0, 3), dtype=np.float64))

    def _sample_surface(draw_count: int, *, stream_idx: int) -> np.ndarray:
        kwargs: dict[str, int] = {}
        if seed is not None:
            draw_seed = int(
                np.random.SeedSequence([int(seed), int(stream_idx)]).generate_state(
                    1,
                    dtype=np.uint32,
                )[0]
            )
            kwargs["seed"] = draw_seed
        pts, _ = trimesh.sample.sample_surface(mesh, int(draw_count), **kwargs)
        return np.asarray(pts, dtype=np.float64)

    # No filtering at all
    if island_index < 0 and height_band is None and not filter_by_island:
        pts = _sample_surface(int(n_samples), stream_idx=0)
        diagnostics["sampled_surface_points_count"] = int(len(pts))
        diagnostics["stage_counts"].extend(
            [
                _stage("sampled_surface_points", int(len(pts))),
                _stage("final_gt_points", int(len(pts))),
            ]
        )
        return _return(pts)

    # ── Phase 1: resolve height band (NO navmesh snapping) ───────────────────
    if height_band is not None:
        y_lo = float(height_band[0])
        y_hi = float(height_band[1])
        y_mid = float(height_band[2])
    elif island_index < 0:
        pts = _sample_surface(int(n_samples), stream_idx=0)
        diagnostics["sampled_surface_points_count"] = int(len(pts))
        diagnostics["stage_counts"].extend(
            [
                _stage("sampled_surface_points", int(len(pts))),
                _stage("final_gt_points", int(len(pts))),
            ]
        )
        return _return(pts)
    else:
        y_lo, y_hi, y_mid = _compute_island_filter_band(
            vertices,
            island_index,
            start_y=start_y,
        )
    diagnostics["height_band"] = {
        "y_lo": float(y_lo),
        "y_hi": float(y_hi),
        "y_mid": float(y_mid),
    }

    # Iterative rejection-sampling to accumulate n_samples height-accepted points
    accepted_chunks: list[np.ndarray] = []
    accepted_count = 0
    sampled_count = 0
    max_rounds = 8
    min_batch_size = 32768
    max_batch_size = max(min_batch_size, min(1_000_000, 4 * int(n_samples)))

    for round_idx in range(max_rounds):
        remaining = int(n_samples) - accepted_count
        if remaining <= 0:
            break

        if accepted_count == 0 or sampled_count == 0:
            draw_count = min(max_batch_size, max(min_batch_size, 2 * remaining))
        else:
            acceptance_ratio = accepted_count / max(sampled_count, 1)
            draw_count = int(np.ceil(remaining / max(acceptance_ratio, 1e-3) * 1.25))
            draw_count = min(max_batch_size, max(min_batch_size, draw_count))

        pts = _sample_surface(draw_count, stream_idx=round_idx)
        sampled_count += len(pts)

        height_mask = _mask_mesh_points_by_height_band(pts, y_lo=y_lo, y_hi=y_hi)
        accepted = pts[height_mask]
        diagnostics["surface_sample_rounds"].append(
            {
                "round_index": int(round_idx),
                "requested_draw_count": int(draw_count),
                "sampled_surface_points": int(len(pts)),
                "height_filtered_points": int(len(accepted)),
            }
        )
        if len(accepted) > 0:
            accepted_chunks.append(accepted)
            accepted_count += len(accepted)

    diagnostics["sampled_surface_points_count"] = int(sampled_count)
    diagnostics["height_filtered_points_before_cap"] = int(accepted_count)
    if not accepted_chunks:
        diagnostics["stage_counts"].extend(
            [
                _stage("sampled_surface_points", int(sampled_count)),
                _stage("height_filtered_points_before_cap", int(accepted_count)),
                _stage("height_filtered_points", 0),
                _stage("final_gt_points", 0),
            ]
        )
        return _return(np.empty((0, 3), dtype=np.float64))

    height_filtered = np.concatenate(accepted_chunks, axis=0)
    if len(height_filtered) > int(n_samples):
        height_filtered = height_filtered[:int(n_samples)]
    diagnostics["height_filtered_points_count"] = int(len(height_filtered))
    diagnostics["stage_counts"].extend(
        [
            _stage("sampled_surface_points", int(sampled_count)),
            _stage("height_filtered_points_before_cap", int(accepted_count)),
            _stage("height_filtered_points", int(len(height_filtered))),
        ]
    )

    # ── Phase 2: island membership filter (navmesh snap, only if requested) ──
    if not filter_by_island or island_index < 0:
        diagnostics["island_filtered_points_count"] = int(len(height_filtered))
        diagnostics["stage_counts"].extend(
            [
                _stage("island_filtered_points", int(len(height_filtered)), applied=False),
                _stage("final_gt_points", int(len(height_filtered))),
            ]
        )
        return _return(height_filtered)

    projection_y = (
        float(start_y)
        if start_y is not None and np.isfinite(float(start_y))
        else float(y_mid)
    )
    island_mask = _classify_mesh_points_to_island(
        height_filtered,
        pathfinder,
        island_index,
        y_lo,
        y_hi,
        projection_y,
    )
    island_filtered = height_filtered[island_mask]
    diagnostics["island_filtered_points_count"] = int(len(island_filtered))
    diagnostics["stage_counts"].extend(
        [
            _stage("island_filtered_points", int(len(island_filtered)), applied=True),
            _stage("final_gt_points", int(len(island_filtered))),
        ]
    )
    return _return(island_filtered)


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics with the same pixel-center convention as the env's `c2w` rays."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int | None = None
    height: int | None = None

    @classmethod
    def from_hfov(
        cls,
        width: int,
        height: int,
        hfov_deg: float,
    ) -> "CameraIntrinsics":
        hfov = np.deg2rad(float(hfov_deg))
        fx = (float(width) / 2.0) / np.tan(hfov / 2.0)
        vfov = 2.0 * np.arctan((float(height) / float(width)) * np.tan(hfov / 2.0))
        fy = (float(height) / 2.0) / np.tan(vfov / 2.0)
        cx = (float(width) - 1.0) / 2.0
        cy = (float(height) - 1.0) / 2.0
        return cls(
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
            width=int(width),
            height=int(height),
        )

    def as_vector(self) -> np.ndarray:
        return np.asarray([self.fx, self.fy, self.cx, self.cy], dtype=np.float64)

    def as_matrix(self) -> np.ndarray:
        return np.asarray(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class CompletenessFrame:
    """One frame of observations for completeness eval.

    `camera_to_world` must map OpenCV-style camera coordinates
    (`+x` right, `+y` down, `+z` forward) into Habitat scene coordinates.
    `modules/environment/env.py` already exposes such a matrix as `obs["c2w"]`.
    """

    depth: np.ndarray
    camera_to_world: np.ndarray
    intrinsics: CameraIntrinsics | np.ndarray | tuple[float, float, float, float]
    valid_mask: np.ndarray | None = None


@dataclass(frozen=True)
class CompletenessMetrics:
    mean_distance_m: float
    ratio_within_threshold: float
    threshold_m: float
    num_gt_points: int
    num_observed_points: int
    num_frames: int


@dataclass(frozen=True)
class EpisodeCompletenessResult:
    final_metrics: CompletenessMetrics
    per_step_metrics: list[CompletenessMetrics]
    min_distances_m: np.ndarray
    gt_points: np.ndarray


def _coerce_intrinsics(
    intrinsics: CameraIntrinsics | np.ndarray | Sequence[float] | Mapping[str, Any],
    *,
    depth_shape: tuple[int, int] | None = None,
) -> CameraIntrinsics:
    if isinstance(intrinsics, CameraIntrinsics):
        return intrinsics

    if isinstance(intrinsics, Mapping):
        keys = set(intrinsics)
        if {"fx", "fy", "cx", "cy"} <= keys:
            return CameraIntrinsics(
                fx=float(intrinsics["fx"]),
                fy=float(intrinsics["fy"]),
                cx=float(intrinsics["cx"]),
                cy=float(intrinsics["cy"]),
                width=int(intrinsics["width"]) if "width" in intrinsics and intrinsics["width"] is not None else None,
                height=int(intrinsics["height"]) if "height" in intrinsics and intrinsics["height"] is not None else None,
            )
        raise ValueError(
            "Expected intrinsics mapping to contain fx/fy/cx/cy keys, "
            f"got keys={sorted(str(k) for k in keys)}"
        )

    intr_arr = np.asarray(intrinsics, dtype=np.float64)
    if intr_arr.shape == (4,):
        width = int(depth_shape[1]) if depth_shape is not None else None
        height = int(depth_shape[0]) if depth_shape is not None else None
        return CameraIntrinsics(
            fx=float(intr_arr[0]),
            fy=float(intr_arr[1]),
            cx=float(intr_arr[2]),
            cy=float(intr_arr[3]),
            width=width,
            height=height,
        )
    if intr_arr.shape == (3, 3):
        width = int(depth_shape[1]) if depth_shape is not None else None
        height = int(depth_shape[0]) if depth_shape is not None else None
        return CameraIntrinsics(
            fx=float(intr_arr[0, 0]),
            fy=float(intr_arr[1, 1]),
            cx=float(intr_arr[0, 2]),
            cy=float(intr_arr[1, 2]),
            width=width,
            height=height,
        )

    raise ValueError(
        "Unsupported intrinsics format. Expected CameraIntrinsics, dict, "
        f"(fx, fy, cx, cy), or 3x3 K matrix, got shape={intr_arr.shape}"
    )


def _coerce_frame(frame: CompletenessFrame | Mapping[str, Any]) -> CompletenessFrame:
    if isinstance(frame, CompletenessFrame):
        return frame
    if not isinstance(frame, Mapping):
        raise TypeError(
            "Frames must be CompletenessFrame instances or mappings with "
            "depth/camera_to_world/intrinsics keys."
        )

    if "depth" not in frame:
        raise KeyError("Frame is missing required key 'depth'")

    camera_to_world = frame.get("camera_to_world")
    if camera_to_world is None and "c2w" in frame:
        camera_to_world = frame["c2w"]
    if camera_to_world is None:
        raise KeyError("Frame is missing required key 'camera_to_world' or alias 'c2w'")

    intrinsics = frame.get("intrinsics")
    if intrinsics is None and "fxfycxcy" in frame:
        intrinsics = frame["fxfycxcy"]
    if intrinsics is None and "K" in frame:
        intrinsics = frame["K"]
    if intrinsics is None:
        raise KeyError(
            "Frame is missing required key 'intrinsics' or aliases 'fxfycxcy' / 'K'"
        )

    valid_mask = frame.get("valid_mask")
    return CompletenessFrame(
        depth=np.asarray(frame["depth"]),
        camera_to_world=np.asarray(camera_to_world),
        intrinsics=intrinsics,
        valid_mask=np.asarray(valid_mask) if valid_mask is not None else None,
    )


def _extract_camera_position(camera_to_world: np.ndarray) -> np.ndarray:
    tf = np.asarray(camera_to_world, dtype=np.float64)
    if tf.shape != (4, 4):
        raise ValueError(f"Expected camera_to_world shape (4, 4), got {tf.shape}")
    return tf[:3, 3].copy()


def infer_episode_start_position(
    frames: Sequence[CompletenessFrame | Mapping[str, Any]],
    *,
    camera_height_m: float | None = None,
) -> np.ndarray:
    """Infer an approximate start position from the first frame.

    If `camera_height_m` is provided, returns an agent-floor estimate by subtracting the
    camera height from the first frame's camera translation along Habitat's +y axis.
    """
    if len(frames) == 0:
        raise ValueError("Cannot infer start position from an empty frame sequence")

    first_frame = _coerce_frame(frames[0])
    start_position = _extract_camera_position(first_frame.camera_to_world)
    if camera_height_m is not None:
        start_position[1] -= float(camera_height_m)
    return start_position


def infer_episode_island_index(
    frames: Sequence[CompletenessFrame | Mapping[str, Any]],
    pathfinder,
    *,
    camera_height_m: float | None = None,
    start_position: np.ndarray | Sequence[float] | None = None,
) -> tuple[int, np.ndarray]:
    """Infer the reachable navmesh island from the first frame pose.

    Explicit `island_index` from episode metadata is still preferable when available.
    Inference is provided as a convenience for baseline logs that only store sensor poses.
    """
    import magnum as mn

    candidates: list[np.ndarray] = []
    if start_position is not None:
        candidates.append(np.asarray(start_position, dtype=np.float64).reshape(3))
    else:
        camera_position = infer_episode_start_position(frames, camera_height_m=None)
        if camera_height_m is not None:
            lowered = camera_position.copy()
            lowered[1] -= float(camera_height_m)
            candidates.append(lowered)
        candidates.append(camera_position)

    best_island: int | None = None
    best_snapped: np.ndarray | None = None
    best_distance: float | None = None
    for candidate in candidates:
        snapped = pathfinder.snap_point(
            mn.Vector3(float(candidate[0]), float(candidate[1]), float(candidate[2]))
        )
        snapped_np = np.asarray(snapped, dtype=np.float64)
        if snapped_np.shape != (3,) or not np.all(np.isfinite(snapped_np)):
            continue
        island_index = int(pathfinder.get_island(snapped))
        snap_distance = float(np.linalg.norm(snapped_np - candidate))
        if best_distance is None or snap_distance < best_distance:
            best_island = island_index
            best_snapped = snapped_np
            best_distance = snap_distance

    if best_island is None or best_snapped is None:
        raise RuntimeError("Unable to infer episode island index from the provided frames")

    return best_island, best_snapped


def _iter_local_floor_raycast_origins(
    camera_position: np.ndarray,
    *,
    sample_radius_m: float,
) -> list[np.ndarray]:
    origins: list[np.ndarray] = []
    for ring_radius, ring_count in zip(
        _LOCAL_FLOOR_RAYCAST_RING_RADII_M,
        _LOCAL_FLOOR_RAYCAST_RING_COUNTS,
        strict=True,
    ):
        if ring_radius - float(sample_radius_m) > 1e-6:
            break
        if ring_radius <= 1e-9:
            origins.append(np.asarray(camera_position, dtype=np.float64).copy())
            continue

        angles = np.linspace(0.0, 2.0 * np.pi, num=int(ring_count), endpoint=False, dtype=np.float64)
        for angle in angles:
            origin = np.asarray(camera_position, dtype=np.float64).copy()
            origin[0] += float(ring_radius) * float(np.cos(angle))
            origin[2] += float(ring_radius) * float(np.sin(angle))
            origins.append(origin)
    return origins


def _snap_point_to_floor_island(
    pathfinder,
    point_xyz: np.ndarray,
    *,
    floor_y: float | None = None,
) -> tuple[int, np.ndarray] | None:
    import magnum as mn

    point_xyz = np.asarray(point_xyz, dtype=np.float64).reshape(3)
    if floor_y is not None:
        point_xyz = point_xyz.copy()
        point_xyz[1] = float(floor_y)
    snapped = pathfinder.snap_point(
        mn.Vector3(float(point_xyz[0]), float(point_xyz[1]), float(point_xyz[2]))
    )
    snapped_np = _vector3_to_numpy(snapped)
    if snapped_np.shape != (3,) or not np.all(np.isfinite(snapped_np)):
        return None
    return int(pathfinder.get_island(snapped)), snapped_np


def infer_island_from_points(
    points: np.ndarray,
    pathfinder,
    *,
    max_samples: int = 512,
    floor_y: float | None = None,
) -> int:
    """Infer the dominant navmesh island by majority vote over a sample of points.

    If ``floor_y`` is provided, each point is first projected to ``(x, floor_y, z)``
    before snapping. This keeps the vote rule consistent with the final GT island
    filter, which also snaps points from the inferred local floor height.

    Returns -1 if no valid snaps are found.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return -1

    # Deterministic subsample for speed
    if len(pts) > int(max_samples):
        idx = np.linspace(0, len(pts) - 1, num=int(max_samples), dtype=np.int64)
        pts = pts[idx]

    island_votes: dict[int, int] = {}
    for pt in pts:
        result = _snap_point_to_floor_island(pathfinder, pt, floor_y=floor_y)
        if result is None:
            continue
        island_idx, _ = result
        island_votes[island_idx] = island_votes.get(island_idx, 0) + 1

    if not island_votes:
        return -1

    return max(island_votes, key=lambda k: island_votes[k])


def infer_local_floor_island_from_raycast(
    sim,
    pathfinder,
    *,
    start_position: np.ndarray | Sequence[float],
    camera_height_m: float,
    sample_radius_m: float = LOCAL_FLOOR_RAYCAST_RADIUS_M,
    height_bin_m: float = LOCAL_FLOOR_RAYCAST_HEIGHT_BIN_M,
    downward_max_distance_m: float = LOCAL_FLOOR_RAYCAST_DOWN_MAX_DIST_M,
    visibility_epsilon_m: float = LOCAL_FLOOR_RAYCAST_VISIBILITY_EPS_M,
    floor_normal_y_min: float = LOCAL_FLOOR_RAYCAST_FLOOR_NORMAL_Y_MIN,
) -> tuple[int, float, dict[str, Any]]:
    """Infer floor height and target island from visible disk-raycast good samples."""
    import habitat_sim
    import magnum as mn

    start_position = np.asarray(start_position, dtype=np.float64).reshape(3)
    camera_position = start_position.copy()
    camera_position[1] += float(camera_height_m)
    origins = _iter_local_floor_raycast_origins(
        camera_position,
        sample_radius_m=float(sample_radius_m),
    )

    visible_candidate_count = 0
    accepted_candidate_count = 0
    rejected_blocked_count = 0
    rejected_downward_miss_count = 0
    rejected_non_floor_count = 0

    # height_bin -> list of visible downward first-hit points in (x, y, z)
    height_bin_hits: dict[int, list[np.ndarray]] = {}

    camera_origin = mn.Vector3(
        float(camera_position[0]),
        float(camera_position[1]),
        float(camera_position[2]),
    )
    for origin in origins:
        # ── horizontal visibility check (unchanged) ──────────────────────────
        horizontal_delta = np.asarray(origin - camera_position, dtype=np.float64)
        horizontal_delta[1] = 0.0
        horizontal_dist = float(np.linalg.norm(horizontal_delta))
        if horizontal_dist > float(visibility_epsilon_m):
            direction = horizontal_delta / horizontal_dist
            vis_ray = habitat_sim.geo.Ray(
                camera_origin,
                mn.Vector3(float(direction[0]), 0.0, float(direction[2])),
            )
            max_distance = max(0.0, horizontal_dist - float(visibility_epsilon_m))
            vis_hit = sim.cast_ray(vis_ray, max_distance=max_distance)
            if vis_hit.has_hits():
                nearest = min(float(hit.ray_distance) for hit in vis_hit.hits)
                if nearest < max_distance:
                    rejected_blocked_count += 1
                    continue

        visible_candidate_count += 1

        # ── downward ray ─────────────────────────────────────────────────────
        down_ray = habitat_sim.geo.Ray(
            mn.Vector3(float(origin[0]), float(origin[1]), float(origin[2])),
            mn.Vector3(0.0, -1.0, 0.0),
        )
        down_hit = sim.cast_ray(down_ray, max_distance=float(downward_max_distance_m))
        if not down_hit.has_hits():
            rejected_downward_miss_count += 1
            continue

        best_hit = min(down_hit.hits, key=lambda hit: float(hit.ray_distance))
        hit_point = _vector3_to_numpy(best_hit.point)
        if not np.all(np.isfinite(hit_point)):
            rejected_downward_miss_count += 1
            continue

        # ── floor-normal check ───────────────────────────────────────────────
        normal = getattr(best_hit, "normal", None)
        if normal is not None:
            normal_np = _vector3_to_numpy(normal)
            normal_norm = float(np.linalg.norm(normal_np))
            if normal_norm > 1e-12:
                if float(normal_np[1]) / normal_norm < float(floor_normal_y_min):
                    rejected_non_floor_count += 1
                    continue

        # ── pure height clustering — NO navmesh snap ─────────────────────────
        height_bin = int(np.round(float(hit_point[1]) / float(height_bin_m)))
        height_bin_hits.setdefault(height_bin, []).append(hit_point.astype(np.float64, copy=True))
        accepted_candidate_count += 1

    diagnostics: dict[str, Any] = {
        "mode": "local_floor_raycast",
        "candidate_count": len(origins),
        "visible_candidate_count": int(visible_candidate_count),
        "accepted_candidate_count": int(accepted_candidate_count),
        "rejected_blocked_count": int(rejected_blocked_count),
        "rejected_downward_miss_count": int(rejected_downward_miss_count),
        "rejected_non_floor_count": int(rejected_non_floor_count),
        "cluster_count": len(height_bin_hits),
    }

    if not height_bin_hits:
        raise RuntimeError("No visible downward floor hits found for local floor inference")

    cluster_summaries: list[dict[str, Any]] = []
    for hbin, entries in height_bin_hits.items():
        hits_arr = np.asarray(entries, dtype=np.float64).reshape(-1, 3)
        cluster_summaries.append({
            "height_bin": int(hbin),
            "support_count": int(len(hits_arr)),
            "height_median": float(np.median(hits_arr[:, 1])),
        })

    diagnostics["cluster_supports"] = [
        {k: c[k] for k in ("height_bin", "support_count", "height_median")}
        for c in sorted(cluster_summaries,
                        key=lambda c: (-c["support_count"], -c["height_median"]))[:8]
    ]
    winning = max(
        cluster_summaries,
        key=lambda c: (c["support_count"], c["height_median"], -c["height_bin"]),
    )
    floor_entries = height_bin_hits[int(winning["height_bin"])]
    good_samples = np.asarray(floor_entries, dtype=np.float64).reshape(-1, 3)
    floor_y = float(np.median(good_samples[:, 1]))

    diagnostics["winning_support_count"] = int(winning["support_count"])
    diagnostics["winning_height_bin"] = int(winning["height_bin"])
    diagnostics["floor_y"] = floor_y
    diagnostics["mode"] = "local_floor_good_samples"
    diagnostics["good_sample_count"] = int(len(good_samples))
    diagnostics["island_vote_input_points"] = int(len(good_samples))

    island_index = infer_island_from_points(
        good_samples,
        pathfinder,
        floor_y=floor_y,
    )
    if island_index < 0:
        raise RuntimeError("Unable to infer local floor island from good samples")

    diagnostics["island_vote_mode"] = "good_samples_majority"
    diagnostics["island_index"] = int(island_index)
    return int(island_index), floor_y, diagnostics


def depth_to_camera_points(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics | np.ndarray | Sequence[float] | Mapping[str, Any],
    *,
    valid_mask: np.ndarray | None = None,
    min_depth: float | None = None,
    max_depth: float | None = None,
    depth_scale: float = 1.0,
    stride: int = 1,
) -> np.ndarray:
    """Unproject a depth image into OpenCV-style camera coordinates.

    The output coordinate convention is `+x` right, `+y` down, `+z` forward.
    """
    depth_arr = np.asarray(depth, dtype=np.float64)
    if depth_arr.ndim == 3 and depth_arr.shape[-1] == 1:
        depth_arr = depth_arr[..., 0]
    if depth_arr.ndim != 2:
        raise ValueError(f"Expected depth image with shape (H, W), got {depth_arr.shape}")

    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    intr = _coerce_intrinsics(intrinsics, depth_shape=depth_arr.shape)
    depth_m = depth_arr * float(depth_scale)
    valid = np.isfinite(depth_m)

    if valid_mask is not None:
        valid_mask_arr = np.asarray(valid_mask, dtype=bool)
        if valid_mask_arr.shape != depth_arr.shape:
            raise ValueError(
                f"valid_mask shape must match depth shape {depth_arr.shape}, "
                f"got {valid_mask_arr.shape}"
            )
        valid &= valid_mask_arr

    if min_depth is not None:
        valid &= depth_m >= float(min_depth)
    if max_depth is not None:
        valid &= depth_m <= float(max_depth)

    rows = np.arange(0, depth_arr.shape[0], stride, dtype=np.float64)
    cols = np.arange(0, depth_arr.shape[1], stride, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(cols, rows, indexing="xy")

    depth_sub = depth_m[::stride, ::stride]
    valid_sub = valid[::stride, ::stride]
    if not np.any(valid_sub):
        return np.empty((0, 3), dtype=np.float32)

    z = depth_sub
    x = ((grid_x + 0.5) - intr.cx) * z / intr.fx
    y = ((grid_y + 0.5) - intr.cy) * z / intr.fy
    points = np.stack([x, y, z], axis=-1)
    return points[valid_sub].astype(np.float32, copy=False)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 row-vector transform to 3D points."""
    pts = np.asarray(points, dtype=np.float64)
    tf = np.asarray(transform, dtype=np.float64)
    if tf.shape != (4, 4):
        raise ValueError(f"Expected transform shape (4, 4), got {tf.shape}")
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected points shape (N, 3), got {pts.shape}")
    if len(pts) == 0:
        return np.empty((0, 3), dtype=np.float32)
    transformed = pts @ tf[:3, :3].T + tf[:3, 3]
    return transformed.astype(np.float32, copy=False)


def depth_to_world_points(
    depth: np.ndarray,
    camera_to_world: np.ndarray,
    intrinsics: CameraIntrinsics | np.ndarray | Sequence[float] | Mapping[str, Any],
    *,
    valid_mask: np.ndarray | None = None,
    min_depth: float | None = None,
    max_depth: float | None = None,
    depth_scale: float = 1.0,
    stride: int = 1,
) -> np.ndarray:
    """Unproject depth into Habitat scene coordinates.

    `camera_to_world` must map OpenCV-style camera coordinates into the Habitat scene frame.
    This is exactly what obs["c2w"] from the environment provides.
    """
    camera_points = depth_to_camera_points(
        depth,
        intrinsics,
        valid_mask=valid_mask,
        min_depth=min_depth,
        max_depth=max_depth,
        depth_scale=depth_scale,
        stride=stride,
    )
    return transform_points(camera_points, camera_to_world)


def _resolve_nn_backend(backend: str, *, torch_device: str | None = None) -> str:
    backend_name = str(backend).strip().lower()
    if backend_name not in _SUPPORTED_NN_BACKENDS:
        raise ValueError(
            f"Unsupported nn backend {backend!r}. "
            f"Expected one of {sorted(_SUPPORTED_NN_BACKENDS)}."
        )
    if backend_name != "auto":
        return backend_name

    if torch_device:
        try:
            import torch

            device = torch.device(torch_device)
            if device.type == "cuda" and torch.cuda.is_available():
                return "torch_cuda_exact"
        except Exception:
            pass

    return "scipy_ckdtree"


def _torch_cuda_exact_min_to_reference_tensors(
    query_t,
    query_sq,
    reference_t,
    reference_sq,
    *,
    query_chunk_size: int = DEFAULT_TORCH_NN_QUERY_CHUNK_SIZE,
    reference_chunk_size: int = DEFAULT_TORCH_NN_REFERENCE_CHUNK_SIZE,
    return_indices: bool = False,
):
    import torch

    if query_chunk_size < 1:
        raise ValueError(f"query_chunk_size must be >= 1, got {query_chunk_size}")
    if reference_chunk_size < 1:
        raise ValueError(
            f"reference_chunk_size must be >= 1, got {reference_chunk_size}"
        )
    if query_t.ndim != 2 or query_t.shape[1] != 3:
        raise ValueError(f"Expected query_t shape (N, 3), got {tuple(query_t.shape)}")
    if reference_t.ndim != 2 or reference_t.shape[1] != 3:
        raise ValueError(
            f"Expected reference_t shape (M, 3), got {tuple(reference_t.shape)}"
        )
    if query_t.device.type != "cuda" or reference_t.device.type != "cuda":
        raise ValueError("torch CUDA exact NN requires CUDA tensors")
    if query_t.device != reference_t.device:
        raise ValueError("query_t and reference_t must be on the same device")

    out = torch.empty((len(query_t),), dtype=torch.float32, device=query_t.device)
    out_idx = (
        torch.empty((len(query_t),), dtype=torch.int64, device=query_t.device)
        if return_indices
        else None
    )

    for q_start in range(0, len(query_t), int(query_chunk_size)):
        q_stop = min(len(query_t), q_start + int(query_chunk_size))
        q_chunk = query_t[q_start:q_stop]
        q_sq_chunk = query_sq[q_start:q_stop]
        best_d2 = torch.full(
            (q_stop - q_start,),
            float("inf"),
            dtype=torch.float32,
            device=query_t.device,
        )
        best_idx = (
            torch.full(
                (q_stop - q_start,),
                -1,
                dtype=torch.int64,
                device=query_t.device,
            )
            if return_indices
            else None
        )

        for r_start in range(0, len(reference_t), int(reference_chunk_size)):
            r_stop = min(len(reference_t), r_start + int(reference_chunk_size))
            r_chunk = reference_t[r_start:r_stop]
            d2 = (
                q_sq_chunk[:, None]
                + reference_sq[r_start:r_stop][None, :]
                - 2.0 * torch.matmul(q_chunk, r_chunk.T)
            )
            d2 = torch.clamp(d2, min=0.0)
            chunk_best_d2, chunk_best_idx = torch.min(d2, dim=1)
            better = chunk_best_d2 < best_d2
            best_d2 = torch.where(better, chunk_best_d2, best_d2)
            if return_indices:
                best_idx = torch.where(
                    better,
                    chunk_best_idx.to(torch.int64) + int(r_start),
                    best_idx,
                )

        out[q_start:q_stop] = torch.sqrt(best_d2)
        if return_indices:
            out_idx[q_start:q_stop] = best_idx

    return out, out_idx


def _torch_cuda_exact_distances_and_indices_to_reference(
    query_points: np.ndarray,
    reference_points: np.ndarray,
    *,
    torch_device: str,
    query_chunk_size: int = DEFAULT_TORCH_NN_QUERY_CHUNK_SIZE,
    reference_chunk_size: int = DEFAULT_TORCH_NN_REFERENCE_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    device = torch.device(torch_device)
    if device.type != "cuda":
        raise ValueError(
            f"torch_cuda_exact requires a CUDA device, got {torch_device!r}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("torch_cuda_exact requested, but CUDA is not available")

    query = np.asarray(query_points, dtype=np.float32)
    reference = np.asarray(reference_points, dtype=np.float32)
    if len(query) == 0:
        return (
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )

    finite_reference = reference[np.isfinite(reference).all(axis=1)]
    if len(finite_reference) == 0:
        return (
            np.full(len(query), np.inf, dtype=np.float64),
            np.full(len(query), -1, dtype=np.int64),
        )

    with torch.no_grad():
        query_t = torch.as_tensor(query, dtype=torch.float32, device=device)
        reference_t = torch.as_tensor(finite_reference, dtype=torch.float32, device=device)
        query_sq = torch.sum(query_t * query_t, dim=1)
        reference_sq = torch.sum(reference_t * reference_t, dim=1)
        out, out_idx = _torch_cuda_exact_min_to_reference_tensors(
            query_t,
            query_sq,
            reference_t,
            reference_sq,
            query_chunk_size=query_chunk_size,
            reference_chunk_size=reference_chunk_size,
            return_indices=True,
        )

        return (
            out.detach().cpu().numpy().astype(np.float64, copy=False),
            out_idx.detach().cpu().numpy().astype(np.int64, copy=False),
        )


def _nearest_neighbor_distances(
    query_points: np.ndarray,
    reference_points: np.ndarray,
    *,
    workers: int | None = None,
) -> np.ndarray:
    distances, _ = _nearest_neighbor_distances_and_indices(
        query_points,
        reference_points,
        workers=workers,
    )
    return distances


def _nearest_neighbor_distances_and_indices(
    query_points: np.ndarray,
    reference_points: np.ndarray,
    *,
    workers: int | None = None,
    backend: str = "scipy_ckdtree",
    torch_device: str | None = None,
    torch_query_chunk_size: int = DEFAULT_TORCH_NN_QUERY_CHUNK_SIZE,
    torch_reference_chunk_size: int = DEFAULT_TORCH_NN_REFERENCE_CHUNK_SIZE,
    return_backend: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, str]:
    query = np.asarray(query_points, dtype=np.float64)
    reference = np.asarray(reference_points, dtype=np.float64)
    resolved_backend = _resolve_nn_backend(backend, torch_device=torch_device)

    if query.ndim != 2 or query.shape[1] != 3:
        raise ValueError(f"Expected query_points shape (N, 3), got {query.shape}")
    if reference.ndim != 2 or reference.shape[1] != 3:
        raise ValueError(
            f"Expected reference_points shape (M, 3), got {reference.shape}"
        )
    if len(query) == 0:
        result = (
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )
        if return_backend:
            return result[0], result[1], "empty_query"
        return result

    reference = reference[np.isfinite(reference).all(axis=1)]
    if len(reference) == 0:
        result = (
            np.full(len(query), np.inf, dtype=np.float64),
            np.full(len(query), -1, dtype=np.int64),
        )
        if return_backend:
            return result[0], result[1], "empty_reference"
        return result

    if resolved_backend == "torch_cuda_exact":
        if torch_device is None:
            raise ValueError("torch_cuda_exact requires torch_device to be set")
        distances, indices = _torch_cuda_exact_distances_and_indices_to_reference(
            query,
            reference,
            torch_device=torch_device,
            query_chunk_size=torch_query_chunk_size,
            reference_chunk_size=torch_reference_chunk_size,
        )
        if return_backend:
            return distances, indices, resolved_backend
        return distances, indices

    if resolved_backend == "numpy_bruteforce":
        out = np.empty(len(query), dtype=np.float64)
        out_idx = np.empty(len(query), dtype=np.int64)
        chunk_size = 4096
        for start in range(0, len(query), chunk_size):
            stop = min(len(query), start + chunk_size)
            delta = query[start:stop, None, :] - reference[None, :, :]
            norms = np.linalg.norm(delta, axis=2)
            min_idx = np.argmin(norms, axis=1)
            out[start:stop] = norms[np.arange(len(min_idx)), min_idx]
            out_idx[start:stop] = min_idx
        if return_backend:
            return out, out_idx, "numpy_bruteforce"
        return out, out_idx

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(reference)
        query_kwargs: dict[str, Any] = {"k": 1}
        if workers is not None:
            query_kwargs["workers"] = int(workers)
        try:
            distances, indices = tree.query(query, **query_kwargs)
        except TypeError:
            query_kwargs.pop("workers", None)
            distances, indices = tree.query(query, **query_kwargs)
        result = (
            np.asarray(distances, dtype=np.float64),
            np.asarray(indices, dtype=np.int64),
        )
        if return_backend:
            return result[0], result[1], "scipy_ckdtree"
        return result
    except Exception:
        out = np.empty(len(query), dtype=np.float64)
        out_idx = np.empty(len(query), dtype=np.int64)
        chunk_size = 4096
        for start in range(0, len(query), chunk_size):
            stop = min(len(query), start + chunk_size)
            delta = query[start:stop, None, :] - reference[None, :, :]
            norms = np.linalg.norm(delta, axis=2)
            min_idx = np.argmin(norms, axis=1)
            out[start:stop] = norms[np.arange(len(min_idx)), min_idx]
            out_idx[start:stop] = min_idx
        if return_backend:
            return out, out_idx, "numpy_bruteforce"
        return out, out_idx


def gt_to_observed_distances(
    gt_points: np.ndarray,
    observed_points: np.ndarray,
    *,
    workers: int | None = None,
    backend: str = "scipy_ckdtree",
    torch_device: str | None = None,
    torch_query_chunk_size: int = DEFAULT_TORCH_NN_QUERY_CHUNK_SIZE,
    torch_reference_chunk_size: int = DEFAULT_TORCH_NN_REFERENCE_CHUNK_SIZE,
) -> np.ndarray:
    """Backward completeness distances: GT samples -> nearest observed point."""
    gt = np.asarray(gt_points, dtype=np.float64)
    if gt.ndim != 2 or gt.shape[1] != 3:
        raise ValueError(f"Expected gt_points shape (N, 3), got {gt.shape}")
    distances, _ = _nearest_neighbor_distances_and_indices(
        gt,
        observed_points,
        workers=workers,
        backend=backend,
        torch_device=torch_device,
        torch_query_chunk_size=torch_query_chunk_size,
        torch_reference_chunk_size=torch_reference_chunk_size,
    )
    return distances


def nearest_observed_point(
    query_point: np.ndarray,
    observed_points: np.ndarray,
    *,
    workers: int | None = None,
    backend: str = "scipy_ckdtree",
    torch_device: str | None = None,
    torch_query_chunk_size: int = DEFAULT_TORCH_NN_QUERY_CHUNK_SIZE,
    torch_reference_chunk_size: int = DEFAULT_TORCH_NN_REFERENCE_CHUNK_SIZE,
) -> tuple[np.ndarray | None, float, int | None]:
    """Return the nearest observed point to a single query point using the metric's NN path."""
    query = np.asarray(query_point, dtype=np.float64).reshape(1, 3)
    observed = np.asarray(observed_points, dtype=np.float64)
    if observed.ndim != 2 or observed.shape[1] != 3:
        raise ValueError(
            f"Expected observed_points shape (N, 3), got {observed.shape}"
        )
    if len(observed) == 0:
        return None, float("inf"), None

    finite_mask = np.isfinite(observed).all(axis=1)
    finite_observed = observed[finite_mask]
    if len(finite_observed) == 0:
        return None, float("inf"), None

    distances, indices = _nearest_neighbor_distances_and_indices(
        query,
        finite_observed,
        workers=workers,
        backend=backend,
        torch_device=torch_device,
        torch_query_chunk_size=torch_query_chunk_size,
        torch_reference_chunk_size=torch_reference_chunk_size,
    )
    nearest_idx = int(indices[0])
    if nearest_idx < 0:
        return None, float("inf"), None
    return (
        finite_observed[nearest_idx].astype(np.float32, copy=False),
        float(distances[0]),
        nearest_idx,
    )


def summarize_completeness(
    min_distances: np.ndarray,
    *,
    threshold_m: float = 0.05,
    num_observed_points: int = 0,
    num_frames: int = 0,
) -> CompletenessMetrics:
    """Aggregate per-GT-point minimum distances into completeness metrics."""
    dists = np.asarray(min_distances, dtype=np.float64).reshape(-1)
    num_gt_points = int(dists.shape[0])
    if num_gt_points == 0:
        mean_distance = float("nan")
        ratio = float("nan")
    else:
        mean_distance = float(np.mean(dists))
        ratio = float(np.mean(dists < float(threshold_m)))

    return CompletenessMetrics(
        mean_distance_m=mean_distance,
        ratio_within_threshold=ratio,
        threshold_m=float(threshold_m),
        num_gt_points=num_gt_points,
        num_observed_points=int(num_observed_points),
        num_frames=int(num_frames),
    )


class RunningCompletenessEvaluator:
    """Streaming completeness evaluator using per-GT-point running minima."""

    def __init__(
        self,
        gt_points: np.ndarray,
        *,
        threshold_m: float = 0.05,
        nn_workers: int | None = None,
        nn_backend: str = "scipy_ckdtree",
        torch_device: str | None = None,
        torch_query_chunk_size: int = DEFAULT_TORCH_NN_QUERY_CHUNK_SIZE,
        torch_reference_chunk_size: int = DEFAULT_TORCH_NN_REFERENCE_CHUNK_SIZE,
        track_history: bool = False,
    ) -> None:
        self.gt_points = np.asarray(gt_points, dtype=np.float64)
        if self.gt_points.ndim != 2 or self.gt_points.shape[1] != 3:
            raise ValueError(f"Expected gt_points shape (N, 3), got {self.gt_points.shape}")
        self.threshold_m = float(threshold_m)
        self.nn_workers = nn_workers
        self.nn_backend = _resolve_nn_backend(nn_backend, torch_device=torch_device)
        self.torch_device = torch_device
        self.torch_query_chunk_size = int(torch_query_chunk_size)
        self.torch_reference_chunk_size = int(torch_reference_chunk_size)
        if self.torch_query_chunk_size < 1:
            raise ValueError(
                f"torch_query_chunk_size must be >= 1, got {self.torch_query_chunk_size}"
            )
        if self.torch_reference_chunk_size < 1:
            raise ValueError(
                "torch_reference_chunk_size must be >= 1, "
                f"got {self.torch_reference_chunk_size}"
            )
        if self.nn_backend == "torch_cuda_exact" and self.torch_device is None:
            raise ValueError("torch_cuda_exact requires torch_device to be set")
        self._min_distances_m_np = np.full(len(self.gt_points), np.inf, dtype=np.float64)
        self._torch_gt_points = None
        self._torch_gt_points_sq = None
        self._torch_min_distances_m = None
        self._min_distances_sync_needed = False
        if self.nn_backend == "torch_cuda_exact":
            import torch

            device = torch.device(self.torch_device)
            self._torch_gt_points = torch.as_tensor(
                self.gt_points,
                dtype=torch.float32,
                device=device,
            )
            self._torch_gt_points_sq = torch.sum(
                self._torch_gt_points * self._torch_gt_points,
                dim=1,
            )
            self._torch_min_distances_m = torch.full(
                (len(self.gt_points),),
                float("inf"),
                dtype=torch.float32,
                device=device,
            )
        self.num_frames = 0
        self.num_observed_points = 0
        self.last_update_debug: dict[str, float | int | str] = {}
        self.track_history = bool(track_history)
        self._history_mean_values: list[Any] = []
        self._history_ratio_values: list[Any] = []
        self._history_num_frames: list[int] = []
        self._history_num_observed_points: list[int] = []

    @property
    def min_distances_m(self) -> np.ndarray:
        if self._torch_min_distances_m is None:
            return self._min_distances_m_np

        if self._min_distances_sync_needed:
            self._min_distances_m_np = (
                self._torch_min_distances_m.detach().cpu().numpy().astype(np.float64, copy=False)
            )
            self._min_distances_sync_needed = False
        return self._min_distances_m_np

    def sample_min_distances(self, indices: np.ndarray | list[int]) -> np.ndarray:
        sample_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if self._torch_min_distances_m is not None:
            import torch

            with torch.no_grad():
                sample_idx_t = torch.as_tensor(
                    sample_indices,
                    dtype=torch.long,
                    device=self._torch_min_distances_m.device,
                )
                return (
                    self._torch_min_distances_m.index_select(0, sample_idx_t)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
        return self._min_distances_m_np[sample_indices].astype(np.float32, copy=False)

    def update_observed_points(
        self,
        observed_points: np.ndarray,
        *,
        compute_metrics: bool = True,
    ) -> CompletenessMetrics | None:
        observed_dtype = np.float32 if self.nn_backend == "torch_cuda_exact" else np.float64
        observed = np.asarray(observed_points, dtype=observed_dtype)
        if observed.ndim != 2 or observed.shape[1] != 3:
            raise ValueError(
                f"Expected observed_points shape (N, 3), got {observed.shape}"
            )

        nn_t0 = time.perf_counter()
        nn_backend = self.nn_backend
        frame_distances = None
        if self.nn_backend == "torch_cuda_exact":
            import torch

            finite_observed = observed[np.isfinite(observed).all(axis=1)]
            if len(finite_observed) > 0:
                with torch.no_grad():
                    reference_t = torch.as_tensor(
                        finite_observed,
                        dtype=torch.float32,
                        device=self._torch_gt_points.device,
                    )
                    reference_sq = torch.sum(reference_t * reference_t, dim=1)
                    frame_distances_t, _ = _torch_cuda_exact_min_to_reference_tensors(
                        self._torch_gt_points,
                        self._torch_gt_points_sq,
                        reference_t,
                        reference_sq,
                        query_chunk_size=self.torch_query_chunk_size,
                        reference_chunk_size=self.torch_reference_chunk_size,
                        return_indices=False,
                    )
            else:
                nn_backend = "empty_reference"
                frame_distances_t = None
        else:
            frame_distances, _, nn_backend = _nearest_neighbor_distances_and_indices(
                self.gt_points,
                observed,
                workers=self.nn_workers,
                backend=self.nn_backend,
                torch_device=self.torch_device,
                torch_query_chunk_size=self.torch_query_chunk_size,
                torch_reference_chunk_size=self.torch_reference_chunk_size,
                return_backend=True,
            )
        nn_query_s = time.perf_counter() - nn_t0

        running_min_t0 = time.perf_counter()
        if self.nn_backend == "torch_cuda_exact":
            if frame_distances_t is not None:
                import torch

                torch.minimum(
                    self._torch_min_distances_m,
                    frame_distances_t,
                    out=self._torch_min_distances_m,
                )
                self._min_distances_sync_needed = True
        else:
            self._min_distances_m_np = np.minimum(self._min_distances_m_np, frame_distances)
        self.num_frames += 1
        self.num_observed_points += int(len(observed))
        running_min_update_s = time.perf_counter() - running_min_t0

        summary_t0 = time.perf_counter()
        metrics = None
        if self.track_history:
            if self._torch_min_distances_m is not None:
                import torch

                with torch.no_grad():
                    self._history_mean_values.append(self._torch_min_distances_m.mean().detach())
                    self._history_ratio_values.append(
                        (self._torch_min_distances_m < float(self.threshold_m)).float().mean().detach()
                    )
            else:
                history_metrics = summarize_completeness(
                    self._min_distances_m_np,
                    threshold_m=self.threshold_m,
                    num_observed_points=self.num_observed_points,
                    num_frames=self.num_frames,
                )
                self._history_mean_values.append(float(history_metrics.mean_distance_m))
                self._history_ratio_values.append(float(history_metrics.ratio_within_threshold))
            self._history_num_frames.append(int(self.num_frames))
            self._history_num_observed_points.append(int(self.num_observed_points))
        if compute_metrics:
            metrics = self.summary()
        summary_s = time.perf_counter() - summary_t0
        self.last_update_debug = {
            "nn_backend": str(nn_backend),
            "gt_points": int(len(self.gt_points)),
            "observed_points": int(len(observed)),
            "nn_workers": int(self.nn_workers) if self.nn_workers is not None else "default",
            "torch_device": str(self.torch_device) if self.torch_device is not None else "",
            "torch_query_chunk_size": int(self.torch_query_chunk_size),
            "torch_reference_chunk_size": int(self.torch_reference_chunk_size),
            "nn_query_s": float(nn_query_s),
            "running_min_update_s": float(running_min_update_s),
            "summary_s": float(summary_s),
        }
        return metrics

    def update_frame(
        self,
        frame: CompletenessFrame | Mapping[str, Any],
        *,
        min_depth: float | None = None,
        max_depth: float | None = None,
        depth_scale: float = 1.0,
        stride: int = 1,
        compute_metrics: bool = True,
    ) -> CompletenessMetrics | None:
        fr = _coerce_frame(frame)
        depth_t0 = time.perf_counter()
        observed_points = depth_to_world_points(
            fr.depth,
            fr.camera_to_world,
            fr.intrinsics,
            valid_mask=fr.valid_mask,
            min_depth=min_depth,
            max_depth=max_depth,
            depth_scale=depth_scale,
            stride=stride,
        )
        depth_to_world_s = time.perf_counter() - depth_t0
        metrics = self.update_observed_points(
            observed_points,
            compute_metrics=compute_metrics,
        )
        self.last_update_debug = {
            **self.last_update_debug,
            "depth_to_world_s": float(depth_to_world_s),
        }
        return metrics

    def materialize_history(self) -> list[CompletenessMetrics]:
        if not self.track_history:
            return []

        if self._torch_min_distances_m is not None:
            import torch

            if not self._history_mean_values:
                return []
            with torch.no_grad():
                mean_values = (
                    torch.stack(self._history_mean_values)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=False)
                )
                ratio_values = (
                    torch.stack(self._history_ratio_values)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64, copy=False)
                )
        else:
            mean_values = np.asarray(self._history_mean_values, dtype=np.float64)
            ratio_values = np.asarray(self._history_ratio_values, dtype=np.float64)

        history: list[CompletenessMetrics] = []
        for idx, (mean_value, ratio_value) in enumerate(zip(mean_values, ratio_values, strict=True)):
            history.append(
                CompletenessMetrics(
                    mean_distance_m=float(mean_value),
                    ratio_within_threshold=float(ratio_value),
                    threshold_m=float(self.threshold_m),
                    num_gt_points=int(len(self.gt_points)),
                    num_observed_points=int(self._history_num_observed_points[idx]),
                    num_frames=int(self._history_num_frames[idx]),
                )
            )
        return history

    def summary(self) -> CompletenessMetrics:
        if self._torch_min_distances_m is not None:
            import torch

            with torch.no_grad():
                dists = self._torch_min_distances_m
                num_gt_points = int(dists.numel())
                if num_gt_points == 0:
                    mean_distance = float("nan")
                    ratio = float("nan")
                else:
                    mean_distance = float(dists.mean().item())
                    ratio = float((dists < float(self.threshold_m)).float().mean().item())
            return CompletenessMetrics(
                mean_distance_m=mean_distance,
                ratio_within_threshold=ratio,
                threshold_m=float(self.threshold_m),
                num_gt_points=num_gt_points,
                num_observed_points=int(self.num_observed_points),
                num_frames=int(self.num_frames),
            )

        return summarize_completeness(
            self._min_distances_m_np,
            threshold_m=self.threshold_m,
            num_observed_points=self.num_observed_points,
            num_frames=self.num_frames,
        )


def load_mesh_for_completeness(mesh_or_path):
    """Load a triangle mesh from a path, or pass through an existing mesh object."""
    if not isinstance(mesh_or_path, (str, os.PathLike)):
        return mesh_or_path

    import trimesh

    loaded = trimesh.load(os.fspath(mesh_or_path), force="scene", process=False)
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.dump(concatenate=True)
    else:
        mesh = loaded

    faces = getattr(mesh, "faces", None)
    if faces is None:
        faces = getattr(mesh, "triangles", None)
    if faces is None or len(faces) == 0:
        raise ValueError(
            f"Loaded mesh has no triangle faces: mesh_or_path={mesh_or_path!r}"
        )
    return mesh


def prepare_reachable_gt_points(
    mesh,
    pathfinder,
    island_index: int,
    *,
    sim=None,
    start_y: float | None = None,
    n_samples: int = 200000,
    return_metadata: bool = False,
):
    """Convert a raw mesh to Habitat scene coordinates and sample reachable GT surface points.
    
    IMPORTANT: Always pass ``sim`` to ensure the mesh transform matches Habitat's observations.
    Without ``sim``, the fixed GLB->scene transform is used which may not match.
    """
    mesh = load_mesh_for_completeness(mesh)
    mesh_scene, metadata = mesh_to_habitat_scene(
        mesh,
        sim=sim,
        pathfinder=pathfinder,
        island_index=island_index,
        return_metadata=True,
    )
    gt_points = sample_mesh_and_filter(
        mesh_scene,
        pathfinder,
        island_index,
        n_samples=n_samples,
        start_y=start_y,
    )
    if return_metadata:
        return gt_points, mesh_scene, metadata
    return gt_points

