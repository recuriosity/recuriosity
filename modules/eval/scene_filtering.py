from __future__ import annotations

"""Barebones scene filtering up to chosen-island extraction."""

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

MAP_SCALE_M = 0.05
BOUNDS_PAD_M = 1.0
SWEEP_HEIGHT_OFFSETS_M = (-0.10, 0.0, 0.10)
GAP_REPAIR_M = 0.15
HEIGHT_PROBE_SPACING_M = 0.50
HEIGHT_MODE_BIN_M = 0.05
FLOOR_HEIGHT_BAND_KEEP_TOLERANCE_M = 0.10
CEILING_HEIGHT_BAND_KEEP_TOLERANCE_M = 0.08
ISLAND_PROJECTION_TOLERANCE_M = 0.15


@dataclass
class SceneFilterState:
    min_x: float
    min_z: float
    map_scale: float
    camera_height_m: float
    slice_reference_y: float
    raw_scene_mask: np.ndarray
    closed_boundary_mask: np.ndarray
    scene_mask: np.ndarray
    island_mask: np.ndarray
    component_labels: np.ndarray
    component_label: int
    start_row: int
    start_col: int

    def grid_to_world(
        self,
        row_indices: np.ndarray | Sequence[int],
        col_indices: np.ndarray | Sequence[int],
    ) -> np.ndarray:
        rows = np.asarray(row_indices, dtype=np.float64)
        cols = np.asarray(col_indices, dtype=np.float64)
        x = float(self.min_x) + (cols + 0.5) * float(self.map_scale)
        z = float(self.min_z) + (rows + 0.5) * float(self.map_scale)
        return np.stack([x, z], axis=1)


def _vector3_to_numpy(vec3) -> np.ndarray:
    try:
        return np.asarray([float(vec3[0]), float(vec3[1]), float(vec3[2])], dtype=np.float64)
    except Exception:
        return np.asarray([float(vec3.x), float(vec3.y), float(vec3.z)], dtype=np.float64)


def _disk(radius_cells: int) -> np.ndarray:
    radius = max(1, int(radius_cells))
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def _resize_nearest(frame: np.ndarray, image_size: int) -> np.ndarray:
    if frame.shape[0] == int(image_size) and frame.shape[1] == int(image_size):
        return frame.astype(np.uint8, copy=False)
    rows = np.linspace(0, frame.shape[0] - 1, num=int(image_size), dtype=np.int64)
    cols = np.linspace(0, frame.shape[1] - 1, num=int(image_size), dtype=np.int64)
    return frame[rows[:, None], cols[None, :]].astype(np.uint8, copy=False)


def _draw_line(mask: np.ndarray, row0: int, col0: int, row1: int, col1: int) -> None:
    steps = int(max(abs(int(row1) - int(row0)), abs(int(col1) - int(col0)))) + 1
    rows = np.rint(np.linspace(int(row0), int(row1), num=steps, dtype=np.float64)).astype(np.int32)
    cols = np.rint(np.linspace(int(col0), int(col1), num=steps, dtype=np.float64)).astype(np.int32)
    rows = np.clip(rows, 0, mask.shape[0] - 1)
    cols = np.clip(cols, 0, mask.shape[1] - 1)
    mask[rows, cols] = True


def _build_outer_hull_mask(mask: np.ndarray, thickness: int) -> np.ndarray:
    from scipy import ndimage
    from scipy.spatial import ConvexHull, QhullError

    hull_mask = np.zeros_like(mask, dtype=bool)
    hull_mask[[0, -1], :] = True
    hull_mask[:, [0, -1]] = True

    points = np.column_stack(np.nonzero(mask))
    if len(points) < 3:
        return hull_mask

    coords = np.column_stack([points[:, 1].astype(np.float64), points[:, 0].astype(np.float64)])
    try:
        hull = ConvexHull(coords)
    except QhullError:
        return hull_mask

    vertices = coords[hull.vertices]
    for idx in range(len(vertices)):
        col0, row0 = vertices[idx]
        col1, row1 = vertices[(idx + 1) % len(vertices)]
        _draw_line(hull_mask, int(round(row0)), int(round(col0)), int(round(row1)), int(round(col1)))

    return ndimage.binary_dilation(hull_mask, structure=_disk(max(1, int(thickness))))


def _uniform_subsample(
    points: np.ndarray,
    target_count: int,
    *,
    seed: int | None = None,
) -> np.ndarray:
    points = np.asarray(points)
    if target_count <= 0 or len(points) == 0:
        return points[:0]
    if len(points) <= int(target_count):
        return points
    rng = np.random.default_rng(None if seed is None else int(seed))
    keep = np.sort(rng.choice(len(points), size=int(target_count), replace=False))
    return points[keep]


def _sample_island_cells(scene_filter: SceneFilterState, spacing_m: float) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.nonzero(scene_filter.island_mask)
    if len(rows) == 0:
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)
    spacing_cells = max(1, int(round(float(spacing_m) / float(scene_filter.map_scale))))
    bins = np.stack([rows // spacing_cells, cols // spacing_cells], axis=1)
    _, keep = np.unique(bins, axis=0, return_index=True)
    keep = np.sort(keep)
    sampled_rows = np.asarray([scene_filter.start_row, *rows[keep]], dtype=np.int32)
    sampled_cols = np.asarray([scene_filter.start_col, *cols[keep]], dtype=np.int32)
    sampled_rows, sampled_cols = np.unique(np.stack([sampled_rows, sampled_cols], axis=1), axis=0).T
    return sampled_rows, sampled_cols


def _binned_mode(values: np.ndarray, bin_size_m: float) -> float:
    if len(values) == 0:
        raise RuntimeError("Cannot compute mode of empty sample set")
    bins = np.rint(np.asarray(values, dtype=np.float64) / float(bin_size_m)).astype(np.int64)
    unique_bins, counts = np.unique(bins, return_counts=True)
    best_bin = int(unique_bins[int(np.argmax(counts))])
    return float(best_bin) * float(bin_size_m)


def _estimate_height_band_from_island(
    sim,
    scene_filter: SceneFilterState,
    *,
    probe_spacing_m: float = HEIGHT_PROBE_SPACING_M,
    mode_bin_m: float = HEIGHT_MODE_BIN_M,
    floor_keep_tolerance_m: float = FLOOR_HEIGHT_BAND_KEEP_TOLERANCE_M,
    ceiling_keep_tolerance_m: float = CEILING_HEIGHT_BAND_KEEP_TOLERANCE_M,
) -> dict[str, Any]:
    import habitat_sim
    import magnum as mn

    probe_rows, probe_cols = _sample_island_cells(scene_filter, float(probe_spacing_m))
    if len(probe_rows) == 0:
        raise RuntimeError("Chosen island has no probe cells")

    down_hits_y: list[float] = []
    up_hits_y: list[float] = []
    for x, z in scene_filter.grid_to_world(probe_rows, probe_cols):
        origin = mn.Vector3(float(x), float(scene_filter.slice_reference_y), float(z))
        down_hits = sim.cast_ray(
            habitat_sim.geo.Ray(origin, mn.Vector3(0.0, -1.0, 0.0)),
            max_distance=8.0,
        )
        if down_hits.has_hits():
            down_hits_y.append(float(min(down_hits.hits, key=lambda hit: float(hit.ray_distance)).point[1]))

        up_hits = sim.cast_ray(
            habitat_sim.geo.Ray(origin, mn.Vector3(0.0, 1.0, 0.0)),
            max_distance=8.0,
        )
        if up_hits.has_hits():
            up_hits_y.append(float(min(up_hits.hits, key=lambda hit: float(hit.ray_distance)).point[1]))

    if not down_hits_y or not up_hits_y:
        raise RuntimeError("Failed to estimate floor/ceiling height band from island probes")

    floor_y = _binned_mode(np.asarray(down_hits_y, dtype=np.float64), float(mode_bin_m))
    ceiling_y = _binned_mode(np.asarray(up_hits_y, dtype=np.float64), float(mode_bin_m))
    if ceiling_y <= floor_y:
        raise RuntimeError("Estimated ceiling is not above estimated floor")

    return {
        "probe_count": int(len(probe_rows)),
        "probe_spacing_m": float(probe_spacing_m),
        "height_mode_bin_m": float(mode_bin_m),
        "floor_keep_tolerance_m": float(floor_keep_tolerance_m),
        "ceiling_keep_tolerance_m": float(ceiling_keep_tolerance_m),
        "island_projection_tolerance_m": float(ISLAND_PROJECTION_TOLERANCE_M),
        "floor_mode_y": float(floor_y),
        "ceiling_mode_y": float(ceiling_y),
        "down_hit_count": int(len(down_hits_y)),
        "up_hit_count": int(len(up_hits_y)),
    }


def _filter_points_by_height_band(
    points: np.ndarray,
    *,
    floor_y: float,
    ceiling_y: float,
    floor_tolerance_m: float,
    ceiling_tolerance_m: float,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return points
    keep = (
        (points[:, 1] >= float(floor_y) - float(floor_tolerance_m))
        & (points[:, 1] <= float(ceiling_y) + float(ceiling_tolerance_m))
    )
    return points[keep]


def _filter_points_by_island_projection(
    points: np.ndarray,
    scene_filter: SceneFilterState,
    *,
    tolerance_m: float = ISLAND_PROJECTION_TOLERANCE_M,
) -> np.ndarray:
    from scipy import ndimage

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return points

    tolerance_cells = max(
        0,
        int(math.ceil(float(tolerance_m) / max(float(scene_filter.map_scale), 1e-9))),
    )
    expanded_island_mask = (
        ndimage.binary_dilation(scene_filter.island_mask, structure=_disk(tolerance_cells))
        if tolerance_cells > 0
        else scene_filter.island_mask
    )

    col_coords = (points[:, 0] - float(scene_filter.min_x)) / float(scene_filter.map_scale)
    row_coords = (points[:, 2] - float(scene_filter.min_z)) / float(scene_filter.map_scale)
    cell_cols = np.floor(col_coords).astype(np.int64)
    cell_rows = np.floor(row_coords).astype(np.int64)

    in_bounds = (
        (cell_rows >= 0)
        & (cell_rows < expanded_island_mask.shape[0])
        & (cell_cols >= 0)
        & (cell_cols < expanded_island_mask.shape[1])
    )
    keep = np.zeros(len(points), dtype=bool)
    keep[in_bounds] = expanded_island_mask[cell_rows[in_bounds], cell_cols[in_bounds]]
    return points[keep]


def _build_raw_scene_mask_from_sweeps(
    sim,
    pathfinder,
    *,
    start_y: float,
    camera_height_m: float,
    map_scale_m: float,
    bounds_pad_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    import habitat_sim
    import magnum as mn

    bounds = pathfinder.get_bounds()
    min_x = float(bounds[0][0]) - float(bounds_pad_m)
    max_x = float(bounds[1][0]) + float(bounds_pad_m)
    min_z = float(bounds[0][2]) - float(bounds_pad_m)
    max_z = float(bounds[1][2]) + float(bounds_pad_m)

    width = int(math.ceil((max_x - min_x) / float(map_scale_m))) + 1
    height = int(math.ceil((max_z - min_z) / float(map_scale_m))) + 1
    raw_mask = np.zeros((height, width), dtype=bool)

    def mark_hits(hits) -> None:
        if not hits.has_hits():
            return
        for hit in hits.hits:
            point = _vector3_to_numpy(hit.point)
            col = int((float(point[0]) - float(min_x)) / float(map_scale_m))
            row = int((float(point[2]) - float(min_z)) / float(map_scale_m))
            if 0 <= row < height and 0 <= col < width:
                raw_mask[row, col] = True

    x_centers = min_x + (np.arange(width, dtype=np.float64) + 0.5) * float(map_scale_m)
    z_centers = min_z + (np.arange(height, dtype=np.float64) + 0.5) * float(map_scale_m)
    sample_ys = [float(start_y) + float(camera_height_m) + offset for offset in SWEEP_HEIGHT_OFFSETS_M]

    x_origin = min_x - 1.0
    z_origin = min_z - 1.0
    max_x_dist = (max_x - min_x) + 2.0
    max_z_dist = (max_z - min_z) + 2.0

    for y in sample_ys:
        for z in z_centers:
            hits = sim.cast_ray(
                habitat_sim.geo.Ray(mn.Vector3(float(x_origin), float(y), float(z)), mn.Vector3(1.0, 0.0, 0.0)),
                max_distance=float(max_x_dist),
            )
            mark_hits(hits)
        for x in x_centers:
            hits = sim.cast_ray(
                habitat_sim.geo.Ray(mn.Vector3(float(x), float(y), float(z_origin)), mn.Vector3(0.0, 0.0, 1.0)),
                max_distance=float(max_z_dist),
            )
            mark_hits(hits)

    return raw_mask, {
        "min_x": float(min_x),
        "max_x": float(max_x),
        "min_z": float(min_z),
        "max_z": float(max_z),
        "map_scale_m": float(map_scale_m),
        "slice_y_samples": [float(y) for y in sample_ys],
        "raw_scene_cell_count": int(np.count_nonzero(raw_mask)),
    }


def extract_scene_island_from_binary_map(
    scene_mask: np.ndarray,
    *,
    start_position: np.ndarray | Sequence[float],
    min_x: float,
    min_z: float,
    map_scale_m: float,
    slice_reference_y: float,
    agent_radius_m: float,
    return_diagnostics: bool = False,
) -> SceneFilterState | tuple[SceneFilterState, dict[str, Any]]:
    from scipy import ndimage

    mask = np.asarray(scene_mask, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        raise RuntimeError("Scene sweep mask is empty or invalid")

    start = np.asarray(start_position, dtype=np.float64).reshape(3)
    start_col = int(np.clip(np.floor((float(start[0]) - float(min_x)) / float(map_scale_m)), 0, mask.shape[1] - 1))
    start_row = int(np.clip(np.floor((float(start[2]) - float(min_z)) / float(map_scale_m)), 0, mask.shape[0] - 1))

    close_radius = max(1, int(math.ceil((float(agent_radius_m) + float(GAP_REPAIR_M)) / float(map_scale_m))))
    closed = ndimage.binary_closing(mask, structure=_disk(close_radius))
    hull = _build_outer_hull_mask(closed, max(1, close_radius // 2))
    closed = closed | hull
    free_space = ~closed

    labels, _ = ndimage.label(free_space)
    scene_labels = labels
    scene_mask_final = scene_labels > 0

    component_label = int(scene_labels[start_row, start_col])
    used_nearest_component_fallback = False
    fallback_distance_cells = 0.0
    if component_label <= 0:
        component_rows, component_cols = np.nonzero(scene_labels > 0)
        if len(component_rows) == 0:
            raise RuntimeError(
                "Start position does not fall inside any convex-hull-bounded free-space component. "
                "The scene sweep or rasterized start cell is invalid."
            )
        row_offsets = component_rows.astype(np.float64) - float(start_row)
        col_offsets = component_cols.astype(np.float64) - float(start_col)
        nearest_idx = int(np.argmin(row_offsets * row_offsets + col_offsets * col_offsets))
        component_label = int(scene_labels[component_rows[nearest_idx], component_cols[nearest_idx]])
        used_nearest_component_fallback = True
        fallback_distance_cells = float(
            math.hypot(float(row_offsets[nearest_idx]), float(col_offsets[nearest_idx]))
        )

    island_mask = scene_labels == int(component_label)
    component_count = int(len(np.unique(scene_labels[scene_labels > 0])))
    state = SceneFilterState(
        min_x=float(min_x),
        min_z=float(min_z),
        map_scale=float(map_scale_m),
        camera_height_m=float(slice_reference_y) - float(start[1]),
        slice_reference_y=float(slice_reference_y),
        raw_scene_mask=mask,
        closed_boundary_mask=closed,
        scene_mask=scene_mask_final,
        island_mask=island_mask,
        component_labels=scene_labels.astype(np.int32),
        component_label=int(component_label),
        start_row=int(start_row),
        start_col=int(start_col),
    )
    diagnostics = {
        "mode": "scene_sweep_convex_hull_component",
        "component_count": int(component_count),
        "component_label": int(component_label),
        "closing_radius_cells": int(close_radius),
        "hull_cell_count": int(np.count_nonzero(hull)),
        "raw_scene_cell_count": int(np.count_nonzero(mask)),
        "closed_boundary_cell_count": int(np.count_nonzero(closed)),
        "navigable_scene_cell_count": int(np.count_nonzero(scene_mask_final)),
        "island_cell_count": int(np.count_nonzero(island_mask)),
        "used_nearest_component_fallback": bool(used_nearest_component_fallback),
        "fallback_distance_cells": float(fallback_distance_cells),
        "fallback_distance_m": float(fallback_distance_cells * float(map_scale_m)),
    }
    return (state, diagnostics) if return_diagnostics else state


def infer_scene_filter_from_sweep(
    sim,
    pathfinder,
    *,
    start_position: np.ndarray | Sequence[float],
    camera_height_m: float,
    agent_radius_m: float,
    map_scale_m: float = MAP_SCALE_M,
    bounds_pad_m: float = BOUNDS_PAD_M,
) -> tuple[SceneFilterState, dict[str, Any]]:
    start = np.asarray(start_position, dtype=np.float64).reshape(3)
    raw_mask, sweep = _build_raw_scene_mask_from_sweeps(
        sim,
        pathfinder,
        start_y=float(start[1]),
        camera_height_m=float(camera_height_m),
        map_scale_m=float(map_scale_m),
        bounds_pad_m=float(bounds_pad_m),
    )
    state, island = extract_scene_island_from_binary_map(
        raw_mask,
        start_position=start,
        min_x=float(sweep["min_x"]),
        min_z=float(sweep["min_z"]),
        map_scale_m=float(map_scale_m),
        slice_reference_y=float(start[1]) + float(camera_height_m),
        agent_radius_m=float(agent_radius_m),
        return_diagnostics=True,
    )
    return state, {"mode": "scene_sweep_island_filter", **sweep, **island}


def sample_mesh_and_filter_with_scene_island(
    mesh,
    scene_filter: SceneFilterState | None,
    *,
    sim=None,
    n_samples: int = 200000,
    filter_by_island: bool = True,
    seed: int | None = None,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    import trimesh

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = getattr(mesh, "faces", getattr(mesh, "triangles", None))
    faces = np.empty((0, 3), dtype=np.int64) if faces is None else np.asarray(faces, dtype=np.int64)

    diagnostics = {
        "filter_mode": "unfiltered" if scene_filter is None else "scene_height_band_and_island_projection",
        "requested_points": int(n_samples),
        "mesh_vertex_count": int(len(vertices)),
        "mesh_face_count": int(len(faces)),
        "scene_component_label": None if scene_filter is None else int(scene_filter.component_label),
        "height_band": None,
        "sampled_surface_points_count": 0,
        "height_band_points_before_cap": 0,
        "projected_island_points_before_cap": 0,
        "final_points_count": 0,
        "stage_counts": [],
    }

    def finish(points: np.ndarray):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        diagnostics["final_points_count"] = int(len(points))
        diagnostics["stage_counts"].append({"stage": "final_gt_points", "count": int(len(points))})
        return (points, diagnostics) if return_diagnostics else points

    def sample_surface(draw_count: int, stream_idx: int) -> np.ndarray:
        kwargs: dict[str, int] = {}
        if seed is not None:
            kwargs["seed"] = int(
                np.random.SeedSequence([int(seed), int(stream_idx)]).generate_state(1, dtype=np.uint32)[0]
            )
        points, _ = trimesh.sample.sample_surface(mesh, int(draw_count), **kwargs)
        return np.asarray(points, dtype=np.float64)

    if n_samples <= 0:
        return finish(np.empty((0, 3), dtype=np.float64))

    if scene_filter is None and not filter_by_island:
        if len(vertices) == 0 or len(faces) == 0:
            return finish(np.empty((0, 3), dtype=np.float64))
        points = sample_surface(int(n_samples), 0)
        diagnostics["sampled_surface_points_count"] = int(len(points))
        return finish(points[: int(n_samples)])

    if scene_filter is None or sim is None:
        raise ValueError("scene filtering requires scene_filter and sim")

    height_band = _estimate_height_band_from_island(sim, scene_filter)
    diagnostics["height_band"] = height_band

    accepted_chunks: list[np.ndarray] = []
    sampled_total = 0
    in_band_total = 0
    projected_total = 0
    kept_total = 0
    for round_idx in range(8):
        remaining = int(n_samples) - int(kept_total)
        if remaining <= 0:
            break
        draw_count = min(1_000_000, max(32768, 4 * remaining))
        sampled = sample_surface(draw_count, round_idx)
        sampled_total += int(len(sampled))
        if len(sampled) == 0:
            continue
        in_band = _filter_points_by_height_band(
            sampled,
            floor_y=float(height_band["floor_mode_y"]),
            ceiling_y=float(height_band["ceiling_mode_y"]),
            floor_tolerance_m=float(height_band["floor_keep_tolerance_m"]),
            ceiling_tolerance_m=float(height_band["ceiling_keep_tolerance_m"]),
        )
        in_band_total += int(len(in_band))
        projected = _filter_points_by_island_projection(
            in_band,
            scene_filter,
            tolerance_m=float(height_band["island_projection_tolerance_m"]),
        )
        projected_total += int(len(projected))
        kept_total += int(len(projected))
        if len(projected) > 0:
            accepted_chunks.append(np.asarray(projected, dtype=np.float64))

    diagnostics["sampled_surface_points_count"] = int(sampled_total)
    diagnostics["height_band_points_before_cap"] = int(in_band_total)
    diagnostics["projected_island_points_before_cap"] = int(projected_total)
    if not accepted_chunks:
        return finish(np.empty((0, 3), dtype=np.float64))

    filtered = np.concatenate(accepted_chunks, axis=0)
    filtered = np.asarray(_uniform_subsample(filtered, int(n_samples), seed=seed), dtype=np.float64)
    return finish(filtered)


def render_scene_filter_debug_images(
    scene_filter: SceneFilterState | None,
    *,
    image_size: int = 512,
) -> dict[str, np.ndarray]:
    from scipy import ndimage

    if scene_filter is None:
        return {}

    def paint_cell(frame: np.ndarray, row: int, col: int, color: np.ndarray) -> None:
        frame[max(0, row - 1) : min(frame.shape[0], row + 2), max(0, col - 1) : min(frame.shape[1], col + 2)] = color

    def mark_points(frame: np.ndarray) -> np.ndarray:
        frame = frame.copy()
        start_row = int(np.clip(scene_filter.start_row, 0, frame.shape[0] - 1))
        start_col = int(np.clip(scene_filter.start_col, 0, frame.shape[1] - 1))
        paint_cell(frame, start_row, start_col, np.asarray([255, 48, 48], dtype=np.uint8))
        return frame

    sweep = np.zeros((*scene_filter.raw_scene_mask.shape, 3), dtype=np.uint8)
    sweep[scene_filter.raw_scene_mask] = np.asarray([230, 230, 230], dtype=np.uint8)

    closed = np.zeros((*scene_filter.closed_boundary_mask.shape, 3), dtype=np.uint8)
    closed[scene_filter.closed_boundary_mask] = np.asarray([230, 230, 230], dtype=np.uint8)
    closed[scene_filter.scene_mask] = np.asarray([70, 110, 170], dtype=np.uint8)
    island_outline = scene_filter.island_mask & ~ndimage.binary_erosion(scene_filter.island_mask, structure=np.ones((3, 3), dtype=np.uint8))
    closed[island_outline] = np.asarray([90, 220, 120], dtype=np.uint8)

    components = np.zeros((*scene_filter.component_labels.shape, 3), dtype=np.uint8)
    labels = [int(label) for label in np.unique(scene_filter.component_labels) if int(label) > 0]
    chosen_color = np.asarray([90, 220, 120], dtype=np.uint8)
    other_color = np.asarray([95, 95, 95], dtype=np.uint8)
    for label in labels:
        color = chosen_color if int(label) == int(scene_filter.component_label) else other_color
        components[scene_filter.component_labels == int(label)] = color

    return {
        "sweep_map": _resize_nearest(mark_points(sweep), int(image_size)),
        "closed_map": _resize_nearest(mark_points(closed), int(image_size)),
        "components_map": _resize_nearest(mark_points(components), int(image_size)),
    }


__all__ = [
    "SceneFilterState",
    "extract_scene_island_from_binary_map",
    "infer_scene_filter_from_sweep",
    "render_scene_filter_debug_images",
    "sample_mesh_and_filter_with_scene_island",
]
