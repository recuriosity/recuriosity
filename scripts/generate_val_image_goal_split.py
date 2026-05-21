#!/usr/bin/env python3
"""Generate HM3D val episodes with fixed image-goal target camera poses.

This mirrors `generate_val_apples_split.py` behavior:
- Read the input episodes payload as-is.
- Keep one output episode per input episode (same scene/start metadata).
- Augment each episode with `goal_position` and `goal_rotation` sampled through
  `HabitatMP3DEnv` image-goal reset logic so constraints match environment rules
  (scene sweep island, boundary margin, geometry clearance, reachability).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPT_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from modules.environment.env_image_goal import HabitatMP3DEnv
from modules.eval.eval_utils import derive_navmesh_path, load_episode_specs, resolve_hm3d_stage_path
from scripts.dl_hm3d_data import resolve_hm3d_root


def _warn(message: str) -> None:
    print(f"[generate_val_image_goal_split] warning: {message}", file=sys.stderr, flush=True)


def _load_json(path: str) -> dict[str, Any]:
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _validate_episode_fields(episodes: Sequence[dict[str, Any]]) -> None:
    required = (
        "episode_id",
        "scene_id",
        "scene_name",
        "start_position",
        "start_rotation",
    )
    for idx, episode in enumerate(episodes):
        for key in required:
            if key not in episode:
                raise ValueError(f"episodes[{idx}] missing required key '{key}'")


def _to_builtin_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin_json(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_to_builtin_json(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _extract_vector(state: dict[str, Any], key: str, expected_size: int) -> np.ndarray:
    if key not in state:
        raise RuntimeError(f"reset state missing '{key}'")
    vec = np.asarray(state[key], dtype=np.float32).reshape(-1)
    if vec.size != int(expected_size):
        raise RuntimeError(
            f"reset state '{key}' has size={vec.size}, expected {expected_size}"
        )
    if not np.all(np.isfinite(vec)):
        raise RuntimeError(f"reset state '{key}' contains non-finite values")
    return vec.astype(np.float32, copy=False)


def _depth_min_for_clearance(
    depth: np.ndarray,
    valid: np.ndarray,
    center_frac: float,
) -> tuple[float, float]:
    """Return (central_min_depth, full_frame_min_depth) over valid depth pixels."""
    min_all = float(depth[valid].min())
    if center_frac >= 0.999:
        return min_all, min_all
    h, w = depth.shape
    frac = float(np.clip(center_frac, 0.05, 1.0))
    margin_h = int(round(h * (1.0 - frac) * 0.5))
    margin_w = int(round(w * (1.0 - frac) * 0.5))
    h0, h1 = margin_h, h - margin_h
    w0, w1 = margin_w, w - margin_w
    if h1 <= h0 or w1 <= w0:
        return min_all, min_all
    depth_roi = depth[h0:h1, w0:w1]
    valid_roi = valid[h0:h1, w0:w1]
    if not np.any(valid_roi):
        return min_all, min_all
    return float(depth_roi[valid_roi].min()), min_all


def _compute_goal_view_quality_metrics(
    *,
    goal_rgb: np.ndarray,
    goal_depth: np.ndarray,
    depth_center_frac: float,
) -> dict[str, float]:
    depth_map = np.asarray(goal_depth, dtype=np.float32).squeeze()
    if depth_map.ndim != 2:
        raise RuntimeError(f"goal_depth must be HxW; got shape={depth_map.shape}")
    valid_depth = np.isfinite(depth_map) & (depth_map > 0.0)
    valid_depth_frac = float(np.count_nonzero(valid_depth) / max(int(depth_map.size), 1))
    if np.any(valid_depth):
        depth_min_clear, depth_min_full = _depth_min_for_clearance(
            depth_map,
            valid_depth,
            float(depth_center_frac),
        )
    else:
        depth_min_clear, depth_min_full = float("nan"), float("nan")

    rgb = np.asarray(goal_rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise RuntimeError(f"goal_rgb must be HxWx3; got shape={rgb.shape}")
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0
    gray = np.clip(gray, 0.0, 1.0)
    gray_flat = gray.reshape(-1)
    gray_p95_p5 = float(np.percentile(gray_flat, 95.0) - np.percentile(gray_flat, 5.0))
    if gray.shape[0] >= 2 and gray.shape[1] >= 2:
        dx = np.abs(gray[:, 1:] - gray[:, :-1])
        dy = np.abs(gray[1:, :] - gray[:-1, :])
        texture_score = float(0.5 * (float(dx.mean()) + float(dy.mean())))
    else:
        texture_score = 0.0

    return {
        "goal_view_valid_depth_fraction": float(valid_depth_frac),
        "goal_view_depth_min_clearance_m": float(depth_min_clear),
        "goal_view_depth_min_full_m": float(depth_min_full),
        "goal_view_gray_p95_p5": float(gray_p95_p5),
        "goal_view_texture_score": float(texture_score),
    }


def _goal_view_quality_fail_reasons(
    *,
    metrics: dict[str, float],
    goal_view_min_clear_depth_m: float,
    goal_view_min_valid_depth_frac: float,
    goal_view_min_gray_p95_p5: float,
    goal_view_min_texture_score: float,
) -> list[str]:
    reasons: list[str] = []

    valid_depth_frac = float(metrics.get("goal_view_valid_depth_fraction", float("nan")))
    if goal_view_min_valid_depth_frac > 0.0:
        if (not np.isfinite(valid_depth_frac)) or (
            valid_depth_frac < float(goal_view_min_valid_depth_frac)
        ):
            reasons.append(
                "valid_depth_fraction="
                f"{valid_depth_frac:.3f} < {float(goal_view_min_valid_depth_frac):.3f}"
            )

    depth_min_clear = float(metrics.get("goal_view_depth_min_clearance_m", float("nan")))
    if goal_view_min_clear_depth_m > 0.0:
        if (not np.isfinite(depth_min_clear)) or (depth_min_clear < float(goal_view_min_clear_depth_m)):
            reasons.append(
                "min_clear_depth_m="
                f"{depth_min_clear:.3f} < {float(goal_view_min_clear_depth_m):.3f}"
            )

    gray_p95_p5 = float(metrics.get("goal_view_gray_p95_p5", float("nan")))
    if goal_view_min_gray_p95_p5 > 0.0:
        if (not np.isfinite(gray_p95_p5)) or (gray_p95_p5 < float(goal_view_min_gray_p95_p5)):
            reasons.append(
                "gray_p95_p5="
                f"{gray_p95_p5:.3f} < {float(goal_view_min_gray_p95_p5):.3f}"
            )

    texture_score = float(metrics.get("goal_view_texture_score", float("nan")))
    if goal_view_min_texture_score > 0.0:
        if (not np.isfinite(texture_score)) or (
            texture_score < float(goal_view_min_texture_score)
        ):
            reasons.append(
                "texture_score="
                f"{texture_score:.4f} < {float(goal_view_min_texture_score):.4f}"
            )

    return reasons


def _per_episode_seed(base_seed: int, episode_id: str, episode_idx: int) -> int:
    mix = int(base_seed) * 1_000_003 + int(episode_idx) * 97 + sum(ord(c) for c in episode_id)
    return int(mix % (2**31 - 1))


def _build_scene_entry(normalized_episode: dict[str, Any]) -> tuple[dict[str, str], tuple[str, str | None]]:
    scene_path = os.path.abspath(str(normalized_episode["scene_id"]))
    navmesh_path = derive_navmesh_path(scene_path)
    stage_glb_path = resolve_hm3d_stage_path(scene_path, navmesh_path)
    scene = {
        "scene_name": str(normalized_episode["scene_name"]),
        "glb_path": stage_glb_path,
        "navmesh": navmesh_path,
    }
    return scene, (stage_glb_path, navmesh_path)


def _sample_goal_for_episode(
    *,
    env: HabitatMP3DEnv,
    scene: dict[str, str],
    start_position: np.ndarray,
    start_rotation: np.ndarray,
    episode_seed: int,
    episode_sample_attempts: int,
    goal_sample_min_start_distance_m: float,
    goal_sample_max_start_distance_m: float | None,
    goal_view_min_clear_depth_m: float,
    goal_view_min_valid_depth_frac: float,
    goal_view_depth_center_frac: float,
    goal_view_min_gray_p95_p5: float,
    goal_view_min_texture_score: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], int]:
    last_error: Exception | None = None

    for attempt_idx in range(int(episode_sample_attempts)):
        reset_seed = int(episode_seed + attempt_idx)
        random.seed(reset_seed)
        np.random.seed(reset_seed)

        try:
            obs, info = env.reset(
                seed=reset_seed,
                options={
                    "restore_state": {
                        "scene": scene,
                        "position": start_position.astype(np.float32, copy=False),
                        "rotation": start_rotation.astype(np.float32, copy=False),
                    }
                },
            )
        except Exception as exc:
            last_error = exc
            continue

        state = env.get_scene_and_agent_state()
        goal_position = _extract_vector(state, "goal_position", 3)
        goal_rotation = _extract_vector(state, "goal_rotation", 4)

        distance_m = float(
            np.linalg.norm(goal_position.astype(np.float64) - start_position.astype(np.float64))
        )
        min_distance_ok = distance_m >= float(goal_sample_min_start_distance_m)
        max_distance_ok = True
        if goal_sample_max_start_distance_m is not None:
            max_distance_ok = distance_m <= float(goal_sample_max_start_distance_m)

        goal_rgb = obs.get("goal_rgb")
        goal_depth = obs.get("goal_depth")
        if goal_rgb is None or goal_depth is None:
            last_error = RuntimeError("reset observation missing goal_rgb/goal_depth")
            continue
        view_metrics = _compute_goal_view_quality_metrics(
            goal_rgb=np.asarray(goal_rgb),
            goal_depth=np.asarray(goal_depth),
            depth_center_frac=float(goal_view_depth_center_frac),
        )
        view_quality_failures = _goal_view_quality_fail_reasons(
            metrics=view_metrics,
            goal_view_min_clear_depth_m=float(goal_view_min_clear_depth_m),
            goal_view_min_valid_depth_frac=float(goal_view_min_valid_depth_frac),
            goal_view_min_gray_p95_p5=float(goal_view_min_gray_p95_p5),
            goal_view_min_texture_score=float(goal_view_min_texture_score),
        )

        if min_distance_ok and max_distance_ok and not view_quality_failures:
            info_with_distance = dict(info)
            info_with_distance["image_goal_start_goal_distance_m"] = float(distance_m)
            info_with_distance.update(view_metrics)
            return goal_position, goal_rotation, info_with_distance, reset_seed

        max_distance_label = (
            f"{float(goal_sample_max_start_distance_m):.3f}"
            if goal_sample_max_start_distance_m is not None
            else "inf"
        )
        if not (min_distance_ok and max_distance_ok):
            last_error = RuntimeError(
                "Sampled image-goal distance outside requested range: "
                f"{distance_m:.3f}m (min={float(goal_sample_min_start_distance_m):.3f}, "
                f"max={max_distance_label})"
            )
        else:
            last_error = RuntimeError(
                "Sampled image-goal view failed quality checks: "
                + "; ".join(view_quality_failures)
            )

    if last_error is not None:
        raise RuntimeError(
            f"Failed to sample image-goal target satisfying distance/view-quality checks after "
            f"{episode_sample_attempts} attempts"
        ) from last_error
    raise RuntimeError(
        f"Failed to sample image-goal target satisfying distance/view-quality checks after "
        f"{episode_sample_attempts} attempts"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate HM3D val image-goal split by augmenting each input episode with goal pose"
    )
    parser.add_argument(
        "--episodes-json",
        type=str,
        default=os.path.join(_ROOT, "data", "splits", "hm3d", "val", "val.json.gz"),
        help="Input episodes JSON/JSON.GZ (typically data/splits/hm3d/val/val.json.gz)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=os.path.join(_ROOT, "data", "splits", "hm3d", "val", "val_image_goal.json"),
        help="Output JSON/JSON.GZ path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global seed")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of episodes to process",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--check-holes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable depth/raycast hole mismatch checks in safety validation",
    )
    parser.add_argument(
        "--episode-sample-attempts",
        type=int,
        default=8,
        help="How many reset seeds to try per episode before failing",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1024,
        help="Environment max_steps used while generating",
    )
    parser.add_argument("--goal-sample-min-start-distance-m", type=float, default=1.0)
    parser.add_argument(
        "--goal-sample-max-start-distance-m",
        type=float,
        default=None,
        help="Optional max start->goal distance in meters; when omitted no max is enforced",
    )
    parser.add_argument("--goal-sample-max-tries", type=int, default=192)
    parser.add_argument("--goal-boundary-margin-m", type=float, default=0.60)
    parser.add_argument("--goal-geometry-clearance-m", type=float, default=0.35)
    parser.add_argument("--goal-max-height-delta-m", type=float, default=0.03)
    parser.add_argument("--goal-reachability-projection-tolerance-m", type=float, default=0.15)
    parser.add_argument(
        "--goal-view-min-clear-depth-m",
        type=float,
        default=1.0,
        help=(
            "Minimum valid central depth in goal view (meters); "
            "<=0 disables this check"
        ),
    )
    parser.add_argument(
        "--goal-view-min-valid-depth-frac",
        type=float,
        default=0.30,
        help=(
            "Minimum valid-pixel fraction in goal depth map; "
            "<=0 disables this check"
        ),
    )
    parser.add_argument(
        "--goal-view-depth-center-frac",
        type=float,
        default=0.70,
        help="Center crop fraction used for goal depth clearance checks",
    )
    parser.add_argument(
        "--goal-view-min-gray-p95-p5",
        type=float,
        default=0.12,
        help=(
            "Minimum grayscale percentile spread (p95-p5, [0,1]) in goal RGB; "
            "<=0 disables this check"
        ),
    )
    parser.add_argument(
        "--goal-view-min-texture-score",
        type=float,
        default=0.015,
        help=(
            "Minimum average abs-gradient texture score in goal RGB; "
            "<=0 disables this check"
        ),
    )
    parser.add_argument(
        "--goal-require-navmesh-path",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    if int(args.episode_sample_attempts) <= 0:
        raise ValueError("--episode-sample-attempts must be > 0")
    if float(args.goal_sample_min_start_distance_m) < 0.0:
        raise ValueError("--goal-sample-min-start-distance-m must be >= 0")
    if args.goal_sample_max_start_distance_m is not None:
        if float(args.goal_sample_max_start_distance_m) <= 0.0:
            raise ValueError("--goal-sample-max-start-distance-m must be > 0 when provided")
        if float(args.goal_sample_max_start_distance_m) < float(args.goal_sample_min_start_distance_m):
            raise ValueError(
                "--goal-sample-max-start-distance-m must be >= --goal-sample-min-start-distance-m"
            )
    if float(args.goal_view_min_clear_depth_m) < 0.0:
        raise ValueError("--goal-view-min-clear-depth-m must be >= 0")
    if (
        float(args.goal_view_min_valid_depth_frac) < 0.0
        or float(args.goal_view_min_valid_depth_frac) > 1.0
    ):
        raise ValueError("--goal-view-min-valid-depth-frac must be in [0, 1]")
    if (
        float(args.goal_view_depth_center_frac) <= 0.0
        or float(args.goal_view_depth_center_frac) > 1.0
    ):
        raise ValueError("--goal-view-depth-center-frac must be in (0, 1]")
    if float(args.goal_view_min_gray_p95_p5) < 0.0:
        raise ValueError("--goal-view-min-gray-p95-p5 must be >= 0")
    if float(args.goal_view_min_texture_score) < 0.0:
        raise ValueError("--goal-view-min-texture-score must be >= 0")

    resolve_hm3d_root()

    normalized_episodes = load_episode_specs(args.episodes_json)
    normalized_episode_count = len(normalized_episodes)
    _validate_episode_fields(normalized_episodes)
    if not normalized_episodes:
        raise RuntimeError(f"No episodes loaded from {args.episodes_json}")

    input_payload = _load_json(args.episodes_json)
    input_episodes = input_payload.get("episodes")
    if not isinstance(input_episodes, list) or len(input_episodes) != int(normalized_episode_count):
        _warn(
            "Input JSON payload episode count differs from normalized list; "
            "output will use normalized episodes"
        )
        input_episodes = normalized_episodes

    if args.limit is not None:
        episode_limit = max(0, int(args.limit))
        normalized_episodes = normalized_episodes[:episode_limit]
        input_episodes = input_episodes[:episode_limit]

    if not normalized_episodes:
        raise RuntimeError("No episodes found after applying --limit")

    output_episodes: list[dict[str, Any]] = []
    goal_sample_attempt_counts: list[int] = []
    goal_start_distance_values_m: list[float] = []
    goal_view_depth_min_clearance_values_m: list[float] = []
    goal_view_gray_p95_p5_values: list[float] = []
    goal_view_texture_score_values: list[float] = []

    active_scene_key: tuple[str, str | None] | None = None
    active_env: HabitatMP3DEnv | None = None

    try:
        for episode_idx, episode in enumerate(normalized_episodes):
            raw_episode = dict(input_episodes[episode_idx])
            episode_id = str(episode["episode_id"])

            scene, scene_key = _build_scene_entry(episode)
            if active_scene_key != scene_key:
                if active_env is not None:
                    active_env.close()
                    active_env = None

                active_env = HabitatMP3DEnv(
                    [scene],
                    max_steps=max(1, int(args.max_steps)),
                    render_mode="rgb_array",
                    gpu_id=int(args.gpu_id),
                    check_holes=bool(args.check_holes),
                    goal_sample_min_start_distance_m=float(args.goal_sample_min_start_distance_m),
                    goal_sample_max_tries=int(args.goal_sample_max_tries),
                    goal_boundary_margin_m=float(args.goal_boundary_margin_m),
                    goal_geometry_clearance_m=float(args.goal_geometry_clearance_m),
                    goal_max_height_delta_m=float(args.goal_max_height_delta_m),
                    goal_reachability_projection_tolerance_m=float(
                        args.goal_reachability_projection_tolerance_m
                    ),
                    goal_require_navmesh_path=bool(args.goal_require_navmesh_path),
                    goal_scene_retry_limit=1,
                )
                active_scene_key = scene_key

            if active_env is None:
                raise RuntimeError("Internal error: active_env was not initialized")

            start_position = np.asarray(episode["start_position"], dtype=np.float32).reshape(3)
            start_rotation = np.asarray(episode["start_rotation"], dtype=np.float32).reshape(4)
            episode_seed = _per_episode_seed(int(args.seed), episode_id, episode_idx)
            (
                goal_position,
                goal_rotation,
                info,
                used_reset_seed,
            ) = _sample_goal_for_episode(
                env=active_env,
                scene=scene,
                start_position=start_position,
                start_rotation=start_rotation,
                episode_seed=episode_seed,
                episode_sample_attempts=int(args.episode_sample_attempts),
                goal_sample_min_start_distance_m=float(args.goal_sample_min_start_distance_m),
                goal_sample_max_start_distance_m=(
                    float(args.goal_sample_max_start_distance_m)
                    if args.goal_sample_max_start_distance_m is not None
                    else None
                ),
                goal_view_min_clear_depth_m=float(args.goal_view_min_clear_depth_m),
                goal_view_min_valid_depth_frac=float(args.goal_view_min_valid_depth_frac),
                goal_view_depth_center_frac=float(args.goal_view_depth_center_frac),
                goal_view_min_gray_p95_p5=float(args.goal_view_min_gray_p95_p5),
                goal_view_min_texture_score=float(args.goal_view_min_texture_score),
            )
            goal_start_distance_m = float(
                np.linalg.norm(goal_position.astype(np.float64) - start_position.astype(np.float64))
            )

            goal_sample_attempts = int(info.get("image_goal_sample_attempts", 0))
            goal_sample_attempt_counts.append(goal_sample_attempts)
            goal_start_distance_values_m.append(goal_start_distance_m)
            goal_view_depth_min_clearance_values_m.append(
                float(info.get("goal_view_depth_min_clearance_m", float("nan")))
            )
            goal_view_gray_p95_p5_values.append(
                float(info.get("goal_view_gray_p95_p5", float("nan")))
            )
            goal_view_texture_score_values.append(
                float(info.get("goal_view_texture_score", float("nan")))
            )

            raw_episode["goal_position"] = [float(v) for v in goal_position]
            raw_episode["goal_rotation"] = [float(v) for v in goal_rotation]
            raw_episode["goal_start_distance_m"] = float(goal_start_distance_m)
            raw_episode["goal_sampling_seed"] = int(episode_seed)
            raw_episode["goal_sampling_reset_seed"] = int(used_reset_seed)
            raw_episode["goal_sampling_attempts"] = int(goal_sample_attempts)
            raw_episode["goal_sampling_info"] = {
                "image_goal_reachability_mode": info.get("image_goal_reachability_mode"),
                "image_goal_scene_filter_mode": info.get("image_goal_scene_filter_mode"),
                "image_goal_scene_island_cell_count": info.get("image_goal_scene_island_cell_count"),
                "image_goal_scene_component_label": info.get("image_goal_scene_component_label"),
                "image_goal_boundary_margin_m": info.get("image_goal_boundary_margin_m"),
                "image_goal_boundary_margin_used_m": info.get("image_goal_boundary_margin_used_m"),
                "image_goal_geometry_clearance_m": info.get("image_goal_geometry_clearance_m"),
                "image_goal_interior_cell_count": info.get("image_goal_interior_cell_count"),
                "image_goal_max_height_delta_m": info.get("image_goal_max_height_delta_m"),
                "goal_view_valid_depth_fraction": info.get("goal_view_valid_depth_fraction"),
                "goal_view_depth_min_clearance_m": info.get("goal_view_depth_min_clearance_m"),
                "goal_view_depth_min_full_m": info.get("goal_view_depth_min_full_m"),
                "goal_view_gray_p95_p5": info.get("goal_view_gray_p95_p5"),
                "goal_view_texture_score": info.get("goal_view_texture_score"),
            }
            output_episodes.append(_to_builtin_json(raw_episode))

            print(
                "[generate_val_image_goal_split] "
                f"episode {episode_idx + 1}/{len(normalized_episodes)} "
                f"id={episode_id} scene={episode['scene_name']} "
                f"goal_attempts={goal_sample_attempts} "
                f"start_goal_dist_m={goal_start_distance_m:.3f}",
                flush=True,
            )
    finally:
        if active_env is not None:
            active_env.close()

    output_payload = dict(input_payload)
    output_payload["episodes"] = output_episodes
    output_payload["image_goal_generation"] = {
        "source_episodes_json": os.path.abspath(args.episodes_json),
        "episode_count": int(len(output_episodes)),
        "seed": int(args.seed),
        "episode_sample_attempts": int(args.episode_sample_attempts),
        "goal_sample_min_start_distance_m": float(args.goal_sample_min_start_distance_m),
        "goal_sample_max_start_distance_m": (
            float(args.goal_sample_max_start_distance_m)
            if args.goal_sample_max_start_distance_m is not None
            else None
        ),
        "goal_sample_max_tries": int(args.goal_sample_max_tries),
        "goal_boundary_margin_m": float(args.goal_boundary_margin_m),
        "goal_geometry_clearance_m": float(args.goal_geometry_clearance_m),
        "goal_max_height_delta_m": float(args.goal_max_height_delta_m),
        "goal_reachability_projection_tolerance_m": float(
            args.goal_reachability_projection_tolerance_m
        ),
        "goal_require_navmesh_path": bool(args.goal_require_navmesh_path),
        "goal_view_min_clear_depth_m": float(args.goal_view_min_clear_depth_m),
        "goal_view_min_valid_depth_frac": float(args.goal_view_min_valid_depth_frac),
        "goal_view_depth_center_frac": float(args.goal_view_depth_center_frac),
        "goal_view_min_gray_p95_p5": float(args.goal_view_min_gray_p95_p5),
        "goal_view_min_texture_score": float(args.goal_view_min_texture_score),
        "goal_sampling_attempts_stats": {
            "mean": (
                float(np.mean(goal_sample_attempt_counts))
                if goal_sample_attempt_counts
                else float("nan")
            ),
            "min": (
                int(np.min(goal_sample_attempt_counts))
                if goal_sample_attempt_counts
                else -1
            ),
            "max": (
                int(np.max(goal_sample_attempt_counts))
                if goal_sample_attempt_counts
                else -1
            ),
        },
        "goal_start_distance_m_stats": {
            "mean": (
                float(np.mean(goal_start_distance_values_m))
                if goal_start_distance_values_m
                else float("nan")
            ),
            "min": (
                float(np.min(goal_start_distance_values_m))
                if goal_start_distance_values_m
                else float("nan")
            ),
            "max": (
                float(np.max(goal_start_distance_values_m))
                if goal_start_distance_values_m
                else float("nan")
            ),
        },
        "goal_view_depth_min_clearance_m_stats": {
            "mean": (
                float(np.nanmean(goal_view_depth_min_clearance_values_m))
                if goal_view_depth_min_clearance_values_m
                else float("nan")
            ),
            "min": (
                float(np.nanmin(goal_view_depth_min_clearance_values_m))
                if goal_view_depth_min_clearance_values_m
                else float("nan")
            ),
            "max": (
                float(np.nanmax(goal_view_depth_min_clearance_values_m))
                if goal_view_depth_min_clearance_values_m
                else float("nan")
            ),
        },
        "goal_view_gray_p95_p5_stats": {
            "mean": (
                float(np.nanmean(goal_view_gray_p95_p5_values))
                if goal_view_gray_p95_p5_values
                else float("nan")
            ),
            "min": (
                float(np.nanmin(goal_view_gray_p95_p5_values))
                if goal_view_gray_p95_p5_values
                else float("nan")
            ),
            "max": (
                float(np.nanmax(goal_view_gray_p95_p5_values))
                if goal_view_gray_p95_p5_values
                else float("nan")
            ),
        },
        "goal_view_texture_score_stats": {
            "mean": (
                float(np.nanmean(goal_view_texture_score_values))
                if goal_view_texture_score_values
                else float("nan")
            ),
            "min": (
                float(np.nanmin(goal_view_texture_score_values))
                if goal_view_texture_score_values
                else float("nan")
            ),
            "max": (
                float(np.nanmax(goal_view_texture_score_values))
                if goal_view_texture_score_values
                else float("nan")
            ),
        },
    }

    _write_json(args.output_json, output_payload)
    print(
        f"Wrote {len(output_episodes)} episodes with image-goal targets to {args.output_json}",
        flush=True,
    )


if __name__ == "__main__":
    main()
