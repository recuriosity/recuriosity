from __future__ import annotations

"""Policy-agnostic utilities shared by checkpoint evaluators."""

import gzip
import hashlib
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_MODULE_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


from modules.eval.completeness_utils import (
    depth_to_world_points,
    infer_episode_island_index,
    load_mesh_for_completeness,
    mesh_to_habitat_scene,
    validate_mesh_observation_alignment,
)
# Swap this import to `modules.eval.scene_filtering_old` to restore the legacy filter pipeline.
from modules.eval.scene_filtering import (
    SceneFilterState,
    infer_scene_filter_from_sweep,
    render_scene_filter_debug_images,
    sample_mesh_and_filter_with_scene_island,
)


def _normalize_seed_key_part(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_normalize_seed_key_part(item) for item in value.tolist()]
    if isinstance(value, tuple):
        return [_normalize_seed_key_part(item) for item in value]
    if isinstance(value, list):
        return [_normalize_seed_key_part(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_seed_key_part(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def derive_mesh_sampling_seed(base_seed: int | None, sampling_key: tuple[Any, ...]) -> int | None:
    if base_seed is None:
        return None
    payload = json.dumps(
        _normalize_seed_key_part(sampling_key),
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    key_bits = int.from_bytes(digest[:4], byteorder="little", signed=False)
    return int((key_bits ^ (int(base_seed) & 0xFFFFFFFF)) & 0xFFFFFFFF)


def _make_gt_stage_entry(stage: str, count: int, **extra: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {"stage": stage, "count": int(count)}
    for key, value in extra.items():
        if value is not None:
            entry[key] = _normalize_seed_key_part(value)
    return entry


def _infer_navmesh_island_from_start(pathfinder, start_position: np.ndarray) -> int:
    try:
        island_index, _ = infer_episode_island_index(
            [],
            pathfinder,
            start_position=np.asarray(start_position, dtype=np.float64),
        )
    except Exception:
        return -1
    return int(island_index)


def _format_gt_stage_counts(stage_counts: list[dict[str, Any]]) -> str:
    if not stage_counts:
        return "no_stage_counts"
    return " | ".join(f"{entry['stage']}={int(entry['count'])}" for entry in stage_counts)


def _validate_gt_points_or_raise(
    *,
    episode_id: str,
    gt_points: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    gt_points_arr = np.asarray(gt_points, dtype=np.float32)
    gt_sampling_stats = metadata.get("gt_sampling_stats", {})
    stage_counts = gt_sampling_stats.get("stage_counts", [])
    non_finite_rows = int(np.count_nonzero(~np.isfinite(gt_points_arr).all(axis=1)))
    gt_sampling_stats["non_finite_rows"] = non_finite_rows
    gt_sampling_stats["final_gt_points_count"] = int(len(gt_points_arr))

    if non_finite_rows == 0 and len(gt_points_arr) > 0:
        return

    floor_info = metadata.get("floor_inference", {})
    failure_reason = "empty_gt_points" if len(gt_points_arr) == 0 else "non_finite_gt_points"
    message = (
        f"[eval] episode_id={episode_id} invalid_gt_points "
        f"reason={failure_reason} "
        f"final_gt_points={int(len(gt_points_arr))} "
        f"non_finite_rows={non_finite_rows} "
        f"floor_mode={floor_info.get('mode', 'unknown')} "
        f"floor_y={float(floor_info.get('floor_y', float('nan'))):.3f} "
        f"stage_counts={_format_gt_stage_counts(stage_counts)}"
    )
    print(message, flush=True)
    raise RuntimeError(message)


def init_distributed() -> tuple[int, int, int]:
    import torch
    import torch.distributed as dist

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        from datetime import timedelta

        # Rank 0 performs heavy wandb I/O between barriers (images, videos,
        # point clouds, file syncs). The default NCCL timeout is far too short
        # and causes silent crashes on the waiting ranks.
        timeout = timedelta(hours=2)
        dist.init_process_group(backend=backend, init_method="env://", timeout=timeout)
    return rank, world_size, local_rank


def distributed_barrier() -> None:
    import torch
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def destroy_process_group_quietly() -> None:
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return
    try:
        dist.destroy_process_group()
    except Exception as exc:
        print(f"[eval] warning: failed to destroy process group: {exc!r}", flush=True)


def broadcast_object(value: Any, *, src: int = 0) -> Any:
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return value
    payload = [value if dist.get_rank() == int(src) else None]
    dist.broadcast_object_list(payload, src=int(src))
    return payload[0]


def make_timestamped_run_name(*, checkpoint_path: str, requested_run_name: str | None, rank: int) -> str:
    if rank == 0:
        base_name = requested_run_name or f"eval_{Path(checkpoint_path).stem}"
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        run_name = f"{base_name}_{timestamp}"
    else:
        run_name = None
    return str(broadcast_object(run_name, src=0))


def render_terminal_eval_frame_without_camera_beam(env) -> np.ndarray:
    """Render one eval frame with camera beam/marker overlays suppressed.

    Used for the final duplicate frame appended to eval videos/panels so the
    terminal view is clean.
    """

    def _iter_env_chain(root_env):
        queue = [root_env]
        seen_ids: set[int] = set()
        while queue:
            current = queue.pop(0)
            current_id = id(current)
            if current_id in seen_ids:
                continue
            seen_ids.add(current_id)
            yield current
            for attr in ("env", "_env", "unwrapped", "raw_env", "_wrapped_env", "habitat_env"):
                try:
                    nested = getattr(current, attr)
                except Exception:
                    nested = None
                if nested is None:
                    continue
                if id(nested) not in seen_ids:
                    queue.append(nested)

    patched_flags: list[tuple[Any, str, bool, bool]] = []
    for target_env in _iter_env_chain(env):
        for attr_name in ("_hide_progress_observed_points",):
            had_attr = hasattr(target_env, attr_name)
            prev_value = bool(getattr(target_env, attr_name, False))
            try:
                setattr(target_env, attr_name, True)
            except Exception:
                continue
            patched_flags.append((target_env, attr_name, had_attr, prev_value))
    try:
        frame = np.asarray(env.render())
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        return np.array(frame, dtype=np.uint8, copy=True)
    finally:
        for target_env, attr_name, had_attr, prev_value in reversed(patched_flags):
            try:
                if had_attr:
                    setattr(target_env, attr_name, prev_value)
                else:
                    delattr(target_env, attr_name)
            except Exception:
                continue


def write_rank_failure_trace(
    *,
    output_root: str,
    rank: int,
    local_rank: int,
    exception: BaseException,
    episode_context: dict[str, Any] | None = None,
) -> str | None:
    trace_text = "".join(
        traceback.format_exception(type(exception), exception, exception.__traceback__)
    )
    failure_dir = os.path.join(output_root, "rank_failures")
    try:
        os.makedirs(failure_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        trace_path = os.path.join(
            failure_dir,
            f"rank{int(rank):03d}_local{int(local_rank):03d}_pid{os.getpid()}_{timestamp}.log",
        )
        with open(trace_path, "w", encoding="utf-8") as handle:
            handle.write(
                "\n".join(
                    [
                        f"rank={int(rank)}",
                        f"local_rank={int(local_rank)}",
                        f"pid={os.getpid()}",
                        f"exception_type={type(exception).__name__}",
                        f"exception_repr={exception!r}",
                        (
                            f"episode_context={json.dumps(episode_context, sort_keys=True)}"
                            if episode_context is not None
                            else "episode_context=null"
                        ),
                        "",
                        trace_text,
                    ]
                )
            )
        return trace_path
    except Exception as exc:
        print(
            f"[eval rank={rank}] warning: failed to write rank failure trace: {exc!r}",
            flush=True,
        )
        return None


def merge_eval_results_from_shared_storage(
    *,
    progress_dir: str,
    local_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the full eval result list without NCCL ``all_gather_object``.

    Each rank writes ``live_results/episode_<index>.json`` atomically. After all
    ranks finish their episode loop, one barrier ensures files are visible; every
    rank then reads the same directory (requires a shared ``output_root``, e.g.
    session-mounted storage on the cluster).
    """
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return list(local_results)

    distributed_barrier()
    return load_live_result_records(progress_dir)


def _normalize_live_record_metrics_inplace(record: dict[str, Any]) -> None:
    metrics_by_step = record.get("metrics_by_step", {})
    if isinstance(metrics_by_step, dict):
        record["metrics_by_step"] = {int(step): values for step, values in metrics_by_step.items()}
    record["episode_index"] = int(record["episode_index"])


def load_single_live_result_file(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        record = json.load(handle)
    _normalize_live_record_metrics_inplace(record)
    return record


def merge_incremental_live_results(
    progress_dir: str,
    cache: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Load only new ``episode_*.json`` files into ``cache``; return sorted merge."""
    if not os.path.isdir(progress_dir):
        return sorted(cache.values(), key=lambda item: int(item["episode_index"]))

    for entry in os.listdir(progress_dir):
        if not entry.endswith(".json"):
            continue
        path = os.path.join(progress_dir, entry)
        stem = entry[: -len(".json")]
        if stem.startswith("episode_"):
            mid = stem[len("episode_") :]
            if mid.isdigit() and len(mid) == 6:
                idx = int(mid)
                if idx in cache:
                    continue
        record = load_single_live_result_file(path)
        cache[int(record["episode_index"])] = record

    return sorted(cache.values(), key=lambda item: int(item["episode_index"]))


def shard_episodes(episodes: list[dict[str, Any]], rank: int, world_size: int) -> list[tuple[int, dict[str, Any]]]:
    return [
        (index, episode)
        for index, episode in enumerate(episodes)
        if index % world_size == rank
    ]


def select_eval_episodes(
    episodes: list[dict[str, Any]],
    *,
    episode_offset: int = 0,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    start_index = int(episode_offset)
    if start_index < 0:
        raise ValueError(f"episode_offset must be >= 0, got {start_index}")

    selected = episodes[start_index:]
    if limit is None:
        return selected

    max_episodes = int(limit)
    if max_episodes < 0:
        raise ValueError(f"limit must be >= 0, got {max_episodes}")
    return selected[:max_episodes]


def load_episode_specs(path: str) -> list[dict[str, Any]]:
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"Expected JSON to contain an 'episodes' list, got {type(episodes).__name__}")
    return [normalize_episode_paths(episode) for episode in episodes]


def normalize_episode_paths(episode: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(episode)
    scene_id = normalized.get("scene_id")
    if isinstance(scene_id, str):
        normalized["scene_id"] = remap_scene_path(scene_id)
    return normalized


def remap_scene_path(scene_path: str) -> str:
    abs_path = os.path.abspath(scene_path)
    if os.path.isfile(abs_path):
        return abs_path

    gibson_path = _resolve_gibson_scene_path(scene_path)
    if gibson_path is not None:
        return gibson_path

    return remap_hm3d_scene_path(scene_path)


_GIBSON_GLB_INDEX_CACHE: dict[str, dict[str, str]] = {}
_GIBSON_SCENE_MISS_CACHE: set[tuple[str, str]] = set()


def _read_int_env(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)


def _read_float_env(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def _index_gibson_scene_glbs(scene_root: str) -> dict[str, str]:
    index: dict[str, str] = {}
    if not scene_root or not os.path.isdir(scene_root):
        return index

    for dirpath, _, filenames in os.walk(scene_root):
        for fname in filenames:
            if not fname.endswith(".glb"):
                continue
            scene_name = os.path.splitext(fname)[0]
            index[scene_name] = os.path.abspath(os.path.join(dirpath, fname))
    return index


def _direct_gibson_scene_candidates(scene_root: str, scene_name: str) -> tuple[str, ...]:
    scene_file = f"{scene_name}.glb"
    return (
        os.path.abspath(os.path.join(scene_root, scene_file)),
        os.path.abspath(os.path.join(scene_root, "gibson", scene_file)),
    )


def _refresh_gibson_index(scene_root: str) -> dict[str, str]:
    index = _index_gibson_scene_glbs(scene_root)
    _GIBSON_GLB_INDEX_CACHE[scene_root] = index
    return index


def _resolve_gibson_scene_path(scene_path: str) -> str | None:
    normalized = scene_path.replace("\\", "/")
    gibson_prefix = "data/scene_datasets/gibson/"
    if not normalized.startswith(gibson_prefix):
        return None

    scene_name = os.path.splitext(os.path.basename(normalized))[0]
    if not scene_name:
        return None

    gibson_root = os.environ.get("GIBSON_SCENE_ROOT", "").strip()
    if not gibson_root:
        return None

    scene_root = os.path.abspath(gibson_root)
    missing_key = (scene_root, scene_name)
    if missing_key in _GIBSON_SCENE_MISS_CACHE:
        return None

    for candidate in _direct_gibson_scene_candidates(scene_root, scene_name):
        if os.path.isfile(candidate):
            return candidate

    index = _GIBSON_GLB_INDEX_CACHE.get(scene_root)
    if index is None:
        index = _refresh_gibson_index(scene_root)
    resolved = index.get(scene_name)
    if resolved and os.path.isfile(resolved):
        return resolved

    # Azure/FUSE mounts can appear empty on first access; retry index refreshes before failing.
    retry_attempts = max(0, _read_int_env("GIBSON_SCENE_LOOKUP_RETRIES", 20))
    retry_delay_s = max(0.0, _read_float_env("GIBSON_SCENE_LOOKUP_RETRY_DELAY_S", 1.0))
    for _ in range(retry_attempts):
        if retry_delay_s > 0:
            time.sleep(retry_delay_s)
        for candidate in _direct_gibson_scene_candidates(scene_root, scene_name):
            if os.path.isfile(candidate):
                return candidate
        refreshed = _refresh_gibson_index(scene_root)
        resolved = refreshed.get(scene_name)
        if resolved and os.path.isfile(resolved):
            return resolved

    _GIBSON_SCENE_MISS_CACHE.add(missing_key)
    return None


def remap_hm3d_scene_path(scene_path: str) -> str:
    abs_path = os.path.abspath(scene_path)
    if os.path.isfile(abs_path):
        return abs_path

    hm3d_root = os.environ.get("HM3D_DATA_ROOT", "").strip()
    if not hm3d_root:
        return abs_path

    for marker in ("/hm3d-val/", "/hm3d/"):
        marker_idx = abs_path.find(marker)
        if marker_idx < 0:
            continue
        rel_suffix = abs_path[marker_idx + 1 :]
        candidate = os.path.join(hm3d_root, rel_suffix)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    return abs_path


def resolve_hm3d_stage_path(scene_path: str, navmesh_path: str | None) -> str:
    abs_scene_path = os.path.abspath(scene_path)
    if abs_scene_path.endswith(".basis.glb"):
        return abs_scene_path

    if navmesh_path:
        navmesh_abs = os.path.abspath(navmesh_path)
        if navmesh_abs.endswith(".basis.navmesh"):
            basis_from_navmesh = navmesh_abs[:-len(".basis.navmesh")] + ".basis.glb"
            if os.path.isfile(basis_from_navmesh):
                return basis_from_navmesh

    if "/hm3d_glb/" in abs_scene_path:
        basis_path = abs_scene_path.replace("/hm3d_glb/", "/hm3d_nav/")
        basis_path = os.path.splitext(basis_path)[0] + ".basis.glb"
        if os.path.isfile(basis_path):
            return basis_path

    return abs_scene_path


def derive_navmesh_path(scene_id: str) -> str | None:
    scene_path = remap_scene_path(scene_id)
    if "/hm3d_glb/" in scene_path:
        navmesh_path = scene_path.replace("/hm3d_glb/", "/hm3d_nav/")
        navmesh_path = os.path.splitext(navmesh_path)[0] + ".basis.navmesh"
        return navmesh_path if os.path.isfile(navmesh_path) else None

    scene_dir = os.path.dirname(scene_path)
    scene_name = os.path.splitext(os.path.basename(scene_path))[0]
    candidates = (
        os.path.join(scene_dir, f"{scene_name}.navmesh"),
        f"{scene_path}.navmesh",
        os.path.join(scene_dir, f"{scene_name}.basis.navmesh"),
    )
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def aggregate_results(results: list[dict[str, Any]], metric_steps: list[int]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "num_episodes": len(results),
        "metric_steps": [int(step) for step in metric_steps],
        "steps": {},
    }

    for step in metric_steps:
        mean_values = []
        ratio_values = []
        for result in results:
            metrics = result["metrics_by_step"].get(int(step))
            if metrics is None:
                continue
            mean_values.append(metrics["mean_distance_m"])
            ratio_values.append(metrics["ratio_within_threshold"])

        aggregate["steps"][str(step)] = {
            "mean_distance_m": float(np.mean(mean_values)) if mean_values else float("nan"),
            "ratio_within_threshold": float(np.mean(ratio_values)) if ratio_values else float("nan"),
        }

    aggregate["final"] = {
        "mean_distance_m": float(np.mean([result["final_mean_distance_m"] for result in results])) if results else float("nan"),
        "ratio_within_threshold": float(np.mean([result["final_ratio_within_threshold"] for result in results])) if results else float("nan"),
    }
    aggregate["avg_final_completeness"] = aggregate["final"]["ratio_within_threshold"]
    return aggregate


def _result_to_json_record(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": result["episode_id"],
        "scene_name": result["scene_name"],
        "episode_index": int(result["episode_index"]),
        "final_completeness": result["final_ratio_within_threshold"],
        "final_mean_distance_m": result["final_mean_distance_m"],
        "final_ratio_within_threshold": result["final_ratio_within_threshold"],
        "num_frames": result["num_frames"],
        "num_observed_points": result["num_observed_points"],
        "metrics_by_step": result["metrics_by_step"],
        "timing": result.get("timing", {}),
        "gt_sampling_stats": result.get("gt_sampling_stats"),
    }


def write_json_atomic(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)


def write_progress_jsons(
    *,
    output_root: str,
    results: list[dict[str, Any]],
    metric_steps: list[int],
    module_name: str,
    checkpoint_path: str,
    checkpoint_step: int,
    episodes_json: str,
    world_size: int,
    total_episodes: int,
) -> tuple[str, str, str]:
    ordered_results = sorted(results, key=lambda item: int(item["episode_index"]))
    per_episode_results = [_result_to_json_record(result) for result in ordered_results]
    aggregate = aggregate_results(ordered_results, metric_steps)
    aggregate["completed_episodes"] = len(ordered_results)
    aggregate["total_episodes"] = int(total_episodes)

    per_episode_json_path = os.path.join(output_root, "per_episode_results.json")
    write_json_atomic(per_episode_json_path, per_episode_results)

    aggregate_json_path = os.path.join(output_root, "aggregate_results.json")
    write_json_atomic(aggregate_json_path, aggregate)

    summary = {
        "ppo_module": module_name,
        "checkpoint_path": checkpoint_path,
        "checkpoint_step": int(checkpoint_step),
        "episodes_json": os.path.abspath(episodes_json),
        "world_size": world_size,
        "completed_episodes": len(ordered_results),
        "total_episodes": int(total_episodes),
        "aggregate": aggregate,
        "episodes": per_episode_results,
    }
    summary_path = os.path.join(output_root, "summary.json")
    write_json_atomic(summary_path, summary)
    return per_episode_json_path, aggregate_json_path, summary_path


def _result_to_live_record(result: dict[str, Any]) -> dict[str, Any]:
    return {
        **_result_to_json_record(result),
        "start_rgb_path": result["start_rgb_path"],
        "final_rgb_path": result["final_rgb_path"],
        "topdown_path": result["topdown_path"],
        "scene_filter_sweep_map_path": result.get("scene_filter_sweep_map_path"),
        "scene_filter_closed_map_path": result.get("scene_filter_closed_map_path"),
        "scene_filter_components_map_path": result.get("scene_filter_components_map_path"),
        "panel_path": result["panel_path"],
        "video_path": result["video_path"],
        "point_cloud_path": result.get("point_cloud_path"),
        "point_cloud_progress_video_path": result.get("point_cloud_progress_video_path"),
        "gt_sampling_stats_path": result.get("gt_sampling_stats_path"),
        "actions_path": result.get("actions_path"),
        "running_history": result["running_history"],
    }


def write_live_result_record(progress_dir: str, result: dict[str, Any]) -> str:
    os.makedirs(progress_dir, exist_ok=True)
    episode_index = int(result["episode_index"])
    path = os.path.join(progress_dir, f"episode_{episode_index:06d}.json")
    write_json_atomic(path, _result_to_live_record(result))
    return path


def load_live_result_records(progress_dir: str) -> list[dict[str, Any]]:
    if not os.path.isdir(progress_dir):
        return []

    records: list[dict[str, Any]] = []
    for entry in sorted(os.listdir(progress_dir)):
        if not entry.endswith(".json"):
            continue
        path = os.path.join(progress_dir, entry)
        records.append(load_single_live_result_file(path))

    records.sort(key=lambda item: int(item["episode_index"]))
    return records


def sync_progress_files_to_wandb(
    *,
    output_root: str,
    progress_dir: str,
    per_episode_json_path: str,
    aggregate_json_path: str,
    summary_path: str,
    include_live_snapshots: bool = True,
) -> None:
    import wandb

    paths = [per_episode_json_path, aggregate_json_path, summary_path]
    if include_live_snapshots and os.path.isdir(progress_dir):
        for entry in sorted(os.listdir(progress_dir)):
            if entry.endswith(".json"):
                paths.append(os.path.join(progress_dir, entry))

    for path in paths:
        wandb.save(path, base_path=output_root)


def log_running_aggregate_to_wandb(
    *,
    results: list[dict[str, Any]],
    metric_steps: list[int],
    log_step: int,
) -> None:
    import wandb

    if not results:
        return

    aggregate = aggregate_results(results, metric_steps)
    payload: dict[str, Any] = {
        "aggregate/completed_episodes": len(results),
        "aggregate/running_avg_final_score": aggregate["final"]["ratio_within_threshold"],
    }
    for step in metric_steps:
        values = aggregate["steps"].get(str(step))
        if values is None:
            continue
        payload[f"aggregate/running_avg_score_t{step}"] = values["ratio_within_threshold"]

    wandb.log(payload, step=int(log_step))


def make_running_average_history(results: list[dict[str, Any]]) -> list[dict[str, float]]:
    step_sums: dict[int, dict[str, float]] = {}
    for result in results:
        history = result.get("running_history", [])
        if not isinstance(history, list):
            continue
        for row in history:
            step = int(row["step"])
            bucket = step_sums.setdefault(
                step,
                {
                    "mean_distance_m_sum": 0.0,
                    "ratio_within_threshold_sum": 0.0,
                    "count": 0.0,
                },
            )
            bucket["mean_distance_m_sum"] += float(row["mean_distance_m"])
            bucket["ratio_within_threshold_sum"] += float(row["ratio_within_threshold"])
            bucket["count"] += 1.0

    averaged_history: list[dict[str, float]] = []
    for step in sorted(step_sums):
        bucket = step_sums[step]
        count = max(int(bucket["count"]), 1)
        averaged_history.append(
            {
                "step": int(step),
                "mean_distance_m": float(bucket["mean_distance_m_sum"] / count),
                "ratio_within_threshold": float(bucket["ratio_within_threshold_sum"] / count),
                "num_episodes": int(count),
            }
        )
    return averaged_history


def patch_wandb_apikey() -> None:
    try:
        import wandb.sdk.lib.apikey as apikey_mod

        original_write_key = apikey_mod.Apikey.write_key

        def _patched(settings, key):
            try:
                return original_write_key(settings, key)
            except ValueError as exc:
                if "40 characters" in str(exc) and len(key) == 86:
                    os.environ["WANDB_API_KEY"] = key
                    return
                raise

        apikey_mod.Apikey.write_key = staticmethod(_patched)
    except Exception:
        pass


def make_gt_point_cloud_object(
    *,
    gt_points: np.ndarray,
    min_distances_m: np.ndarray,
    threshold_m: float,
    max_points: int,
    seed: int,
):
    import wandb

    points = np.asarray(gt_points, dtype=np.float32)
    dists = np.asarray(min_distances_m, dtype=np.float32).reshape(-1)
    if len(points) != len(dists):
        raise ValueError("gt_points and min_distances_m must have the same length")
    if len(points) == 0:
        return None

    if len(points) > int(max_points):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(points), size=int(max_points), replace=False)
        points = points[idx]
        dists = dists[idx]

    gray = np.asarray([150, 150, 150], dtype=np.float32)
    green = np.asarray([20, 220, 40], dtype=np.float32)
    colors = np.where(dists[:, None] < float(threshold_m), green[None, :], gray[None, :])
    return wandb.Object3D(np.concatenate([points, colors], axis=1))


def select_history_frame_indices(num_frames: int, max_frames: int) -> np.ndarray:
    if num_frames <= 0:
        return np.zeros((0,), dtype=np.int64)
    if num_frames <= max_frames:
        return np.arange(num_frames, dtype=np.int64)

    indices = np.linspace(0, num_frames - 1, num=max_frames, dtype=np.int64)
    indices[0] = 0
    indices[-1] = num_frames - 1
    return np.unique(indices)


def pack_point_cloud_history_frames(frame_points_history: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(frame_points_history) + 1, dtype=np.int64)
    if not frame_points_history:
        return np.empty((0, 3), dtype=np.float32), offsets

    flattened_frames: list[np.ndarray] = []
    total_points = 0
    for idx, frame_points in enumerate(frame_points_history, start=1):
        points = np.asarray(frame_points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"Expected frame points shape (N, 3), got {points.shape}")
        flattened_frames.append(points)
        total_points += len(points)
        offsets[idx] = total_points

    if total_points == 0:
        return np.empty((0, 3), dtype=np.float32), offsets
    return np.concatenate(flattened_frames, axis=0), offsets


def extract_camera_position_from_obs(obs: dict[str, Any]) -> np.ndarray:
    camera_to_world = np.asarray(obs["c2w"], dtype=np.float32)
    if camera_to_world.shape != (4, 4):
        raise ValueError(f"Expected obs['c2w'] shape (4, 4), got {camera_to_world.shape}")
    return camera_to_world[:3, 3].copy()


def select_point_cloud_visualization_subset(
    gt_points: np.ndarray,
    *,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    gt_points_arr = np.asarray(gt_points, dtype=np.float32)
    if len(gt_points_arr) == 0:
        return None, None

    vis_count = min(int(max_points), int(len(gt_points_arr)))
    if vis_count <= 0:
        return None, None

    rng = np.random.default_rng(int(seed))
    if vis_count >= len(gt_points_arr):
        vis_indices = np.arange(len(gt_points_arr), dtype=np.int64)
    else:
        vis_indices = np.sort(
            rng.choice(len(gt_points_arr), size=vis_count, replace=False).astype(np.int64)
        )
    vis_points = np.asarray(gt_points_arr[vis_indices], dtype=np.float32)
    return vis_points, vis_indices


def unproject_observed_points_from_obs(obs: dict[str, Any], *, depth_stride: int) -> np.ndarray:
    return depth_to_world_points(
        obs["depth"],
        obs["c2w"],
        obs["fxfycxcy"],
        valid_mask=obs.get("valid_mask"),
        stride=depth_stride,
    )


def render_point_cloud_topdown_progress_video(
    *,
    points: np.ndarray,
    min_distance_history_m: np.ndarray,
    steps: np.ndarray,
    threshold_m: float,
    observed_points: np.ndarray | None = None,
    observed_point_offsets: np.ndarray | None = None,
    camera_positions: np.ndarray | None = None,
    output_path: str,
    image_size: int = 512,
    max_frames: int | None = None,
    point_radius_px: int = 1,
    observed_point_radius_px: int = 1,
    camera_radius_px: int = 4,
    include_observed_points: bool = True,
    include_trajectory: bool = True,
    include_camera_marker: bool = True,
    include_progress_bar: bool = True,
) -> None:
    import imageio.v2 as imageio
    from matplotlib import cm
    from PIL import Image, ImageDraw

    pts = np.asarray(points, dtype=np.float32)
    history = np.asarray(min_distance_history_m, dtype=np.float32)
    step_values = np.asarray(steps, dtype=np.int32).reshape(-1)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected points shape (N, 3), got {pts.shape}")
    if history.ndim != 2 or history.shape[1] != len(pts):
        raise ValueError(
            "Expected min_distance_history_m shape (T, N) matching points, "
            f"got {history.shape} and N={len(pts)}"
        )
    if len(step_values) != len(history):
        raise ValueError("steps and min_distance_history_m must have the same first dimension")

    observed_points_arr = None
    observed_offsets_arr = None
    if observed_points is not None or observed_point_offsets is not None:
        if observed_points is None or observed_point_offsets is None:
            raise ValueError("observed_points and observed_point_offsets must be provided together")
        observed_points_arr = np.asarray(observed_points, dtype=np.float32)
        observed_offsets_arr = np.asarray(observed_point_offsets, dtype=np.int64).reshape(-1)
        if observed_points_arr.ndim != 2 or observed_points_arr.shape[1] != 3:
            raise ValueError(f"Expected observed_points shape (K, 3), got {observed_points_arr.shape}")
        if len(observed_offsets_arr) != len(step_values) + 1:
            raise ValueError(
                "observed_point_offsets must have length len(steps) + 1, "
                f"got {len(observed_offsets_arr)} for {len(step_values)} steps"
            )
        if observed_offsets_arr[0] != 0 or observed_offsets_arr[-1] != len(observed_points_arr):
            raise ValueError("observed_point_offsets must start at 0 and end at len(observed_points)")

    camera_positions_arr = None
    if camera_positions is not None:
        camera_positions_arr = np.asarray(camera_positions, dtype=np.float32)
        if camera_positions_arr.ndim != 2 or camera_positions_arr.shape[1] != 3:
            raise ValueError(f"Expected camera_positions shape (T, 3), got {camera_positions_arr.shape}")
        if len(camera_positions_arr) != len(step_values):
            raise ValueError(
                "camera_positions must have the same number of entries as steps, "
                f"got {len(camera_positions_arr)} and {len(step_values)}"
            )

    if max_frames is not None and int(max_frames) > 0:
        frame_indices = select_history_frame_indices(len(step_values), max_frames=int(max_frames))
    else:
        frame_indices = np.arange(len(step_values), dtype=np.int64)

    bounds_sources = []
    if len(pts) > 0:
        bounds_sources.append(pts[:, [0, 2]])
    if observed_points_arr is not None and len(observed_points_arr) > 0:
        bounds_sources.append(observed_points_arr[:, [0, 2]])
    if camera_positions_arr is not None and len(camera_positions_arr) > 0:
        bounds_sources.append(camera_positions_arr[:, [0, 2]])
    if not bounds_sources:
        raise ValueError("Expected at least one GT point, observed point, or camera position to render")
    all_xz = np.concatenate(bounds_sources, axis=0)
    mins = all_xz.min(axis=0)
    maxs = all_xz.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = float(max(maxs[0] - mins[0], maxs[1] - mins[1], 1e-6))
    pad = 24
    scale = float((image_size - 2 * pad - 1) / span)

    def _project_points(points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts_xz = np.asarray(points_xyz, dtype=np.float32)[:, [0, 2]]
        coords = (pts_xz - center[None, :]) * scale
        px = np.clip(np.rint(coords[:, 0] + image_size / 2.0), 0, image_size - 1).astype(np.int32)
        py = np.clip(np.rint(image_size / 2.0 + coords[:, 1]), 0, image_size - 1).astype(np.int32)
        return px, py

    gt_px, gt_py = _project_points(pts)

    bg = np.asarray([255, 255, 255], dtype=np.uint8)
    grey = np.asarray([140, 140, 140], dtype=np.uint8)
    green = np.asarray([46, 139, 87], dtype=np.uint8)
    red = np.asarray([232, 73, 64], dtype=np.uint8)
    blue = np.asarray([66, 156, 255], dtype=np.uint8)
    bar_bg = np.asarray([230, 230, 230], dtype=np.uint8)

    def _paint(
        frame: np.ndarray,
        xs: np.ndarray,
        ys: np.ndarray,
        color: np.ndarray,
        *,
        radius_px: int,
        alpha: float = 0.70,
    ) -> None:
        if len(xs) == 0:
            return
        color_arr = np.asarray(color, dtype=np.float32).reshape(1, 3)
        alpha_f = float(np.clip(alpha, 0.0, 1.0))
        if alpha_f <= 0.0:
            return
        for dy in range(-int(radius_px), int(radius_px) + 1):
            for dx in range(-int(radius_px), int(radius_px) + 1):
                yy = np.clip(ys + dy, 0, image_size - 1)
                xx = np.clip(xs + dx, 0, image_size - 1)
                base = frame[yy, xx].astype(np.float32, copy=False)
                blended = np.rint((1.0 - alpha_f) * base + alpha_f * color_arr)
                frame[yy, xx] = np.clip(blended, 0.0, 255.0).astype(np.uint8, copy=False)

    def _paint_gt(frame: np.ndarray, mask: np.ndarray, color: np.ndarray) -> None:
        if not np.any(mask):
            return
        _paint(frame, gt_px[mask], gt_py[mask], color, radius_px=point_radius_px, alpha=0.70)

    def _frame_observed_points(frame_idx: int) -> np.ndarray:
        if observed_points_arr is None or observed_offsets_arr is None:
            return np.empty((0, 3), dtype=np.float32)
        start = int(observed_offsets_arr[frame_idx])
        stop = int(observed_offsets_arr[frame_idx + 1])
        return observed_points_arr[start:stop]

    def _render_frame(
        frame_idx: int,
        *,
        draw_camera_marker: bool,
        draw_observed_points: bool,
    ) -> np.ndarray:
        dists = history[int(frame_idx)]
        completed = dists < float(threshold_m)
        frame = np.empty((image_size, image_size, 3), dtype=np.uint8)
        frame[...] = bg
        _paint_gt(frame, ~completed, grey)
        _paint_gt(frame, completed, green)

        current_observed_points = _frame_observed_points(int(frame_idx))
        if draw_observed_points and len(current_observed_points) > 0:
            obs_px, obs_py = _project_points(current_observed_points)
            _paint(
                frame,
                obs_px,
                obs_py,
                red,
                radius_px=observed_point_radius_px,
                alpha=0.70,
            )

        if camera_positions_arr is not None and (include_trajectory or draw_camera_marker):
            trajectory = camera_positions_arr[: int(frame_idx) + 1]
            traj_px, traj_py = _project_points(trajectory)
            if include_trajectory and len(traj_px) > 1:
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
                n = len(traj_px) - 1
                denom = max(1, n - 1)
                cmap = cm.get_cmap("plasma")
                for idx in range(n):
                    t01 = float(idx) / float(denom)
                    r, g, b, _ = cmap(t01)
                    traj_color = (int(255 * r), int(255 * g), int(255 * b))
                    draw.line(
                        [
                            (int(traj_px[idx]), int(traj_py[idx])),
                            (int(traj_px[idx + 1]), int(traj_py[idx + 1])),
                        ],
                        fill=traj_color,
                        width=6,
                    )
                frame = np.array(img, dtype=np.uint8, copy=True)
            if draw_camera_marker:
                cam_px, cam_py = _project_points(trajectory[-1].reshape(1, 3))
                _paint(frame, cam_px, cam_py, blue, radius_px=camera_radius_px, alpha=0.70)

        if include_progress_bar:
            progress = float(step_values[int(frame_idx)]) / float(max(int(step_values[-1]), 1))
            bar_x0 = pad
            bar_x1 = image_size - pad
            bar_y0 = image_size - 14
            bar_y1 = image_size - 8
            frame[bar_y0:bar_y1, bar_x0:bar_x1] = bar_bg
            fill_x1 = bar_x0 + int(round((bar_x1 - bar_x0) * np.clip(progress, 0.0, 1.0)))
            frame[bar_y0:bar_y1, bar_x0:fill_x1] = green
        return frame

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    writer = imageio.get_writer(output_path, fps=12, codec="libx264", bitrate="4M")
    try:
        for frame_idx in frame_indices:
            writer.append_data(
                _render_frame(
                    int(frame_idx),
                    draw_camera_marker=bool(include_camera_marker),
                    draw_observed_points=bool(include_observed_points),
                )
            )
        if len(frame_indices) > 0:
            terminal_idx = int(frame_indices[-1])
            writer.append_data(
                _render_frame(
                    terminal_idx,
                    draw_camera_marker=bool(include_camera_marker),
                    draw_observed_points=False,
                )
            )
    finally:
        writer.close()



def make_running_plots(history: list[dict[str, float]], title: str):
    import wandb

    table = wandb.Table(
        data=[
            [
                int(row["step"]),
                float(row["mean_distance_m"]),
                float(row["ratio_within_threshold"]),
                int(row.get("num_episodes", 1)),
            ]
            for row in history
        ],
        columns=["step", "mean_distance_m", "ratio_within_threshold", "num_episodes"],
    )
    ratio_plot = wandb.plot.line(
        table,
        "step",
        "ratio_within_threshold",
        title=title,
    )
    mean_plot = wandb.plot.line(
        table,
        "step",
        "mean_distance_m",
        title=f"{title} mean distance",
    )
    return table, ratio_plot, mean_plot


def save_episode_media(
    *,
    media_dir: str,
    episode_id: str,
    media: dict[str, Any],
) -> dict[str, Any]:
    import imageio.v2 as imageio

    episode_dir = os.path.join(media_dir, episode_id)
    os.makedirs(episode_dir, exist_ok=True)

    start_rgb_path = os.path.join(episode_dir, "start_rgb.png")
    final_rgb_path = os.path.join(episode_dir, "final_rgb.png")
    topdown_path = os.path.join(episode_dir, "topdown.png")
    topdown_start_with_apples_path = os.path.join(episode_dir, "topdown_start_with_apples.png")
    topdown_trajectory_only_path = os.path.join(episode_dir, "topdown_trajectory_only.png")
    scene_filter_sweep_map_path = os.path.join(episode_dir, "scene_filter_sweep_map.png")
    scene_filter_closed_map_path = os.path.join(episode_dir, "scene_filter_closed_map.png")
    scene_filter_components_map_path = os.path.join(episode_dir, "scene_filter_components_map.png")
    panel_path = os.path.join(episode_dir, "panel.png")

    imageio.imwrite(start_rgb_path, np.asarray(media["start_rgb"], dtype=np.uint8))
    imageio.imwrite(final_rgb_path, np.asarray(media["final_rgb"], dtype=np.uint8))
    imageio.imwrite(topdown_path, np.asarray(media["topdown"], dtype=np.uint8))
    if media.get("topdown_start_with_apples") is not None:
        imageio.imwrite(
            topdown_start_with_apples_path,
            np.asarray(media["topdown_start_with_apples"], dtype=np.uint8),
        )
    if media.get("topdown_trajectory_only") is not None:
        imageio.imwrite(
            topdown_trajectory_only_path,
            np.asarray(media["topdown_trajectory_only"], dtype=np.uint8),
        )
    if media.get("scene_filter_sweep_map") is not None:
        imageio.imwrite(
            scene_filter_sweep_map_path,
            np.asarray(media["scene_filter_sweep_map"], dtype=np.uint8),
        )
    if media.get("scene_filter_closed_map") is not None:
        imageio.imwrite(
            scene_filter_closed_map_path,
            np.asarray(media["scene_filter_closed_map"], dtype=np.uint8),
        )
    if media.get("scene_filter_components_map") is not None:
        imageio.imwrite(
            scene_filter_components_map_path,
            np.asarray(media["scene_filter_components_map"], dtype=np.uint8),
        )
    imageio.imwrite(panel_path, np.asarray(media["panel"], dtype=np.uint8))

    saved = {
        "start_rgb_path": start_rgb_path,
        "final_rgb_path": final_rgb_path,
        "topdown_path": topdown_path,
        "topdown_start_with_apples_path": (
            topdown_start_with_apples_path
            if media.get("topdown_start_with_apples") is not None
            else None
        ),
        "topdown_trajectory_only_path": (
            topdown_trajectory_only_path if media.get("topdown_trajectory_only") is not None else None
        ),
        "scene_filter_sweep_map_path": (
            scene_filter_sweep_map_path if media.get("scene_filter_sweep_map") is not None else None
        ),
        "scene_filter_closed_map_path": (
            scene_filter_closed_map_path
            if media.get("scene_filter_closed_map") is not None
            else None
        ),
        "scene_filter_components_map_path": (
            scene_filter_components_map_path
            if media.get("scene_filter_components_map") is not None
            else None
        ),
        "panel_path": panel_path,
        "running_history": media["running_history"],
    }

    point_cloud_data = media.get("point_cloud_data")
    if point_cloud_data is not None:
        point_cloud_path = os.path.join(episode_dir, "point_cloud.npz")
        payload = {
            "gt_points": np.asarray(point_cloud_data["gt_points"], dtype=np.float32),
            "min_distances_m": np.asarray(point_cloud_data["min_distances_m"], dtype=np.float32),
        }
        vis_indices = point_cloud_data.get("vis_indices")
        if vis_indices is not None:
            payload["vis_indices"] = np.asarray(vis_indices, dtype=np.int64)
        np.savez_compressed(point_cloud_path, **payload)
        saved["point_cloud_path"] = point_cloud_path

    point_cloud_progress_data = media.get("point_cloud_progress_data")
    if point_cloud_progress_data is not None:
        point_cloud_progress_path = os.path.join(episode_dir, "point_cloud_progress.npz")
        np.savez_compressed(
            point_cloud_progress_path,
            points=np.asarray(point_cloud_progress_data["points"], dtype=np.float32),
            steps=np.asarray(point_cloud_progress_data["steps"], dtype=np.int32),
            min_distance_history_m=np.asarray(
                point_cloud_progress_data["min_distance_history_m"],
                dtype=np.float32,
            ),
            observed_points=np.asarray(point_cloud_progress_data["observed_points"], dtype=np.float32),
            observed_point_offsets=np.asarray(
                point_cloud_progress_data["observed_point_offsets"],
                dtype=np.int64,
            ),
            camera_positions=np.asarray(point_cloud_progress_data["camera_positions"], dtype=np.float32),
        )
        point_cloud_progress_video_path = os.path.join(episode_dir, "point_cloud_topdown_progress.mp4")
        render_point_cloud_topdown_progress_video(
            points=point_cloud_progress_data["points"],
            min_distance_history_m=point_cloud_progress_data["min_distance_history_m"],
            steps=point_cloud_progress_data["steps"],
            threshold_m=float(point_cloud_progress_data["threshold_m"]),
            observed_points=point_cloud_progress_data["observed_points"],
            observed_point_offsets=point_cloud_progress_data["observed_point_offsets"],
            camera_positions=point_cloud_progress_data["camera_positions"],
            output_path=point_cloud_progress_video_path,
        )
        saved["point_cloud_progress_path"] = point_cloud_progress_path
        saved["point_cloud_progress_video_path"] = point_cloud_progress_video_path

    gt_sampling_stats = media.get("gt_sampling_stats")
    if gt_sampling_stats is not None:
        gt_sampling_stats_path = os.path.join(episode_dir, "gt_sampling_stats.json")
        write_json_atomic(gt_sampling_stats_path, _normalize_seed_key_part(gt_sampling_stats))
        saved["gt_sampling_stats_path"] = gt_sampling_stats_path
    actions_trace = media.get("actions_trace")
    if actions_trace is not None:
        actions_path = os.path.join(episode_dir, "actions.json")
        write_json_atomic(actions_path, _normalize_seed_key_part(actions_trace))
        saved["actions_path"] = actions_path

    return saved


def load_point_cloud_object(point_cloud_path: str, threshold_m: float, max_points: int, seed: int):
    payload = np.load(point_cloud_path)
    gt_points = payload["gt_points"]
    min_distances_m = payload["min_distances_m"]
    if "vis_indices" in payload:
        vis_indices = np.asarray(payload["vis_indices"], dtype=np.int64).reshape(-1)
        gt_points = gt_points[vis_indices]
        min_distances_m = min_distances_m[vis_indices]
        max_points = max(int(max_points), int(len(gt_points)))
    return make_gt_point_cloud_object(
        gt_points=gt_points,
        min_distances_m=min_distances_m,
        threshold_m=threshold_m,
        max_points=max_points,
        seed=seed,
    )


def log_episode_to_wandb(
    *,
    result: dict[str, Any],
    completed_results: list[dict[str, Any]] | None,
    metric_steps: list[int],
    threshold_m: float,
    max_gt_points_vis: int,
    seed: int,
    log_step: int,
    log_video: bool,
) -> None:
    import wandb

    history_results = completed_results if completed_results is not None else [result]
    running_average_history = make_running_average_history(history_results)
    running_plot_title = (
        f"running completeness avg over {len(history_results)} episode"
        f"{'' if len(history_results) == 1 else 's'}"
    )
    running_table, running_ratio_plot, running_mean_plot = make_running_plots(
        running_average_history,
        running_plot_title,
    )
    log_payload: dict[str, Any] = {
        "episode/index": int(result["episode_index"]) + 1,
        "episode/final_mean_distance_m": result["final_mean_distance_m"],
        "episode/final_score": result["final_ratio_within_threshold"],
        "episode/start_rgb": wandb.Image(result["start_rgb_path"]),
        "episode/final_rgb": wandb.Image(result["final_rgb_path"]),
        "episode/topdown_trajectory": wandb.Image(result["topdown_path"]),
        "episode/final_panel": wandb.Image(result["panel_path"]),
        "episode/running_completeness_table": running_table,
        "episode/running_completeness": running_ratio_plot,
        "episode/running_mean_distance_m": running_mean_plot,
    }
    gt_sampling_stats = result.get("gt_sampling_stats")
    if gt_sampling_stats:
        visualization_counts = gt_sampling_stats.get("visualization_counts", {})
        log_payload["gt/final_points"] = int(gt_sampling_stats.get("final_gt_points_count", 0))
        log_payload["gt/non_finite_rows"] = int(gt_sampling_stats.get("non_finite_rows", 0))
        log_payload["gt/point_cloud_completion_points"] = int(
            visualization_counts.get("gt_point_cloud_completion", 0)
        )
        log_payload["gt/point_cloud_topdown_progress_points"] = int(
            visualization_counts.get("gt_point_cloud_topdown_progress", 0)
        )
    for step in metric_steps:
        step_metrics = result["metrics_by_step"].get(step)
        if step_metrics is None:
            continue
        log_payload[f"completeness/score_t{step}"] = step_metrics["ratio_within_threshold"]
        log_payload[f"completeness/mean_distance_m_t{step}"] = step_metrics["mean_distance_m"]
    for stage, values in result.get("timing", {}).items():
        safe_stage = stage.replace("/", "_")
        log_payload[f"timing/{safe_stage}_total_s"] = values["total_s"]
        log_payload[f"timing/{safe_stage}_mean_s"] = values["mean_s"]
    if result.get("scene_filter_sweep_map_path"):
        log_payload["episode/scene_filter_sweep_map"] = wandb.Image(
            result["scene_filter_sweep_map_path"]
        )
    if result.get("scene_filter_closed_map_path"):
        log_payload["episode/scene_filter_closed_map"] = wandb.Image(
            result["scene_filter_closed_map_path"]
        )
    if result.get("scene_filter_components_map_path"):
        log_payload["episode/scene_filter_components_map"] = wandb.Image(
            result["scene_filter_components_map_path"]
        )
    if result.get("point_cloud_path"):
        log_payload["episode/gt_point_cloud_completion"] = load_point_cloud_object(
            result["point_cloud_path"],
            threshold_m=threshold_m,
            max_points=max_gt_points_vis,
            seed=seed + int(result["episode_index"]),
        )
    if result.get("point_cloud_progress_video_path"):
        log_payload["episode/gt_point_cloud_topdown_progress"] = wandb.Video(
            result["point_cloud_progress_video_path"],
            format="mp4",
        )
    if log_video:
        log_payload["eval/video"] = wandb.Video(result["video_path"], format="mp4")
    wandb.log(log_payload, step=log_step)
    if result.get("gt_sampling_stats_path"):
        wandb.save(
            result["gt_sampling_stats_path"],
            base_path=os.path.dirname(os.path.dirname(result["gt_sampling_stats_path"])),
        )
    if result.get("actions_path"):
        wandb.save(
            result["actions_path"],
            base_path=os.path.dirname(os.path.dirname(result["actions_path"])),
        )


def _disable_wandb_after_failure(*, context: str, exc: BaseException) -> None:
    print(
        f"[eval] warning: disabling W&B logging after {context} failed: {exc!r}",
        flush=True,
    )


def try_log_episode_to_wandb(
    *,
    run,
    result: dict[str, Any],
    completed_results: list[dict[str, Any]] | None,
    metric_steps: list[int],
    threshold_m: float,
    max_gt_points_vis: int,
    seed: int,
    log_step: int,
    log_video: bool,
):
    if run is None:
        return None
    try:
        log_episode_to_wandb(
            result=result,
            completed_results=completed_results,
            metric_steps=metric_steps,
            threshold_m=threshold_m,
            max_gt_points_vis=max_gt_points_vis,
            seed=seed,
            log_step=log_step,
            log_video=log_video,
        )
        return run
    except Exception as exc:
        _disable_wandb_after_failure(context="episode logging", exc=exc)
        return None


def try_log_running_aggregate_to_wandb(
    *,
    run,
    results: list[dict[str, Any]],
    metric_steps: list[int],
    log_step: int,
):
    if run is None:
        return None
    try:
        log_running_aggregate_to_wandb(
            results=results,
            metric_steps=metric_steps,
            log_step=log_step,
        )
        return run
    except Exception as exc:
        _disable_wandb_after_failure(context="running aggregate logging", exc=exc)
        return None


def try_sync_progress_files_to_wandb(
    *,
    run,
    output_root: str,
    progress_dir: str,
    per_episode_json_path: str,
    aggregate_json_path: str,
    summary_path: str,
    include_live_snapshots: bool = True,
):
    if run is None:
        return None
    try:
        sync_progress_files_to_wandb(
            output_root=output_root,
            progress_dir=progress_dir,
            per_episode_json_path=per_episode_json_path,
            aggregate_json_path=aggregate_json_path,
            summary_path=summary_path,
            include_live_snapshots=include_live_snapshots,
        )
        return run
    except Exception as exc:
        _disable_wandb_after_failure(context="artifact sync", exc=exc)
        return None


def try_log_final_aggregate_to_wandb(
    *,
    run,
    aggregate: dict[str, Any],
    results: list[dict[str, Any]],
):
    if run is None:
        return None

    import wandb

    try:
        for step, values in aggregate["steps"].items():
            run.summary[f"aggregate/mean_distance_m@{step}"] = values["mean_distance_m"]
            run.summary[f"aggregate/score@{step}"] = values["ratio_within_threshold"]
        run.summary["aggregate/final_mean_distance_m"] = aggregate["final"]["mean_distance_m"]
        run.summary["aggregate/final_score"] = aggregate["final"]["ratio_within_threshold"]
        wandb.log(
            {
                "aggregate/final_mean_distance_m": aggregate["final"]["mean_distance_m"],
                "aggregate/final_score": aggregate["final"]["ratio_within_threshold"],
            },
            step=len(results) + 1,
        )
        return run
    except Exception as exc:
        _disable_wandb_after_failure(context="final aggregate logging", exc=exc)
        return None


def try_annotate_wandb_run_metadata(
    *,
    run,
    checkpoint_path: str,
    checkpoint_step: int | None = None,
    episodes_json: str | None = None,
    extra_summary_fields: dict[str, Any] | None = None,
):
    if run is None:
        return None
    try:
        run.summary["eval/checkpoint_path"] = os.path.abspath(checkpoint_path)
        run.summary["eval/checkpoint_name"] = os.path.basename(checkpoint_path)
        if checkpoint_step is not None:
            run.summary["eval/checkpoint_step"] = int(checkpoint_step)
        if episodes_json is not None:
            run.summary["eval/episodes_json"] = os.path.abspath(episodes_json)
        if extra_summary_fields:
            for key, value in extra_summary_fields.items():
                run.summary[str(key)] = _normalize_seed_key_part(value)
        return run
    except Exception as exc:
        _disable_wandb_after_failure(context="run metadata annotation", exc=exc)
        return None


def try_finish_wandb(*, rank: int) -> None:
    try:
        import wandb

        wandb.finish()
    except Exception as exc:
        if rank == 0:
            print(f"[eval] warning: wandb.finish() failed: {exc!r}", flush=True)


def install_graceful_shutdown_handler(*, rank: int) -> None:
    """Handle SIGTERM so wandb.finish() runs before the process exits.

    Without this, torchrun kills workers on cancel/preemption and wandb
    never gets a clean shutdown - causing the run to show as "crashed".
    """
    original_handler = signal.getsignal(signal.SIGTERM)

    def _handler(signum, frame):
        print(f"[eval rank={rank}] received SIGTERM, finishing wandb gracefully", flush=True)
        try_finish_wandb(rank=rank)
        if callable(original_handler) and original_handler not in (
            signal.SIG_DFL,
            signal.SIG_IGN,
        ):
            original_handler(signum, frame)
        else:
            raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handler)


def prepare_reachable_gt_points_with_sim(
    scene_path: str,
    env,
    island_index: int,
    n_samples: int,
    *,
    disable_gt_filtering: bool = False,
    disable_island_filter: bool = False,
    mesh_sample_seed: int | None = None,
    scene_filter_state: SceneFilterState | None = None,
    scene_filter_diagnostics: dict[str, Any] | None = None,
    validate_alignment: bool = False,
    initial_obs: dict[str, Any] | None = None,
    depth_stride: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Prepare GT points ensuring consistency with Habitat observations.

    Filtering pipeline:
      1. Load mesh and transform to Habitat scene coordinates.
      2. Build a reachable scene island from the sweep map.
      3. Probe the island plane with vertical rays and estimate floor/ceiling mode heights.
      4. Sample GT mesh points, keep points inside that height band, and keep only
         points whose vertical projection lands on the chosen island.
    """
    mesh = load_mesh_for_completeness(scene_path)
    mesh_scene, transform_metadata = mesh_to_habitat_scene(
        mesh,
        sim=env.sim,
        pathfinder=env.sim.pathfinder,
        island_index=island_index,
        return_metadata=True,
    )
    mesh_vertices = np.asarray(mesh_scene.vertices, dtype=np.float64)
    mesh_faces = getattr(mesh_scene, "faces", None)
    if mesh_faces is None:
        mesh_faces = getattr(mesh_scene, "triangles", None)
    mesh_faces_arr = (
        np.empty((0, 3), dtype=np.int64)
        if mesh_faces is None
        else np.asarray(mesh_faces, dtype=np.int64)
    )

    gt_sampling_stats: dict[str, Any] = {
        "scene_path": scene_path,
        "gt_filter_mode": "none" if disable_gt_filtering else "filtered",
        "mesh_sample_seed": None if mesh_sample_seed is None else int(mesh_sample_seed),
        "mesh_transform_mode": transform_metadata.get("mode"),
        "mesh_vertex_count": int(len(mesh_vertices)),
        "mesh_face_count": int(len(mesh_faces_arr)),
        "requested_gt_samples": int(n_samples),
        "scene_filter": _normalize_seed_key_part(scene_filter_diagnostics or {}),
        "sample_calls": [],
        "stage_counts": [
            _make_gt_stage_entry("mesh_vertices", int(len(mesh_vertices))),
            _make_gt_stage_entry("mesh_faces", int(len(mesh_faces_arr))),
        ],
    }

    def _record_sample_call(label: str, diagnostics: dict[str, Any]) -> None:
        gt_sampling_stats["sample_calls"].append(
            {
                "label": label,
                **_normalize_seed_key_part(diagnostics),
            }
        )
        for stage in diagnostics.get("stage_counts", []):
            stage_name = str(stage.get("stage", "unknown"))
            if stage_name in {"mesh_vertices", "mesh_faces"}:
                continue
            gt_sampling_stats["stage_counts"].append(
                _make_gt_stage_entry(
                    f"{label}/{stage_name}",
                    int(stage.get("count", 0)),
                    **{key: value for key, value in stage.items() if key not in {"stage", "count"}},
                )
            )

    if disable_gt_filtering:
        gt_points, sample_diagnostics = sample_mesh_and_filter_with_scene_island(
            mesh_scene,
            None,
            n_samples=n_samples,
            filter_by_island=False,
            seed=mesh_sample_seed,
            return_diagnostics=True,
        )
        _record_sample_call("unfiltered_final", sample_diagnostics)
        inferred_island_index = -1
        gt_sampling_stats["inferred_island_index"] = int(inferred_island_index)
    else:
        if scene_filter_state is None:
            raise ValueError("scene filtering requires scene_filter_state")

        inferred_island_index = int(scene_filter_state.component_label)
        if not disable_island_filter and inferred_island_index < 0:
            raise RuntimeError(
                f"[eval] invalid scene component label={inferred_island_index}"
            )
        gt_sampling_stats["inferred_island_index"] = (
            int(inferred_island_index) if not disable_island_filter else -1
        )
        gt_points, oracle_diagnostics = sample_mesh_and_filter_with_scene_island(
            mesh_scene,
            scene_filter_state,
            sim=env.sim,
            n_samples=n_samples,
            filter_by_island=not disable_island_filter,
            seed=mesh_sample_seed,
            return_diagnostics=True,
        )
        _record_sample_call(
            "final_height_band_and_island_projection"
            if not disable_island_filter
            else "final_height_band_no_island_projection",
            oracle_diagnostics,
        )
        if disable_island_filter:
            inferred_island_index = -1

    gt_points = np.asarray(gt_points, dtype=np.float32)
    gt_sampling_stats["stage_counts"].append(
        _make_gt_stage_entry("pipeline_final_gt_points", int(len(gt_points)))
    )
    gt_sampling_stats["final_gt_points_count"] = int(len(gt_points))

    metadata = {
        "transform": transform_metadata,
        "num_gt_points": len(gt_points),
        "scene_path": scene_path,
        "gt_filter_mode": "none" if disable_gt_filtering else "filtered",
        "inferred_island_index": int(inferred_island_index),
        "mesh_sample_seed": None if mesh_sample_seed is None else int(mesh_sample_seed),
        "gt_sampling_stats": gt_sampling_stats,
    }
    if scene_filter_state is not None:
        metadata["scene_filter_component_label"] = int(scene_filter_state.component_label)

    if validate_alignment and initial_obs is not None:
        observed_points = unproject_observed_points_from_obs(initial_obs, depth_stride=depth_stride)
        validation_result = validate_mesh_observation_alignment(
            mesh_scene.vertices,
            observed_points,
            tolerance_m=0.5,
        )
        metadata["alignment_validation"] = validation_result
        if not validation_result.get("valid", False):
            print(
                f"[WARNING] Mesh-observation alignment validation failed: "
                f"within_tolerance={validation_result.get('within_tolerance_ratio', 0):.2%} "
                f"(need >50%)",
                flush=True,
            )

    return gt_points, metadata


def get_gt_points(
    *,
    episode: dict[str, Any],
    env,
    scene_path: str,
    gt_cache: dict[tuple[Any, ...], tuple[np.ndarray, dict[str, Any]]],
    n_gt_samples: int,
    mesh_sampling_base_seed: int | None = None,
    disable_gt_filtering: bool = False,
    disable_island_filter: bool = False,
    validate_alignment: bool = False,
    initial_obs: dict[str, Any] | None = None,
    depth_stride: int = 2,
) -> tuple[np.ndarray, int, dict[str, Any] | None]:
    """Get GT points for an episode, using cache if available."""
    from modules.environment.env import AGENT_HEIGHT, AGENT_RADIUS

    start_position = np.asarray(episode["start_position"], dtype=np.float64)
    metadata_island_index = episode.get("island_index")
    scene_filter_state: SceneFilterState | None = None
    scene_filter_diagnostics: dict[str, Any] | None = None

    if disable_gt_filtering:
        filter_island_index = -1
        navmesh_island_index = -1 if metadata_island_index is None else int(metadata_island_index)
        start_y = float(start_position[1])
        floor_diagnostics = {
            "mode": "disabled_gt_filtering",
            "island_index": int(navmesh_island_index),
            "floor_y": float(start_y),
        }
    else:
        scene_filter_state, scene_filter_diagnostics = infer_scene_filter_from_sweep(
            env.sim,
            env.sim.pathfinder,
            start_position=start_position,
            camera_height_m=AGENT_HEIGHT,
            agent_radius_m=AGENT_RADIUS,
        )
        filter_island_index = int(scene_filter_state.component_label)
        start_y = float(start_position[1])
        if metadata_island_index is not None:
            navmesh_island_index = int(metadata_island_index)
        else:
            navmesh_island_index = _infer_navmesh_island_from_start(
                env.sim.pathfinder,
                start_position,
            )
        floor_diagnostics = {
            **(scene_filter_diagnostics or {}),
            "mode": "scene_sweep_island_filter",
            "island_index": int(filter_island_index),
            "scene_component_label": int(filter_island_index),
            "navmesh_island_index": int(navmesh_island_index),
            "floor_y": float(start_y),
        }

    filter_island_index = int(filter_island_index)
    navmesh_island_index = int(navmesh_island_index)
    start_y = float(start_y)
    start_position_key = (
        tuple(np.round(start_position.astype(np.float64), 3).tolist())
        if not disable_gt_filtering
        else None
    )
    sampling_key = (
        scene_path,
        -1 if disable_gt_filtering else filter_island_index,
        bool(disable_gt_filtering),
        bool(disable_island_filter),
        start_position_key,
    )
    mesh_sample_seed = derive_mesh_sampling_seed(mesh_sampling_base_seed, sampling_key)
    cache_key = (*sampling_key, mesh_sampling_base_seed)

    if cache_key in gt_cache:
        cached_points, cached_metadata = gt_cache[cache_key]
        cached_metadata = dict(cached_metadata)
        _validate_gt_points_or_raise(
            episode_id=str(episode["episode_id"]),
            gt_points=cached_points,
            metadata=cached_metadata,
        )
        return cached_points, navmesh_island_index, cached_metadata

    gt_points, metadata = prepare_reachable_gt_points_with_sim(
        scene_path,
        env,
        filter_island_index,
        n_gt_samples,
        mesh_sample_seed=mesh_sample_seed,
        disable_gt_filtering=disable_gt_filtering,
        disable_island_filter=disable_island_filter,
        scene_filter_state=scene_filter_state,
        scene_filter_diagnostics=scene_filter_diagnostics,
        validate_alignment=validate_alignment,
        initial_obs=initial_obs,
        depth_stride=depth_stride,
    )
    gt_sampling_stats = metadata.get("gt_sampling_stats", {})
    if scene_filter_state is not None:
        floor_diagnostics["scene_component_label"] = int(scene_filter_state.component_label)
    local_floor_good_samples = floor_diagnostics.get("good_sample_count")
    if local_floor_good_samples is not None:
        gt_sampling_stats.setdefault("stage_counts", []).append(
            _make_gt_stage_entry(
                "local_floor_good_samples",
                int(local_floor_good_samples),
            )
        )
    metadata["floor_inference"] = floor_diagnostics
    metadata["navmesh_island_index"] = int(navmesh_island_index)
    metadata["metadata_island_index"] = (
        None if metadata_island_index is None else int(metadata_island_index)
    )
    metadata["scene_filter_visualizations"] = render_scene_filter_debug_images(scene_filter_state)
    metadata.setdefault("gt_sampling_stats", {})["floor_inference"] = floor_diagnostics
    metadata["gt_sampling_stats"]["metadata_island_index"] = (
        None if metadata_island_index is None else int(metadata_island_index)
    )

    _validate_gt_points_or_raise(
        episode_id=str(episode["episode_id"]),
        gt_points=gt_points,
        metadata=metadata,
    )

    gt_cache[cache_key] = (gt_points, dict(metadata))
    return gt_points, navmesh_island_index, metadata


__all__ = [
    "_disable_wandb_after_failure",
    "_format_gt_stage_counts",
    "_make_gt_stage_entry",
    "_normalize_seed_key_part",
    "_validate_gt_points_or_raise",
    "aggregate_results",
    "broadcast_object",
    "derive_mesh_sampling_seed",
    "derive_navmesh_path",
    "destroy_process_group_quietly",
    "distributed_barrier",
    "extract_camera_position_from_obs",
    "get_gt_points",
    "init_distributed",
    "install_graceful_shutdown_handler",
    "load_episode_specs",
    "load_live_result_records",
    "load_single_live_result_file",
    "log_episode_to_wandb",
    "log_running_aggregate_to_wandb",
    "make_gt_point_cloud_object",
    "make_running_plots",
    "make_timestamped_run_name",
    "merge_eval_results_from_shared_storage",
    "merge_incremental_live_results",
    "normalize_episode_paths",
    "pack_point_cloud_history_frames",
    "patch_wandb_apikey",
    "prepare_reachable_gt_points_with_sim",
    "render_terminal_eval_frame_without_camera_beam",
    "remap_hm3d_scene_path",
    "remap_scene_path",
    "render_point_cloud_topdown_progress_video",
    "resolve_hm3d_stage_path",
    "save_episode_media",
    "select_point_cloud_visualization_subset",
    "select_eval_episodes",
    "select_history_frame_indices",
    "shard_episodes",
    "sync_progress_files_to_wandb",
    "try_finish_wandb",
    "try_annotate_wandb_run_metadata",
    "try_log_episode_to_wandb",
    "try_log_final_aggregate_to_wandb",
    "try_log_running_aggregate_to_wandb",
    "try_sync_progress_files_to_wandb",
    "unproject_observed_points_from_obs",
    "write_live_result_record",
    "write_progress_jsons",
    "write_rank_failure_trace",
]
