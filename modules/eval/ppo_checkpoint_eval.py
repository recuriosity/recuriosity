from __future__ import annotations

"""Evaluate PPO checkpoints on fixed episode specs (HM3D and Gibson-supported paths).

Execution model:
- Single-process launch: one GPU, one episode rollout at a time.
- `torchrun --nproc_per_node=N`: N worker processes, typically one per GPU.
  Episodes are sharded by global episode index, each worker runs its shard
  serially, and rank 0 logs all gathered results into one W&B run.
- Multi-GPU merges results by reading shared ``live_results/episode_*.json`` after
  a barrier (no ``all_gather_object``), so slow ranks do not trigger NCCL timeouts.
"""

import argparse
import gc
import importlib
import os
import random
import sys
import time
from contextlib import nullcontext
from typing import Any

import numpy as np

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_MODULE_DIR))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


from modules.eval.completeness_utils import RunningCompletenessEvaluator
from modules.eval.backface_mesh_fix import ensure_double_sided_glb_cached
from modules.eval.eval_utils import (
    _disable_wandb_after_failure,
    _format_gt_stage_counts,
    _make_gt_stage_entry,
    _normalize_seed_key_part,
    aggregate_results,
    derive_navmesh_path,
    destroy_process_group_quietly,
    distributed_barrier,
    extract_camera_position_from_obs,
    get_gt_points,
    init_distributed,
    install_graceful_shutdown_handler,
    load_episode_specs,
    make_timestamped_run_name,
    merge_eval_results_from_shared_storage,
    merge_incremental_live_results,
    pack_point_cloud_history_frames,
    patch_wandb_apikey,
    render_terminal_eval_frame_without_camera_beam,
    resolve_hm3d_stage_path,
    save_episode_media,
    select_point_cloud_visualization_subset,
    select_eval_episodes,
    shard_episodes,
    try_finish_wandb,
    try_annotate_wandb_run_metadata,
    try_log_episode_to_wandb,
    try_log_final_aggregate_to_wandb,
    try_log_running_aggregate_to_wandb,
    try_sync_progress_files_to_wandb,
    unproject_observed_points_from_obs,
    write_live_result_record,
    write_progress_jsons,
    write_rank_failure_trace,
)
from scripts.dl_hm3d_data import resolve_hm3d_root


def normalize_ppo_module_spec(module_spec: str) -> str:
    spec = module_spec.strip()
    if not spec:
        raise ValueError("PPO module spec must not be empty")

    if spec.endswith(".py"):
        spec = spec[:-3]
    if os.path.sep in spec:
        if os.path.exists(spec):
            spec = os.path.relpath(os.path.abspath(spec), _ROOT)
        spec = spec.replace(os.path.sep, ".")
    return spec.lstrip(".")


def load_ppo_module(module_spec: str):
    module_name = normalize_ppo_module_spec(module_spec)
    return module_name, importlib.import_module(module_name)


def infer_policy_in_channels(*, pre, device, hw: int) -> int:
    import torch

    dummy_images = torch.zeros((1, 1, 3, hw, hw), device=device, dtype=torch.float32)
    dummy_ray_o = torch.zeros((1, 1, 3, hw, hw), device=device, dtype=torch.float32)
    dummy_ray_d = torch.zeros((1, 1, 3, hw, hw), device=device, dtype=torch.float32)
    current_view = pre.build_visual_input(dummy_images, dummy_ray_o, dummy_ray_d)
    return int(current_view.shape[2] + 6)


def select_eval_amp_dtype(device):
    import torch

    if device.type != "cuda" or not torch.cuda.is_available():
        return torch.float32

    major, minor = torch.cuda.get_device_capability(device)
    # bf16 attention kernels are reliably available on SM80 A100 and newer Hopper.
    # Older cards (e.g. T4, A10) should use fp16 for xformers compatibility.
    if major > 8 or (major == 8 and minor == 0):
        return torch.bfloat16
    return torch.float16


def build_components(ppo_module, *, device, max_steps: int, hw: int, attn_window: int):
    import torch

    from modules.environment.env import STEP_METERS, YAW_DEG

    # `pre` inside PPO main() is just a local instance. We create the same instance here
    # from the module-level PoseProcess class exported by the PPO module.
    pre = ppo_module.PoseProcess(out_hw=hw, scene_scale_factor=1.35)
    policy_in_channels = infer_policy_in_channels(pre=pre, device=device, hw=hw)
    agent = ppo_module.NavAgent(
        image_size=hw,
        patch_size=32,
        in_channels=policy_in_channels,
        d_model=768,
        d_head=64,
        n_layer_spatial_cross=4,
        n_layer_temporal=24,
        use_qk_norm=True,
        n_act=4,
        max_v=int(max_steps) + 1,
        attn_window=attn_window,
        checkpoint_every=0,
        dino_model_name="dinov2_vitb14",
        dino_freeze=True,
        frame_token_mode="learnable_query",
    ).to(device)
    action_T = torch.stack(
        [
            ppo_module.se3_from_translation_rotation(dz=STEP_METERS, device=device),
            ppo_module.se3_from_translation_rotation(yaw_deg=YAW_DEG, device=device),
            ppo_module.se3_from_translation_rotation(yaw_deg=-YAW_DEG, device=device),
            torch.eye(4, dtype=torch.float32).to(device),
        ],
        dim=0,
    )
    return pre, agent, action_T, select_eval_amp_dtype(device)


def build_policy_input(
    *,
    ppo_module,
    pre,
    agent,
    action_T,
    amp_dtype,
    rgb,
    k,
    c2w,
    rgb_ref,
    k_ref,
    c2w_ref,
    prev_action,
    step_idx: int,
    device,
):
    import torch

    rgb_win = torch.stack([rgb_ref, rgb], dim=0).unsqueeze(0)
    k_win = torch.stack([k_ref, k], dim=0).unsqueeze(0)
    c2w_win = torch.stack([c2w_ref, c2w], dim=0).unsqueeze(0)

    rgb_n, k_n, c2w_n, _ = pre.process_window(
        rgb_win,
        k_win,
        c2w_win,
        device=device,
        amp_dtype=amp_dtype,
        outHW=pre.out_hw,
    )
    rgb_n = rgb_n[:, -1:]
    k_n = k_n[:, -1:]
    c2w_n = c2w_n[:, -1:]

    ray_o, ray_d = pre.compute_rays(c2w_n, k_n, h=pre.out_hw, w=pre.out_hw, device=device)
    current_view = pre.build_visual_input(rgb_n, ray_o, ray_d)
    action_pose = ppo_module.action_pose_from_indices(
        prev_action.unsqueeze(1),
        action_T,
        k_n,
        pre.out_hw,
        device,
        pre,
        zero_first_step=(step_idx == 1),
    )
    policy_input = torch.cat([current_view, action_pose], dim=2).squeeze(1)

    expected_channels = int(agent.in_channels)
    if policy_input.shape[1] != expected_channels:
        raise RuntimeError(
            f"Built {policy_input.shape[1]} policy channels, but agent expects {expected_channels}. "
            "For the no-camera-pose module this should be 9 = RGB(3) + action pose(6)."
        )
    return policy_input


def evaluate_episode(
    *,
    episode: dict[str, Any],
    progress_label: str,
    ppo_module,
    agent,
    pre,
    action_T,
    device,
    amp_dtype,
    max_steps: int,
    metric_steps: list[int],
    threshold_m: float,
    depth_stride: int,
    nn_workers: int | None,
    nn_backend: str,
    torch_nn_query_chunk_size: int,
    torch_nn_reference_chunk_size: int,
    gpu_id: int,
    greedy: bool,
    uniform_actions: bool,
    video_dir: str,
    gt_cache: dict[
        tuple[str, int, float, bool, bool, tuple[float, float, float] | None, int | None],
        tuple[np.ndarray, dict[str, Any]],
    ],
    n_gt_samples: int,
    log_point_cloud: bool,
    max_gt_points_vis: int,
    point_cloud_seed: int,
    mesh_sampling_base_seed: int,
    disable_gt_filtering: bool = False,
    disable_island_filter: bool = False,
    validate_alignment: bool = False,
    check_holes: bool = False,
    eval_hole_fix: bool = False,
    eval_hole_fix_cache_dir: str | None = None,
    eval_hole_fix_force_white: bool = False,
    eval_hole_fix_backface_black: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import imageio.v2 as imageio
    import torch
    from torch.distributions.categorical import Categorical

    from modules.environment.env import HabitatMP3DEnv
    from modules.environment.render_log_utils import render_topdown

    scene_path = os.path.abspath(episode["scene_id"])
    navmesh_path = derive_navmesh_path(scene_path)
    start_position = np.asarray(episode["start_position"], dtype=np.float64)
    scene_glb_path = resolve_hm3d_stage_path(scene_path, navmesh_path)
    if eval_hole_fix:
        fixed = ensure_double_sided_glb_cached(
            scene_glb_path,
            cache_dir=eval_hole_fix_cache_dir,
            force_white=bool(eval_hole_fix_force_white),
            backface_black=bool(eval_hole_fix_backface_black),
        )
        scene_glb_path = fixed["output_path"]
        cache_status = "cache_hit" if bool(fixed.get("reused_existing")) else "rewritten"
        print(
            f"[eval] {progress_label} eval_hole_fix status={cache_status} "
            f"src={fixed['source_path']} dst={fixed['output_path']}",
            flush=True,
        )
    scene = {
        "scene_name": episode["scene_name"],
        "glb_path": scene_glb_path,
        "navmesh": navmesh_path,
    }

    env = HabitatMP3DEnv(
        [scene],
        max_steps=int(max_steps) + 1,
        render_mode="rgb_array",
        gpu_id=gpu_id,
        check_holes=check_holes,
    )
    env._eval_panel_layout = True

    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"{episode['episode_id']}.mp4")
    writer = imageio.get_writer(video_path, fps=12, codec="libx264", bitrate="4M")

    kv_caches = agent.init_kv_cache(batch_size=1)
    prev_action = torch.zeros(1, device=device, dtype=torch.long)
    metric_steps_set = set(int(step) for step in metric_steps)
    timing_totals: dict[str, float] = {}
    timing_counts: dict[str, int] = {}
    episode_t0 = time.perf_counter()
    point_cloud_vis_points = None
    point_cloud_vis_indices = None
    point_cloud_history_steps: list[int] = []
    point_cloud_history_dists: list[np.ndarray] = []
    point_cloud_history_observed_points: list[np.ndarray] = []
    point_cloud_history_camera_positions: list[np.ndarray] = []
    episode_camera_positions: list[np.ndarray] = []
    action_labels = ("F", "L", "R", "S")
    actions_trace: list[dict[str, Any]] = []

    try:
        t0 = time.perf_counter()
        obs, _ = env.reset(
            options={
                "restore_state": {
                    "scene": scene,
                    "position": start_position.astype(np.float32, copy=False),
                    "rotation": np.asarray(episode["start_rotation"], dtype=np.float32),
                }
            }
        )
        timing_totals["setup/env_reset"] = timing_totals.get("setup/env_reset", 0.0) + (time.perf_counter() - t0)
        timing_counts["setup/env_reset"] = timing_counts.get("setup/env_reset", 0) + 1
        start_rgb = np.asarray(obs["rgb"], dtype=np.uint8)
        episode_camera_positions.append(extract_camera_position_from_obs(obs))

        t0 = time.perf_counter()
        gt_points, island_index, gt_metadata = get_gt_points(
            episode=episode,
            env=env,
            scene_path=scene_path,
            gt_cache=gt_cache,
            n_gt_samples=n_gt_samples,
            mesh_sampling_base_seed=mesh_sampling_base_seed,
            disable_gt_filtering=disable_gt_filtering,
            disable_island_filter=disable_island_filter,
            validate_alignment=validate_alignment,
            initial_obs=obs,
            depth_stride=depth_stride,
        )
        timing_totals["setup/get_gt_points"] = timing_totals.get("setup/get_gt_points", 0.0) + (time.perf_counter() - t0)
        timing_counts["setup/get_gt_points"] = timing_counts.get("setup/get_gt_points", 0) + 1
        gt_metadata = dict(gt_metadata or {})
        gt_metadata["gt_sampling_stats"] = _normalize_seed_key_part(gt_metadata.get("gt_sampling_stats", {}))
        
        # Log transform metadata on first encounter of this scene
        if gt_metadata is not None:
            transform_info = gt_metadata.get("transform", {})
            floor_info = gt_metadata.get("floor_inference", {})
            gt_sampling_stats = gt_metadata.get("gt_sampling_stats", {})
            floor_alignment_error = transform_info.get("floor_alignment_error")
            if floor_alignment_error is None:
                floor_alignment_error = float("nan")
            print(
                f"[eval] {progress_label} mesh_transform "
                f"mode={transform_info.get('mode', 'unknown')} "
                f"rotation={transform_info.get('rotation_name', 'unknown')} "
                f"floor_align_err={floor_alignment_error:.4f}m "
                f"extent_err={transform_info.get('extent_error', float('nan')):.4f}m "
                f"reachable_hits={transform_info.get('reachable_hits', -1)}",
                flush=True,
            )
            if floor_info:
                print(
                    f"[eval] {progress_label} floor_inference "
                    f"mode={floor_info.get('mode', 'unknown')} "
                    f"island_index={floor_info.get('island_index', -1)} "
                    f"floor_y={floor_info.get('floor_y', float('nan')):.3f} "
                    f"support={floor_info.get('winning_support_count', 0)} "
                    f"accepted={floor_info.get('accepted_candidate_count', 0)} "
                    f"blocked={floor_info.get('rejected_blocked_count', 0)}",
                    flush=True,
                )
            # Print bounds comparison for debugging alignment
            mesh_min = transform_info.get('transformed_bounds_min')
            mesh_max = transform_info.get('transformed_bounds_max')
            nav_min = transform_info.get('navmesh_bounds_min')
            nav_max = transform_info.get('navmesh_bounds_max')
            if mesh_min is not None and nav_min is not None:
                print(
                    f"[eval] {progress_label} bounds_check "
                    f"mesh=[{mesh_min[0]:.2f},{mesh_min[1]:.2f},{mesh_min[2]:.2f}]-"
                    f"[{mesh_max[0]:.2f},{mesh_max[1]:.2f},{mesh_max[2]:.2f}] "
                    f"nav=[{nav_min[0]:.2f},{nav_min[1]:.2f},{nav_min[2]:.2f}]-"
                    f"[{nav_max[0]:.2f},{nav_max[1]:.2f},{nav_max[2]:.2f}]",
                    flush=True,
                )
            if "alignment_validation" in gt_metadata:
                av = gt_metadata["alignment_validation"]
                print(
                    f"[eval] {progress_label} alignment_validation "
                    f"valid={av.get('valid', False)} "
                    f"within_tol={av.get('within_tolerance_ratio', 0):.2%} "
                    f"mean_dist={av.get('mean_distance_m', float('nan')):.3f}m",
                    flush=True,
                )
            if gt_sampling_stats:
                print(
                    f"[eval] {progress_label} gt_sampling "
                    f"mode={gt_sampling_stats.get('gt_filter_mode', 'unknown')} "
                    f"seed={gt_sampling_stats.get('mesh_sample_seed', 'none')} "
                    f"stages={_format_gt_stage_counts(gt_sampling_stats.get('stage_counts', []))}",
                    flush=True,
                )
        
        point_cloud_vis_points, point_cloud_vis_indices = select_point_cloud_visualization_subset(
            gt_points,
            max_points=max_gt_points_vis,
            seed=point_cloud_seed,
        )
        t0 = time.perf_counter()
        evaluator = RunningCompletenessEvaluator(
            gt_points,
            threshold_m=threshold_m,
            nn_workers=nn_workers,
            nn_backend=nn_backend,
            torch_device=str(device) if device.type == "cuda" else None,
            torch_query_chunk_size=torch_nn_query_chunk_size,
            torch_reference_chunk_size=torch_nn_reference_chunk_size,
            track_history=True,
        )
        timing_totals["setup/init_evaluator"] = timing_totals.get("setup/init_evaluator", 0.0) + (time.perf_counter() - t0)
        timing_counts["setup/init_evaluator"] = timing_counts.get("setup/init_evaluator", 0) + 1

        t0 = time.perf_counter()
        initial_observed_points = unproject_observed_points_from_obs(obs, depth_stride=depth_stride)
        initial_depth_to_world_s = time.perf_counter() - t0
        timing_totals["setup/initial_depth_to_world"] = timing_totals.get("setup/initial_depth_to_world", 0.0) + initial_depth_to_world_s
        timing_counts["setup/initial_depth_to_world"] = timing_counts.get("setup/initial_depth_to_world", 0) + 1
        t0 = time.perf_counter()
        evaluator.update_observed_points(initial_observed_points, compute_metrics=False)
        timing_totals["setup/initial_completeness"] = timing_totals.get("setup/initial_completeness", 0.0) + (time.perf_counter() - t0)
        timing_counts["setup/initial_completeness"] = timing_counts.get("setup/initial_completeness", 0) + 1
        initial_progress_min_dists = None
        if point_cloud_vis_indices is not None and point_cloud_vis_points is not None:
            initial_progress_min_dists = evaluator.sample_min_distances(point_cloud_vis_indices).astype(
                np.float32, copy=False
            )
            point_cloud_history_steps.append(0)
            point_cloud_history_dists.append(initial_progress_min_dists)
            point_cloud_history_observed_points.append(
                initial_observed_points.astype(np.float32, copy=False)
            )
            point_cloud_history_camera_positions.append(extract_camera_position_from_obs(obs))
        initial_comp_debug = dict(evaluator.last_update_debug)
        print(
            f"[eval] {progress_label} completeness_setup "
            f"gt_pts={int(initial_comp_debug.get('gt_points', len(gt_points)))} "
            f"obs_pts0={int(initial_comp_debug.get('observed_points', 0))} "
            f"depth0={initial_depth_to_world_s:.3f}s "
            f"depth_stride={int(depth_stride)} "
            f"nn_workers={initial_comp_debug.get('nn_workers', 'default')} "
            f"nn={str(initial_comp_debug.get('nn_backend', 'unknown'))} "
            f"nn_dev={str(initial_comp_debug.get('torch_device', '')) or 'cpu'} "
            f"nn_qchunk={int(initial_comp_debug.get('torch_query_chunk_size', 0))} "
            f"nn_rchunk={int(initial_comp_debug.get('torch_reference_chunk_size', 0))}",
            flush=True,
        )

        t0 = time.perf_counter()
        env.update_meta(
                [
                    {
                        "rgb_gt": np.asarray(obs["rgb"], dtype=np.float32) / 255.0,
                        "mode": int(2 if uniform_actions else 3),
                        "episode_island_index": island_index,
                        "point_cloud_topdown_progress": (
                        {
                            "points": point_cloud_vis_points,
                            "min_distances_m": initial_progress_min_dists,
                            "threshold_m": float(threshold_m),
                            "step": 0,
                            "max_step": int(max_steps),
                            "observed_points": initial_observed_points.astype(
                                np.float32, copy=False
                            ),
                        }
                        if initial_progress_min_dists is not None
                        else None
                    ),
                }
            ]
        )
        initial_frame = np.asarray(env.render())
        if initial_frame.shape[-1] == 4:
            initial_frame = initial_frame[..., :3]
        writer.append_data(initial_frame)
        timing_totals["setup/initial_render_and_video"] = timing_totals.get(
            "setup/initial_render_and_video", 0.0
        ) + (time.perf_counter() - t0)
        timing_counts["setup/initial_render_and_video"] = timing_counts.get(
            "setup/initial_render_and_video", 0
        ) + 1

        t0 = time.perf_counter()
        rgb, k, c2w, _ = ppo_module.obs_to_img_pose(obs, device)
        timing_totals["setup/obs_to_img_pose"] = timing_totals.get("setup/obs_to_img_pose", 0.0) + (time.perf_counter() - t0)
        timing_counts["setup/obs_to_img_pose"] = timing_counts.get("setup/obs_to_img_pose", 0) + 1
        rgb_ref = rgb.clone()
        k_ref = k.clone()
        c2w_ref = c2w.clone()

        for step_idx in range(1, int(max_steps) + 1):
            t0 = time.perf_counter()
            policy_input = build_policy_input(
                ppo_module=ppo_module,
                pre=pre,
                agent=agent,
                action_T=action_T,
                amp_dtype=amp_dtype,
                rgb=rgb,
                k=k,
                c2w=c2w,
                rgb_ref=rgb_ref,
                k_ref=k_ref,
                c2w_ref=c2w_ref,
                prev_action=prev_action,
                step_idx=step_idx,
                device=device,
            )
            build_policy_input_s = time.perf_counter() - t0
            timing_totals["step/build_policy_input"] = timing_totals.get("step/build_policy_input", 0.0) + build_policy_input_s
            timing_counts["step/build_policy_input"] = timing_counts.get("step/build_policy_input", 0) + 1

            t0 = time.perf_counter()
            with torch.no_grad():
                with (
                    torch.cuda.amp.autocast(dtype=amp_dtype)
                    if device.type == "cuda"
                    else nullcontext()
                ):
                    logits, _ = agent.forward_step(
                        policy_input,
                        time_idx=step_idx - 1,
                        kv_caches=kv_caches,
                    )
            policy_forward_s = time.perf_counter() - t0
            timing_totals["step/policy_forward"] = timing_totals.get("step/policy_forward", 0.0) + policy_forward_s
            timing_counts["step/policy_forward"] = timing_counts.get("step/policy_forward", 0) + 1

            t0 = time.perf_counter()
            if uniform_actions:
                action = torch.randint(
                    0,
                    logits.shape[-1],
                    (logits.shape[0],),
                    device=device,
                    dtype=torch.long,
                )
            else:
                action = torch.argmax(logits, dim=-1) if greedy else Categorical(logits=logits).sample()
            prev_action = action
            action_index = int(action.item())
            actions_trace.append(
                {
                    "step": int(step_idx),
                    "action": action_index,
                    "action_name": (
                        action_labels[action_index]
                        if 0 <= action_index < len(action_labels)
                        else str(action_index)
                    ),
                }
            )
            select_action_s = time.perf_counter() - t0
            timing_totals["step/select_action"] = timing_totals.get("step/select_action", 0.0) + select_action_s
            timing_counts["step/select_action"] = timing_counts.get("step/select_action", 0) + 1

            t0 = time.perf_counter()
            obs, _, term, trunc, _ = env.step(action_index)
            env_step_s = time.perf_counter() - t0
            timing_totals["step/env_step"] = timing_totals.get("step/env_step", 0.0) + env_step_s
            timing_counts["step/env_step"] = timing_counts.get("step/env_step", 0) + 1
            t0 = time.perf_counter()
            observed_points = unproject_observed_points_from_obs(obs, depth_stride=depth_stride)
            depth_to_world_s = time.perf_counter() - t0
            timing_totals["step/depth_to_world"] = timing_totals.get("step/depth_to_world", 0.0) + depth_to_world_s
            timing_counts["step/depth_to_world"] = timing_counts.get("step/depth_to_world", 0) + 1

            t0 = time.perf_counter()
            evaluator.update_observed_points(observed_points, compute_metrics=False)
            completeness_update_s = time.perf_counter() - t0
            timing_totals["step/completeness_update"] = timing_totals.get("step/completeness_update", 0.0) + completeness_update_s
            timing_counts["step/completeness_update"] = timing_counts.get("step/completeness_update", 0) + 1
            comp_debug = dict(evaluator.last_update_debug)
            comp_depth_s = float(depth_to_world_s)
            comp_nn_s = float(comp_debug.get("nn_query_s", float("nan")))
            comp_min_s = float(comp_debug.get("running_min_update_s", float("nan")))
            comp_summary_s = float(comp_debug.get("summary_s", float("nan")))
            comp_obs_points = int(comp_debug.get("observed_points", -1))
            comp_gt_points = int(comp_debug.get("gt_points", len(gt_points)))
            comp_backend = str(comp_debug.get("nn_backend", "unknown"))
            current_progress_min_dists = None
            if point_cloud_vis_indices is not None:
                current_progress_min_dists = evaluator.sample_min_distances(
                    point_cloud_vis_indices
                ).astype(np.float32, copy=False)
                point_cloud_history_steps.append(int(step_idx))
                point_cloud_history_dists.append(current_progress_min_dists)
                point_cloud_history_observed_points.append(observed_points.astype(np.float32, copy=False))
                point_cloud_history_camera_positions.append(extract_camera_position_from_obs(obs))
            episode_camera_positions.append(extract_camera_position_from_obs(obs))

            t0 = time.perf_counter()
            env.update_meta(
                [
                    {
                        "rgb_gt": np.asarray(obs["rgb"], dtype=np.float32) / 255.0,
                        "policy_logits": logits[0].float().detach().cpu().numpy(),
                        "mode": int(2 if uniform_actions else 3),
                        "episode_island_index": island_index,
                        "point_cloud_topdown_progress": (
                            {
                                "points": point_cloud_vis_points,
                                "min_distances_m": current_progress_min_dists,
                                "threshold_m": float(threshold_m),
                                "step": int(step_idx),
                                "max_step": int(max_steps),
                                "observed_points": observed_points.astype(np.float32, copy=False),
                            }
                            if current_progress_min_dists is not None
                            else None
                        ),
                    }
                ]
            )
            frame = np.asarray(env.render())
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            writer.append_data(frame)
            render_and_video_s = time.perf_counter() - t0
            timing_totals["step/render_and_video"] = timing_totals.get("step/render_and_video", 0.0) + render_and_video_s
            timing_counts["step/render_and_video"] = timing_counts.get("step/render_and_video", 0) + 1

            t0 = time.perf_counter()
            rgb, k, c2w, _ = ppo_module.obs_to_img_pose(obs, device)
            obs_to_img_pose_s = time.perf_counter() - t0
            timing_totals["step/obs_to_img_pose"] = timing_totals.get("step/obs_to_img_pose", 0.0) + obs_to_img_pose_s
            timing_counts["step/obs_to_img_pose"] = timing_counts.get("step/obs_to_img_pose", 0) + 1
            step_total_s = (
                build_policy_input_s
                + policy_forward_s
                + select_action_s
                + env_step_s
                + completeness_update_s
                + render_and_video_s
                + obs_to_img_pose_s
            )
            print(
                f"[eval] {progress_label} step {step_idx}/{int(max_steps)} "
                f"total={step_total_s:.3f}s "
                f"fwd={policy_forward_s:.3f}s "
                f"env={env_step_s:.3f}s "
                f"comp={completeness_update_s:.3f}s "
                f"comp_depth={comp_depth_s:.3f}s "
                f"comp_nn={comp_nn_s:.3f}s "
                f"comp_min={comp_min_s:.3f}s "
                f"comp_sum={comp_summary_s:.3f}s "
                f"obs_pts={comp_obs_points} "
                f"gt_pts={comp_gt_points} "
                f"nn={comp_backend} "
                f"render={render_and_video_s:.3f}s",
                flush=True,
            )
            if term or trunc:
                break

        t0 = time.perf_counter()
        history_metrics = evaluator.materialize_history()
        timing_totals["final/materialize_history"] = timing_totals.get("final/materialize_history", 0.0) + (time.perf_counter() - t0)
        timing_counts["final/materialize_history"] = timing_counts.get("final/materialize_history", 0) + 1
        if history_metrics:
            final_metrics = history_metrics[-1]
        else:
            t0 = time.perf_counter()
            final_metrics = evaluator.summary()
            timing_totals["final/evaluator_summary"] = timing_totals.get("final/evaluator_summary", 0.0) + (time.perf_counter() - t0)
            timing_counts["final/evaluator_summary"] = timing_counts.get("final/evaluator_summary", 0) + 1
        running_history = [
            {
                "step": int(step_idx),
                "mean_distance_m": float(step_metrics.mean_distance_m),
                "ratio_within_threshold": float(step_metrics.ratio_within_threshold),
            }
            for step_idx, step_metrics in enumerate(history_metrics)
        ]
        metrics_by_step: dict[int, dict[str, float]] = {}
        for step_idx in metric_steps_set:
            if step_idx < len(history_metrics):
                step_metrics = history_metrics[step_idx]
                metrics_by_step[step_idx] = {
                    "mean_distance_m": float(step_metrics.mean_distance_m),
                    "ratio_within_threshold": float(step_metrics.ratio_within_threshold),
                    "num_frames": int(step_metrics.num_frames),
                    "num_observed_points": int(step_metrics.num_observed_points),
                }
        print(
            f"[eval] {progress_label} finished at step {int(final_metrics.num_frames)}/{int(max_steps)}",
            flush=True,
        )
        t0 = time.perf_counter()
        terminal_panel_frame = render_terminal_eval_frame_without_camera_beam(env)
        writer.append_data(terminal_panel_frame)
        topdown_image = render_topdown(env)
        final_rgb = np.asarray(obs["rgb"], dtype=np.uint8)
        panel_image = terminal_panel_frame
        timing_totals["final/render_outputs"] = timing_totals.get("final/render_outputs", 0.0) + (time.perf_counter() - t0)
        timing_counts["final/render_outputs"] = timing_counts.get("final/render_outputs", 0) + 1
        timing_totals["episode/total"] = timing_totals.get("episode/total", 0.0) + (time.perf_counter() - episode_t0)
        timing_counts["episode/total"] = timing_counts.get("episode/total", 0) + 1
        timing_summary = {
            stage: {
                "total_s": float(total_s),
                "count": int(timing_counts.get(stage, 0)),
                "mean_s": float(total_s / max(timing_counts.get(stage, 0), 1)),
            }
            for stage, total_s in sorted(timing_totals.items(), key=lambda item: item[1], reverse=True)
        }
        print(f"[eval] {progress_label} timing summary:", flush=True)
        for stage, values in timing_summary.items():
            print(
                f"[eval]   {stage}: total={values['total_s']:.3f}s "
                f"count={int(values['count'])} mean={values['mean_s']:.4f}s",
                flush=True,
            )

        media = {
            "start_rgb": start_rgb,
            "final_rgb": final_rgb,
            "topdown": topdown_image,
            "scene_filter_sweep_map": gt_metadata.get("scene_filter_visualizations", {}).get("sweep_map"),
            "scene_filter_closed_map": gt_metadata.get("scene_filter_visualizations", {}).get("closed_map"),
            "scene_filter_components_map": gt_metadata.get("scene_filter_visualizations", {}).get("components_map"),
            "panel": panel_image,
            "running_history": running_history,
            "actions_trace": actions_trace,
        }
        gt_sampling_stats = dict(gt_metadata.get("gt_sampling_stats", {}))
        completion_vis_count = int(len(point_cloud_vis_indices)) if point_cloud_vis_indices is not None else 0
        topdown_progress_count = (
            int(len(point_cloud_vis_points))
            if point_cloud_vis_points is not None and point_cloud_history_dists
            else 0
        )
        gt_sampling_stats["visualization_counts"] = {
            "gt_point_cloud_completion": int(completion_vis_count),
            "gt_point_cloud_topdown_progress": int(topdown_progress_count),
        }
        gt_sampling_stats["stage_counts"] = list(gt_sampling_stats.get("stage_counts", [])) + [
            _make_gt_stage_entry("visualization/gt_point_cloud_completion", int(completion_vis_count)),
            _make_gt_stage_entry("visualization/gt_point_cloud_topdown_progress", int(topdown_progress_count)),
        ]
        print(
            f"[eval] {progress_label} gt_visualization "
            f"completion_points={completion_vis_count} "
            f"topdown_progress_points={topdown_progress_count}",
            flush=True,
        )
        media["gt_sampling_stats"] = gt_sampling_stats
        if log_point_cloud:
            media["point_cloud_data"] = {
                "gt_points": gt_points,
                "min_distances_m": evaluator.min_distances_m.copy(),
                "vis_indices": point_cloud_vis_indices,
            }
        if point_cloud_vis_points is not None and point_cloud_history_dists:
            observed_points, observed_point_offsets = pack_point_cloud_history_frames(
                point_cloud_history_observed_points
            )
            media["point_cloud_progress_data"] = {
                "points": point_cloud_vis_points,
                "steps": np.asarray(point_cloud_history_steps, dtype=np.int32),
                "min_distance_history_m": np.stack(point_cloud_history_dists, axis=0),
                "observed_points": observed_points,
                "observed_point_offsets": observed_point_offsets,
                "camera_positions": np.stack(point_cloud_history_camera_positions, axis=0),
                "threshold_m": float(threshold_m),
            }

        return {
            "episode_id": str(episode["episode_id"]),
            "scene_name": str(episode["scene_name"]),
            "video_path": video_path,
            "metrics_by_step": metrics_by_step,
            "final_mean_distance_m": float(final_metrics.mean_distance_m),
            "final_ratio_within_threshold": float(final_metrics.ratio_within_threshold),
            "num_frames": int(final_metrics.num_frames),
            "num_observed_points": int(final_metrics.num_observed_points),
            "timing": timing_summary,
            "gt_sampling_stats": gt_sampling_stats,
        }, media
    finally:
        writer.close()
        env.close()
        agent.clear_kv_cache(kv_caches)
        if device.type == "cuda":
            torch.cuda.empty_cache()

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a PPO checkpoint on fixed HM3D episodes.")
    parser.add_argument("--ppo-module", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument(
        "--episodes-json",
        type=str,
        default=os.path.join(_ROOT, "data", "splits", "hm3d", "val", "val.json.gz"),
    )
    parser.add_argument("--output-dir", type=str, default=os.path.join(_ROOT, "eval_outputs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--metric-steps", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--threshold-m", type=float, default=0.05)
    parser.add_argument("--depth-stride", type=int, default=2)
    parser.add_argument("--n-gt-samples", type=int, default=200_000)
    parser.add_argument("--nn-workers", type=int, default=None)
    parser.add_argument(
        "--nn-backend",
        type=str,
        default="torch_cuda_exact",
        choices=["auto", "scipy_ckdtree", "torch_cuda_exact", "numpy_bruteforce"],
    )
    parser.add_argument("--torch-nn-query-chunk-size", type=int, default=2048)
    parser.add_argument("--torch-nn-reference-chunk-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--hw", type=int, default=64)
    parser.add_argument("--attn-window", type=int, default=64)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--uniform-actions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Ignore policy sampling and execute uniformly random actions at each step",
    )
    parser.add_argument("--wandb-project", type=str, default="recuriosity")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default=os.getenv("WANDB_MODE", "online"))
    parser.add_argument(
        "--disable-gt-filtering",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Temporarily disable all GT mesh filtering and sample the full transformed mesh surface",
    )
    parser.add_argument(
        "--disable-island-filter",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Temporarily skip navmesh-island membership filtering and keep only the inferred height band",
    )
    parser.add_argument(
        "--check-holes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable depth/raycast hole mismatch checks in Habitat start-position safety validation",
    )
    parser.add_argument(
        "--eval-hole-fix",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Rewrite scene GLBs to cached double-sided copies before Habitat loads them "
            "(eval-only; leaves source meshes unchanged)"
        ),
    )
    parser.add_argument(
        "--eval-hole-fix-cache-dir",
        type=str,
        default="",
        help="Optional writable cache directory for eval-hole-fix rewritten GLBs",
    )
    parser.add_argument(
        "--eval-hole-fix-force-white",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use white backside primitives for backfaces while preserving front-face materials/textures "
            "(ignored when --eval-hole-fix-backface-black is enabled)"
        ),
    )
    parser.add_argument(
        "--eval-hole-fix-backface-black",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use black backside primitives for backfaces while preserving front-face materials/textures"
        ),
    )
    parser.add_argument(
        "--log-point-cloud",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-gt-points-vis", type=int, default=20000)
    parser.add_argument(
        "--validate-alignment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Validate mesh-observation alignment on first frame of each scene",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.uniform_actions and args.greedy:
        print(
            "[eval] --uniform-actions enabled; ignoring --greedy and sampling uniform actions",
            flush=True,
        )
    if args.depth_stride < 1:
        raise ValueError(f"--depth-stride must be >= 1, got {args.depth_stride}")
    if args.torch_nn_query_chunk_size < 1:
        raise ValueError(
            f"--torch-nn-query-chunk-size must be >= 1, got {args.torch_nn_query_chunk_size}"
        )
    if args.torch_nn_reference_chunk_size < 1:
        raise ValueError(
            "--torch-nn-reference-chunk-size must be >= 1, "
            f"got {args.torch_nn_reference_chunk_size}"
        )

    resolve_hm3d_root()
    patch_wandb_apikey()

    import torch
    import torch.distributed as dist
    import wandb

    rank, world_size, local_rank = init_distributed()
    output_root: str | None = None
    current_episode_context: dict[str, Any] | None = None
    encountered_exception = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    module_name, ppo_module = load_ppo_module(args.ppo_module)
    loaded_episodes = load_episode_specs(args.episodes_json)
    episodes = select_eval_episodes(
        loaded_episodes,
        episode_offset=args.episode_offset,
        limit=args.limit,
    )
    if not episodes:
        raise RuntimeError(
            f"No episodes found in {args.episodes_json} after episode_offset={int(args.episode_offset)} "
            f"limit={args.limit!r}"
        )

    if rank == 0:
        torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    distributed_barrier()

    metric_steps = [int(step) for step in args.metric_steps]
    max_steps = max(int(args.max_steps), max(metric_steps))
    if torch.cuda.is_available():
        gpu_id = local_rank if world_size > 1 else int(args.gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        gpu_id = 0
        device = torch.device("cpu")

    pre, agent, action_T, amp_dtype = build_components(
        ppo_module,
        device=device,
        max_steps=max_steps,
        hw=args.hw,
        attn_window=args.attn_window,
    )
    agent.eval()

    checkpoint_path = os.path.abspath(args.checkpoint_path)
    checkpoint_step, _ = agent.load_ckpt(checkpoint_path, optimizer=None, strict=False)

    run_name = make_timestamped_run_name(
        checkpoint_path=checkpoint_path,
        requested_run_name=args.run_name,
        rank=rank,
    )
    output_root = os.path.join(os.path.abspath(args.output_dir), run_name)
    video_dir = os.path.join(output_root, "videos")
    media_dir = os.path.join(output_root, "media")
    progress_dir = os.path.join(output_root, "live_results")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(media_dir, exist_ok=True)
    os.makedirs(progress_dir, exist_ok=True)
    if rank == 0:
        print(
            f"[eval] output_root={output_root} "
            f"videos={video_dir} media={media_dir} "
            f"per_episode_json={os.path.join(output_root, 'per_episode_results.json')} "
            f"aggregate_json={os.path.join(output_root, 'aggregate_results.json')}",
            flush=True,
        )
        if args.disable_gt_filtering:
            print(
                "[eval] warning: all GT filtering disabled; sampling uses the full transformed mesh surface",
                flush=True,
            )
        elif args.disable_island_filter:
            print(
                "[eval] warning: island filtering disabled; GT sampling will use the inferred height band only",
                flush=True,
            )
        if args.eval_hole_fix:
            cache_dir = (
                args.eval_hole_fix_cache_dir.strip()
                if args.eval_hole_fix_cache_dir
                else ""
            )
            print(
                "[eval] eval-hole-fix enabled: scene GLBs will be rewritten to cached "
                "hole-fix copies before sim load "
                f"(cache_dir={cache_dir or 'default_tmp'} "
                f"backface_black={bool(args.eval_hole_fix_backface_black)})",
                flush=True,
            )

    run = None
    if rank == 0:
        import tempfile

        wandb_local_dir = tempfile.mkdtemp(prefix="wandb_eval_")
        print(f"[eval] wandb local dir: {wandb_local_dir}", flush=True)
        try:
            run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                mode=args.wandb_mode,
                dir=wandb_local_dir,
                config={
                    "ppo_module": module_name,
                    "checkpoint_path": checkpoint_path,
                    "episodes_json": os.path.abspath(args.episodes_json),
                    "max_steps": max_steps,
                    "metric_steps": metric_steps,
                    "checkpoint_step": int(checkpoint_step),
                    "episode_offset": int(args.episode_offset),
                    "selected_num_episodes": len(episodes),
                    "loaded_num_episodes": len(loaded_episodes),
                    "world_size": world_size,
                    "disable_gt_filtering": bool(args.disable_gt_filtering),
                    "disable_island_filter": bool(args.disable_island_filter),
                    "check_holes": bool(args.check_holes),
                    "uniform_actions": bool(args.uniform_actions),
                    "eval_hole_fix": bool(args.eval_hole_fix),
                    "eval_hole_fix_cache_dir": args.eval_hole_fix_cache_dir or "",
                    "eval_hole_fix_force_white": bool(args.eval_hole_fix_force_white),
                    "eval_hole_fix_backface_black": bool(args.eval_hole_fix_backface_black),
                },
            )
        except Exception as exc:
            _disable_wandb_after_failure(context="wandb.init", exc=exc)
            run = None
        run = try_annotate_wandb_run_metadata(
            run=run,
            checkpoint_path=checkpoint_path,
            checkpoint_step=int(checkpoint_step),
            episodes_json=args.episodes_json,
            extra_summary_fields={
                "eval/ppo_module": module_name,
            },
        )
    else:
        wandb.init(mode="disabled")
    install_graceful_shutdown_handler(rank=rank)
    local_results: list[dict[str, Any]] = []
    logged_episode_indices: set[int] = set()
    gt_cache: dict[
        tuple[str, int, float, bool, bool, tuple[float, float, float] | None, int | None],
        tuple[np.ndarray, dict[str, Any]],
    ] = {}
    rank_episodes = shard_episodes(episodes, rank, world_size)
    max_rank_episodes = max(1, (len(episodes) + max(world_size, 1) - 1) // max(world_size, 1))
    live_results_cache: dict[int, dict[str, Any]] = {}

    try:
        for local_episode_idx0 in range(max_rank_episodes):
            current_episode_context = None
            if local_episode_idx0 < len(rank_episodes):
                episode_index0, episode = rank_episodes[local_episode_idx0]
                local_episode_idx = local_episode_idx0 + 1
                global_episode_index0 = int(args.episode_offset) + int(episode_index0)
                current_episode_context = {
                    "episode_index": int(global_episode_index0),
                    "episode_id": str(episode["episode_id"]),
                    "scene_name": str(episode["scene_name"]),
                    "local_episode_index": int(local_episode_idx),
                    "rank_episode_count": int(len(rank_episodes)),
                }
                print(
                    f"[eval rank={rank}] episode {local_episode_idx}/{len(rank_episodes)} "
                    f"id={episode['episode_id']} scene={episode['scene_name']} "
                    f"source_index={global_episode_index0}",
                    flush=True,
                )
                result, media = evaluate_episode(
                    episode=episode,
                    progress_label=(
                        f"rank={rank} episode={global_episode_index0 + 1}/{len(loaded_episodes)} "
                        f"id={episode['episode_id']}"
                    ),
                    ppo_module=ppo_module,
                    agent=agent,
                    pre=pre,
                    action_T=action_T,
                    device=device,
                    amp_dtype=amp_dtype,
                    max_steps=max_steps,
                    metric_steps=metric_steps,
                    threshold_m=args.threshold_m,
                    depth_stride=args.depth_stride,
                    nn_workers=args.nn_workers,
                    nn_backend=args.nn_backend,
                    torch_nn_query_chunk_size=args.torch_nn_query_chunk_size,
                    torch_nn_reference_chunk_size=args.torch_nn_reference_chunk_size,
                    gpu_id=gpu_id,
                    greedy=bool(args.greedy),
                    uniform_actions=bool(args.uniform_actions),
                    video_dir=video_dir,
                    gt_cache=gt_cache,
                    n_gt_samples=args.n_gt_samples,
                    log_point_cloud=bool(args.log_point_cloud),
                    max_gt_points_vis=args.max_gt_points_vis,
                    point_cloud_seed=args.seed + global_episode_index0,
                    mesh_sampling_base_seed=int(args.seed),
                    disable_gt_filtering=bool(args.disable_gt_filtering),
                    disable_island_filter=bool(args.disable_island_filter),
                    validate_alignment=bool(args.validate_alignment),
                    check_holes=bool(args.check_holes),
                    eval_hole_fix=bool(args.eval_hole_fix),
                    eval_hole_fix_cache_dir=(
                        args.eval_hole_fix_cache_dir.strip()
                        if args.eval_hole_fix_cache_dir
                        else None
                    ),
                    eval_hole_fix_force_white=bool(args.eval_hole_fix_force_white),
                    eval_hole_fix_backface_black=bool(args.eval_hole_fix_backface_black),
                )
                saved_media = save_episode_media(
                    media_dir=media_dir,
                    episode_id=result["episode_id"],
                    media=media,
                )
                local_results.append(
                    {
                        **result,
                        **saved_media,
                        "episode_index": global_episode_index0,
                    }
                )
                write_live_result_record(progress_dir, local_results[-1])
            else:
                print(
                    f"[eval rank={rank}] round {local_episode_idx0 + 1}/{max_rank_episodes} idle",
                    flush=True,
                )
            current_episode_context = None
            distributed_barrier()

            if rank == 0:
                live_results = merge_incremental_live_results(progress_dir, live_results_cache)
                for live_result_idx, live_result in enumerate(live_results):
                    episode_index = int(live_result["episode_index"])
                    if episode_index in logged_episode_indices:
                        continue
                    run = try_log_episode_to_wandb(
                        run=run,
                        result=live_result,
                        completed_results=live_results[: live_result_idx + 1],
                        metric_steps=metric_steps,
                        threshold_m=args.threshold_m,
                        max_gt_points_vis=args.max_gt_points_vis,
                        seed=args.seed,
                        log_step=episode_index + 1,
                        log_video=True,
                    )
                    if run is not None:
                        logged_episode_indices.add(episode_index)

                per_episode_json_path, aggregate_json_path, summary_path = write_progress_jsons(
                    output_root=output_root,
                    results=live_results,
                    metric_steps=metric_steps,
                    module_name=module_name,
                    checkpoint_path=checkpoint_path,
                    checkpoint_step=int(checkpoint_step),
                    episodes_json=args.episodes_json,
                    world_size=world_size,
                    total_episodes=len(episodes),
                )
                if live_results:
                    run = try_log_running_aggregate_to_wandb(
                        run=run,
                        results=live_results,
                        metric_steps=metric_steps,
                        log_step=max(int(result["episode_index"]) for result in live_results) + 1,
                    )
                run = try_sync_progress_files_to_wandb(
                    run=run,
                    output_root=output_root,
                    progress_dir=progress_dir,
                    per_episode_json_path=per_episode_json_path,
                    aggregate_json_path=aggregate_json_path,
                    summary_path=summary_path,
                    include_live_snapshots=False,
                )
                print(
                    f"[eval] updated results after round {local_episode_idx0 + 1}/{max_rank_episodes} "
                    f"episodes_done={len(live_results)}/{len(episodes)} "
                    f"per_episode_json={per_episode_json_path} "
                    f"aggregate_json={aggregate_json_path} "
                    f"summary_json={summary_path}",
                    flush=True,
                )
            gc.collect()
            distributed_barrier()

        results = merge_eval_results_from_shared_storage(
            progress_dir=progress_dir,
            local_results=local_results,
        )
        if rank == 0 and world_size > 1:
            results.sort(key=lambda item: int(item["episode_index"]))
            for result_idx, result in enumerate(results):
                if int(result["episode_index"]) in logged_episode_indices:
                    continue
                run = try_log_episode_to_wandb(
                    run=run,
                    result=result,
                    completed_results=results[: result_idx + 1],
                    metric_steps=metric_steps,
                    threshold_m=args.threshold_m,
                    max_gt_points_vis=args.max_gt_points_vis,
                    seed=args.seed,
                    log_step=int(result["episode_index"]) + 1,
                    log_video=True,
                )
                if run is not None:
                    logged_episode_indices.add(int(result["episode_index"]))
        # Non-zero ranks must not reach ``finally``'s barrier while rank 0 is still logging here.
        distributed_barrier()

        aggregate = aggregate_results(results, metric_steps)
        if rank == 0:
            run = try_log_final_aggregate_to_wandb(
                run=run,
                aggregate=aggregate,
                results=results,
            )

        if rank == 0:
            per_episode_json_path, aggregate_json_path, summary_path = write_progress_jsons(
                output_root=output_root,
                results=results,
                metric_steps=metric_steps,
                module_name=module_name,
                checkpoint_path=checkpoint_path,
                checkpoint_step=int(checkpoint_step),
                episodes_json=args.episodes_json,
                world_size=world_size,
                total_episodes=len(episodes),
            )
            run = try_sync_progress_files_to_wandb(
                run=run,
                output_root=output_root,
                progress_dir=progress_dir,
                per_episode_json_path=per_episode_json_path,
                aggregate_json_path=aggregate_json_path,
                summary_path=summary_path,
            )
            print(f"[eval] wrote summary to {summary_path}", flush=True)
            print(f"[eval] wrote per-episode results to {per_episode_json_path}", flush=True)
            print(f"[eval] wrote aggregate results to {aggregate_json_path}", flush=True)
        gc.collect()
        # Same: workers must not block in ``finally`` while rank 0 uploads via W&B above.
        distributed_barrier()

        return 0
    except Exception as exc:
        encountered_exception = True
        failure_root = output_root or os.path.abspath(args.output_dir)
        failure_path = write_rank_failure_trace(
            output_root=failure_root,
            rank=rank,
            local_rank=local_rank,
            exception=exc,
            episode_context=current_episode_context,
        )
        failure_bits = [f"[eval rank={rank}] fatal exception"]
        if current_episode_context is not None:
            failure_bits.append(
                f"episode_id={current_episode_context['episode_id']} "
                f"scene={current_episode_context['scene_name']} "
                f"episode_index={current_episode_context['episode_index']}"
            )
        if failure_path is not None:
            failure_bits.append(f"trace_path={failure_path}")
        print(" ".join(failure_bits), file=sys.stderr, flush=True)
        raise
    finally:
        if not encountered_exception:
            distributed_barrier()
        try_finish_wandb(rank=rank)
        if not encountered_exception:
            distributed_barrier()
        destroy_process_group_quietly()


if __name__ == "__main__":
    raise SystemExit(main())
