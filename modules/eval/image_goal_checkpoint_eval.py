from __future__ import annotations

"""Evaluate PPO checkpoints on fixed HM3D image-goal episodes.

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
import heapq
import importlib
import inspect
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


from modules.eval.backface_mesh_fix import ensure_double_sided_glb_cached
from modules.eval.eval_utils import (
    _disable_wandb_after_failure,
    derive_navmesh_path,
    destroy_process_group_quietly,
    distributed_barrier,
    init_distributed,
    install_graceful_shutdown_handler,
    load_episode_specs,
    make_timestamped_run_name,
    merge_eval_results_from_shared_storage,
    merge_incremental_live_results,
    patch_wandb_apikey,
    render_terminal_eval_frame_without_camera_beam,
    resolve_hm3d_stage_path,
    save_episode_media,
    select_eval_episodes,
    shard_episodes,
    try_annotate_wandb_run_metadata,
    try_finish_wandb,
    try_sync_progress_files_to_wandb,
    write_json_atomic,
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
    if major > 8 or (major == 8 and minor == 0):
        return torch.bfloat16
    return torch.float16


def build_components(ppo_module, *, device, max_steps: int, hw: int, attn_window: int):
    import torch

    from modules.environment.env_image_goal import STEP_METERS, YAW_DEG

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
    action_t = torch.stack(
        [
            ppo_module.se3_from_translation_rotation(dz=STEP_METERS, device=device),
            ppo_module.se3_from_translation_rotation(yaw_deg=YAW_DEG, device=device),
            ppo_module.se3_from_translation_rotation(yaw_deg=-YAW_DEG, device=device),
            torch.eye(4, dtype=torch.float32).to(device),
        ],
        dim=0,
    )
    return pre, agent, action_t, select_eval_amp_dtype(device)


def build_policy_input(
    *,
    ppo_module,
    pre,
    agent,
    action_t,
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
        action_t,
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
            f"Built {policy_input.shape[1]} policy channels, but agent expects {expected_channels}."
        )
    return policy_input


def _goal_rgb_from_obs(*, obs: dict[str, Any], device, default_hw: tuple[int, int] | None):
    import torch

    goal = obs.get("goal_rgb")
    if goal is None:
        if default_hw is None:
            raise KeyError("Image-goal observation is missing 'goal_rgb'")
        h, w = default_hw
        return torch.zeros((1, 3, h, w), device=device, dtype=torch.uint8)

    goal_rgb = torch.as_tensor(goal, device=device, dtype=torch.uint8)
    if goal_rgb.ndim == 4:
        return goal_rgb.permute(0, 3, 1, 2).contiguous()
    if goal_rgb.ndim == 3:
        return goal_rgb.permute(2, 0, 1).unsqueeze(0).contiguous()
    raise ValueError(f"Unexpected goal_rgb shape: {tuple(goal_rgb.shape)}")


def _goal_rgb_to_policy_resolution(*, pre, goal_rgb_u8, amp_dtype, hw: int):
    return pre.preprocess_images(goal_rgb_u8[:, None], amp_dtype, outHW=hw)[:, 0]


def _camera_position_from_obs(obs: dict[str, Any]) -> np.ndarray | None:
    c2w = np.asarray(obs.get("c2w"), dtype=np.float64)
    if c2w.shape != (4, 4):
        return None
    pos = np.asarray(c2w[:3, 3], dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(pos)):
        return None
    return pos


def _compute_spl(
    *,
    success: bool,
    shortest_path_m: float,
    traversed_path_m: float,
) -> float:
    if not success:
        return 0.0
    d_star = float(shortest_path_m)
    if not np.isfinite(d_star) or d_star < 0.0:
        return float("nan")
    if d_star <= 1e-6:
        return 1.0
    path_len = max(float(traversed_path_m), 0.0)
    return float(min(1.0, d_star / max(path_len, d_star)))


def _project_to_navigable_cell(
    *,
    row: int,
    col: int,
    navigable_mask: np.ndarray,
    max_radius_cells: int,
) -> tuple[int, int] | None:
    mask = np.asarray(navigable_mask, dtype=bool)
    if mask.ndim != 2 or mask.size == 0:
        return None
    h, w = mask.shape
    if h <= 0 or w <= 0:
        return None

    row0 = int(np.clip(row, 0, h - 1))
    col0 = int(np.clip(col, 0, w - 1))
    if bool(mask[row0, col0]):
        return row0, col0

    for radius in range(1, max(1, int(max_radius_cells)) + 1):
        r_min = max(0, row0 - radius)
        r_max = min(h, row0 + radius + 1)
        c_min = max(0, col0 - radius)
        c_max = min(w, col0 + radius + 1)
        patch = mask[r_min:r_max, c_min:c_max]
        if not np.any(patch):
            continue
        rr, cc = np.nonzero(patch)
        rr = rr + r_min
        cc = cc + c_min
        dist2 = (rr - row0) ** 2 + (cc - col0) ** 2
        best_idx = int(np.argmin(dist2))
        return int(rr[best_idx]), int(cc[best_idx])
    return None


def _grid_geodesic_distance_from_scene_filter(
    *,
    scene_filter,
    navigable_mask: np.ndarray,
    start_position: np.ndarray,
    goal_position: np.ndarray,
    projection_tolerance_m: float,
) -> float:
    if scene_filter is None:
        return float("nan")

    mask = np.asarray(navigable_mask, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        return float("nan")

    scale_m = max(float(getattr(scene_filter, "map_scale", 0.0)), 1e-9)
    min_x = float(scene_filter.min_x)
    min_z = float(scene_filter.min_z)

    start = np.asarray(start_position, dtype=np.float64).reshape(3)
    goal = np.asarray(goal_position, dtype=np.float64).reshape(3)

    start_col = int(np.floor((float(start[0]) - min_x) / scale_m))
    start_row = int(np.floor((float(start[2]) - min_z) / scale_m))
    goal_col = int(np.floor((float(goal[0]) - min_x) / scale_m))
    goal_row = int(np.floor((float(goal[2]) - min_z) / scale_m))

    proj_radius_cells = max(
        1,
        int(np.ceil(float(projection_tolerance_m) / scale_m)),
    )
    start_cell = _project_to_navigable_cell(
        row=start_row,
        col=start_col,
        navigable_mask=mask,
        max_radius_cells=proj_radius_cells,
    )
    goal_cell = _project_to_navigable_cell(
        row=goal_row,
        col=goal_col,
        navigable_mask=mask,
        max_radius_cells=proj_radius_cells,
    )
    if start_cell is None or goal_cell is None:
        return float("nan")
    if start_cell == goal_cell:
        return 0.0

    goal_row_i, goal_col_i = goal_cell
    start_row_i, start_col_i = start_cell
    neighbours = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, float(np.sqrt(2.0))),
        (-1, 1, float(np.sqrt(2.0))),
        (1, -1, float(np.sqrt(2.0))),
        (1, 1, float(np.sqrt(2.0))),
    ]
    height, width = mask.shape

    frontier: list[tuple[float, int, int]] = [(0.0, start_row_i, start_col_i)]
    best_cost: dict[tuple[int, int], float] = {(start_row_i, start_col_i): 0.0}
    closed: set[tuple[int, int]] = set()

    while frontier:
        cost, row, col = heapq.heappop(frontier)
        node = (row, col)
        if node in closed:
            continue
        closed.add(node)

        if row == goal_row_i and col == goal_col_i:
            return float(cost * scale_m)

        for d_row, d_col, d_cost in neighbours:
            n_row = row + d_row
            n_col = col + d_col
            if n_row < 0 or n_row >= height or n_col < 0 or n_col >= width:
                continue
            if not bool(mask[n_row, n_col]):
                continue
            n_cost = cost + d_cost
            n_node = (n_row, n_col)
            prev_cost = best_cost.get(n_node, float("inf"))
            if n_cost >= prev_cost:
                continue
            best_cost[n_node] = n_cost
            heapq.heappush(frontier, (n_cost, n_row, n_col))

    return float("nan")


def _compute_cumulative_metrics_by_step(
    *,
    running_history: list[dict[str, float]],
    metric_steps: list[int],
) -> dict[int, dict[str, float]]:
    """Aggregate prefix metrics so every t-step value means "up to and including t"."""
    metrics_by_step: dict[int, dict[str, float]] = {}
    if not running_history:
        return metrics_by_step

    sorted_steps = sorted(set(int(step) for step in metric_steps))
    history_index = 0
    running_step_success = 0.0
    running_episode_success = 0.0
    running_goal_visible_ratio = 0.0
    running_goal_distance_m = float("inf")
    running_spl = 0.0

    for metric_step in sorted_steps:
        while history_index < len(running_history) and int(running_history[history_index]["step"]) <= metric_step:
            row = running_history[history_index]
            running_step_success = max(running_step_success, float(row["step_success"]))
            running_episode_success = max(running_episode_success, float(row["episode_success"]))
            running_goal_visible_ratio = max(
                running_goal_visible_ratio,
                float(row["goal_visible_ratio"]),
            )
            goal_distance = float(row["goal_distance_m"])
            if np.isfinite(goal_distance):
                running_goal_distance_m = min(running_goal_distance_m, goal_distance)
            spl_value = float(row.get("spl", 0.0))
            if np.isfinite(spl_value):
                running_spl = max(running_spl, spl_value)
            history_index += 1

        metrics_by_step[metric_step] = {
            "step_success": float(running_step_success),
            "episode_success": float(running_episode_success),
            "goal_visible_ratio": float(running_goal_visible_ratio),
            "goal_distance_m": (
                float(running_goal_distance_m)
                if np.isfinite(running_goal_distance_m)
                else float("nan")
            ),
            "spl": float(running_spl),
        }

    return metrics_by_step


def evaluate_episode(
    *,
    episode: dict[str, Any],
    progress_label: str,
    ppo_module,
    agent,
    pre,
    action_t,
    device,
    amp_dtype,
    max_steps: int,
    metric_steps: list[int],
    gpu_id: int,
    greedy: bool,
    video_dir: str,
    check_holes: bool,
    goal_terminate_on_success: bool,
    episode_seed: int,
    eval_hole_fix: bool = False,
    eval_hole_fix_cache_dir: str | None = None,
    eval_hole_fix_force_white: bool = False,
    eval_hole_fix_backface_black: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import imageio.v2 as imageio
    import torch
    from torch.distributions.categorical import Categorical

    from modules.environment.env_image_goal import HabitatMP3DEnv
    from modules.environment.render_log_utils import render_topdown

    scene_path = os.path.abspath(episode["scene_id"])
    navmesh_path = derive_navmesh_path(scene_path)
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

    start_position = np.asarray(episode["start_position"], dtype=np.float64)
    start_rotation = np.asarray(episode["start_rotation"], dtype=np.float32)
    goal_position = np.asarray(episode["goal_position"], dtype=np.float64)
    goal_rotation = np.asarray(episode["goal_rotation"], dtype=np.float32)
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
        goal_terminate_on_success=bool(goal_terminate_on_success),
    )
    # Keep the default render layout so image-goal eval videos include
    # target-view and status/condition panels (same visual style as training).
    env._eval_panel_layout = False

    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"{episode['episode_id']}.mp4")
    writer = imageio.get_writer(video_path, fps=12, codec="libx264", bitrate="4M")

    kv_caches = agent.init_kv_cache(batch_size=1)
    prev_action = torch.zeros(1, device=device, dtype=torch.long)
    action_labels = ("F", "L", "R", "S")
    actions_trace: list[dict[str, Any]] = []
    forward_sig = inspect.signature(agent.forward_step)
    supports_goal_rgb = "goal_rgb" in forward_sig.parameters
    supports_global_state = "global_state" in forward_sig.parameters
    global_state = None
    if supports_global_state and hasattr(agent, "init_global_state"):
        global_state = agent.init_global_state(
            batch_size=1,
            device=device,
            dtype=amp_dtype if device.type == "cuda" else torch.float32,
        )

    timing_totals: dict[str, float] = {}
    timing_counts: dict[str, int] = {}
    episode_t0 = time.perf_counter()

    try:
        restore_state = {
            "scene": scene,
            "position": start_position.astype(np.float32, copy=False),
            "rotation": start_rotation,
            "goal_position": goal_position.astype(np.float32, copy=False),
            "goal_rotation": goal_rotation,
        }

        t0 = time.perf_counter()
        obs, info = env.reset(
            seed=int(episode_seed),
            options={"restore_state": restore_state},
        )
        timing_totals["setup/env_reset"] = timing_totals.get("setup/env_reset", 0.0) + (
            time.perf_counter() - t0
        )
        timing_counts["setup/env_reset"] = timing_counts.get("setup/env_reset", 0) + 1

        start_rgb = np.asarray(obs["rgb"], dtype=np.uint8)
        step_success = bool(info.get("image_goal_success", False))
        episode_success = bool(step_success)
        goal_visible_ratio = float(info.get("image_goal_visible_ratio", 0.0))
        goal_distance_m = float(info.get("image_goal_distance_m", float("nan")))
        scene_filter = getattr(env, "_goal_scene_filter", None)
        navigable_mask = getattr(env, "_goal_interior_mask", None)
        if navigable_mask is None or not np.any(navigable_mask):
            if scene_filter is None:
                navigable_mask = np.empty((0, 0), dtype=bool)
            else:
                navigable_mask = np.asarray(scene_filter.island_mask, dtype=bool)
        goal_geodesic_distance_m = _grid_geodesic_distance_from_scene_filter(
            scene_filter=scene_filter,
            navigable_mask=navigable_mask,
            start_position=start_position,
            goal_position=goal_position,
            projection_tolerance_m=float(getattr(env, "goal_reachability_projection_tolerance_m", 0.15)),
        )
        if not np.isfinite(goal_geodesic_distance_m):
            goal_geodesic_distance_m = float(
                np.linalg.norm(goal_position.astype(np.float64) - start_position.astype(np.float64))
            )

        path_length_m = 0.0
        path_length_to_first_success_m = 0.0 if step_success else float("nan")
        previous_camera_position = _camera_position_from_obs(obs)
        initial_spl = _compute_spl(
            success=episode_success,
            shortest_path_m=goal_geodesic_distance_m,
            traversed_path_m=path_length_to_first_success_m if np.isfinite(path_length_to_first_success_m) else 0.0,
        )

        running_history: list[dict[str, float]] = [
            {
                "step": 0,
                "step_success": float(1.0 if step_success else 0.0),
                "episode_success": float(1.0 if episode_success else 0.0),
                "goal_visible_ratio": float(goal_visible_ratio),
                "goal_distance_m": float(goal_distance_m),
                "path_length_m": float(path_length_m),
                "spl": float(initial_spl),
            }
        ]
        t0 = time.perf_counter()
        env.update_meta(
            [
                {
                    "rgb_gt": np.asarray(obs["rgb"], dtype=np.float32) / 255.0,
                    "mode": 3,
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
        timing_totals["setup/obs_to_img_pose"] = timing_totals.get("setup/obs_to_img_pose", 0.0) + (
            time.perf_counter() - t0
        )
        timing_counts["setup/obs_to_img_pose"] = timing_counts.get("setup/obs_to_img_pose", 0) + 1
        rgb_ref = rgb.clone()
        k_ref = k.clone()
        c2w_ref = c2w.clone()

        goal_rgb_n = None
        if supports_goal_rgb:
            goal_rgb_u8 = _goal_rgb_from_obs(
                obs=obs,
                device=device,
                default_hw=(int(rgb.shape[-2]), int(rgb.shape[-1])),
            )
            goal_rgb_n = _goal_rgb_to_policy_resolution(
                pre=pre,
                goal_rgb_u8=goal_rgb_u8,
                amp_dtype=amp_dtype,
                hw=int(pre.out_hw),
            )
            if hasattr(agent, "set_episode_goal_rgb"):
                agent.set_episode_goal_rgb(goal_rgb_n)

        for step_idx in range(1, int(max_steps) + 1):
            t0 = time.perf_counter()
            policy_input = build_policy_input(
                ppo_module=ppo_module,
                pre=pre,
                agent=agent,
                action_t=action_t,
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
            timing_totals["step/build_policy_input"] = timing_totals.get(
                "step/build_policy_input", 0.0
            ) + build_policy_input_s
            timing_counts["step/build_policy_input"] = timing_counts.get("step/build_policy_input", 0) + 1

            forward_kwargs: dict[str, Any] = {}
            if supports_global_state:
                forward_kwargs["global_state"] = global_state
            if supports_goal_rgb and goal_rgb_n is not None:
                forward_kwargs["goal_rgb"] = goal_rgb_n

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
                        **forward_kwargs,
                    )
            policy_forward_s = time.perf_counter() - t0
            timing_totals["step/policy_forward"] = timing_totals.get("step/policy_forward", 0.0) + policy_forward_s
            timing_counts["step/policy_forward"] = timing_counts.get("step/policy_forward", 0) + 1

            t0 = time.perf_counter()
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
            obs, _reward, term, trunc, info = env.step(action_index)
            env_step_s = time.perf_counter() - t0
            timing_totals["step/env_step"] = timing_totals.get("step/env_step", 0.0) + env_step_s
            timing_counts["step/env_step"] = timing_counts.get("step/env_step", 0) + 1

            step_success = bool(info.get("image_goal_success", False))
            episode_success = bool(episode_success or step_success)
            goal_visible_ratio = float(info.get("image_goal_visible_ratio", 0.0))
            goal_distance_m = float(info.get("image_goal_distance_m", float("nan")))
            camera_position = _camera_position_from_obs(obs)
            if previous_camera_position is not None and camera_position is not None:
                step_distance_m = float(np.linalg.norm(camera_position - previous_camera_position))
                if np.isfinite(step_distance_m) and step_distance_m > 0.0:
                    path_length_m += step_distance_m
            previous_camera_position = camera_position

            if step_success and not np.isfinite(path_length_to_first_success_m):
                path_length_to_first_success_m = float(path_length_m)
            spl_path_length_m = (
                float(path_length_to_first_success_m)
                if np.isfinite(path_length_to_first_success_m)
                else float(path_length_m)
            )
            spl_value = _compute_spl(
                success=episode_success,
                shortest_path_m=goal_geodesic_distance_m,
                traversed_path_m=spl_path_length_m,
            )

            running_history.append(
                {
                    "step": int(step_idx),
                    "step_success": float(1.0 if step_success else 0.0),
                    "episode_success": float(1.0 if episode_success else 0.0),
                    "goal_visible_ratio": float(goal_visible_ratio),
                    "goal_distance_m": float(goal_distance_m),
                    "path_length_m": float(path_length_m),
                    "spl": float(spl_value),
                }
            )
            t0 = time.perf_counter()
            env.update_meta(
                [
                    {
                        "rgb_gt": np.asarray(obs["rgb"], dtype=np.float32) / 255.0,
                        "policy_logits": logits[0].float().detach().cpu().numpy(),
                        "mode": 3,
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
                + render_and_video_s
                + obs_to_img_pose_s
            )
            print(
                f"[eval] {progress_label} step {step_idx}/{int(max_steps)} "
                f"total={step_total_s:.3f}s "
                f"fwd={policy_forward_s:.3f}s "
                f"env={env_step_s:.3f}s "
                f"render={render_and_video_s:.3f}s "
                f"step_success={int(step_success)} "
                f"episode_success={int(episode_success)} "
                f"goal_vis={goal_visible_ratio:.3f} "
                f"goal_dist={goal_distance_m:.3f}m "
                f"spl={spl_value:.3f}",
                flush=True,
            )
            if term or trunc:
                break

        metrics_by_step = _compute_cumulative_metrics_by_step(
            running_history=running_history,
            metric_steps=metric_steps,
        )

        final_step = int(running_history[-1]["step"])
        final_step_success = bool(running_history[-1]["step_success"] > 0.5)
        final_episode_success = bool(running_history[-1]["episode_success"] > 0.5)
        final_goal_visible_ratio = float(running_history[-1]["goal_visible_ratio"])
        final_goal_distance_m = float(running_history[-1]["goal_distance_m"])
        final_path_length_m = float(running_history[-1].get("path_length_m", path_length_m))
        final_path_length_to_first_success_m = (
            float(path_length_to_first_success_m)
            if np.isfinite(path_length_to_first_success_m)
            else float("nan")
        )
        final_spl = _compute_spl(
            success=final_episode_success,
            shortest_path_m=goal_geodesic_distance_m,
            traversed_path_m=(
                final_path_length_to_first_success_m
                if np.isfinite(final_path_length_to_first_success_m)
                else final_path_length_m
            ),
        )

        print(
            f"[eval] {progress_label} finished at step {final_step}/{int(max_steps)} "
            f"success={int(final_episode_success)} spl={final_spl:.3f} "
            f"path={final_path_length_m:.3f}m shortest={goal_geodesic_distance_m:.3f}m",
            flush=True,
        )

        t0 = time.perf_counter()
        terminal_panel_frame = render_terminal_eval_frame_without_camera_beam(env)
        writer.append_data(terminal_panel_frame)
        topdown_image = render_topdown(env)
        final_rgb = np.asarray(obs["rgb"], dtype=np.uint8)
        panel_image = terminal_panel_frame
        timing_totals["final/render_outputs"] = timing_totals.get("final/render_outputs", 0.0) + (
            time.perf_counter() - t0
        )
        timing_counts["final/render_outputs"] = timing_counts.get("final/render_outputs", 0) + 1

        timing_totals["episode/total"] = timing_totals.get("episode/total", 0.0) + (
            time.perf_counter() - episode_t0
        )
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
            "panel": panel_image,
            "running_history": running_history,
            "actions_trace": actions_trace,
        }

        return {
            "episode_id": str(episode["episode_id"]),
            "scene_name": str(episode["scene_name"]),
            "video_path": video_path,
            "metrics_by_step": metrics_by_step,
            "final_step_success": int(final_step_success),
            "final_episode_success": int(final_episode_success),
            "final_goal_visible_ratio": float(final_goal_visible_ratio),
            "final_goal_distance_m": float(final_goal_distance_m),
            "goal_geodesic_distance_m": float(goal_geodesic_distance_m),
            "path_length_to_first_success_m": float(final_path_length_to_first_success_m),
            "final_path_length_m": float(final_path_length_m),
            "final_spl": float(final_spl),
            "num_frames": int(final_step),
            "timing": timing_summary,
        }, media
    finally:
        writer.close()
        env.close()
        agent.clear_kv_cache(kv_caches)
        if global_state is not None and hasattr(agent, "clear_global_state"):
            agent.clear_global_state(global_state)
        if device.type == "cuda":
            torch.cuda.empty_cache()


def aggregate_results(results: list[dict[str, Any]], metric_steps: list[int]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "num_episodes": len(results),
        "metric_steps": [int(step) for step in metric_steps],
        "steps": {},
    }

    for step in metric_steps:
        step_success_values: list[float] = []
        episode_success_values: list[float] = []
        goal_visible_ratio_values: list[float] = []
        goal_distance_values: list[float] = []
        spl_values: list[float] = []
        for result in results:
            metrics_by_step = result.get("metrics_by_step", {})
            metrics = metrics_by_step.get(int(step))
            if metrics is None:
                metrics = metrics_by_step.get(str(int(step)))
            if metrics is None:
                continue
            step_success_values.append(float(metrics["step_success"]))
            episode_success_values.append(float(metrics["episode_success"]))
            goal_visible_ratio_values.append(float(metrics["goal_visible_ratio"]))
            goal_distance_values.append(float(metrics["goal_distance_m"]))
            spl_value = float(metrics.get("spl", float("nan")))
            if np.isfinite(spl_value):
                spl_values.append(spl_value)

        aggregate["steps"][str(step)] = {
            "step_success_rate": (
                float(np.mean(step_success_values)) if step_success_values else float("nan")
            ),
            "episode_success_rate": (
                float(np.mean(episode_success_values)) if episode_success_values else float("nan")
            ),
            "goal_visible_ratio": (
                float(np.mean(goal_visible_ratio_values)) if goal_visible_ratio_values else float("nan")
            ),
            "goal_distance_m": (
                float(np.mean(goal_distance_values)) if goal_distance_values else float("nan")
            ),
            "spl": float(np.mean(spl_values)) if spl_values else float("nan"),
            "num_episodes": int(len(episode_success_values)),
        }

    final_spl_values = [
        float(result.get("final_spl", float("nan")))
        for result in results
        if np.isfinite(float(result.get("final_spl", float("nan"))))
    ]
    aggregate["final"] = {
        "success_rate": (
            float(np.mean([result["final_episode_success"] for result in results]))
            if results
            else float("nan")
        ),
        "step_success_rate": (
            float(np.mean([result["final_step_success"] for result in results]))
            if results
            else float("nan")
        ),
        "goal_visible_ratio": (
            float(np.mean([result["final_goal_visible_ratio"] for result in results]))
            if results
            else float("nan")
        ),
        "goal_distance_m": (
            float(np.mean([result["final_goal_distance_m"] for result in results]))
            if results
            else float("nan")
        ),
        "spl": float(np.mean(final_spl_values)) if final_spl_values else float("nan"),
    }
    aggregate["success_rate_percent"] = 100.0 * float(aggregate["final"]["success_rate"])
    return aggregate


def _result_to_json_record(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": result["episode_id"],
        "scene_name": result["scene_name"],
        "episode_index": int(result["episode_index"]),
        "final_step_success": int(result["final_step_success"]),
        "final_episode_success": int(result["final_episode_success"]),
        "final_goal_visible_ratio": float(result["final_goal_visible_ratio"]),
        "final_goal_distance_m": float(result["final_goal_distance_m"]),
        "goal_geodesic_distance_m": float(result["goal_geodesic_distance_m"]),
        "path_length_to_first_success_m": float(result["path_length_to_first_success_m"]),
        "final_path_length_m": float(result["final_path_length_m"]),
        "final_spl": float(result["final_spl"]),
        "num_frames": int(result["num_frames"]),
        "metrics_by_step": result["metrics_by_step"],
        "timing": result.get("timing", {}),
    }


def _result_to_live_record(result: dict[str, Any]) -> dict[str, Any]:
    return {
        **_result_to_json_record(result),
        "start_rgb_path": result["start_rgb_path"],
        "final_rgb_path": result["final_rgb_path"],
        "topdown_path": result["topdown_path"],
        "panel_path": result["panel_path"],
        "video_path": result["video_path"],
        "actions_path": result.get("actions_path"),
        "running_history": result["running_history"],
    }


def write_live_result_record_image_goal(progress_dir: str, result: dict[str, Any]) -> str:
    os.makedirs(progress_dir, exist_ok=True)
    episode_index = int(result["episode_index"])
    path = os.path.join(progress_dir, f"episode_{episode_index:06d}.json")
    write_json_atomic(path, _result_to_live_record(result))
    return path


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
                    "step_success_sum": 0.0,
                    "episode_success_sum": 0.0,
                    "count": 0.0,
                },
            )
            bucket["step_success_sum"] += float(row["step_success"])
            bucket["episode_success_sum"] += float(row["episode_success"])
            bucket["count"] += 1.0

    averaged_history: list[dict[str, float]] = []
    for step in sorted(step_sums):
        bucket = step_sums[step]
        count = max(int(bucket["count"]), 1)
        averaged_history.append(
            {
                "step": int(step),
                "step_success": float(bucket["step_success_sum"] / count),
                "episode_success": float(bucket["episode_success_sum"] / count),
                "num_episodes": int(count),
            }
        )
    return averaged_history


def make_running_plots(history: list[dict[str, float]], title: str):
    import wandb

    table = wandb.Table(
        data=[
            [
                int(row["step"]),
                float(row["step_success"]),
                float(row["episode_success"]),
                int(row.get("num_episodes", 1)),
            ]
            for row in history
        ],
        columns=["step", "step_success", "episode_success", "num_episodes"],
    )
    step_success_plot = wandb.plot.line(
        table,
        "step",
        "step_success",
        title=title,
    )
    episode_success_plot = wandb.plot.line(
        table,
        "step",
        "episode_success",
        title=f"{title} cumulative",
    )
    return table, step_success_plot, episode_success_plot


def log_episode_to_wandb(
    *,
    result: dict[str, Any],
    completed_results: list[dict[str, Any]] | None,
    metric_steps: list[int],
    log_step: int,
    log_video: bool,
) -> None:
    import wandb

    history_results = completed_results if completed_results is not None else [result]
    running_average_history = make_running_average_history(history_results)
    running_plot_title = (
        f"running image-goal success over {len(history_results)} episode"
        f"{'' if len(history_results) == 1 else 's'}"
    )
    running_table, running_step_success_plot, running_episode_success_plot = make_running_plots(
        running_average_history,
        running_plot_title,
    )
    log_payload: dict[str, Any] = {
        "episode/index": int(result["episode_index"]) + 1,
        "episode/final_step_success": int(result["final_step_success"]),
        "episode/final_episode_success": int(result["final_episode_success"]),
        "episode/final_goal_visible_ratio": float(result["final_goal_visible_ratio"]),
        "episode/final_goal_distance_m": float(result["final_goal_distance_m"]),
        "episode/final_goal_geodesic_distance_m": float(result["goal_geodesic_distance_m"]),
        "episode/final_path_length_m": float(result["final_path_length_m"]),
        "episode/final_path_length_to_first_success_m": float(result["path_length_to_first_success_m"]),
        "episode/final_spl": float(result["final_spl"]),
        "episode/start_rgb": wandb.Image(result["start_rgb_path"]),
        "episode/final_rgb": wandb.Image(result["final_rgb_path"]),
        "episode/topdown_trajectory": wandb.Image(result["topdown_path"]),
        "episode/final_panel": wandb.Image(result["panel_path"]),
        "episode/final_panel_frame": wandb.Image(result["panel_path"]),
        "episode/running_image_goal_table": running_table,
        "episode/running_step_success": running_step_success_plot,
        "episode/running_episode_success": running_episode_success_plot,
    }
    for step in metric_steps:
        metrics_by_step = result.get("metrics_by_step", {})
        step_metrics = metrics_by_step.get(step)
        if step_metrics is None:
            step_metrics = metrics_by_step.get(str(int(step)))
        if step_metrics is None:
            continue
        log_payload[f"image_goal/step_success_t{step}"] = float(step_metrics["step_success"])
        log_payload[f"image_goal/episode_success_t{step}"] = float(step_metrics["episode_success"])
        log_payload[f"image_goal/visible_ratio_t{step}"] = float(step_metrics["goal_visible_ratio"])
        log_payload[f"image_goal/distance_t{step}"] = float(step_metrics["goal_distance_m"])
        log_payload[f"image_goal/spl_t{step}"] = float(step_metrics.get("spl", float("nan")))
    for stage, values in result.get("timing", {}).items():
        safe_stage = stage.replace("/", "_")
        log_payload[f"timing/{safe_stage}_total_s"] = values["total_s"]
        log_payload[f"timing/{safe_stage}_mean_s"] = values["mean_s"]
    if log_video:
        log_payload["eval/video"] = wandb.Video(result["video_path"], format="mp4")
    wandb.log(log_payload, step=log_step)
    if result.get("actions_path"):
        wandb.save(
            result["actions_path"],
            base_path=os.path.dirname(os.path.dirname(result["actions_path"])),
        )


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
        "aggregate/running_success_rate": aggregate["final"]["success_rate"],
        "aggregate/running_success_rate_percent": aggregate["success_rate_percent"],
        "aggregate/running_spl": aggregate["final"]["spl"],
    }
    for step in metric_steps:
        values = aggregate["steps"].get(str(step))
        if values is None:
            continue
        payload[f"aggregate/running_step_success_t{step}"] = values["step_success_rate"]
        payload[f"aggregate/running_episode_success_t{step}"] = values["episode_success_rate"]
        payload[f"aggregate/running_spl_t{step}"] = values["spl"]
    wandb.log(payload, step=int(log_step))


def try_log_episode_to_wandb(
    *,
    run,
    result: dict[str, Any],
    completed_results: list[dict[str, Any]] | None,
    metric_steps: list[int],
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
            run.summary[f"aggregate/step_success_rate@{step}"] = values["step_success_rate"]
            run.summary[f"aggregate/episode_success_rate@{step}"] = values["episode_success_rate"]
            run.summary[f"aggregate/spl@{step}"] = values["spl"]
        run.summary["aggregate/final_success_rate"] = aggregate["final"]["success_rate"]
        run.summary["aggregate/final_success_rate_percent"] = aggregate["success_rate_percent"]
        run.summary["aggregate/final_goal_visible_ratio"] = aggregate["final"]["goal_visible_ratio"]
        run.summary["aggregate/final_goal_distance_m"] = aggregate["final"]["goal_distance_m"]
        run.summary["aggregate/final_spl"] = aggregate["final"]["spl"]
        wandb.log(
            {
                "aggregate/final_success_rate": aggregate["final"]["success_rate"],
                "aggregate/final_success_rate_percent": aggregate["success_rate_percent"],
                "aggregate/final_goal_visible_ratio": aggregate["final"]["goal_visible_ratio"],
                "aggregate/final_goal_distance_m": aggregate["final"]["goal_distance_m"],
                "aggregate/final_spl": aggregate["final"]["spl"],
            },
            step=len(results) + 1,
        )
        return run
    except Exception as exc:
        _disable_wandb_after_failure(context="final aggregate logging", exc=exc)
        return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a PPO checkpoint on fixed HM3D image-goal episodes."
    )
    parser.add_argument("--ppo-module", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument(
        "--episodes-json",
        type=str,
        default=os.path.join(_ROOT, "data", "splits", "hm3d", "val", "val_image_goal.json"),
    )
    parser.add_argument("--output-dir", type=str, default=os.path.join(_ROOT, "eval_outputs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--metric-steps", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--hw", type=int, default=64)
    parser.add_argument("--attn-window", type=int, default=64)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--wandb-project", type=str, default="recuriosity")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-mode", type=str, default=os.getenv("WANDB_MODE", "online"))
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
        "--goal-terminate-on-success",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    resolve_hm3d_root()
    patch_wandb_apikey()

    import torch
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
    max_steps = max(int(args.max_steps), max(metric_steps) if metric_steps else int(args.max_steps))
    if torch.cuda.is_available():
        gpu_id = local_rank if world_size > 1 else int(args.gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        gpu_id = 0
        device = torch.device("cpu")

    pre, agent, action_t, amp_dtype = build_components(
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
                    "check_holes": bool(args.check_holes),
                    "eval_hole_fix": bool(args.eval_hole_fix),
                    "eval_hole_fix_cache_dir": args.eval_hole_fix_cache_dir or "",
                    "eval_hole_fix_force_white": bool(args.eval_hole_fix_force_white),
                    "eval_hole_fix_backface_black": bool(args.eval_hole_fix_backface_black),
                    "goal_terminate_on_success": bool(args.goal_terminate_on_success),
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
                "eval/task": "image_goal",
            },
        )
    else:
        wandb.init(mode="disabled")

    install_graceful_shutdown_handler(rank=rank)
    local_results: list[dict[str, Any]] = []
    logged_episode_indices: set[int] = set()
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
                    action_t=action_t,
                    device=device,
                    amp_dtype=amp_dtype,
                    max_steps=max_steps,
                    metric_steps=metric_steps,
                    gpu_id=gpu_id,
                    greedy=bool(args.greedy),
                    video_dir=video_dir,
                    check_holes=bool(args.check_holes),
                    goal_terminate_on_success=bool(args.goal_terminate_on_success),
                    episode_seed=int(args.seed) + int(global_episode_index0),
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
                write_live_result_record_image_goal(progress_dir, local_results[-1])
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
                    log_step=int(result["episode_index"]) + 1,
                    log_video=True,
                )
                if run is not None:
                    logged_episode_indices.add(int(result["episode_index"]))
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
