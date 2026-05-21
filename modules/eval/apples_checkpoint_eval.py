from __future__ import annotations

"""Evaluate PPO checkpoints on fixed HM3D apple-collection episodes.

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
from pathlib import Path
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

    from modules.environment.env_apples import STEP_METERS, YAW_DEG

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
    num_apples: int,
    apple_asset_path: str,
    apple_collect_radius_m: float,
    apple_diameter_m: float,
    apple_step_penalty: float,
    apple_terminate_on_completion: bool,
    apple_height_offset_m: float,
    episode_seed: int,
    eval_hole_fix: bool = False,
    eval_hole_fix_cache_dir: str | None = None,
    eval_hole_fix_force_white: bool = False,
    eval_hole_fix_backface_black: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import imageio.v2 as imageio
    import torch
    from torch.distributions.categorical import Categorical

    from modules.environment.env_apples import HabitatMP3DEnv
    from modules.environment.render_log_utils import (
        render_topdown,
        render_topdown_trajectory_layer,
    )

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
    scene = {
        "scene_name": episode["scene_name"],
        "glb_path": scene_glb_path,
        "navmesh": navmesh_path,
    }

    episode_apple_positions = episode.get("apple_positions")

    env = HabitatMP3DEnv(
        [scene],
        max_steps=int(max_steps) + 1,
        render_mode="rgb_array",
        gpu_id=gpu_id,
        check_holes=check_holes,
        num_apples=int(num_apples),
        apple_asset_path=str(apple_asset_path),
        apple_collect_radius_m=float(apple_collect_radius_m),
        apple_diameter_m=float(apple_diameter_m),
        apple_step_penalty=float(apple_step_penalty),
        apple_terminate_on_completion=bool(apple_terminate_on_completion),
        apple_height_offset_m=float(apple_height_offset_m),
        apple_seed=int(episode_seed),
    )
    env._eval_panel_layout = True

    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"{episode['episode_id']}.mp4")
    writer = imageio.get_writer(video_path, fps=12, codec="libx264", bitrate="4M")

    kv_caches = agent.init_kv_cache(batch_size=1)
    prev_action = torch.zeros(1, device=device, dtype=torch.long)
    metric_steps_set = set(int(step) for step in metric_steps)
    action_labels = ("F", "L", "R", "S")
    actions_trace: list[dict[str, Any]] = []

    timing_totals: dict[str, float] = {}
    timing_counts: dict[str, int] = {}
    episode_t0 = time.perf_counter()

    try:
        restore_state = {
            "scene": scene,
            "position": start_position.astype(np.float32, copy=False),
            "rotation": start_rotation,
        }
        if episode_apple_positions is not None:
            restore_state["apple_positions"] = episode_apple_positions

        t0 = time.perf_counter()
        obs, info = env.reset(options={"restore_state": restore_state})
        timing_totals["setup/env_reset"] = timing_totals.get("setup/env_reset", 0.0) + (
            time.perf_counter() - t0
        )
        timing_counts["setup/env_reset"] = timing_counts.get("setup/env_reset", 0) + 1

        start_rgb = np.asarray(obs["rgb"], dtype=np.uint8)
        t0 = time.perf_counter()
        topdown_start_with_apples = render_topdown(env)
        timing_totals["setup/render_topdown_start_with_apples"] = timing_totals.get(
            "setup/render_topdown_start_with_apples", 0.0
        ) + (time.perf_counter() - t0)
        timing_counts["setup/render_topdown_start_with_apples"] = timing_counts.get(
            "setup/render_topdown_start_with_apples", 0
        ) + 1
        num_apples_total = int(info.get("num_apples_total", int(num_apples)))
        apples_remaining = int(info.get("apples_remaining", num_apples_total))
        apples_collected_total = max(0, int(num_apples_total - apples_remaining))
        apples_collected_total = min(apples_collected_total, max(num_apples_total, 0))

        running_history: list[dict[str, float]] = [
            {
                "step": 0,
                "apples_collected": float(apples_collected_total),
                "apples_collected_ratio": (
                    float(apples_collected_total / num_apples_total) if num_apples_total > 0 else 0.0
                ),
                "apples_remaining": float(apples_remaining),
            }
        ]
        metrics_by_step: dict[int, dict[str, float]] = {}
        if 0 in metric_steps_set:
            metrics_by_step[0] = {
                "apples_collected": float(apples_collected_total),
                "apples_collected_ratio": (
                    float(apples_collected_total / num_apples_total) if num_apples_total > 0 else 0.0
                ),
                "apples_remaining": float(apples_remaining),
            }

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

            apples_collected_step = int(info.get("apples_collected_step", 0))
            apples_remaining = int(info.get("apples_remaining", max(num_apples_total - apples_collected_total, 0)))
            apples_collected_total += apples_collected_step
            apples_collected_total = max(apples_collected_total, max(num_apples_total - apples_remaining, 0))
            apples_collected_total = min(apples_collected_total, max(num_apples_total, 0))
            apples_collected_ratio = (
                float(apples_collected_total / num_apples_total) if num_apples_total > 0 else 0.0
            )

            running_history.append(
                {
                    "step": int(step_idx),
                    "apples_collected": float(apples_collected_total),
                    "apples_collected_ratio": float(apples_collected_ratio),
                    "apples_remaining": float(apples_remaining),
                }
            )
            if step_idx in metric_steps_set:
                metrics_by_step[step_idx] = {
                    "apples_collected": float(apples_collected_total),
                    "apples_collected_ratio": float(apples_collected_ratio),
                    "apples_remaining": float(apples_remaining),
                }

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
                f"apples_step={apples_collected_step} "
                f"apples_total={apples_collected_total}/{num_apples_total}",
                flush=True,
            )
            if term or trunc:
                break

        final_step = int(running_history[-1]["step"])
        final_apples_collected = int(running_history[-1]["apples_collected"])
        final_apples_collected_ratio = float(running_history[-1]["apples_collected_ratio"])
        final_apples_remaining = int(running_history[-1]["apples_remaining"])

        print(
            f"[eval] {progress_label} finished at step {final_step}/{int(max_steps)} "
            f"apples={final_apples_collected}/{num_apples_total}",
            flush=True,
        )

        t0 = time.perf_counter()
        terminal_panel_frame = render_terminal_eval_frame_without_camera_beam(env)
        writer.append_data(terminal_panel_frame)
        topdown_image = render_topdown(env)
        topdown_trajectory_only = render_topdown_trajectory_layer(env)
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
            "topdown_start_with_apples": topdown_start_with_apples,
            "topdown_trajectory_only": topdown_trajectory_only,
            "panel": panel_image,
            "running_history": running_history,
            "actions_trace": actions_trace,
        }

        return {
            "episode_id": str(episode["episode_id"]),
            "scene_name": str(episode["scene_name"]),
            "video_path": video_path,
            "metrics_by_step": metrics_by_step,
            "final_apples_collected": int(final_apples_collected),
            "final_apples_collected_ratio": float(final_apples_collected_ratio),
            "final_apples_remaining": int(final_apples_remaining),
            "num_apples_total": int(num_apples_total),
            "num_frames": int(final_step),
            "timing": timing_summary,
        }, media
    finally:
        writer.close()
        env.close()
        agent.clear_kv_cache(kv_caches)
        if device.type == "cuda":
            torch.cuda.empty_cache()


def aggregate_results(results: list[dict[str, Any]], metric_steps: list[int]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "num_episodes": len(results),
        "metric_steps": [int(step) for step in metric_steps],
        "steps": {},
    }

    for step in metric_steps:
        apples_collected_values = []
        apples_collected_ratio_values = []
        apples_remaining_values = []
        for result in results:
            metrics = result["metrics_by_step"].get(int(step))
            if metrics is None:
                continue
            apples_collected_values.append(float(metrics["apples_collected"]))
            apples_collected_ratio_values.append(float(metrics["apples_collected_ratio"]))
            apples_remaining_values.append(float(metrics["apples_remaining"]))

        aggregate["steps"][str(step)] = {
            "apples_collected": (
                float(np.mean(apples_collected_values)) if apples_collected_values else float("nan")
            ),
            "apples_collected_ratio": (
                float(np.mean(apples_collected_ratio_values)) if apples_collected_ratio_values else float("nan")
            ),
            "apples_remaining": (
                float(np.mean(apples_remaining_values)) if apples_remaining_values else float("nan")
            ),
            "num_episodes": int(len(apples_collected_values)),
        }

    aggregate["final"] = {
        "apples_collected": (
            float(np.mean([result["final_apples_collected"] for result in results]))
            if results
            else float("nan")
        ),
        "apples_collected_ratio": (
            float(np.mean([result["final_apples_collected_ratio"] for result in results]))
            if results
            else float("nan")
        ),
        "apples_remaining": (
            float(np.mean([result["final_apples_remaining"] for result in results]))
            if results
            else float("nan")
        ),
    }
    aggregate["avg_final_apples_collected"] = aggregate["final"]["apples_collected"]
    return aggregate


def _result_to_json_record(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": result["episode_id"],
        "scene_name": result["scene_name"],
        "episode_index": int(result["episode_index"]),
        "final_apples_collected": int(result["final_apples_collected"]),
        "final_apples_collected_ratio": float(result["final_apples_collected_ratio"]),
        "final_apples_remaining": int(result["final_apples_remaining"]),
        "num_apples_total": int(result["num_apples_total"]),
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
        "topdown_start_with_apples_path": result.get("topdown_start_with_apples_path"),
        "topdown_trajectory_only_path": result.get("topdown_trajectory_only_path"),
        "panel_path": result["panel_path"],
        "video_path": result["video_path"],
        "actions_path": result.get("actions_path"),
        "running_history": result["running_history"],
    }


def write_live_result_record_apple(progress_dir: str, result: dict[str, Any]) -> str:
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
                    "apples_collected_sum": 0.0,
                    "apples_collected_ratio_sum": 0.0,
                    "count": 0.0,
                },
            )
            bucket["apples_collected_sum"] += float(row["apples_collected"])
            bucket["apples_collected_ratio_sum"] += float(row["apples_collected_ratio"])
            bucket["count"] += 1.0

    averaged_history: list[dict[str, float]] = []
    for step in sorted(step_sums):
        bucket = step_sums[step]
        count = max(int(bucket["count"]), 1)
        averaged_history.append(
            {
                "step": int(step),
                "apples_collected": float(bucket["apples_collected_sum"] / count),
                "apples_collected_ratio": float(bucket["apples_collected_ratio_sum"] / count),
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
                float(row["apples_collected"]),
                float(row["apples_collected_ratio"]),
                int(row.get("num_episodes", 1)),
            ]
            for row in history
        ],
        columns=["step", "apples_collected", "apples_collected_ratio", "num_episodes"],
    )
    apples_plot = wandb.plot.line(
        table,
        "step",
        "apples_collected",
        title=title,
    )
    ratio_plot = wandb.plot.line(
        table,
        "step",
        "apples_collected_ratio",
        title=f"{title} ratio",
    )
    return table, apples_plot, ratio_plot


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
        f"running apples avg over {len(history_results)} episode"
        f"{'' if len(history_results) == 1 else 's'}"
    )
    running_table, running_apples_plot, running_ratio_plot = make_running_plots(
        running_average_history,
        running_plot_title,
    )
    log_payload: dict[str, Any] = {
        "episode/index": int(result["episode_index"]) + 1,
        "episode/final_apples_collected": int(result["final_apples_collected"]),
        "episode/final_apples_collected_ratio": float(result["final_apples_collected_ratio"]),
        "episode/final_apples_remaining": int(result["final_apples_remaining"]),
        "episode/num_apples_total": int(result["num_apples_total"]),
        "episode/start_rgb": wandb.Image(result["start_rgb_path"]),
        "episode/final_rgb": wandb.Image(result["final_rgb_path"]),
        "episode/topdown_trajectory": wandb.Image(result["topdown_path"]),
        "episode/final_panel": wandb.Image(result["panel_path"]),
        "episode/running_apples_table": running_table,
        "episode/running_apples_collected": running_apples_plot,
        "episode/running_apples_collected_ratio": running_ratio_plot,
    }
    if result.get("topdown_trajectory_only_path"):
        log_payload["episode/topdown_trajectory_only"] = wandb.Image(
            result["topdown_trajectory_only_path"]
        )
    if result.get("topdown_start_with_apples_path"):
        log_payload["episode/topdown_start_with_apples"] = wandb.Image(
            result["topdown_start_with_apples_path"]
        )
    for step in metric_steps:
        step_metrics = result["metrics_by_step"].get(step)
        if step_metrics is None:
            continue
        log_payload[f"apples/collected_t{step}"] = float(step_metrics["apples_collected"])
        log_payload[f"apples/collected_ratio_t{step}"] = float(step_metrics["apples_collected_ratio"])
        log_payload[f"apples/remaining_t{step}"] = float(step_metrics["apples_remaining"])
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
        "aggregate/running_avg_final_apples_collected": aggregate["final"]["apples_collected"],
        "aggregate/running_avg_final_apples_collected_ratio": aggregate["final"]["apples_collected_ratio"],
    }
    for step in metric_steps:
        values = aggregate["steps"].get(str(step))
        if values is None:
            continue
        payload[f"aggregate/running_avg_apples_collected_t{step}"] = values["apples_collected"]
        payload[f"aggregate/running_avg_apples_ratio_t{step}"] = values["apples_collected_ratio"]
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
            run.summary[f"aggregate/apples_collected@{step}"] = values["apples_collected"]
            run.summary[f"aggregate/apples_collected_ratio@{step}"] = values["apples_collected_ratio"]
        run.summary["aggregate/final_apples_collected"] = aggregate["final"]["apples_collected"]
        run.summary["aggregate/final_apples_collected_ratio"] = aggregate["final"]["apples_collected_ratio"]
        run.summary["aggregate/final_apples_remaining"] = aggregate["final"]["apples_remaining"]
        wandb.log(
            {
                "aggregate/final_apples_collected": aggregate["final"]["apples_collected"],
                "aggregate/final_apples_collected_ratio": aggregate["final"]["apples_collected_ratio"],
                "aggregate/final_apples_remaining": aggregate["final"]["apples_remaining"],
            },
            step=len(results) + 1,
        )
        return run
    except Exception as exc:
        _disable_wandb_after_failure(context="final aggregate logging", exc=exc)
        return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a PPO checkpoint on fixed HM3D apple episodes.")
    parser.add_argument("--ppo-module", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument(
        "--episodes-json",
        type=str,
        default=os.path.join(_ROOT, "data", "splits", "hm3d", "val", "val_apples.json"),
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
    parser.add_argument("--num-apples", type=int, default=5)
    parser.add_argument("--apple-asset-path", type=str, default=os.path.join(_ROOT, "data", "Apple.glb"))
    parser.add_argument("--apple-collect-radius-m", type=float, default=1.5)
    parser.add_argument("--apple-diameter-m", type=float, default=0.40)
    parser.add_argument("--apple-step-penalty", type=float, default=-2e-4)
    parser.add_argument("--apple-height-offset-m", type=float, default=-0.15)
    parser.add_argument(
        "--apple-terminate-on-completion",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if int(args.num_apples) <= 0:
        raise ValueError(f"--num-apples must be > 0, got {args.num_apples}")

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
                    "num_apples": int(args.num_apples),
                    "apple_asset_path": str(args.apple_asset_path),
                    "apple_collect_radius_m": float(args.apple_collect_radius_m),
                    "apple_diameter_m": float(args.apple_diameter_m),
                    "apple_step_penalty": float(args.apple_step_penalty),
                    "apple_height_offset_m": float(args.apple_height_offset_m),
                    "apple_terminate_on_completion": bool(args.apple_terminate_on_completion),
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
                "eval/task": "apple_collection",
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
                    num_apples=int(args.num_apples),
                    apple_asset_path=str(args.apple_asset_path),
                    apple_collect_radius_m=float(args.apple_collect_radius_m),
                    apple_diameter_m=float(args.apple_diameter_m),
                    apple_step_penalty=float(args.apple_step_penalty),
                    apple_terminate_on_completion=bool(args.apple_terminate_on_completion),
                    apple_height_offset_m=float(args.apple_height_offset_m),
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
                write_live_result_record_apple(progress_dir, local_results[-1])
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
