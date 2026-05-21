"""Forgetful GSplat variant with strict age-based retention.

This module wraps gs_train_slam_fast and overrides env-time ingest/optimize to
keep only splats born within the last N timesteps. Unlike rebuild-from-cache,
this preserves in-place optimization state for retained splats.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from . import gsplat as _base
from .gsplat import *  # noqa: F401,F403


DEFAULT_FORGETFUL_GSPLAT_WINDOW = 16
_ENV_KEY = "FORGETFUL_GSPLAT_WINDOW"


def _resolve_forgetful_window(forgetful_window: int | None = None) -> int:
    value = forgetful_window
    if value is None:
        env_value = os.environ.get(_ENV_KEY, "").strip()
        if env_value:
            value = int(env_value)
    if value is None:
        value = DEFAULT_FORGETFUL_GSPLAT_WINDOW
    return max(1, int(value))


def _sync_birth_tensor(track: dict, *, old_n: int, new_n: int, t_idx: int, device) -> torch.Tensor:
    """Maintain per-splat birth timestep aligned with current splat array length."""
    birth = track.get("forgetful_birth_t")
    if birth is None:
        birth = torch.full((old_n,), int(t_idx), device=device, dtype=torch.long)
    else:
        birth = birth.to(device=device, dtype=torch.long)
        if birth.numel() > old_n:
            birth = birth[:old_n]
        elif birth.numel() < old_n:
            pad = torch.full((old_n - birth.numel(),), int(t_idx), device=device, dtype=torch.long)
            birth = torch.cat([birth, pad], dim=0)

    if new_n > old_n:
        new_birth = torch.full((new_n - old_n,), int(t_idx), device=device, dtype=torch.long)
        birth = torch.cat([birth, new_birth], dim=0)

    return birth


def _age_prune_track(track: dict, *, cutoff_t: int) -> None:
    splats = track.get("splats")
    optimizers = track.get("optimizers")
    birth = track.get("forgetful_birth_t")

    if splats is None or optimizers is None or birth is None:
        return

    n = int(splats["means"].shape[0])
    if n == 0:
        track["splats"] = None
        track["optimizers"] = None
        track["strategy_state"] = None
        track["schedulers"] = None
        track["forgetful_birth_t"] = None
        return

    if birth.numel() != n:
        # Defensive fallback if any external pruning changed count without us.
        if birth.numel() > n:
            birth = birth[:n]
        else:
            pad = torch.full((n - birth.numel(),), int(cutoff_t), device=birth.device, dtype=torch.long)
            birth = torch.cat([birth, pad], dim=0)

    keep = birth >= int(cutoff_t)
    if bool(keep.all()):
        track["forgetful_birth_t"] = birth
        return

    keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
    if keep_idx.numel() == 0:
        track["splats"] = None
        track["optimizers"] = None
        track["strategy_state"] = None
        track["schedulers"] = None
        track["forgetful_birth_t"] = None
        return

    for name in ["means", "scales", "quats", "opacities", "sh0", "shN"]:
        _base._slice_param_and_state_adam(optimizers[name], splats[name], keep_idx)

    track["forgetful_birth_t"] = birth[keep_idx].contiguous()
    # Strategy state can depend on per-splat buffers; re-init after structural prune.
    track["strategy_state"] = None


def ingest_frames_batch_envtime(
    t_idx,
    tracks,
    imgs_big_all,
    depths_big_all,
    c2ws_big_all,
    ks_big_all,
    max_points_per_frame,
    device="cuda",
    cap_max=750_000,
    stride=1,
    lock=None,
    env_mask=None,
    forgetful_window: int | None = None,
):
    """Ingest one timestep and hard-prune splats older than last N timesteps."""
    forgetful_window = _resolve_forgetful_window(forgetful_window)
    cutoff_t = int(t_idx) - forgetful_window + 1

    e_count = imgs_big_all.shape[0]

    images_b = imgs_big_all[:, t_idx].permute(0, 2, 3, 1).contiguous()
    depths_b = depths_big_all[:, t_idx].contiguous()
    ks_b = _base.k_to_intr(ks_big_all[:, t_idx : t_idx + 1].squeeze(1)).squeeze(1)
    c2ws_b = c2ws_big_all[:, t_idx].contiguous()

    images_b = images_b.to(torch.float32)
    depths_b = depths_b.to(torch.float32)

    active = None
    if env_mask is not None:
        active = np.nonzero(env_mask)[0]
        if len(active) == 0:
            return tracks
        images_b = images_b[active]
        depths_b = depths_b[active]
        ks_b = ks_b[active]
        c2ws_b = c2ws_b[active]

    env_indices = active if env_mask is not None else np.arange(e_count)
    points_list, rgbs_list, d_list = _base.depth_points_colors_batch(
        images_b,
        depths_b,
        ks_b,
        c2ws_b,
        stride=stride,
        max_points_per_frame=max_points_per_frame,
    )

    for i, e in enumerate(env_indices):
        tr = tracks[e]
        tr_lock = lock if (lock is not None and e == 0) else None

        old_n = 0 if tr.get("splats") is None else int(tr["splats"]["means"].shape[0])
        tr["splats"], tr["optimizers"] = _base.grow_splats_with_new_points(
            tr.get("splats"),
            tr.get("optimizers"),
            points_list[i],
            rgbs_list[i],
            d_new=d_list[i],
            K_new=ks_b[i],
            stride=stride,
            device=device,
            cap_max=cap_max,
            lock=tr_lock,
        )

        new_n = 0 if tr.get("splats") is None else int(tr["splats"]["means"].shape[0])
        tr["frames_seen"] = max(int(tr.get("frames_seen", 0)), int(t_idx) + 1)

        if new_n == 0:
            tr["forgetful_birth_t"] = None
            tr["strategy_state"] = None
            tr["schedulers"] = None
            continue

        tr["forgetful_birth_t"] = _sync_birth_tensor(
            tr,
            old_n=old_n,
            new_n=new_n,
            t_idx=int(t_idx),
            device=tr["splats"]["means"].device,
        )
        _age_prune_track(tr, cutoff_t=cutoff_t)

    return tracks


def optimize_tracks_every_envtime(
    t_idx,
    optimize_every,
    tracks,
    strategy,
    imgs_big_all,
    depths_big_all,
    c2ws_big_all,
    ks_big_all,
    H,
    W,
    iters_per_opt=50,
    batch_cams=10,
    newest_prob=0.1,
    ssim_lambda=0.2,
    lambda_depth=0.1,
    max_steps=10_000,
    device="cuda",
    window=None,
    env_mask=None,
    forgetful_window: int | None = None,
):
    """Optimize using at most last N frames to match forgetful retention."""
    forgetful_window = _resolve_forgetful_window(forgetful_window)
    if window is None:
        effective_window = forgetful_window
    else:
        effective_window = max(1, min(int(window), forgetful_window))

    return _base.optimize_tracks_every_envtime(
        t_idx=t_idx,
        optimize_every=optimize_every,
        tracks=tracks,
        strategy=strategy,
        imgs_big_all=imgs_big_all,
        depths_big_all=depths_big_all,
        c2ws_big_all=c2ws_big_all,
        ks_big_all=ks_big_all,
        H=H,
        W=W,
        iters_per_opt=iters_per_opt,
        batch_cams=batch_cams,
        newest_prob=newest_prob,
        ssim_lambda=ssim_lambda,
        lambda_depth=lambda_depth,
        max_steps=max_steps,
        device=device,
        window=effective_window,
        env_mask=env_mask,
    )
