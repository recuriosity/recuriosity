# gs_batched_refactor.py
#
# Refactor goals (minimal + clear changes):
# 1) Add fast batched ingest: project E frames -> (points, rgbs, depths) for each frame in one batched pass.
# 2) Keep splats state persistent across grows (NO rebuilding Adam each frame).
# 3) Optimize only every X ingests (user-controlled), reusing your existing optimize_splat().
# 4) Rendering stays single-splat (CUDA kernel limitation).
#
# Key changes vs your original:
# - Removed sklearn KNN from the growth path (it was CPU+sync). Scales init uses depth+intrinsics (GPU).
# - grow_splats_with_new_points now *updates parameters + Adam state* instead of recreating optimizers.
# - Added ingest_frames_batch() and optimize_tracks_every().

import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torch
import json
from gsplat.rendering import rasterization
import math
from fused_ssim import fused_ssim
from gsplat.strategy import MCMCStrategy
import torch.nn.functional as F
import random
import time
import threading


import viser
from .gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState


# ============================================================
# I/O
# ============================================================

def load_basics(scene_id="00000"):
    """
    Folder layout:
      random_/
        images/   scene_id_000.png, ...
        metadata/ scene_id.json
        depths/   scene_id_000.npy, ...
    """
    base = Path("random_")
    image_base = base / "images"
    depth_base = base / "depths"
    meta_base = base / "metadata"

    scene_meta = scene_id + ".json"
    with open(meta_base / scene_meta, "r") as f:
        meta = json.load(f)

    return meta, image_base, depth_base


def load_next(image_base, depth_base, meta, scene_id="00000", i=0, device="cuda"):
    """
    Load RGB, depth, intrinsics, c2w for frame i.
    """
    image_name = scene_id + "_" + str(i).zfill(3) + ".png"
    image_np = plt.imread(image_base / image_name)[..., :3]
    image = torch.tensor(image_np, device=device, dtype=torch.float32)  # (H, W, 3) in [0,1]

    depth_name = scene_id + "_" + str(i).zfill(3) + ".npy"
    depth_np = np.load(depth_base / depth_name)
    depth = torch.from_numpy(depth_np).to(device=device, dtype=torch.float32)  # (H, W)

    fx, fy, cx, cy = meta["frames"][i]["fxfycxcy"]
    k = torch.tensor([[fx, 0., cx],
                      [0., fy, cy],
                      [0., 0., 1.]], device=device, dtype=torch.float32)

    c2w = torch.tensor(meta["frames"][i]["c2w"], device=device, dtype=torch.float32)
    return image, depth, c2w, k


# ============================================================
# Utils
# ============================================================

def rgb_to_sh(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def k_to_intr(k):
    """
    k: (B, 4) -> intr: (B, 3, 3)
    """
    fx, fy, cx, cy = k[:, 0], k[:, 1], k[:, 2], k[:, 3]
    device = k.device
    zeros = torch.zeros_like(fx, device=device)
    ones  = torch.ones_like(fx, device=device)

    row0 = torch.stack([fx, zeros, cx], dim=-1)
    row1 = torch.stack([zeros, fy, cy], dim=-1)
    row2 = torch.stack([zeros, zeros, ones], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)

@torch.no_grad()
def init_scales_from_depth(d, K, stride=1, init_scale=1.0, clamp_max=0.015):
    """
    Fast scale init (GPU): approximate local metric spacing from depth & intrinsics.
    d: (N,) meters
    K: (3,3)
    Returns log-scales: (N,3)
    """
    fx = K[0, 0].clamp(min=1e-6)
    fy = K[1, 1].clamp(min=1e-6)
    sx = d * float(stride) / fx
    sy = d * float(stride) / fy
    s = 0.5 * (sx + sy)
    s = (s * init_scale).clamp(min=1e-6, max=clamp_max)
    return torch.log(s).unsqueeze(-1).repeat(1, 3)


# ============================================================
# Batched depth -> world points + colors (+ depths) for E frames
# ============================================================

@torch.no_grad()
def depth_points_colors_batch(
    images,   # (E,H,W,3) float32 on GPU
    depths,   # (E,H,W)   float32 on GPU
    Ks,       # (E,3,3)
    c2ws,     # (E,4,4)
    stride=1,
    depth_min=0.1,
    max_points_per_frame=None,
):
    """
    Fully batched projection on a fixed grid -> (E,N,3), then per-frame gather valid points.
    Returns lists (len E): points_e (Ne,3), rgbs_e (Ne,3), d_e (Ne,)
    """
    assert images.ndim == 4 and depths.ndim == 3
    E, H, W, _ = images.shape
    device = images.device

    ys = torch.arange(0, H, stride, device=device)
    xs = torch.arange(0, W, stride, device=device)
    ys2d, xs2d = torch.meshgrid(ys, xs, indexing="ij")  # (hs,ws)
    hs, ws = ys2d.shape
    N = hs * ws

    d = depths[:, ::stride, ::stride]                           # (E,hs,ws)
    rgb = images[:, ::stride, ::stride, :].to(torch.float32)    # (E,hs,ws,3)
    valid = (d > depth_min)                                     # (E,hs,ws)

    fx = Ks[:, 0, 0].view(E, 1, 1)
    fy = Ks[:, 1, 1].view(E, 1, 1)
    cx = Ks[:, 0, 2].view(E, 1, 1)
    cy = Ks[:, 1, 2].view(E, 1, 1)

    xs_b = xs2d.view(1, hs, ws).expand(E, hs, ws).to(torch.float32)
    ys_b = ys2d.view(1, hs, ws).expand(E, hs, ws).to(torch.float32)

    z = d.to(torch.float32)
    x_cam = (xs_b - cx) / fx * z
    y_cam = (ys_b - cy) / fy * z

    ones = torch.ones_like(z)
    pts_cam_h = torch.stack([x_cam, y_cam, z, ones], dim=-1).view(E, N, 4)  # (E,N,4)

    # world: right-multiply by c2w^T (ensure float32 for bmm - c2ws may be bfloat16)
    c2ws_f = c2ws.float()
    pts_world_h = torch.bmm(pts_cam_h, c2ws_f.transpose(1, 2))  # (E,N,4)
    pts_world = pts_world_h[..., :3]                          # (E,N,3)

    rgb_flat = rgb.view(E, N, 3)
    d_flat = z.view(E, N)
    v_flat = valid.view(E, N)

    points_list, rgbs_list, depths_list = [], [], []
    for e in range(E):
        m = v_flat[e]
        if not m.any():
            points_list.append(torch.empty((0, 3), device=device))
            rgbs_list.append(torch.empty((0, 3), device=device, dtype=torch.float32))
            depths_list.append(torch.empty((0,), device=device, dtype=torch.float32))
            continue

        pts_e = pts_world[e][m]
        rgb_e = rgb_flat[e][m]
        d_e = d_flat[e][m]

        if (max_points_per_frame is not None) and (pts_e.shape[0] > max_points_per_frame):
            sel = torch.randperm(pts_e.shape[0], device=device)[:max_points_per_frame]
            pts_e = pts_e[sel]
            rgb_e = rgb_e[sel]
            d_e = d_e[sel]

        points_list.append(pts_e)
        rgbs_list.append(rgb_e)
        depths_list.append(d_e)

    return points_list, rgbs_list, depths_list


# ============================================================
# Optimizer state helpers (so we DON'T rebuild Adam each grow)
# ============================================================

def _replace_param_in_optimizer_adam(opt: torch.optim.Optimizer,
                                     old_p: torch.nn.Parameter,
                                     new_p: torch.nn.Parameter):
    """
    Replace old parameter with new parameter in an Adam optimizer,
    migrating exp_avg / exp_avg_sq (prefix copy) and keeping scalar states (e.g. step).
    """
    # swap param in param_groups
    for g in opt.param_groups:
        ps = g["params"]
        for i in range(len(ps)):
            if ps[i] is old_p:
                ps[i] = new_p

    state = opt.state
    old_state = state.pop(old_p, None)
    if old_state is None:
        return

    new_state = {}
    for k, v in old_state.items():
        if not torch.is_tensor(v):
            # rare, but keep as-is
            new_state[k] = v
            continue

        if v.ndim == 0:
            # scalar tensor (e.g. step)
            new_state[k] = v.clone()
            continue

        # tensor state: try to prefix-copy into new shape
        nv = torch.zeros_like(new_p.data)

        if v.shape == nv.shape:
            nv.copy_(v)
        else:
            # prefix copy along dim0; assume remaining dims match
            n = min(v.shape[0], nv.shape[0])
            if v.ndim == nv.ndim and v.shape[1:] == nv.shape[1:]:
                nv[:n].copy_(v[:n])
            else:
                # shape mismatch beyond dim0: just leave zeros (safe fallback)
                pass

        new_state[k] = nv

    state[new_p] = new_state

def _slice_param_and_state_adam(opt: torch.optim.Optimizer, p: torch.nn.Parameter, keep_idx: torch.Tensor):
    """
    Apply pruning indices to a parameter and its Adam state tensors.
    keep_idx: (K,) long tensor on same device.
    """
    # slice param data
    p.data = p.data[keep_idx].contiguous()

    st = opt.state.get(p, None)
    if st is None:
        return
    max_keep_idx = int(keep_idx.max().item()) if keep_idx.numel() > 0 else -1
    for k, v in list(st.items()):
        if not torch.is_tensor(v) or v.ndim == 0:
            continue
        # Typical Adam states (exp_avg / exp_avg_sq) mirror the parameter shape.
        if v.shape[0] > max_keep_idx:
            st[k] = v[keep_idx].contiguous()


def _resolve_cap_max(strategy, cap_max: int | None) -> int | None:
    if cap_max is not None:
        return int(cap_max)

    for attr in ("cap_max", "_cap_max"):
        value = getattr(strategy, attr, None)
        if value is not None:
            return int(value)

    cfg = getattr(strategy, "cfg", None)
    if cfg is not None:
        value = getattr(cfg, "cap_max", None)
        if value is not None:
            return int(value)

    return None


def _hard_prune_splats_by_opacity(
    splats,
    optimizers,
    *,
    cap_max: int | None,
    prune_margin: float = 1.10,
):
    if splats is None or cap_max is None:
        return splats, optimizers

    cap_max = int(cap_max)
    n_splats = int(splats["means"].shape[0])
    if n_splats <= int(float(cap_max) * float(prune_margin)):
        return splats, optimizers

    with torch.no_grad():
        keep_idx = torch.topk(splats["opacities"].data, k=cap_max).indices

    for name in ["means", "scales", "quats", "opacities", "sh0", "shN"]:
        _slice_param_and_state_adam(optimizers[name], splats[name], keep_idx)

    return splats, optimizers


# ============================================================
# Splat creation / growth (FAST + persistent Adam)
# ============================================================

def create_splats_from_points(
    points,
    rgbs,
    d_new=None,
    K_new=None,
    stride=1,
    init_opacity: float = 0.5,
    init_scale: float = 1.0,
    means_lr: float = 6e-4,
    scales_lr: float = 2e-3,
    opacities_lr: float = 5e-2,
    quats_lr: float = 2e-3,
    sh0_lr: float = 5e-3,
    shN_lr: float = 5e-3 / 20,
    sh_degree: int = 3,
    batch_size: int = 1,
    device: str = "cuda",
):
    points = points.to(device=device, dtype=torch.float32)
    rgbs = rgbs.to(device=device, dtype=torch.float32)
    if d_new is not None:
        d_new = d_new.to(device=device, dtype=torch.float32)

    N = points.shape[0]
    if N == 0:
        raise ValueError("No valid depth points to initialize splats. "
                         "Check that the agent is not inside geometry.")

    if (d_new is not None) and (K_new is not None):
        scales = init_scales_from_depth(d_new, K_new.to(device), stride=stride, init_scale=init_scale)
    else:
        s = torch.full((N,), 0.01, device=device, dtype=torch.float32)
        scales = torch.log(s * init_scale).unsqueeze(-1).repeat(1, 3)

    quats = torch.rand((N, 4), device=device, dtype=torch.float32)
    opacities = torch.logit(torch.full((N,), init_opacity, device=device, dtype=torch.float32))

    colors = torch.zeros((N, (sh_degree + 1) ** 2, 3), device=device, dtype=torch.float32)
    colors[:, 0, :] = rgb_to_sh(rgbs)

    params = [
        ("means", torch.nn.Parameter(points), means_lr),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
        ("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr),
        ("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr),
    ]

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)

    BS = batch_size
    optimizers = {}
    for name, _, lr in params:
        opt = torch.optim.Adam(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-8,
            betas=(0.9, 0.999),
        )
        optimizers[name] = opt
    return splats, optimizers


def grow_splats_with_new_points(
    splats,
    optimizers,
    points_new,
    rgbs_new,
    d_new=None,
    K_new=None,
    stride=1,
    device: str = "cuda",
    cap_max: int | None = None,
    init_opacity: float = 0.5,
    init_scale: float = 1.0,
    means_lr: float = 5e-5,
    scales_lr: float = 1e-2,
    opacities_lr: float = 1e-2,
    quats_lr: float = 2e-3,
    sh0_lr: float = 5e-3,
    shN_lr: float = 5e-3 / 20,
    sh_degree: int = 3,
    batch_size: int = 1,
    prune_margin: float = 1.10,  # prune only if N > cap_max*margin
    lock = None,
):
    """
    Append new splats while keeping Adam state.
    - NO sklearn KNN
    - scales initialized from depth+intrinsics if provided
    """
    points_new = points_new.to(device=device, dtype=torch.float32)
    rgbs_new = rgbs_new.to(device=device, dtype=torch.float32)
    if d_new is not None:
        d_new = d_new.to(device=device, dtype=torch.float32)

    N_new = points_new.shape[0]

    # If no valid depth points and splats already exist, just skip this frame
    if N_new == 0 and splats is not None:
        return splats, optimizers

    if splats is None:
        return create_splats_from_points(
            points_new, rgbs_new,
            d_new=d_new, K_new=K_new, stride=stride,
            init_opacity=init_opacity,
            init_scale=init_scale,
            means_lr=means_lr,
            scales_lr=scales_lr,
            opacities_lr=opacities_lr,
            quats_lr=quats_lr,
            sh0_lr=sh0_lr,
            shN_lr=shN_lr,
            sh_degree=sh_degree,
            batch_size=batch_size,
            device=device,
        )

    if N_new == 0:
        return splats, optimizers

    # new params
    if (d_new is not None) and (K_new is not None):
        scales_new = init_scales_from_depth(d_new, K_new.to(device), stride=stride, init_scale=init_scale)
    else:
        s = torch.full((N_new,), 0.01, device=device, dtype=torch.float32)
        scales_new = torch.log(s * init_scale).unsqueeze(-1).repeat(1, 3)

    quats_new = torch.rand((N_new, 4), device=device, dtype=torch.float32)
    opacities_new = torch.logit(torch.full((N_new,), init_opacity, device=device, dtype=torch.float32))

    colors_new = torch.zeros((N_new, (sh_degree + 1) ** 2, 3), device=device, dtype=torch.float32)
    colors_new[:, 0, :] = rgb_to_sh(rgbs_new)

    # concat old + new
    means_all = torch.cat([splats["means"].data, points_new], dim=0)
    scales_all = torch.cat([splats["scales"].data, scales_new], dim=0)
    quats_all = torch.cat([splats["quats"].data, quats_new], dim=0)
    opacities_all = torch.cat([splats["opacities"].data, opacities_new], dim=0)
    sh0_all = torch.cat([splats["sh0"].data, colors_new[:, :1, :]], dim=0)
    shN_all = torch.cat([splats["shN"].data, colors_new[:, 1:, :]], dim=0)

    # replace Parameters (and migrate Adam state)
    def replace(name, new_tensor):
        old_p = splats[name]
        new_p = torch.nn.Parameter(new_tensor.contiguous())
        splats[name] = new_p  # ParameterDict supports assignment

        # migrate optimizer state
        opt = optimizers.get(name, None)
        if opt is None:
            # first time (shouldn't happen if created properly), make optimizer
            BS = batch_size
            lr = dict(means=means_lr, scales=scales_lr, quats=quats_lr, opacities=opacities_lr, sh0=sh0_lr, shN=shN_lr)[name]
            optimizers[name] = torch.optim.Adam(
                [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
                eps=1e-8, betas=(0.9, 0.999),
            )
        else:
            _replace_param_in_optimizer_adam(opt, old_p, new_p)

    if lock is None:
        class _NoLock:
            def __enter__(self): return None
            def __exit__(self, *a): return False
        lock_ctx = _NoLock()
    else:
        lock_ctx = lock

    with lock_ctx:
        
        replace("means", means_all)
        replace("scales", scales_all)
        replace("quats", quats_all)
        replace("opacities", opacities_all)
        replace("sh0", sh0_all)
        replace("shN", shN_all)

    return splats, optimizers


# ============================================================
# Rendering wrappers (unchanged)
# ============================================================

def rasterize_splats(
    splats,
    camtoworlds,
    Ks,
    width: int,
    height: int,
    masks=None,
    rasterize_mode="classic",
    camera_model="pinhole",
    sh_degree=3,
    near_plane=0.01,
    far_plane=1000.0,
    render_mode="RGB",
    **kwargs,
):
    means = splats["means"]              # [N, 3]
    quats = splats["quats"]              # [N, 4]
    scales = torch.clip(torch.exp(splats["scales"]), 0, 0.025)  # [N, 3]
    opacities = torch.sigmoid(splats["opacities"])  # [N,]
    colors = torch.cat([splats["sh0"], splats["shN"]], 1)  # [N, K, 3]

    # linalg.inv does not support BFloat16; convert to float32
    camtoworlds_f = camtoworlds.float()
    render_colors, render_alphas, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=torch.linalg.inv(camtoworlds_f),  # [C, 4, 4]
        Ks=Ks,                                   # [C, 3, 3]
        width=width,
        height=height,
        packed=False,
        absgrad=False,
        sparse_grad=False,
        rasterize_mode=rasterize_mode,
        distributed=False,
        camera_model=camera_model,
        sh_degree=sh_degree,
        near_plane=near_plane,
        far_plane=far_plane,
        render_mode=render_mode,
        **kwargs,
    )

    if masks is not None:
        render_colors[~masks] = 0
    return render_colors, render_alphas, info


@torch.no_grad()
def render_img(splats, c2w, K, width, height, near_plane=0.01, far_plane=1000):
    
    render_colors, render_alphas, info = rasterize_splats(
        splats=splats,
        camtoworlds=c2w[None],
        Ks=K[None],
        width=width,
        height=height,
        sh_degree=3,
        near_plane=near_plane,
        far_plane=far_plane,
        render_mode="RGB+ED",
        rasterize_mode="classic",
        camera_model="pinhole",
    )
    img = render_colors[0, ..., :3].clamp(0.0, 1.0).clone().detach()
    depth = render_colors[..., 3].clone().detach()
    return img, depth


@torch.no_grad()
def make_viewer(splat_holder, device, H, W):
    server = viser.ViserServer(port=8080, verbose=False)

    def viewer_render_fn(camera_state: CameraState, render_tab_state: GsplatRenderTabState):
        width = render_tab_state.viewer_width or W
        height = render_tab_state.viewer_height or H

        c2w = torch.from_numpy(camera_state.c2w).float().to(device)
        K_np = camera_state.get_K((width, height))
        K = torch.from_numpy(K_np).float().to(device)

        with splat_holder["lock"]:
            splats = splat_holder["splats"]
        if splats is None:
            return np.zeros((height, width, 3), dtype=np.float32)

        img = render_img(
            splats, c2w, K, width, height,
            render_tab_state.near_plane, render_tab_state.far_plane
        ).cpu().numpy()
        return img

    viewer = GsplatViewer(
        server=server,
        render_fn=viewer_render_fn,
        output_dir=Path("viewer_outputs"),
        mode="training",
    )
    return viewer



# ============================================================
# Optimization (your code, unchanged except you pass in per-track state)
# ============================================================

def optimize_splat(
    splats,
    optimizers,
    strategy,
    strategy_state,
    schedulers,
    images,   # [T, H, W, 3]
    depths,   # [T, H, W] or None
    c2ws,     # [T, 4, 4]
    ks,       # [T, 3, 3]
    H,
    W,
    iters=50,
    batch_cams=10,
    newest_prob=0.1,
    ssim_lambda=0.2,
    lambda_depth=0.1,
    cap_max: int | None = None,
    prune_margin: float = 1.10,
    device="cuda",
):
    T = images.shape[0]
    assert c2ws.shape[0] == T and ks.shape[0] == T
    if depths is not None:
        assert depths.shape[0] == T

    n_steps = max(1, int(math.ceil(float(iters) / float(batch_cams))))
    renders_done = 0
    for step in range(n_steps):
        B = min(int(batch_cams), int(iters - renders_done))
        B = max(1, B)
        renders_done += B

        if T == 1:
            inds = torch.zeros((B,), device=device, dtype=torch.long)
        else:
            inds_list = []
            for _ in range(B):
                if random.random() < float(newest_prob):
                    inds_list.append(T - 1)
                else:
                    inds_list.append(random.randint(0, T - 2))
            inds = torch.tensor(inds_list, device=device, dtype=torch.long)

        renders, alphas, info = rasterize_splats(
            splats,
            camtoworlds=c2ws[inds],
            Ks=ks[inds],
            width=W,
            height=H,
            sh_degree=3,
            near_plane=0.01,
            far_plane=1000,
            render_mode="RGB+ED",
            masks=None,
        )
        if renders.shape[-1] == 4:
            colors = renders[..., 0:3]
            depths_pred = renders[..., 3]
        else:
            colors = renders
            depths_pred = None

        img_gt = images[inds].float()

        if (depths_pred is not None) and (depths is not None):
            d_gt = depths[inds]
            valid = d_gt > 0
        else:
            valid = torch.ones((B, H, W), device=device, dtype=torch.bool)

        valid_rgb = valid.unsqueeze(-1)

        diff = (colors - img_gt).abs() * valid_rgb
        l1loss = diff.sum() / (valid_rgb.sum() * 3.0 + 1e-8)

        invalid = ~valid
        if invalid.any():
            alpha_map = alphas[..., 0] if alphas.ndim == 4 else alphas
            loss_alpha_holes = alpha_map[invalid].mean()
        else:
            loss_alpha_holes = colors.new_tensor(0.0)

        ssimloss = colors.new_tensor(0.0)

        depth_loss = colors.new_tensor(0.0)
        if (depths_pred is not None) and (depths is not None) and valid.any():
            depth_loss = F.l1_loss(depths_pred[valid].float(), d_gt[valid].float())

        ops = torch.sigmoid(splats["opacities"])
        scales = torch.exp(splats["scales"])

        L_opa = ops.mean()
        L_scale = scales.var(dim=-1).mean()
        L_scale_2 = scales.max()

        lambda_opa = 5e-4
        lambda_scale = 1.0
        lambda_scale_2 = 1.0

        recon_loss = l1loss * (1.0 - ssim_lambda) + ssimloss * ssim_lambda
        loss = (
            recon_loss
            + lambda_opa * L_opa
            + lambda_scale * L_scale
            + lambda_depth * depth_loss
            + lambda_scale_2 * L_scale_2
            + loss_alpha_holes
        )

        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

        for sched in schedulers:
            sched.step()

        lr_now = schedulers[0].get_last_lr()[0] if (schedulers and len(schedulers) > 0) else None
        strategy.step_post_backward(
            params=splats,
            optimizers=optimizers,
            state=strategy_state,
            step=step,
            lr=lr_now,
            info=info,
        )

    splats, optimizers = _hard_prune_splats_by_opacity(
        splats,
        optimizers,
        cap_max=_resolve_cap_max(strategy, cap_max),
        prune_margin=prune_margin,
    )

    return splats, optimizers, strategy, strategy_state, schedulers


# ============================================================
# Batched ingest + periodic optimize
# ============================================================

def _maybe_init_opt_state(track, strategy, max_steps):
    if track["strategy_state"] is None:
        track["strategy_state"] = strategy.initialize_state()
    if track["schedulers"] is None:
        track["schedulers"] = [
            torch.optim.lr_scheduler.ExponentialLR(
                track["optimizers"]["means"],
                gamma=0.01 ** (1.0 / max_steps)
            )
        ]


def ingest_frames_batch(
    idxs,      # list/int tensor length E
    tracks,    # list of track dicts length E
    images, depths, c2ws, ks,
    max_points_per_frame,
    device="cuda",
    cap_max=750_000,
    stride=1,
    lock=None
):
    idxs = torch.as_tensor(idxs, device=device, dtype=torch.long)
    images_b = images[idxs]
    depths_b = depths[idxs]
    ks_b = ks[idxs]
    c2ws_b = c2ws[idxs]

    points_list, rgbs_list, d_list = depth_points_colors_batch(
        images_b, depths_b, ks_b, c2ws_b,
        stride=stride,
        max_points_per_frame=max_points_per_frame,
    )

    for e, tr in enumerate(tracks):
        # grow, but keep optimizer state persistent
        tr["splats"], tr["optimizers"] = grow_splats_with_new_points(
            tr["splats"],
            tr["optimizers"],
            points_list[e],
            rgbs_list[e],
            d_new=d_list[e],
            K_new=ks_b[e],
            stride=stride,
            device=device,
            cap_max=cap_max,
            lock=lock
        )
        tr["frames_seen"] += 1

    return tracks


def optimize_tracks_every(
    step,
    optimize_every,
    tracks,
    strategy,
    images, depths, c2ws, ks,
    H, W,
    iters_per_opt=50,
    batch_cams=10,
    newest_prob=0.1,
    ssim_lambda=0.2,
    lambda_depth=0.1,
    max_steps=10_000,
    device="cuda",
    window=None,   # e.g. 32 to optimize only last 32 frames
):
    if optimize_every <= 0 or (step % optimize_every) != 0:
        return tracks
    for tr in tracks:
        if tr["splats"] is None:
            continue

        _maybe_init_opt_state(tr, strategy, max_steps)

        T_total = images.shape[0]
        hi = min(T_total, tr["frames_seen"])
        if window is None:
            lo = 0
        else:
            lo = max(0, hi - int(window))

        if hi - lo <= 0:
            continue
        tr["splats"], tr["optimizers"], _, tr["strategy_state"], tr["schedulers"] = optimize_splat(
            tr["splats"],
            tr["optimizers"],
            strategy,
            tr["strategy_state"],
            tr["schedulers"],
            images[lo:hi],
            depths[lo:hi] if depths is not None else None,
            c2ws[lo:hi],
            ks[lo:hi],
            H, W,
            iters=iters_per_opt,
            batch_cams=batch_cams,
            newest_prob=newest_prob,
            ssim_lambda=ssim_lambda,
            lambda_depth=lambda_depth,
            device=device,
        )

    return tracks


def ingest_frames_batch_envtime(
    t_idx,     # int, time index in caches (0..T)
    tracks,    # list of len E
    imgs_big_all, depths_big_all, c2ws_big_all, ks_big_all,  # env-time caches
    max_points_per_frame,
    device="cuda",
    cap_max=750_000,
    stride=1,
    lock=None,           # optional threading.Lock if viewer reads track0
    env_mask=None,       # optional (E,) bool, True = process; if None, process all
):
    """
    Ingest ONE timestep t_idx for ALL envs at once.
    Caches:
      imgs_big_all:   (E, T+1, 3, H, W)  (float/amp)
      depths_big_all: (E, T+1, H, W)
      ks_big_all:     (E, T+1, 4)        (your k format)
      c2ws_big_all:   (E, T+1, 4, 4)
    """
    E = imgs_big_all.shape[0]

    # gather batch (E, ...)
    images_b = imgs_big_all[:, t_idx].permute(0, 2, 3, 1).contiguous()  # (E,H,W,3)
    depths_b = depths_big_all[:, t_idx].contiguous()                    # (E,H,W)
    Ks_b = k_to_intr(ks_big_all[:, t_idx:t_idx+1].squeeze(1)).squeeze(1)           # (E,3,3)
    c2ws_b = c2ws_big_all[:, t_idx].contiguous()                        # (E,4,4)

    # (optional) stabilize math
    images_b = images_b.to(torch.float32)
    depths_b = depths_b.to(torch.float32)

    active = None
    if env_mask is not None:
        active = np.nonzero(env_mask)[0]
        if len(active) == 0:
            return tracks
        images_b = images_b[active]
        depths_b = depths_b[active]
        Ks_b = Ks_b[active]
        c2ws_b = c2ws_b[active]

    env_indices = active if env_mask is not None else np.arange(E)
    points_list, rgbs_list, d_list = depth_points_colors_batch(
        images_b, depths_b, Ks_b, c2ws_b,
        stride=stride,
        max_points_per_frame=max_points_per_frame,
    )

    # grow per env (cheap loop, heavy proj already batched)
    for i, e in enumerate(env_indices):
        tr = tracks[e]
        tr_lock = lock if (lock is not None and e == 0) else None  # only lock track0 if it's the one being viewed
        tr["splats"], tr["optimizers"] = grow_splats_with_new_points(
            tr["splats"],
            tr["optimizers"],
            points_list[i],
            rgbs_list[i],
            d_new=d_list[i],
            K_new=Ks_b[i],
            stride=stride,
            device=device,
            cap_max=cap_max,
            lock=tr_lock,   # only works if you added lock support inside grow()
        )
        tr["frames_seen"] = max(tr["frames_seen"], t_idx + 1)

    return tracks


def optimize_tracks_every_envtime(
    t_idx,
    optimize_every,
    tracks,
    strategy,
    imgs_big_all, depths_big_all, c2ws_big_all, ks_big_all,
    H, W,
    iters_per_opt=50,
    batch_cams=10,
    newest_prob=0.1,
    ssim_lambda=0.2,
    lambda_depth=0.1,
    max_steps=10_000,
    device="cuda",
    window=None,   # recommend finite window
    env_mask=None,  # optional (E,) bool, True = process; if None, process all
):
    """
    Optimize each env’s splats every optimize_every steps using its cache history.
    """
    if optimize_every <= 0 or (t_idx % optimize_every) != 0:
        return tracks

    E = imgs_big_all.shape[0]

    for e in range(E):
        if env_mask is not None and not env_mask[e]:
            continue
        tr = tracks[e]
        if tr["splats"] is None:
            continue

        if tr["strategy_state"] is None:
            tr["strategy_state"] = strategy.initialize_state()
        if tr["schedulers"] is None:
            tr["schedulers"] = [
                torch.optim.lr_scheduler.ExponentialLR(
                    tr["optimizers"]["means"],
                    gamma=0.01 ** (1.0 / max_steps),
                ),
            ]

        hi = tr["frames_seen"]  # number of frames ingested so far for this env
        if hi <= 0:
            continue
        lo = 0 if (window is None) else max(0, hi - int(window))

        imgs_seq = imgs_big_all[e, lo:hi].permute(0, 2, 3, 1).contiguous().to(torch.float32)  # (L,H,W,3)
        deps_seq = depths_big_all[e, lo:hi].contiguous().to(torch.float32)                     # (L,H,W)
        c2w_seq  = c2ws_big_all[e, lo:hi].contiguous()                                         # (L,4,4)
        K_seq    = k_to_intr(ks_big_all[e, lo:hi]).contiguous()                                # (L,3,3)

        tr["splats"], tr["optimizers"], _, tr["strategy_state"], tr["schedulers"] = optimize_splat(
            tr["splats"],
            tr["optimizers"],
            strategy,
            tr["strategy_state"],
            tr["schedulers"],
            imgs_seq,
            deps_seq,
            c2w_seq,
            K_seq,
            H,
            W,
            iters=iters_per_opt,
            batch_cams=batch_cams,
            newest_prob=newest_prob,
            ssim_lambda=ssim_lambda,
            lambda_depth=lambda_depth,
            device=device,
        )
        torch.cuda.empty_cache()

    return tracks


# ============================================================
# Main
# ============================================================

def main_gs(scene_id="00000"):
    device = "cuda"

    meta, image_base, depth_base = load_basics(scene_id=scene_id)
    step = 1
    N_images = len(meta["frames"]) // step

    H_orig, W_orig = 360, 640
    H, W = 256, 256
    cap_max = 1_000_000

    images = torch.empty((N_images, H, W, 3), device=device, dtype=torch.float32)
    depths = torch.empty((N_images, H, W), device=device, dtype=torch.float32)
    c2ws = torch.empty((N_images, 4, 4), device=device, dtype=torch.float32)
    ks = torch.empty((N_images, 3, 3), device=device, dtype=torch.float32)

    max_steps = N_images * 10

    # -------- batched tracks settings --------
    E_TRACKS = 1            # set >1 if you truly have multiple splat objects in parallel
    OPTIMIZE_EVERY = 16      # optimize every X ingests (set 0 to disable)
    # WINDOW = 32             # optimize only last WINDOW frames (None = full history)
    ITERS_PER_OPT = 10
    BATCH_CAMS = 10

    tracks = [
        dict(splats=None, optimizers=None, schedulers=None, strategy_state=None, frames_seen=0)
        for _ in range(E_TRACKS)
    ]

    splat_holder = {"splats": None, "lock": threading.Lock()}

    strategy = MCMCStrategy(
        cap_max=cap_max,
        noise_lr=5e5,
        refine_start_iter=0,
        refine_stop_iter=10**9,
        refine_every=3,
        min_opacity=0.001,
        verbose=True,
    )

    viewer = None
    anchor_c2w = None

    max_points_per_frame = 20_000

    # -------- load sequence --------
    for i in range(0, N_images * step, step):
        img, depth, c2w, k = load_next(image_base, depth_base, meta, scene_id=scene_id, i=i, device=device)

        # resize image/depth
        img = img.permute(2, 0, 1).unsqueeze(0)
        img = F.interpolate(img, size=(H, W), mode="bilinear", align_corners=False)
        img = img.squeeze(0).permute(1, 2, 0)

        depth = depth.unsqueeze(0).unsqueeze(0)
        depth = F.interpolate(depth, size=(H, W), mode="nearest")
        depth = depth.squeeze(0).squeeze(0)

        # scale intrinsics
        sx = float(W) / float(W_orig)
        sy = float(H) / float(H_orig)
        fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        fx = fx * sx
        fy = fy * sy
        cx = cx * sx
        cy = cy * sy
        k = torch.tensor([[fx, 0., cx],
                          [0., fy, cy],
                          [0., 0., 1.]], device=device, dtype=torch.float32)

        idx = i // step
        images[idx] = img
        depths[idx] = depth
        ks[idx] = k

        if anchor_c2w is None:
            anchor_c2w = c2w.clone()
        c2ws[idx] = torch.linalg.inv(anchor_c2w) @ c2w

        # -------- batched ingest (E tracks, E indices) --------
        # If E_TRACKS>1, choose a policy here. Two common ones:
        # (a) all tracks ingest the same idx: idxs = [idx]*E
        # (b) shard: idxs = [idx*E + e] ... (if you have enough frames)
        idxs = [idx] * E_TRACKS

        tracks = ingest_frames_batch(
            idxs,
            tracks,
            images,
            depths,
            c2ws,
            ks,
            max_points_per_frame=max_points_per_frame,
            device=device,
            cap_max=cap_max,
            stride=1,
            lock=splat_holder["lock"],
        )

        # optimize periodically (per track)
        tracks = optimize_tracks_every(
            step=idx,
            optimize_every=OPTIMIZE_EVERY,
            tracks=tracks,
            strategy=strategy,
            images=images,
            depths=depths,
            c2ws=c2ws,
            ks=ks,
            H=H,
            W=W,
            iters_per_opt=ITERS_PER_OPT,
            batch_cams=BATCH_CAMS,
            newest_prob=0.1,
            ssim_lambda=0.2,
            lambda_depth=0.1,
            max_steps=max_steps,
            device=device,
            window=None,
        )

        with splat_holder["lock"]:
            splat_holder["splats"] = tracks[0]["splats"]

        # viewer uses track0
        splat_holder["splats"] = tracks[0]["splats"]
        if viewer is None and splat_holder["splats"] is not None:
            viewer = make_viewer(splat_holder, device, H, W)
            print("Viewer started on port 8080")

        print(f"idx={idx} | Nsplats={(tracks[0]['splats']['means'].shape[0] if tracks[0]['splats'] is not None else 0)}")

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    main_gs(scene_id="00003")
