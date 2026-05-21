
import os
import sys
_ppo_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_ppo_dir))
if _root not in sys.path:
    sys.path.insert(0, _root)
import json
import time, random, math
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import wandb
from modules.environment.env import HabitatMP3DEnv, list_scene_glbs, PoseProcess, reward_from_mse_threshold, mse_lowfreq_blur_then_downsample
from modules.environment.env import STEP_METERS, YAW_DEG
from modules.environment.gsplat import *
from gsplat.strategy import MCMCStrategy

from modules.agent.agent_base import NavAgent

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta


import warnings
warnings.filterwarnings("ignore")

# ============================================
# Distributed
# ============================================

def init_distributed():
    """Initialize torch.distributed if launched with torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        
        # Increase timeout from default 30min to e.g. 2 hours
        timeout = timedelta(minutes=120)
        dist.init_process_group(
            backend="nccl", 
            init_method="env://",
            timeout=timeout
        )
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    return rank, world_size, local_rank


def explained_variance(y_pred, y_true):
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    var_y = np.var(y_true)
    return float("nan") if var_y == 0.0 else 1.0 - np.var(y_true - y_pred) / (var_y + 1e-8)


# ============================================
# Obs utils
# ============================================

def obs_to_img_pose(obs, device):
    rgb   = torch.as_tensor(obs["rgb"], device=device, dtype=torch.uint8)        # (E,H,W,3) uint8
    k     = torch.as_tensor(obs["fxfycxcy"], device=device, dtype=torch.float32) # (E,4)
    c2w   = torch.as_tensor(obs["c2w"], device=device, dtype=torch.float32)      # (E,4,4)
    depth = torch.as_tensor(obs["depth"], device=device, dtype=torch.float32)    # (E,H,W)

    if len(rgb.shape) == 4:
        rgb_cf = rgb.permute(0, 3, 1, 2).contiguous()  # (E,3,H,W)
    else:
        rgb_cf = rgb.permute(2, 0, 1).contiguous()
    return rgb_cf, k, c2w, depth


def se3_from_translation_rotation(dx=0.0, dy=0.0, dz=0.0, yaw_deg=0.0, pitch_deg=0.0, device="cuda"):
    yaw   = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)

    R_yaw = torch.tensor([[ cy, 0., sy],
                          [0.,  1., 0.],
                          [-sy, 0., cy]], dtype=torch.float32)

    R_pitch = torch.tensor([[1., 0.,  0.],
                            [0., cp, -sp],
                            [0., sp,  cp]], dtype=torch.float32)

    R = R_yaw @ R_pitch
    t = torch.tensor([dx, dy, dz], dtype=torch.float32)

    T = torch.eye(4, dtype=torch.float32)
    T[:3, :3] = R
    T[:3, 3]  = t
    return T.to(device)



def action_pose_from_indices(
    act_idx,              # (B, L) long
    action_T,             # (A, 4, 4)
    k_for_action,         # (B, L, 4)
    hw,
    device,
    pre,
    zero_first_step=True,
):
    """
    returns: (B, L, 6, hw, hw)
    """
    B, L = act_idx.shape
    T = action_T[act_idx]  # (B, L, 4, 4)
    ray_o, ray_d = pre.compute_rays(T, k_for_action, h=hw, w=hw, device=device)
    action_pose = pre.build_visual_input(ray_o=ray_o, ray_d=ray_d)  # (B, L, 6, hw, hw)
    if zero_first_step and L > 0:
        action_pose[:, 0].zero_()
    return action_pose


# ============================================
# BIG-cache canonicalization 
# ============================================

@torch.no_grad()
def build_anchor_w2c(c2w0: torch.Tensor) -> torch.Tensor:
    """
    c2w0: (E,4,4) float
    returns anchor_w2c: (E,4,4) = inverse(c2w0) but computed stably (rigid)
    """
    c2w0 = c2w0.float()
    E = c2w0.shape[0]
    R = c2w0[:, :3, :3]
    t = c2w0[:, :3, 3:4]
    w2c = torch.eye(4, device=c2w0.device, dtype=torch.float32).expand(E, 4, 4).clone()
    w2c[:, :3, :3] = R.transpose(1, 2)
    w2c[:, :3, 3]  = (-w2c[:, :3, :3] @ t)[..., 0]
    return w2c


@torch.no_grad()
def preprocess_big_step(
    pre,
    rgb_u8: torch.Tensor,     # (E,3,H0,W0) uint8
    depth: torch.Tensor,      # (E,H0,W0) float32
    k: torch.Tensor,          # (E,4) float32 (fx,fy,cx,cy) in pixel units for (H0,W0)
    c2w: torch.Tensor,        # (E,4,4) float32
    anchor_w2c: torch.Tensor, # (E,4,4) float32
    big_hw: int,
    amp_dtype: torch.dtype,
):
    """
    Returns:
      rgb_big   (E,3,big,big) float (amp_dtype) in [0,1]
      depth_big (E,big,big)   float (amp_dtype)
      k_big     (E,4)         float32
      c2w_big   (E,4,4)       float32  (canon wrt anchor)
    """
    assert rgb_u8.dtype == torch.uint8
    E, _, H0, W0 = rgb_u8.shape

    # PoseProcess.preprocess_images expects (B,L,C,H,W); returns float in [0,1]
    rgb_big = pre.preprocess_images(rgb_u8[:, None], amp_dtype, outHW=big_hw)[:, 0]  # (E,3,big,big)

    # depth: nearest to avoid smoothing
    depth_big = pre.preprocess_images(
        depth[:, None, None], amp_dtype, interp="nearest", outHW=big_hw
    )[:, 0, 0]  # (E,big,big)

    # scale intrinsics from (H0,W0) -> (big_hw,big_hw)
    sx = float(big_hw) / float(W0)
    sy = float(big_hw) / float(H0)
    k = k.float()
    fx, fy, cx, cy = k[:, 0], k[:, 1], k[:, 2], k[:, 3]
    k_big = torch.stack([fx * sx, fy * sy, cx * sx, cy * sy], dim=-1).contiguous()  # (E,4)

    # canonicalize poses wrt anchor
    c2w_big = (anchor_w2c @ c2w.float()).contiguous()  # (E,4,4)

    return rgb_big, depth_big, k_big, c2w_big


# ============================================
# GSplat helpers
# ============================================



def render_from_gsplat(splats, c2w, intr, big_hw):
    # render_img returns HWC in [0,1]
    img, depth = render_img(splats, c2w, intr, big_hw, big_hw)
    return img.permute(2, 0, 1).detach(), depth.squeeze().detach()  # (3,H,W), (H,W)


# ============================================
# Rollout (policy uses 2-frame canon; GSplat uses full-history BIG cache for higher resolution)
# ============================================

def rollout_with_cache(
    envs,
    agent,
    pre,
    action_T,
    device,
    amp_dtype,
    E,
    T,
    hw,
    big_hw,
    nerf_iters,
    cap_max,
    seed,
    max_splats_steps=10**9,
    max_points_per_frame=5_000,
    rank = 0,
    out_path=None,
    random_action_prob=0.2,
):
    """
    Returns:
      actions   (E,T) long
      logprobs  (E,T) float (log P_agent_old(action))
      rewards   (E,T) float
      dones     (E,T) float
      values    (E,T) float
      imgs_all  (E,T+1,3,H,W) uint8
      ks_all    (E,T+1,4) float
      c2ws_all  (E,T+1,4,4) float
    """
    agent_impl = agent.module if isinstance(agent, DDP) else agent

    actions  = torch.zeros(E, T, device=device, dtype=torch.long)
    logprobs = torch.zeros(E, T, device=device, dtype=torch.float32)
    rewards  = torch.zeros(E, T, device=device, dtype=torch.float32)
    dones    = torch.zeros(E, T, device=device, dtype=torch.float32)
    values   = torch.zeros(E, T, device=device, dtype=torch.float32)

    obs, _ = envs.reset(seed=seed, options={"start": True})
    
     
    rgb, k, c2w, depth = obs_to_img_pose(obs, device)

    H0, W0 = rgb.shape[-2], rgb.shape[-1]

    # reference 0 frame (raw)
    rgb_ref = rgb.clone()
    k_ref   = k.clone()
    c2w_ref = c2w.clone()

    # full history buffers (raw)
    imgs_all   = torch.empty(E, T + 1, 3, H0, W0, device=device, dtype=torch.uint8)
    ks_all     = torch.empty(E, T + 1, 4, device=device, dtype=torch.float32)
    c2ws_all   = torch.empty(E, T + 1, 4, 4, device=device, dtype=torch.float32)
    imgs_all[:, 0].copy_(rgb)
    ks_all[:, 0].copy_(k)
    c2ws_all[:, 0].copy_(c2w)

    # ---- BIG canonicalization cache for GSplat (incremental) ----
    anchor_w2c = build_anchor_w2c(c2w_ref)  # (E,4,4)

    imgs_big_all   = torch.empty(E, T + 1, 3, big_hw, big_hw, device=device, dtype=amp_dtype)
    depths_big_all = torch.empty(E, T + 1, big_hw, big_hw, device=device, dtype=amp_dtype)
    ks_big_all     = torch.empty(E, T + 1, 4, device=device, dtype=torch.float32)
    c2ws_big_all   = torch.empty(E, T + 1, 4, 4, device=device, dtype=torch.float32)

    rgb_big0, depth_big0, k_big0, c2w_big0 = preprocess_big_step(
        pre, rgb, depth, k, c2w, anchor_w2c, big_hw=big_hw, amp_dtype=amp_dtype
    )
    imgs_big_all[:, 0].copy_(rgb_big0)
    depths_big_all[:, 0].copy_(depth_big0)
    ks_big_all[:, 0].copy_(k_big0)
    c2ws_big_all[:, 0].copy_(c2w_big0)


    # init GSplat per env using cached big frame 0 (L=1)
    tracks = [dict(splats=None, optimizers=None, schedulers=None, strategy_state=None, frames_seen=0) for _ in range(E)]
    strategy = MCMCStrategy(
        cap_max=cap_max,
        noise_lr=5e5,
        refine_start_iter=0,
        refine_stop_iter=10**9,
        refine_every=3,
        min_opacity=0.001,
        verbose=(rank == 0),
    )
    # ingest t_idx = 0 for all envs
    tracks = ingest_frames_batch_envtime(
        t_idx=0,
        tracks=tracks,
        imgs_big_all=imgs_big_all,
        depths_big_all=depths_big_all,
        c2ws_big_all=c2ws_big_all,
        ks_big_all=ks_big_all,
        max_points_per_frame=max_points_per_frame,
        device=device,
        cap_max=cap_max,
        stride=1,
        lock=None,  # or splat_holder["lock"] if you have viewer
    )

        
    kv_caches = agent_impl.init_kv_cache(batch_size=E)
    prev_action = torch.zeros(E, device=device, dtype=torch.long)
    next_done = torch.zeros(E, device=device, dtype=torch.float32)
    OPTIMIZE_EVERY = 16

    frames = None
    if rank == 0 and out_path is not None:
            frames = []

    for t in range(T):
        dones[:, t] = next_done

        # ----- policy canonicalization: [ref0, current] only -----
       
        
        rgb_c = imgs_all[:, t]   # (E,3,H0,W0) uint8
        k_c   = ks_all[:, t]     # (E,4)
        c2w_c = c2ws_all[:, t]  # (E,4,4)
        rgb_win2 = torch.stack([rgb_ref, rgb_c], dim=1)   # (E,2,3,H0,W0)
        k_win2   = torch.stack([k_ref,   k_c],   dim=1)   # (E,2,4)
        c2w_win2 = torch.stack([c2w_ref, c2w_c], dim=1)   # (E,2,4,4)

        rgb_n2, k_n2, c2w_n2, _ = pre.process_window(
            rgb_win2, k_win2, c2w_win2,
            device=device, amp_dtype=amp_dtype, outHW=hw
        )

        rgb_n = rgb_n2[:, -1:]     # (E,1,3,hw,hw)
        k_n   = k_n2[:, -1:]       # (E,1,4)
        c2w_n = c2w_n2[:, -1:]     # (E,1,4,4)


        ro, rd = pre.compute_rays(c2w_n, k_n, h=hw, w=hw, device=device)
        posed_9 = pre.build_visual_input(rgb_n, ro, rd)  # (E,1,9,hw,hw)

        act_pose = action_pose_from_indices(
            prev_action.unsqueeze(1),
            action_T,
            k_n,
            hw,
            device,
            pre,
            zero_first_step=(t == 0),
        )  # (E,1,6,hw,hw)

        posed_15 = torch.cat([posed_9, act_pose], dim=2).squeeze(1)  # (E,15,hw,hw)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                logits, value = agent_impl.forward_step(posed_15, time_idx=t, kv_caches=kv_caches)
                probs = Categorical(logits=logits)
                policy_action = probs.sample()
                random_action = torch.randint(0, logits.shape[-1], (E,), device=device, dtype=torch.long)
                use_random = (torch.rand(E, device=device) < float(random_action_prob))
                action = torch.where(use_random, random_action, policy_action)
                logprob = probs.log_prob(action)

        values[:, t].copy_(value.reshape(-1).float())
        actions[:, t].copy_(action)
        logprobs[:, t].copy_(logprob.float())
        prev_action = action

        obs_next, _, term, trunc, infos = envs.step(action.detach().cpu().numpy())
        done_np = np.logical_or(term, trunc)
        next_done = torch.as_tensor(done_np, device=device, dtype=torch.float32)

        rgb1, k1, c2w1, depth1 = obs_to_img_pose(obs_next, device)

        imgs_all[:, t + 1].copy_(rgb1)
        ks_all[:, t + 1].copy_(k1)
        c2ws_all[:, t + 1].copy_(c2w1)
        # ---- fill BIG cache only for the NEW frame (t+1) ----
        rgb_big1, depth_big1, k_big1, c2w_big1 = preprocess_big_step(
            pre, rgb1, depth1, k1, c2w1, anchor_w2c, big_hw=big_hw, amp_dtype=amp_dtype
        )
        imgs_big_all[:, t + 1].copy_(rgb_big1)
        depths_big_all[:, t + 1].copy_(depth_big1)
        ks_big_all[:, t + 1].copy_(k_big1)
        c2ws_big_all[:, t + 1].copy_(c2w_big1)

        # ----- GSplat reward + update: FULL history, NO big pre.process_window -----
        L = t + 2  # frames 0..t+1 inclusive

        imgs_big   = imgs_big_all[:, :L]      # (E,L,3,big,big)
        depths_big = depths_big_all[:, :L]    # (E,L,big,big)
        ks_big     = ks_big_all[:, :L]        # (E,L,4)
        c2ws_big   = c2ws_big_all[:, :L]      # (E,L,4,4)

        # reward: render last pose from current splats
        pred_rgbs = []
        for e in range(E):
            splats_e = tracks[e]["splats"]
            intr_last = k_to_intr(ks_big[e])[-1]
            pred_rgb, _ = render_from_gsplat(splats_e, c2ws_big[e, -1], intr_last, big_hw)
            pred_rgbs.append(pred_rgb)         # (3,H,W)
        
        pred_rgb = torch.stack(pred_rgbs, dim=0).to(torch.float32)  # (E,3,H,W)
        gt_rgb   = imgs_big[:, -1].to(torch.float32)                 # (E,3,H,W)



        mse = mse_lowfreq_blur_then_downsample(
            gt_rgb,            # (E,3,H,W)
            pred_rgb,          # (E,3,H,W)
            factor=16
        )
        rewards[:, t].copy_(reward_from_mse_threshold(mse).float())
        
      

        if rank == 0:
            envs.call("update_meta", [{
                "rgb_pred":   pred_rgb[0].permute(1,2,0).detach().cpu().numpy(),    # (H,W,3) in [0,1]
                "rgb_gt":     gt_rgb[0].permute(1,2,0).detach().cpu().numpy(),      # (H,W,3) in [0,1]
                "policy_logits": logits[0].float().detach().cpu().numpy(),
                "mode": int(2 if bool(use_random[0].item()) else 3),
            }])
            
            envs.call("push_step_metrics", reward=float(rewards[0, t].item()))
        # print("t:", t, "logits:", logits[0].float().detach().cpu().numpy())

       
        # update splats with full history (prevents MCMC forgetting)
        # ingest new frame (t+1) for all envs
        tracks = ingest_frames_batch_envtime(
            t_idx=t+1,
            tracks=tracks,
            imgs_big_all=imgs_big_all,
            depths_big_all=depths_big_all,
            c2ws_big_all=c2ws_big_all,
            ks_big_all=ks_big_all,
            max_points_per_frame=max_points_per_frame,
            device=device,
            cap_max=cap_max,
            stride=1,
            lock=None,
        )
        
        # optimize periodically (e.g. every 8 steps)
        tracks = optimize_tracks_every_envtime(
            t_idx=t+1,
            optimize_every=OPTIMIZE_EVERY,
            tracks=tracks,
            strategy=strategy,
            imgs_big_all=imgs_big_all,
            depths_big_all=depths_big_all,
            c2ws_big_all=c2ws_big_all,
            ks_big_all=ks_big_all,
            H=big_hw,
            W=big_hw,
            iters_per_opt=nerf_iters,  # or smaller
            batch_cams=10,
            newest_prob=0.1,
            ssim_lambda=0.2,
            lambda_depth=0.1,
            max_steps=max_splats_steps,
            device=device,
            window=None,   # keep it bounded
        )


       
        if rank == 0 and frames is not None:
            frame = envs.call("render")[0]
            frame = np.asarray(frame)
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            frames.append(frame)

    
        
    # cleanup
    if rank == 0 and frames is not None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        # writes once (still encodes at the end)
        imageio.mimwrite(out_path, frames, fps=12, codec="libx264", bitrate="4M")
    agent_impl.clear_kv_cache(kv_caches)
    for e in range(E):
        tracks[e].clear()
    del tracks
    torch.cuda.empty_cache()
    return actions, logprobs, rewards, dones, values, imgs_all, c2ws_all, ks_all, out_path


# ============================================
# PPO Update (parallel over all frames)
# ============================================

def ppo_update(
    agent,
    optimizer,
    pre,
    action_T,
    device,
    amp_dtype,
    hw,
    imgs_all,    # (E,T+1,3,H,W) uint8
    c2ws_all,    # (E,T+1,4,4)
    ks_all,      # (E,T+1,4)
    actions,     # (E,T)
    logprobs,    # (E,T)
    advantages,  # (E,T)
    returns,     # (E,T)
    values_old,  # (E,T)
    update_epochs=5,
    clip_coef=0.3,
    vf_coef=0.1,
    max_grad_norm=1.0,
    ent_coef_start=0.05,
    ent_decay_rate=0.99,
    update_idx=1,
    random_action_prob=0.2,
    mbE=1,              # env-minibatch size for PPO
    mbE_pre=None,       # env-minibatch size for pre.process_window (defaults to mbE)
):
    """
    PPO update that avoids recomputing pre.process_window every epoch:
      - preprocess (process_window) ONCE per update (chunked over envs)
      - cache imgs_n/ks_n/c2ws_n
      - each epoch: env-minibatch -> compute rays/posed/action_pose -> agent forward/backward
    """
    agent_impl = agent.module if isinstance(agent, DDP) else agent
    agent_impl.train()

    E, T = actions.shape
    mbE = int(mbE)
    mbE_pre = int(mbE if mbE_pre is None else mbE_pre)
    denom = max(1, math.ceil(E / mbE))  # grad accumulation scaling

    # ----------------------------
    # 1) Cache normalized imgs/poses ONCE (pre.process_window)
    # ----------------------------
    imgs_win = imgs_all[:, :T]   # (E,T,3,H,W) uint8
    ks_win   = ks_all[:, :T]     # (E,T,4)
    c2ws_win = c2ws_all[:, :T]   # (E,T,4,4)
    # Cache outputs of process_window (much smaller than caching posed_15)
    # imgs_n: (E,T,3,hw,hw) amp_dtype; ks_n: (E,T,4) float32; c2ws_n: (E,T,4,4) float32
    imgs_n_all = torch.empty((E, T, 3, hw, hw), device=device, dtype=amp_dtype)
    ks_n_all   = torch.empty((E, T, 4),         device=device, dtype=torch.float32)
    c2ws_n_all = torch.empty((E, T, 4, 4),      device=device, dtype=torch.float32)

    

    with torch.no_grad():
        for e0 in range(0, E, mbE_pre):
            e1 = min(E, e0 + mbE_pre)

            imgs_n, ks_n, c2ws_n, _ = pre.process_window(
                imgs_win[e0:e1],
                ks_win[e0:e1],
                c2ws_win[e0:e1],
                device=device,
                amp_dtype=amp_dtype,
                outHW=hw,
            )
            imgs_n_all[e0:e1].copy_(imgs_n)
            ks_n_all[e0:e1].copy_(ks_n.float())
            c2ws_n_all[e0:e1].copy_(c2ws_n.float())

            del imgs_n, ks_n, c2ws_n
            torch.cuda.empty_cache()

    del imgs_win, ks_win, c2ws_win#, depths_win
    torch.cuda.empty_cache()

    # shifted previous-action channel (once)
    pad = torch.zeros_like(actions[:, :1])
    actions_shifted = torch.cat([pad, actions], dim=1)[:, :T]  # (E,T)

    # ----------------------------
    # 2) PPO epochs (NO process_window inside)
    # ----------------------------
    clipfracs = []
    pg_loss_log = v_loss_log = entropy_log = approx_kl_log = old_kl_log = loss_log = None

    for epoch in range(int(update_epochs)):
        optimizer.zero_grad(set_to_none=True)

        # epoch accumulators
        loss_sum = pg_sum = v_sum = ent_sum = kl_sum = oldkl_sum = 0.0
        n_mb = 0
        clipfracs_epoch = []

        for e0 in range(0, E, mbE):
            e1 = min(E, e0 + mbE)
            mb = e1 - e0

            # slice cached normalized window
            imgs_n = imgs_n_all[e0:e1]    # (mb,T,3,hw,hw) amp_dtype
            ks_n   = ks_n_all[e0:e1]      # (mb,T,4) float32
            c2ws_n = c2ws_n_all[e0:e1]    # (mb,T,4,4) float32

            # build posed_15 on the fly (this is now the main cost per epoch)
            ro, rd = pre.compute_rays(c2ws_n, ks_n, h=hw, w=hw, device=device)
            posed_9 = pre.build_visual_input(imgs_n, ro, rd)  # (mb,T,9,hw,hw)

            act_pose = action_pose_from_indices(
                actions_shifted[e0:e1],  # (mb,T)
                action_T,
                ks_n,
                hw,
                device,
                pre,
                zero_first_step=True,
            )  # (mb,T,6,hw,hw)

            posed_15 = torch.cat([posed_9, act_pose], dim=2)  # (mb,T,15,hw,hw)

            # forward + loss
            act_mb   = actions[e0:e1]
            oldlp_mb = logprobs[e0:e1]
            adv_mb   = advantages[e0:e1]
            ret_mb   = returns[e0:e1]
            vold_mb  = values_old[e0:e1]

            with torch.cuda.amp.autocast(dtype=amp_dtype):
                logits, v_pred = agent_impl(posed_15, last_only=False)  # (mb,T,A), (mb,T)

                logits_f = logits.reshape(mb * T, -1)
                v_pred_f = v_pred.reshape(mb * T).float()

                probs = Categorical(logits=logits_f)
                newlogp = probs.log_prob(act_mb.reshape(mb * T))
                entropy = probs.entropy().mean()

            oldlogp_agent = oldlp_mb.reshape(mb * T)
            oldprob_agent = oldlogp_agent.exp().clamp_min(1e-8)
            uniform_prob = 1.0 / float(logits_f.shape[-1])
            oldprob_mix = float(random_action_prob) * uniform_prob + (1.0 - float(random_action_prob)) * oldprob_agent
            oldlogp = torch.log(oldprob_mix.clamp_min(1e-8))
            logratio = (newlogp.float() - oldlogp)
            ratio = logratio.exp()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > float(clip_coef)).float().mean()
                clipfracs_epoch.append(float(clipfrac.item()))

            adv_f = adv_mb.reshape(mb * T)  # NO normalization
            pg_loss1 = -adv_f * ratio
            pg_loss2 = -adv_f * torch.clamp(ratio, 1 - float(clip_coef), 1 + float(clip_coef))
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            v_target_f = ret_mb.reshape(mb * T).float()
            v_old_f    = vold_mb.reshape(mb * T).float()
            v_loss_unclipped = (v_pred_f - v_target_f) ** 2
            v_clipped = v_old_f + torch.clamp(v_pred_f - v_old_f, -float(clip_coef), float(clip_coef))
            v_loss = 0.5 * v_loss_unclipped.mean()
                       
            ent_coef = max(float(ent_coef_start) * (float(ent_decay_rate) ** int(update_idx)), 1e-3)
            loss = 1. * pg_loss + float(vf_coef) * v_loss - ent_coef * entropy

            # grad accumulation (one optimizer step per epoch)
            (loss / float(denom)).backward()

            # log accum
            loss_sum  += float(loss.detach().item())
            pg_sum    += float(pg_loss.detach().item())
            v_sum     += float(v_loss.detach().item())
            ent_sum   += float(entropy.detach().item())
            kl_sum    += float(approx_kl.detach().item())
            oldkl_sum += float(old_approx_kl.detach().item())
            n_mb += 1

            # cleanup big temps
            del ro, rd, posed_9, act_pose, posed_15
            del logits, v_pred, logits_f, v_pred_f, probs, newlogp, entropy, logratio, ratio
            del adv_f, pg_loss1, pg_loss2, pg_loss, v_target_f, v_old_f, v_loss_unclipped, v_clipped, v_loss, loss
            torch.cuda.empty_cache()

        nn.utils.clip_grad_norm_(agent_impl.parameters(), float(max_grad_norm))
        optimizer.step()

        # epoch logs
        loss_log      = torch.tensor(loss_sum / max(1, n_mb), device=device)
        pg_loss_log   = torch.tensor(pg_sum / max(1, n_mb), device=device)
        v_loss_log    = torch.tensor(v_sum / max(1, n_mb), device=device)
        entropy_log   = torch.tensor(ent_sum / max(1, n_mb), device=device)
        approx_kl_log = torch.tensor(kl_sum / max(1, n_mb), device=device)
        old_kl_log    = torch.tensor(oldkl_sum / max(1, n_mb), device=device)

        clipfracs.append(float(np.mean(clipfracs_epoch)) if clipfracs_epoch else 0.0)

    # free caches
    del imgs_n_all, ks_n_all, c2ws_n_all, actions_shifted
    torch.cuda.empty_cache()

    return {
        "loss": float(loss_log.item()) if loss_log is not None else None,
        "pg_loss": float(pg_loss_log.item()) if pg_loss_log is not None else None,
        "v_loss": float(v_loss_log.item()) if v_loss_log is not None else None,
        "entropy": float(entropy_log.item()) if entropy_log is not None else None,
        "approx_kl": float(approx_kl_log.item()) if approx_kl_log is not None else None,
        "old_approx_kl": float(old_kl_log.item()) if old_kl_log is not None else None,
        "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
    }


# ============================================
# Eval rollout (full length; greedy by default)
# ============================================

def eval_rollout_video(
    agent,
    pre,
    action_T,
    scene_list,
    device,
    amp_dtype,
    roll_length,
    hw,
    big_hw,
    nerf_iters,
    cap_max,
    seed,
    out_dir,
    out_name,
    greedy=True,
    max_splats_steps=10**9,
    max_points_per_frame=5_000,
    optimize_gsplat=True,
    use_gsplat=True,
):
    agent_impl = agent.module if isinstance(agent, DDP) else agent
    agent_impl.eval()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)

    env = HabitatMP3DEnv(scene_list, max_steps=int(roll_length) + 1, render_mode="rgb_array")
    # Eval-only tweaks when GSplat disabled
    if not use_gsplat:
        env._topdown_regen_interval = 20
        env._eval_panel_layout = True  # simplified layout: GT | map | policy logits
    else:
        env._eval_panel_layout = False
    obs, _ = env.reset(seed=seed, options={"start": False})

    rgb, k, c2w, depth = obs_to_img_pose(obs, device)
    H0, W0 = rgb.shape[-2], rgb.shape[-1]

    rgb_ref = rgb.clone()
    k_ref   = k.clone()
    c2w_ref = c2w.clone()

    imgs_all   = torch.empty(1, roll_length + 1, 3, H0, W0, device=device, dtype=torch.uint8)
    ks_all     = torch.empty(1, roll_length + 1, 4, device=device, dtype=torch.float32)
    c2ws_all   = torch.empty(1, roll_length + 1, 4, 4, device=device, dtype=torch.float32)

    imgs_all[:, 0].copy_(rgb)
    ks_all[:, 0].copy_(k)
    c2ws_all[:, 0].copy_(c2w)

    tracks = None
    strategy = None
    if use_gsplat:
        # ---- BIG cache init (E=1) ----
        anchor_w2c = build_anchor_w2c(c2w_ref.unsqueeze(0))[0:1]  # (1,4,4)

        imgs_big_all   = torch.empty(1, roll_length + 1, 3, big_hw, big_hw, device=device, dtype=amp_dtype)
        depths_big_all = torch.empty(1, roll_length + 1, big_hw, big_hw, device=device, dtype=amp_dtype)
        ks_big_all     = torch.empty(1, roll_length + 1, 4, device=device, dtype=torch.float32)
        c2ws_big_all   = torch.empty(1, roll_length + 1, 4, 4, device=device, dtype=torch.float32)

        rgb_big0, depth_big0, k_big0, c2w_big0 = preprocess_big_step(
            pre, rgb.unsqueeze(0), depth.unsqueeze(0), k.unsqueeze(0), c2w.unsqueeze(0), anchor_w2c, big_hw=big_hw, amp_dtype=amp_dtype
        )
        imgs_big_all[:, 0].copy_(rgb_big0)
        depths_big_all[:, 0].copy_(depth_big0)
        ks_big_all[:, 0].copy_(k_big0)
        c2ws_big_all[:, 0].copy_(c2w_big0)

        # ---------- GSplat track (E=1) ----------
        tracks = [dict(splats=None, optimizers=None, schedulers=None, strategy_state=None, frames_seen=0)]

        strategy = MCMCStrategy(
            cap_max=cap_max,
            noise_lr=5e5,
            refine_start_iter=0,
            refine_stop_iter=10**9,
            refine_every=3,
            min_opacity=0.001,
            verbose=False,
        )

        # ingest t_idx=0 from caches
        tracks = ingest_frames_batch_envtime(
            t_idx=0,
            tracks=tracks,
            imgs_big_all=imgs_big_all,
            depths_big_all=depths_big_all,
            c2ws_big_all=c2ws_big_all,
            ks_big_all=ks_big_all,
            max_points_per_frame=max_points_per_frame,
            device=device,
            cap_max=cap_max,
            stride=1,
            lock=None,
        )

    kv_caches = agent_impl.init_kv_cache(batch_size=1)
    prev_action = torch.zeros(1, device=device, dtype=torch.long)

    rewards = []
    video_writer = imageio.get_writer(out_path, fps=12, codec="libx264", bitrate="4M")

    try:
        for t in range(int(roll_length)):
            _t0 = time.perf_counter()

            # policy 2-frame canon
            rgb_c = imgs_all[:, t]
            k_c   = ks_all[:, t]
            c2w_c = c2ws_all[:, t]
            rgb_win2 = torch.stack([rgb_ref.unsqueeze(0), rgb_c], dim=1)  # (1,2,3,H,W)
            k_win2   = torch.stack([k_ref.unsqueeze(0),   k_c],   dim=1)
            c2w_win2 = torch.stack([c2w_ref.unsqueeze(0), c2w_c], dim=1)
            _t1 = time.perf_counter()
            rgb_n2, k_n2, c2w_n2, _ = pre.process_window(
                rgb_win2, k_win2, c2w_win2, device=device, amp_dtype=amp_dtype, outHW=hw
            )
            rgb_n = rgb_n2[:, -1:]
            k_n   = k_n2[:, -1:]
            c2w_n = c2w_n2[:, -1:]

            gtt = rgb_n[0, 0].to(torch.float32).clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
           
            ro, rd = pre.compute_rays(c2w_n, k_n, h=hw, w=hw, device=device)
            posed_9 = pre.build_visual_input(rgb_n, ro, rd)  # (1,1,9,hw,hw)

            act_pose = action_pose_from_indices(
                prev_action.unsqueeze(1),
                action_T,
                k_n,
                hw,
                device,
                pre,
                zero_first_step=(t == 0),
            )
            posed_15 = torch.cat([posed_9, act_pose], dim=2).squeeze(1)  # (1,15,hw,hw)

            _t2 = time.perf_counter()
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    logits, _ = agent_impl.forward_step(posed_15, time_idx=t, kv_caches=kv_caches)

            if greedy:
                action = torch.argmax(logits, dim=-1)
            else:
                action = Categorical(logits=logits).sample()
            prev_action = action

            _t3 = time.perf_counter()
            # env step
            obs_next, _, term, trunc, infos = env.step(int(action.item()))
            rgb, k, c2w, depth = obs_to_img_pose(obs_next, device)

            imgs_all[:, t + 1].copy_(rgb)
            ks_all[:, t + 1].copy_(k)
            c2ws_all[:, t + 1].copy_(c2w)

            _t4 = time.perf_counter()
            if use_gsplat:
                # ---- fill BIG cache only for the NEW frame (t+1) ----
                rgb_big1, depth_big1, k_big1, c2w_big1 = preprocess_big_step(
                    pre, rgb.unsqueeze(0), depth.unsqueeze(0), k.unsqueeze(0), c2w.unsqueeze(0), anchor_w2c, big_hw=big_hw, amp_dtype=amp_dtype
                )
                imgs_big_all[:, t + 1].copy_(rgb_big1)
                depths_big_all[:, t + 1].copy_(depth_big1)
                ks_big_all[:, t + 1].copy_(k_big1)
                c2ws_big_all[:, t + 1].copy_(c2w_big1)

                # GSplat full history (from cache) for reward + update
                L = t + 2
                imgs_big   = imgs_big_all[:, :L]
                depths_big = depths_big_all[:, :L]
                ks_big     = ks_big_all[:, :L]
                c2ws_big   = c2ws_big_all[:, :L]

                intr_last = k_to_intr(ks_big[0])[-1]
                splats0 = tracks[0]["splats"]
                pred_rgb, _ = render_from_gsplat(splats0, c2ws_big[0, -1], intr_last, big_hw)
                pred_rgb = pred_rgb.to(torch.float32)
                gt_rgb = imgs_big[0, -1].to(torch.float32)

                mse = mse_lowfreq_blur_then_downsample(
                    gt_rgb[None],
                    pred_rgb[None],
                    factor=16
                )
                rew = reward_from_mse_threshold(mse).item()
                rewards.append(rew)
                env.update_meta([{
                    "rgb_pred":   pred_rgb.permute(1,2,0).detach().cpu().numpy(),
                    "rgb_gt":     gt_rgb.permute(1,2,0).detach().cpu().numpy(),
                    "policy_logits": logits[0].float().detach().cpu().numpy(),
                    "mode": 3,
                }])
                env.push_step_metrics(reward=rew)

                # ingest only the NEW frame t+1
                tracks = ingest_frames_batch_envtime(
                    t_idx=t+1,
                    tracks=tracks,
                    imgs_big_all=imgs_big_all,
                    depths_big_all=depths_big_all,
                    c2ws_big_all=c2ws_big_all,
                    ks_big_all=ks_big_all,
                    max_points_per_frame=max_points_per_frame,
                    device=device,
                    cap_max=cap_max,
                    stride=1,
                    lock=None,
                )

                # optimize periodically (skip when optimize_gsplat=False for faster eval)
                if optimize_gsplat:
                    OPTIMIZE_EVERY = 16
                    WINDOW = None
                    ITERS_PER_OPT = nerf_iters
                    BATCH_CAMS = 10
                    tracks = optimize_tracks_every_envtime(
                        t_idx=t+1,
                        optimize_every=OPTIMIZE_EVERY,
                        tracks=tracks,
                        strategy=strategy,
                        imgs_big_all=imgs_big_all,
                        depths_big_all=depths_big_all,
                        c2ws_big_all=c2ws_big_all,
                        ks_big_all=ks_big_all,
                        H=big_hw,
                        W=big_hw,
                        iters_per_opt=ITERS_PER_OPT,
                        batch_cams=BATCH_CAMS,
                        newest_prob=0.1,
                        ssim_lambda=0.2,
                        lambda_depth=0.1,
                        max_steps=max_splats_steps,
                        device=device,
                        window=WINDOW,
                    )
            else:
                rewards.append(0.0)
                # Pass GT image for eval panel
                if rgb.dim() == 4:
                    rgb_gt = rgb[0].permute(1, 2, 0).float().cpu().numpy() / 255.0
                else:
                    rgb_gt = rgb.permute(1, 2, 0).float().cpu().numpy() / 255.0
                env.update_meta([{
                    "rgb_gt": rgb_gt,
                    "policy_logits": logits[0].float().detach().cpu().numpy(),
                    "mode": 3,
                }])

            _t5 = time.perf_counter()
            frame = np.asarray(env.render())
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            _t6 = time.perf_counter()
            video_writer.append_data(frame)
            _t7 = time.perf_counter()

            dt_process = _t2 - _t1
            dt_agent = _t3 - _t2
            dt_env = _t4 - _t3
            dt_gsplat = _t5 - _t4
            dt_render = _t6 - _t5
            dt_video = _t7 - _t6
            dt_total = _t7 - _t0
            print(
                f"[eval t={t}] process={dt_process*1000:.0f}ms agent={dt_agent*1000:.0f}ms "
                f"env_step={dt_env*1000:.0f}ms gsplat={dt_gsplat*1000:.0f}ms "
                f"render={dt_render*1000:.0f}ms video={dt_video*1000:.0f}ms total={dt_total*1000:.0f}ms",
                flush=True,
            )

            if term or trunc:
                break

        
    
    finally:
        video_writer.close()
        env.close()
        agent_impl.clear_kv_cache(kv_caches)
        if tracks is not None:
            for tr in tracks:
                tr.clear()
        del tracks
        del strategy
        torch.cuda.empty_cache()

    return out_path, float(np.sum(rewards)), float(np.mean(rewards) if rewards else 0.0)


# Main
# ============================================



def main(
    logdir="runs",
    save_eval_video=True,
    train=True,
    base_ckpt=None,
    eval_on_val=False,
    eval_compile=False,
    greedy_eval=False,
    num_envs=32,
    roll_length=1024,
    eval_roll_length=None,
    learning_rate=1e-5,
    hw=64,
    big_hw=128,
    gamma=0.995,
    gae_lambda=0.97,
    update_epochs=3,
    clip_coef=0.2,
    ent_coef_start=0.1, #change
    ent_decay_rate=0.99,
    vf_coef=0.5,
    max_grad_norm=1.0,
    nerf_iters=10,
    cap_max=750_000,
    attn_window=64,
    eval_every=10,
    checkpoint_path="checkpoints_saved/explore_base.pt",
    random_action_prob=0.2,
    anneal_random_action=True,
    anneal_random_steps=2_500_000,
    anneal_random_duration=500_000,
):
    rank, world_size, local_rank = init_distributed()
    checkpoint_milestone_steps = 20_000_000
    session_dir = os.environ.get("TRAINING_SESSION_DIR", "")
    if not session_dir and logdir and os.path.isabs(logdir):
        session_dir = logdir
    if session_dir:
        checkpoint_path = os.path.join(session_dir, "checkpoints", os.path.basename(checkpoint_path))

    def _milestone_checkpoint_path(base_path: str, step: int) -> str:
        stem, ext = os.path.splitext(base_path)
        suffix = f"_{step // 1_000_000}M"
        return f"{stem}{suffix}{ext}"

    def _save_checkpoint(save_path: str, *, step: int, milestone_step: int | None = None) -> None:
        to_save = agent.module if isinstance(agent, DDP) else agent
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        extra = {"step_is_total": True}
        if milestone_step is not None:
            extra["milestone_step"] = int(milestone_step)
        to_save.save_ckpt(
            save_path,
            optimizer=optimizer,
            step=step,
            extra=extra,
        )

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    amp_dtype = torch.bfloat16

    seed = 42 + rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    run_name = f"train_explore_base__{seed}__{int(time.time())}"
    if not train:
        run_name = f"{run_name}_test_only"
    tb_dir = os.path.join(session_dir, "tensorboard") if session_dir else os.path.join(logdir, run_name)

    # Wandb resume: when resuming from session checkpoint, continue same wandb run
    wandb_run_id = None
    wandb_resume = None
    if session_dir and os.path.exists(checkpoint_path):
        id_file = os.path.join(session_dir, "wandb_run_id.json")
        if os.path.exists(id_file):
            try:
                data = json.load(open(id_file))
                wandb_run_id = data.get("id")
                if wandb_run_id:
                    wandb_resume = "allow"
            except Exception:
                pass

    if rank == 0:
        wandb_init_kwargs = dict(
            project=os.getenv("WANDB_PROJECT", "recuriosity"),
            entity=os.getenv("WANDB_ENTITY", None),
            name=run_name,
            tags=["ppo", "nerf", "nav", "sliding", "simple_hierarchy"],
            notes="",
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
            mode=os.getenv("WANDB_MODE", "online"),
            dir=os.getenv("WANDB_DIR", None),
        )
        if wandb_run_id:
            wandb_init_kwargs["id"] = wandb_run_id
            wandb_init_kwargs["resume"] = wandb_resume
        wandb.init(**wandb_init_kwargs)
        if session_dir and wandb.run is not None:
            os.makedirs(session_dir, exist_ok=True)
            id_file = os.path.join(session_dir, "wandb_run_id.json")
            with open(id_file, "w") as f:
                json.dump({"id": wandb.run.id}, f)
        _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        wandb.run.log_code(root=_root)
        writer = SummaryWriter(tb_dir)
    else:
        wandb.init(mode="disabled")
        writer = None

    
    all_scenes = list_scene_glbs()
    test_scenes = list_scene_glbs(test=eval_on_val)  # True = val/eval scenes, False = train scenes
    # Eval-only: minimize envs to save memory (rollout not used, but envs are still created)
    if not train:
        num_envs = world_size  # 1 per GPU, minimum for assert num_envs % world_size == 0
    rng = np.random.RandomState(12345)
    perm = rng.permutation(len(all_scenes))
    
    def make_env():
        def thunk():
            env = HabitatMP3DEnv(all_scenes, max_steps=int(roll_length) + 1, render_mode="rgb_array", gpu_id=local_rank)
            return gym.wrappers.RecordEpisodeStatistics(env)
        return thunk

    assert num_envs % world_size == 0, "num_envs must be divisible by world_size"
    E = num_envs // world_size
    T = int(roll_length)
    T_eval = int(eval_roll_length) if eval_roll_length is not None else T

    envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(E)], shared_memory=False, context="spawn")

    pre = PoseProcess(out_hw=hw, scene_scale_factor=1.35)

    # Pre-download DINO on rank 0 to avoid FileExistsError when multiple ranks extract concurrently
    if rank == 0:
        torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    if world_size > 1:
        dist.barrier()

    agent = NavAgent(
        image_size=hw,
        patch_size=32, #8
        in_channels=15, #15,
        d_model=768,
        d_head=64,
        n_layer_spatial_cross=4,  # Changed from n_layer_spatial
        n_layer_temporal=24,
        use_qk_norm=True,
        n_act=4,
        max_v=T+1,
        attn_window=attn_window,
        checkpoint_every=0,
        dino_model_name="dinov2_vitb14",  # New parameter
        dino_freeze=True,  # New parameter
        frame_token_mode="learnable_query",  # New parameter (recommended)
    ).to(device)


    if world_size > 1:
        agent = DDP(agent, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    optimizer = optim.AdamW(list(agent.parameters()), lr=float(learning_rate), eps=1e-4, fused=True)

    start_update = 1
    global_step = 0

    # Prefer own session checkpoint first (preemption resume), then base_ckpt (user-specified)
    resume_path = None
    if session_dir and os.path.exists(checkpoint_path):
        resume_path = checkpoint_path
        if rank == 0:
            print(f"[resume] found checkpoint in session dir, resuming from {checkpoint_path}", flush=True)
    if resume_path is None and base_ckpt:
        resume_path = base_ckpt
    if resume_path and not os.path.isabs(resume_path) and session_dir:
        alt = os.path.join(session_dir, "checkpoints", os.path.basename(resume_path))
        if os.path.exists(alt):
            resume_path = alt

    if resume_path and os.path.exists(resume_path):
        to_load = agent.module if isinstance(agent, DDP) else agent
        step, ckpt = to_load.load_ckpt(resume_path, optimizer=optimizer, strict=False)
    
        # Keep checkpoint step in total env-step units.
        steps_per_update = num_envs * T
        saved_step_is_total = ckpt.get("step_is_total")
        if saved_step_is_total is None:
            saved_step_is_total = (world_size == 1) or (step % steps_per_update == 0)
            if rank == 0 and world_size > 1 and not saved_step_is_total:
                print(
                    "[resume] checkpoint missing step metadata; treating stored step as legacy per-rank step",
                    flush=True,
                )

        if saved_step_is_total:
            global_step = step
        else:
            global_step = step * world_size

        update0 = global_step // steps_per_update
        start_update = int(update0) + 1

    
        if rank == 0:
            print(
                f"[resume] step={step} -> global_step={global_step}, "
                f"resume at update={start_update}"
            )

    # Optional: torch.compile for eval (reduces Python overhead, may speed up agent)
    if eval_compile and not train and rank == 0:
        if isinstance(agent, DDP):
            inner = torch.compile(agent.module, mode="reduce-overhead")
            agent = DDP(inner, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        else:
            agent = torch.compile(agent, mode="reduce-overhead")
        print(
            f"[eval] torch.compile enabled (mode=reduce-overhead), "
            f"global_step={global_step}, start_update={start_update}",
            flush=True,
        )
        
    action_T = torch.stack([
        se3_from_translation_rotation(dz= STEP_METERS, device=device),  # forward
        se3_from_translation_rotation(yaw_deg= YAW_DEG, device=device), # look left
        se3_from_translation_rotation(yaw_deg=-YAW_DEG, device=device), # look right
        torch.eye(4, dtype=torch.float32).to(device)                     # stop
    ], dim=0)

    total_timesteps = 1e10
    batch_size = num_envs * T
    n_updates = int(total_timesteps // batch_size)
    start_time = time.time()

    eval_count = 0
    train_video_path = None

    for update in range(start_update, n_updates + 1):
        log_payload = {"global_step": global_step}
        global_step_before_update = global_step
        anneal_progress = (update - 1) * E * T

        if anneal_random_action and anneal_progress >= anneal_random_steps:
            frac = min(1.0, (anneal_progress - anneal_random_steps) / float(anneal_random_duration))
            current_random_action_prob = random_action_prob * (1.0 - frac)
        else:
            current_random_action_prob = random_action_prob

        if train:
            if rank == 0:
                train_video_path = f"train_videos_sliding/train_{run_name}_u{update:04d}.mp4"
    
            
        
            # -------- rollout --------
            agent.eval()
            actions, logprobs, rewards, dones, values, imgs_all, c2ws_all, ks_all, train_video_path = rollout_with_cache(
                envs=envs,
                agent=agent,
                pre=pre,
                action_T=action_T,
                device=device,
                amp_dtype=amp_dtype,
                E=E,
                T=T,
                hw=hw,
                big_hw=big_hw,
                nerf_iters=nerf_iters,
                cap_max=cap_max,
                seed=seed + 1000 * update,
                rank=rank,
                out_path=train_video_path if (update%50==0 ) else None, # every 50 steps logs, super  slow, remove to get faster
                random_action_prob=current_random_action_prob,
            )
            global_step += num_envs * T
    
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    
            
            # -------- GAE (bootstrap v(s_T) with env-minibatching) --------
            with torch.no_grad():
                agent_impl = agent.module if isinstance(agent, DDP) else agent
            
                # compute next_value = v(s_T) for each env, chunked over E
                next_value = torch.empty(E, device=device, dtype=torch.float32)
            
                mbE = 1  # <-- tune this (2,4,6,8...) based on memory
                for e0 in range(0, E, mbE):
                    e1 = min(E, e0 + mbE)
            
                    imgs_win = imgs_all[e0:e1, :T+1]
                    ks_win   = ks_all[e0:e1, :T+1]
                    c2ws_win = c2ws_all[e0:e1, :T+1]
                    imgs_n, ks_n, c2ws_n, _ = pre.process_window(
                        imgs_win, ks_win, c2ws_win,
                        device=device, amp_dtype=amp_dtype, outHW=hw
                    )
                    ro, rd = pre.compute_rays(c2ws_n, ks_n, h=hw, w=hw, device=device)
                    posed_9 = pre.build_visual_input(imgs_n, ro, rd)  # (mbE,T+1,9,hw,hw)
            
                    pad0 = torch.zeros_like(actions[e0:e1, :1])
                    prev_actions = torch.cat([pad0, actions[e0:e1]], dim=1)  # (mbE,T+1)
            
                    act_pose = action_pose_from_indices(
                        prev_actions,
                        action_T,
                        ks_n,
                        hw,
                        device,
                        pre,
                        zero_first_step=True,
                    )  # (mbE,T+1,6,hw,hw)
            
                    posed_15 = torch.cat([posed_9, act_pose], dim=2)  # (mbE,T+1,15,hw,hw)
            
                    with torch.cuda.amp.autocast(dtype=amp_dtype):
                        _, v_all = agent_impl(posed_15, last_only=False)  # (mbE,T+1)
            
                    next_value[e0:e1] = v_all[:, -1].float()
            
                    # optional: free big temporaries early
                    del imgs_n, ks_n, c2ws_n, ro, rd, posed_9, act_pose, posed_15, v_all
                    torch.cuda.empty_cache()
            
                adv = torch.zeros_like(rewards, device=device, dtype=torch.float32)
                lastgaelam = torch.zeros(E, device=device, dtype=torch.float32)  # per-env scalar
                next_done = torch.zeros(E, device=device, dtype=torch.float32)
                
                for t in reversed(range(T)):
                    if t == T - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[:, t]
                        nextvalues = values[:, t + 1]
            
                    delta = rewards[:, t] + float(gamma) * nextvalues * nextnonterminal - values[:, t]
                    lastgaelam = delta + float(gamma) * float(gae_lambda) * nextnonterminal * lastgaelam
                    adv[:, t] = lastgaelam
            
                ret = adv + values
               
    
                adv_norm = (adv - adv.mean(dim=1, keepdim=True)) / (adv.std(dim=1, keepdim=True, unbiased=False) + 1e-8)
            if rank == 0 :
                e_vis = 0
                envs.call(
                    "set_rollout_metrics",
                    rewards=rewards[e_vis].detach().cpu().tolist(),
                    returns=ret[e_vis].detach().cpu().tolist(),
                    advs=adv[e_vis].detach().cpu().tolist(),  # or adv[e_vis]
                )
            
                post_img = np.asarray(envs.call("render")[0])
                if post_img.shape[-1] == 4:
                    post_img = post_img[..., :3]
                wandb.log({"train/post_metrics_frame": wandb.Image(post_img)}, step=update)
            
    
            # -------- PPO update --------
            stats = ppo_update(
                agent=agent,
                optimizer=optimizer,
                pre=pre,
                action_T=action_T,
                device=device,
                amp_dtype=amp_dtype,
                hw=hw,
                imgs_all=imgs_all,
                c2ws_all=c2ws_all,
                ks_all=ks_all,
                actions=actions,
                logprobs=logprobs,
                advantages=adv_norm,
                returns=ret,
                values_old=values,
                update_epochs=update_epochs,
                clip_coef=clip_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                ent_coef_start=ent_coef_start,
                ent_decay_rate=ent_decay_rate,
                update_idx=update,
                random_action_prob=current_random_action_prob,
            )
            if dist.is_initialized():
                print(f"[rank{rank}] update={update} entering barrier_after_ppo", flush=True)
                dist.barrier()
                print(f"[rank{rank}] update={update} passed barrier_after_ppo", flush=True)

            if rank == 0:
                previous_bucket = global_step_before_update // checkpoint_milestone_steps
                current_bucket = global_step // checkpoint_milestone_steps
                for milestone_idx in range(previous_bucket + 1, current_bucket + 1):
                    milestone_step = milestone_idx * checkpoint_milestone_steps
                    milestone_path = _milestone_checkpoint_path(checkpoint_path, milestone_step)
                    print(
                        f"[checkpoint] saving milestone checkpoint at global_step={milestone_step}: {milestone_path}",
                        flush=True,
                    )
                    _save_checkpoint(
                        milestone_path,
                        step=global_step,
                        milestone_step=milestone_step,
                    )
            # -------- Eval + checkpoint --------
            eval_return = None
            eval_reward_mean = None
            eval_video_path = None
            greedy_eval = None
        
        if rank == 0 and save_eval_video and (update == 1 or update % int(eval_every) == 0):
            eval_count += 1
          
            greedy_eval = greedy_eval

            agent.eval()
            eval_video_path, eval_return, eval_reward_mean = eval_rollout_video(
                agent=agent,
                pre=pre,
                action_T=action_T,
                scene_list=test_scenes,
                device=device,
                amp_dtype=amp_dtype,
                roll_length=T_eval,
                hw=hw,
                big_hw=big_hw,
                nerf_iters=nerf_iters,
                cap_max=cap_max,
                seed=seed + 999_000 * update,
                out_dir="eval_videos_sliding",
                out_name=f"eval_{run_name}_u{update:04d}.mp4",
                greedy=greedy_eval,
                optimize_gsplat=train,
                use_gsplat=train,  # skip all GSplat (ingest, render, reward) for eval-only
            )
            agent.train()

            _save_checkpoint(
                checkpoint_path,
                step=global_step,
            )
            if eval_return is not None:
                writer.add_scalar("reward/eval_return", eval_return, update)
                writer.add_scalar("reward/eval_mean_per_step", eval_reward_mean, update)
                if greedy_eval is not None:
                    writer.add_scalar("eval/greedy", float(greedy_eval), update)
                    
            if eval_video_path is not None:
                log_payload["eval/video"] = wandb.Video(eval_video_path, format="mp4")

        # -------- Logging --------
        if rank == 0 and train:
            sps = int((E * T * world_size * update) / (time.time() - start_time))
            writer.add_scalar("charts/SPS", sps, update)
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], update)

            if stats.get("loss") is not None:
                writer.add_scalar("losses/all_loss", stats["loss"], update)
            if stats.get("v_loss") is not None:
                writer.add_scalar("losses/value_loss", stats["v_loss"], update)
            if stats.get("pg_loss") is not None:
                writer.add_scalar("losses/policy_loss", stats["pg_loss"], update)
            if stats.get("entropy") is not None:
                writer.add_scalar("losses/entropy", stats["entropy"], update)
            if stats.get("approx_kl") is not None:
                writer.add_scalar("losses/approx_kl", stats["approx_kl"], update)
            if stats.get("icm_loss") is not None:
                writer.add_scalar("losses/icm_loss", stats["icm_loss"], update)
            if stats.get("icm_inverse_loss") is not None:
                writer.add_scalar("losses/icm_inverse_loss", stats["icm_inverse_loss"], update)
            if stats.get("icm_forward_loss") is not None:
                writer.add_scalar("losses/icm_forward_loss", stats["icm_forward_loss"], update)
            writer.add_scalar("losses/clipfrac", float(stats.get("clipfrac", 0.0)), update)
            writer.add_scalar(
                "losses/explained_variance",
                explained_variance(values.view(-1).float(), ret.view(-1).float()),
                update
            )

            writer.add_scalar("reward/train_mean_per_step", rewards.mean().item(), update)
            writer.add_scalar("reward/train_sum_per_update", rewards.sum().item(), update)
            writer.add_scalar("charts/random_action_prob", current_random_action_prob, update)

            
            if train_video_path is not None:
                log_payload["train/video"] = wandb.Video(train_video_path, format="mp4")


        if rank == 0:
            wandb.log(log_payload, step=update)

        if dist.is_initialized():
            dist.barrier()

        if train:
            # explicit cleanup
            del actions, logprobs, rewards, dones, values, imgs_all, c2ws_all, ks_all, adv, ret
            torch.cuda.empty_cache()
    
    envs.close()

    if writer is not None:
        writer.close()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", type=str, default="runs")
    p.add_argument("--no_eval_video", action="store_true")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--base_ckpt", type=str, default="checkpoints_saved/explore_base.pt")
    p.add_argument("--num_envs", type=int, default=32)
    p.add_argument("--roll_length", type=int, default=1024)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--hw", type=int, default=64)
    p.add_argument("--big_hw", type=int, default=128)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae_lambda", type=float, default=0.97)
    p.add_argument("--update_epochs", type=int, default=3)
    p.add_argument("--clip_coef", type=float, default=0.2)
    p.add_argument("--ent_coef_start", type=float, default=0.1)
    p.add_argument("--ent_decay_rate", type=float, default=0.99)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--nerf_iters", type=int, default=10)
    p.add_argument("--cap_max", type=int, default=750_000)
    p.add_argument("--attn_window", type=int, default=64)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--checkpoint_path", type=str, default="checkpoints_saved/explore_base.pt")
    p.add_argument("--random_action_prob", type=float, default=0.2)
    p.add_argument("--anneal_random_action", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--anneal_random_steps", type=int, default=2_500_000)
    p.add_argument("--anneal_random_duration", type=int, default=750_000)
    p.add_argument("--eval_on_val", action="store_true", help="Use validation/eval scenes for eval (default: train scenes)")
    p.add_argument("--eval_roll_length", type=int, default=None, help="Max steps per eval rollout (default: same as roll_length)")
    p.add_argument("--eval_compile", action="store_true", help="Use torch.compile on agent for eval (PyTorch 2.0+)")
    p.add_argument("--greedy_eval", action="store_true", help="Use greedy (argmax) policy for eval rollout")
    args = p.parse_args()
    main(
        logdir=args.logdir,
        save_eval_video=(not args.no_eval_video),
        train=(not args.eval_only),
        base_ckpt=args.base_ckpt,
        num_envs=args.num_envs,
        roll_length=args.roll_length,
        learning_rate=args.learning_rate,
        hw=args.hw,
        big_hw=args.big_hw,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        update_epochs=args.update_epochs,
        clip_coef=args.clip_coef,
        ent_coef_start=args.ent_coef_start,
        ent_decay_rate=args.ent_decay_rate,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        nerf_iters=args.nerf_iters,
        cap_max=args.cap_max,
        attn_window=args.attn_window,
        eval_every=args.eval_every,
        checkpoint_path=args.checkpoint_path,
        random_action_prob=args.random_action_prob,
        anneal_random_action=args.anneal_random_action,
        anneal_random_steps=args.anneal_random_steps,
        anneal_random_duration=args.anneal_random_duration,
        eval_on_val=args.eval_on_val,
        eval_roll_length=args.eval_roll_length,
        eval_compile=args.eval_compile,
        greedy_eval=args.greedy_eval,
    )
