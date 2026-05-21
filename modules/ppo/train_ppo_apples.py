
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
from modules.environment.env_apples import (
    ACT_LIST,
    STEP_METERS,
    YAW_DEG,
    HabitatMP3DEnv,
    PoseProcess,
    list_scene_glbs,
)

from modules.agent.agent import NavAgent

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


def _checkpoint_path_with_step_suffix(checkpoint_path: str, global_step: int) -> str:
    ckpt_dir = os.path.dirname(checkpoint_path) or "."
    ckpt_name = os.path.basename(checkpoint_path)
    ckpt_stem, ckpt_ext = os.path.splitext(ckpt_name)
    if not ckpt_ext:
        ckpt_ext = ".pt"
    return os.path.join(ckpt_dir, f"{ckpt_stem}_{int(global_step)}{ckpt_ext}")


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
# ============================================
# Rollout (policy uses 2-frame canon; reward comes from environment)
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
    reward_coef=0.5,
):
    """
    Returns:
      actions   (E,T) long
      logprobs  (E,T) float (log P_agent_old(action))
      rewards   (E,T) float
      dones     (E,T) float
      valid_mask(E,T) float (1=real transition, 0=post-terminal padding)
      values    (E,T) float
      imgs_all  (E,T+1,3,H,W) uint8
      ks_all    (E,T+1,4) float
      c2ws_all  (E,T+1,4,4) float
    """
    del big_hw, nerf_iters, cap_max, max_splats_steps, max_points_per_frame

    agent_impl = agent.module if isinstance(agent, DDP) else agent

    actions  = torch.zeros(E, T, device=device, dtype=torch.long)
    logprobs = torch.zeros(E, T, device=device, dtype=torch.float32)
    rewards  = torch.zeros(E, T, device=device, dtype=torch.float32)
    dones    = torch.zeros(E, T, device=device, dtype=torch.float32)
    values   = torch.zeros(E, T, device=device, dtype=torch.float32)
    stop_action_idx = int(len(ACT_LIST) - 1)
    env_finished = np.zeros(E, dtype=bool)
    first_padded_step = np.full(E, T, dtype=np.int32)

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

        
    kv_caches = agent_impl.init_kv_cache(batch_size=E)
    # New: if the agent has a global-memory (TTT/linear-attention) state,
    # create it alongside the KV caches. Backwards-compatible: agents that
    # don't have these methods just skip this branch.
    global_state = None
    if hasattr(agent_impl, "init_global_state"):
        global_state = agent_impl.init_global_state(
            batch_size=E, device=device, dtype=amp_dtype,
        )
    prev_action = torch.zeros(E, device=device, dtype=torch.long)

    frames = None
    if rank == 0 and out_path is not None:
            frames = []

    for t in range(T):
        if env_finished.any():
            prev_action[torch.as_tensor(env_finished, device=device, dtype=torch.bool)] = 0

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
                fs_kwargs = {}
                if global_state is not None:
                    fs_kwargs["global_state"] = global_state
                logits, value = agent_impl.forward_step(
                    posed_15, time_idx=t, kv_caches=kv_caches, **fs_kwargs,
                )
                probs = Categorical(logits=logits)
                action = probs.sample()
                if float(random_action_prob) > 0.0:
                    random_mask = torch.rand(action.shape, device=device) < float(random_action_prob)
                    if random_mask.any():
                        random_actions = torch.randint(
                            low=0,
                            high=logits.shape[-1],
                            size=action.shape,
                            device=device,
                        )
                        action = torch.where(random_mask, random_actions, action)
                logprob = torch.log_softmax(logits, dim=-1).gather(-1, action.unsqueeze(-1)).squeeze(-1)

        values[:, t].copy_(value.reshape(-1).float())
        actions[:, t].copy_(action)
        logprobs[:, t].copy_(logprob.float())
        finished_before = torch.as_tensor(env_finished, device=device, dtype=torch.bool)
        if finished_before.any():
            actions[finished_before, t] = stop_action_idx
            logprobs[finished_before, t] = 0.0
            if t > 0:
                values[finished_before, t] = values[finished_before, t - 1]
            else:
                values[finished_before, t] = 0.0

            
        obs_next, env_reward, term, trunc, infos = envs.step(actions[:, t].detach().cpu().numpy())
        done_np = np.logical_or(term, trunc)
        done_t = torch.as_tensor(done_np, device=device, dtype=torch.float32)
        if finished_before.any():
            done_t[finished_before] = 1.0
        dones[:, t].copy_(done_t)

        rgb1, k1, c2w1, depth1 = obs_to_img_pose(obs_next, device)

        imgs_all[:, t + 1].copy_(rgb1)
        ks_all[:, t + 1].copy_(k1)
        c2ws_all[:, t + 1].copy_(c2w1)
        if finished_before.any():
            imgs_all[finished_before, t + 1].copy_(imgs_all[finished_before, t])
            ks_all[finished_before, t + 1].copy_(ks_all[finished_before, t])
            c2ws_all[finished_before, t + 1].copy_(c2ws_all[finished_before, t])

        env_reward_t = torch.as_tensor(env_reward, device=device, dtype=torch.float32)
        rewards_t = float(reward_coef) * env_reward_t
        if finished_before.any():
            rewards_t[finished_before] = 0.0
        rewards[:, t].copy_(rewards_t)
        prev_action = actions[:, t].clone()
        prev_action[dones[:, t] > 0.5] = 0

        newly_done = np.logical_and(np.logical_not(env_finished), done_np)
        first_padded_step[newly_done] = t + 1
        env_finished = np.logical_or(env_finished, done_np)
        
      

        if rank == 0:
            envs.call("update_meta", [{
                "policy_logits": logits[0].float().detach().cpu().numpy(),
                "mode": 3,
            }])
            
            envs.call("push_step_metrics", reward=float(rewards[0, t].item()))

       
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
    if global_state is not None and hasattr(agent_impl, "clear_global_state"):
        agent_impl.clear_global_state(global_state)
    torch.cuda.empty_cache()

    valid_mask = torch.ones(E, T, device=device, dtype=torch.float32)
    for e in range(E):
        if first_padded_step[e] < T:
            valid_mask[e, first_padded_step[e] : T] = 0.0

    return actions, logprobs, rewards, dones, valid_mask, values, imgs_all, c2ws_all, ks_all, out_path


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
    valid_mask,  # (E,T) 1 for real transitions, 0 for padded post-terminal steps
    update_epochs=5,
    clip_coef=0.3,
    vf_coef=0.1,
    max_grad_norm=1.0,
    ent_coef_start=0.0,
    ent_decay_rate=0.99,
    update_idx=1,
    random_action_prob=0.2,
    mbE=1,              # env-minibatch size for PPO
    mbE_pre=None,       # env-minibatch size for pre.process_window (defaults to mbE)
    target_kl_early_stop=False,
    target_kl=0.02,
):
    """
    PPO update that avoids recomputing pre.process_window every epoch:
      - preprocess (process_window) ONCE per update (chunked over envs)
      - cache imgs_n/ks_n/c2ws_n
      - each epoch: env-minibatch -> compute rays/posed/action_pose -> agent forward/backward
    """
    agent_impl = agent.module if isinstance(agent, DDP) else agent
    agent_impl.train()
    del random_action_prob

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
    epochs_ran = 0
    target_kl_triggered = False

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
            valid_flat = valid_mask[e0:e1].reshape(mb * T)
            n_valid = valid_flat.sum().clamp(min=1.0)

            with torch.cuda.amp.autocast(dtype=amp_dtype):
                logits, v_pred = agent_impl(posed_15, last_only=False)  # (mb,T,A), (mb,T)

                logits_f = logits.reshape(mb * T, -1)
                v_pred_f = v_pred.reshape(mb * T).float()

                probs = Categorical(logits=logits_f)
                newlogp = probs.log_prob(act_mb.reshape(mb * T))
                entropy_per_step = probs.entropy()
                entropy = (entropy_per_step * valid_flat).sum() / n_valid

            oldlogp = oldlp_mb.reshape(mb * T)
            logratio = (newlogp.float() - oldlogp)
            ratio = logratio.exp()

            with torch.no_grad():
                old_approx_kl = ((-logratio) * valid_flat).sum() / n_valid
                approx_kl = (((ratio - 1) - logratio) * valid_flat).sum() / n_valid
                clipfrac = (((ratio - 1.0).abs() > float(clip_coef)).float() * valid_flat).sum() / n_valid
                clipfracs_epoch.append(float(clipfrac.item()))

            adv_f = adv_mb.reshape(mb * T)  # NO normalization
            pg_loss1 = -adv_f * ratio
            pg_loss2 = -adv_f * torch.clamp(ratio, 1 - float(clip_coef), 1 + float(clip_coef))
            pg_loss = (torch.max(pg_loss1, pg_loss2) * valid_flat).sum() / n_valid

            v_target_f = ret_mb.reshape(mb * T).float()
            v_old_f    = vold_mb.reshape(mb * T).float()
            v_loss_unclipped = (v_pred_f - v_target_f) ** 2
            v_clipped = v_old_f + torch.clamp(v_pred_f - v_old_f, -float(clip_coef), float(clip_coef))
            v_loss = 0.5 * (v_loss_unclipped * valid_flat).sum() / n_valid
                       
            ent_coef = float(ent_coef_start) * (float(ent_decay_rate) ** int(update_idx))
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
            del logits, v_pred, logits_f, v_pred_f, probs, newlogp, entropy_per_step, entropy, logratio, ratio
            del adv_f, pg_loss1, pg_loss2, pg_loss, v_target_f, v_old_f, v_loss_unclipped, v_clipped, v_loss, loss
            del valid_flat, n_valid
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
        epochs_ran = int(epoch) + 1

        if bool(target_kl_early_stop) and float(approx_kl_log.item()) > float(target_kl):
            target_kl_triggered = True
            print(
                f"[ppo] early-stop update epoch {epochs_ran}/{int(update_epochs)} "
                f"because approx_kl={float(approx_kl_log.item()):.6f} > target_kl={float(target_kl):.6f}",
                flush=True,
            )
            break

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
        "epochs_ran": int(epochs_ran),
        "target_kl_early_stop": bool(target_kl_triggered),
        "target_kl": float(target_kl),
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
    reward_coef=0.5,
    apple_collect_radius_m=1.5,
):
    del big_hw, nerf_iters, cap_max, max_splats_steps, max_points_per_frame

    agent_impl = agent.module if isinstance(agent, DDP) else agent
    agent_impl.eval()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)

    env = HabitatMP3DEnv(
        scene_list,
        max_steps=int(roll_length) + 1,
        render_mode="rgb_array",
        apple_collect_radius_m=float(apple_collect_radius_m),
    )
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

    kv_caches = agent_impl.init_kv_cache(batch_size=1)
    global_state = None
    if hasattr(agent_impl, "init_global_state"):
        global_state = agent_impl.init_global_state(
            batch_size=1, device=device, dtype=amp_dtype,
        )
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
                    fs_kwargs = {}
                    if global_state is not None:
                        fs_kwargs["global_state"] = global_state
                    logits, _ = agent_impl.forward_step(
                        posed_15, time_idx=t, kv_caches=kv_caches, **fs_kwargs,
                    )
            if greedy:
                action = torch.argmax(logits, dim=-1)
            else:
                action = Categorical(logits=logits).sample()
            prev_action = action

            _t3 = time.perf_counter()
            # env step
            obs_next, env_reward, term, trunc, infos = env.step(int(action.item()))
            rgb, k, c2w, depth = obs_to_img_pose(obs_next, device)

            imgs_all[:, t + 1].copy_(rgb)
            ks_all[:, t + 1].copy_(k)
            c2ws_all[:, t + 1].copy_(c2w)
            rew = float(reward_coef) * float(env_reward)
            rewards.append(rew)

            _t4 = time.perf_counter()
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
            env.push_step_metrics(reward=rew)

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
            dt_meta = _t5 - _t4
            dt_render = _t6 - _t5
            dt_video = _t7 - _t6
            dt_total = _t7 - _t0
            print(
                f"[eval t={t}] process={dt_process*1000:.0f}ms agent={dt_agent*1000:.0f}ms "
                f"env_step={dt_env*1000:.0f}ms meta={dt_meta*1000:.0f}ms "
                f"render={dt_render*1000:.0f}ms video={dt_video*1000:.0f}ms total={dt_total*1000:.0f}ms",
                flush=True,
            )

            if term or trunc:
                break
   
    finally:
        video_writer.close()
        env.close()
        agent_impl.clear_kv_cache(kv_caches)
        if global_state is not None and hasattr(agent_impl, "clear_global_state"):
            agent_impl.clear_global_state(global_state)
        torch.cuda.empty_cache()

    return out_path, float(np.sum(rewards)), float(np.mean(rewards) if rewards else 0.0)


# Main
# ============================================



def main(
    logdir="runs",
    save_eval_video=True,
    train=True,
    base_ckpt=None,
    weights_only_ckpt=None,
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
    ent_coef_start=0.0,
    ent_decay_rate=0.99,
    vf_coef=0.5,
    reward_coef=0.5,
    num_apples=5,
    max_grad_norm=1.0,
    nerf_iters=10,
    cap_max=750_000,
    attn_window=64,
    eval_every=10,
    checkpoint_path="checkpoints_saved/apples.pt",
    random_action_prob=0.0,
    apple_collect_radius_m=1.5,
    anneal_random_action=False,
    anneal_random_steps=2_500_000,
    anneal_random_duration=500_000,
    wandb_name_suffix="",
    archive_checkpoint_interval=200_000,
    target_kl_early_stop=False,
    target_kl=0.02,
    reset_value_head_after_load=False,
):
    rank, world_size, local_rank = init_distributed()
    session_dir = os.environ.get("TRAINING_SESSION_DIR", "")
    cap_max = int(os.environ.get("CAP_MAX", str(cap_max)))
    archive_checkpoint_interval = int(
        os.environ.get("ARCHIVE_CHECKPOINT_INTERVAL", str(archive_checkpoint_interval))
    )
    archive_checkpoint_interval = max(0, archive_checkpoint_interval)
    target_kl_early_stop = (
        os.environ.get("TARGET_KL_EARLY_STOP", str(target_kl_early_stop)).strip().lower()
        in {"1", "true", "yes", "on"}
    )
    target_kl = float(os.environ.get("TARGET_KL", str(target_kl)))
    reset_value_head_after_load = (
        os.environ.get("RESET_VALUE_HEAD_AFTER_LOAD", str(reset_value_head_after_load)).strip().lower()
        in {"1", "true", "yes", "on"}
    )
    reward_coef = float(
        os.environ.get("REWARD_COEF", os.environ.get("APPLE_REWARD_COEF", str(reward_coef)))
    )
    num_apples = int(os.environ.get("NUM_APPLES", os.environ.get("APPLE_NUM_APPLES", str(num_apples))))
    apple_collect_radius_m = float(
        os.environ.get("APPLE_COLLECT_RADIUS_M", str(apple_collect_radius_m))
    )
    weights_only_ckpt = os.environ.get("WEIGHTS_ONLY_CKPT", str(weights_only_ckpt or "")).strip() or None
    wandb_name_suffix = os.environ.get("WANDB_NAME_SUFFIX", wandb_name_suffix).strip()
    if not session_dir and logdir and os.path.isabs(logdir):
        session_dir = logdir
    if session_dir:
        checkpoint_path = os.path.join(session_dir, "checkpoints", os.path.basename(checkpoint_path))

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    amp_dtype = torch.bfloat16

    base_seed = int(os.environ.get("TRAIN_SEED", "42"))
    seed = base_seed + rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    run_name = f"train_apples__{seed}__{int(time.time())}"
    if wandb_name_suffix:
        run_name = f"{run_name}_{wandb_name_suffix}"
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
            tags=["ppo", "apple", "nav", "sliding", "ttt"],
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
            env = HabitatMP3DEnv(
                all_scenes,
                max_steps=int(roll_length) + 1,
                render_mode="rgb_array",
                gpu_id=local_rank,
                num_apples=int(num_apples),
                apple_collect_radius_m=float(apple_collect_radius_m),
            )
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

    # Loading priority:
    # 1) session checkpoint resume (full state: model+optimizer+step)
    # 2) weights_only_ckpt (model only; optimizer/step untouched)
    # 3) base_ckpt resume (full state)
    def _resolve_ckpt_path(path: str | None) -> str | None:
        if not path:
            return None
        resolved = path
        if not os.path.isabs(resolved) and session_dir:
            alt = os.path.join(session_dir, "checkpoints", os.path.basename(resolved))
            if os.path.exists(alt):
                resolved = alt
        return resolved

    resume_path = None
    resumed_from_session_ckpt = False
    loaded_ckpt_source = None
    if session_dir and os.path.exists(checkpoint_path):
        resume_path = checkpoint_path
        resumed_from_session_ckpt = True
        if rank == 0:
            print(f"[resume] found checkpoint in session dir, resuming from {checkpoint_path}", flush=True)
    elif weights_only_ckpt:
        pass
    elif base_ckpt:
        resume_path = base_ckpt
    resume_path = _resolve_ckpt_path(resume_path)

    if resume_path and os.path.exists(resume_path):
        to_load = agent.module if isinstance(agent, DDP) else agent
        step, ckpt = to_load.load_ckpt(resume_path, optimizer=optimizer, strict=False)
        loaded_ckpt_source = "session" if resumed_from_session_ckpt else "base"
    
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
    elif weights_only_ckpt:
        weights_only_path = _resolve_ckpt_path(weights_only_ckpt)
        if not weights_only_path or not os.path.exists(weights_only_path):
            raise FileNotFoundError(
                f"weights_only_ckpt not found: {weights_only_ckpt}"
            )
        to_load = agent.module if isinstance(agent, DDP) else agent
        _loaded_step, _ = to_load.load_ckpt(weights_only_path, optimizer=None, strict=False)
        loaded_ckpt_source = "weights_only"
        if rank == 0:
            print(
                f"[init] loaded weights-only checkpoint from {weights_only_path}; "
                "keeping fresh optimizer state and global_step=0",
                flush=True,
            )
    if (
        reset_value_head_after_load
        and loaded_ckpt_source in {"base", "weights_only"}
    ):
        to_reset = agent.module if isinstance(agent, DDP) else agent
        if hasattr(to_reset, "v_head") and isinstance(to_reset.v_head, nn.Linear):
            nn.init.normal_(to_reset.v_head.weight, mean=0.0, std=1e-2)
            nn.init.zeros_(to_reset.v_head.bias)
            for p in to_reset.v_head.parameters():
                optimizer.state.pop(p, None)
            if rank == 0:
                print(
                    f"[init] reset value head after {loaded_ckpt_source} checkpoint load; "
                    "value head optimizer state cleared",
                    flush=True,
                )
        elif rank == 0:
            print(
                "[init] RESET_VALUE_HEAD_AFTER_LOAD is set, but no nn.Linear v_head was found",
                flush=True,
            )
    if resumed_from_session_ckpt and weights_only_ckpt and rank == 0:
        print(
            "[init] weights_only_ckpt was provided but ignored because session resume is active",
            flush=True,
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

        if train:
            global_step_before_update = global_step
            if rank == 0:
                train_video_path = f"train_videos_sliding/train_{run_name}_u{update:04d}.mp4"
    
            
        
            # -------- rollout --------
            agent.eval()
            actions, logprobs, rewards, dones, valid_mask, values, imgs_all, c2ws_all, ks_all, train_video_path = rollout_with_cache(
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
                random_action_prob=float(random_action_prob),
                reward_coef=reward_coef,
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
                
                for t in reversed(range(T)):
                    if t == T - 1:
                        nextvalues = next_value
                    else:
                        nextvalues = values[:, t + 1]
                    nextnonterminal = 1.0 - dones[:, t]
            
                    delta = rewards[:, t] + float(gamma) * nextvalues * nextnonterminal - values[:, t]
                    lastgaelam = delta + float(gamma) * float(gae_lambda) * nextnonterminal * lastgaelam
                    adv[:, t] = lastgaelam
            
                ret = adv + values

                adv[valid_mask == 0] = 0.0
                ret[valid_mask == 0] = 0.0

                valid_adv = adv[valid_mask == 1]
                if valid_adv.numel() > 0:
                    adv_mean = valid_adv.mean()
                    adv_std = valid_adv.std(unbiased=False) + 1e-8
                    adv_norm = (adv - adv_mean) / adv_std
                else:
                    adv_norm = torch.zeros_like(adv)
                adv_norm[valid_mask == 0] = 0.0
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
                valid_mask=valid_mask,
                update_epochs=update_epochs,
                clip_coef=clip_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                ent_coef_start=ent_coef_start,
                ent_decay_rate=ent_decay_rate,
                update_idx=update,
                target_kl_early_stop=target_kl_early_stop,
                target_kl=target_kl,
            )
            if dist.is_initialized():
                print(f"[rank{rank}] update={update} entering barrier_after_ppo", flush=True)
                dist.barrier()
                print(f"[rank{rank}] update={update} passed barrier_after_ppo", flush=True)

            if rank == 0 and archive_checkpoint_interval > 0:
                crossed_archive_boundary = (
                    global_step // archive_checkpoint_interval
                    > global_step_before_update // archive_checkpoint_interval
                )
                if crossed_archive_boundary:
                    to_save = agent.module if isinstance(agent, DDP) else agent
                    archive_checkpoint_path = _checkpoint_path_with_step_suffix(checkpoint_path, global_step)
                    os.makedirs(os.path.dirname(archive_checkpoint_path) or ".", exist_ok=True)
                    to_save.save_ckpt(
                        archive_checkpoint_path,
                        optimizer=optimizer,
                        step=global_step,
                        extra={
                            "step_is_total": True,
                        },
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
                reward_coef=reward_coef,
                apple_collect_radius_m=apple_collect_radius_m,
            )
            agent.train()

            to_save = agent.module if isinstance(agent, DDP) else agent
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            to_save.save_ckpt(
                checkpoint_path,
                optimizer=optimizer,
                step=global_step,
                extra={
                    "step_is_total": True,
                },
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
            writer.add_scalar("losses/ppo_epochs_ran", float(stats.get("epochs_ran", 0)), update)
            writer.add_scalar(
                "losses/target_kl_early_stop",
                1.0 if bool(stats.get("target_kl_early_stop", False)) else 0.0,
                update,
            )
            writer.add_scalar("losses/clipfrac", float(stats.get("clipfrac", 0.0)), update)
            writer.add_scalar(
                "losses/explained_variance",
                explained_variance(values.view(-1).float(), ret.view(-1).float()),
                update
            )

            writer.add_scalar("reward/train_mean_per_step", rewards.mean().item(), update)
            writer.add_scalar("reward/train_sum_per_update", rewards.sum().item(), update)

            
            if train_video_path is not None:
                log_payload["train/video"] = wandb.Video(train_video_path, format="mp4")


        if rank == 0:
            wandb.log(log_payload, step=update)

        if dist.is_initialized():
            dist.barrier()

        if train:
            # explicit cleanup
            del actions, logprobs, rewards, dones, valid_mask, values, imgs_all, c2ws_all, ks_all, adv, ret
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
    p.add_argument("--base_ckpt", type=str, default="checkpoints_saved/apples.pt")
    p.add_argument(
        "--weights_only_ckpt",
        type=str,
        default=None,
        help="Load model weights only (no optimizer/step restore). Ignored when session resume checkpoint exists.",
    )
    p.add_argument("--num_envs", type=int, default=16)
    p.add_argument("--roll_length", type=int, default=1024)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--hw", type=int, default=64)
    p.add_argument("--big_hw", type=int, default=128)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae_lambda", type=float, default=0.97)
    p.add_argument("--update_epochs", type=int, default=3)
    p.add_argument("--clip_coef", type=float, default=0.2)
    p.add_argument("--ent_coef_start", type=float, default=0.0)
    p.add_argument("--ent_decay_rate", type=float, default=0.99)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--reward_coef", type=float, default=0.5)
    p.add_argument("--num_apples", type=int, default=5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--nerf_iters", type=int, default=10)
    p.add_argument("--cap_max", type=int, default=750_000)
    p.add_argument("--attn_window", type=int, default=64)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--checkpoint_path", type=str, default="checkpoints_saved/apples.pt")
    p.add_argument("--random_action_prob", type=float, default=0.0)
    p.add_argument("--anneal_random_action", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--anneal_random_steps", type=int, default=2_500_000)
    p.add_argument("--anneal_random_duration", type=int, default=750_000)
    p.add_argument("--wandb_name_suffix", type=str, default="")
    p.add_argument("--archive_checkpoint_interval", type=int, default=200_000)
    p.add_argument("--target_kl_early_stop", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--target_kl", type=float, default=0.02)
    p.add_argument("--reset_value_head_after_load", action=argparse.BooleanOptionalAction, default=False)
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
        weights_only_ckpt=args.weights_only_ckpt,
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
        reward_coef=args.reward_coef,
        num_apples=args.num_apples,
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
        wandb_name_suffix=args.wandb_name_suffix,
        archive_checkpoint_interval=args.archive_checkpoint_interval,
        target_kl_early_stop=args.target_kl_early_stop,
        target_kl=args.target_kl,
        reset_value_head_after_load=args.reset_value_head_after_load,
        eval_on_val=args.eval_on_val,
        eval_roll_length=args.eval_roll_length,
        eval_compile=args.eval_compile,
        greedy_eval=args.greedy_eval,
    )
