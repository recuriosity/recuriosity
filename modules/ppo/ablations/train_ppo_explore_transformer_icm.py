"""No-camera-pose PPO with ICM over the policy's per-frame transformer encoder."""

from __future__ import annotations

import math
import os
import sys

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.nn.parallel import DistributedDataParallel as DDP

_ppo_dir = os.path.dirname(os.path.abspath(__file__))
_modules_dir = os.path.dirname(os.path.dirname(_ppo_dir))
_root = os.path.dirname(_modules_dir)
if _root not in sys.path:
    sys.path.insert(0, _root)

import modules.ppo.ablations.train_ppo_explore_base as _base
from modules.agent.ablations.icm_feature import FeatureICM
from modules.ppo.train_ppo_explore_no_pose import (
    PoseProcessNoCameraPose,
    _adapt_patch_embed_weight,
)
from modules.ppo.train_ppo_explore_no_pose import NavAgentNoCameraPose as _BaseNavAgentNoCameraPose


def _build_policy_input_from_raw(
    pre,
    rgb_ref: torch.Tensor,
    k_ref: torch.Tensor,
    c2w_ref: torch.Tensor,
    rgb_cur: torch.Tensor,
    k_cur: torch.Tensor,
    c2w_cur: torch.Tensor,
    prev_actions: torch.Tensor,
    action_T: torch.Tensor,
    device: torch.device,
    amp_dtype: torch.dtype,
    hw: int,
    *,
    zero_first_step: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    rgb_win2 = torch.stack([rgb_ref, rgb_cur], dim=1)
    k_win2 = torch.stack([k_ref, k_cur], dim=1)
    c2w_win2 = torch.stack([c2w_ref, c2w_cur], dim=1)

    rgb_n2, k_n2, c2w_n2, _ = pre.process_window(
        rgb_win2, k_win2, c2w_win2, device=device, amp_dtype=amp_dtype, outHW=hw
    )
    rgb_n = rgb_n2[:, -1:]
    k_n = k_n2[:, -1:]
    c2w_n = c2w_n2[:, -1:]

    ro, rd = pre.compute_rays(c2w_n, k_n, h=hw, w=hw, device=device)
    posed_rgb = pre.build_visual_input(rgb_n, ro, rd)
    act_pose = _base.action_pose_from_indices(
        prev_actions,
        action_T,
        k_n,
        hw,
        device,
        pre,
        zero_first_step=zero_first_step,
    )
    posed_input = torch.cat([posed_rgb, act_pose], dim=2).squeeze(1)
    return posed_input, rgb_n[:, 0]


def _build_policy_input_from_normalized(
    pre,
    imgs_n: torch.Tensor,
    ks_n: torch.Tensor,
    c2ws_n: torch.Tensor,
    prev_actions: torch.Tensor,
    action_T: torch.Tensor,
    device: torch.device,
    hw: int,
    *,
    zero_first_step: bool,
) -> torch.Tensor:
    ro, rd = pre.compute_rays(c2ws_n, ks_n, h=hw, w=hw, device=device)
    posed_rgb = pre.build_visual_input(imgs_n.to(torch.float32), ro, rd)
    act_pose = _base.action_pose_from_indices(
        prev_actions,
        action_T,
        ks_n,
        hw,
        device,
        pre,
        zero_first_step=zero_first_step,
    )
    return torch.cat([posed_rgb, act_pose], dim=2)


class NavAgentNoCameraPoseICMFrameEncoder(_BaseNavAgentNoCameraPose):
    def __init__(
        self,
        *args,
        icm_hidden_dim: int = 256,
        icm_forward_loss_weight: float = 0.2,
        icm_reward_scale: float = 0.01,
        icm_loss_scale: float = 10.0,
        **kwargs,
    ):
        kwargs["in_channels"] = 9
        super().__init__(*args, **kwargs)
        self.icm = FeatureICM(
            feature_dim=self.d_model,
            action_dim=self.n_act,
            hidden_dim=icm_hidden_dim,
            forward_loss_weight=icm_forward_loss_weight,
            reward_scale=icm_reward_scale,
            loss_scale=icm_loss_scale,
        )

    @torch.no_grad()
    def load_ckpt(self, load_path, optimizer=None, strict=False):
        if os.path.isdir(load_path):
            ckpt_names = sorted(fn for fn in os.listdir(load_path) if fn.endswith(".pt"))
            assert ckpt_names, f"no checkpoint files found in {load_path}"
            ckpt_path = os.path.join(load_path, ckpt_names[-1])
        else:
            ckpt_path = load_path

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        target_state = self.state_dict()

        filtered_state = {}
        skipped_keys = []
        adapted_keys = []
        unexpected_keys = []

        for key, value in ckpt["model"].items():
            target_value = target_state.get(key)
            if target_value is None:
                unexpected_keys.append(key)
                continue
            if value.shape == target_value.shape:
                filtered_state[key] = value
                continue

            adapted_value = None
            if key == "patch_embed.1.weight":
                adapted_value = _adapt_patch_embed_weight(value, target_value)

            if adapted_value is not None:
                filtered_state[key] = adapted_value
                adapted_keys.append(key)
                continue

            skipped_keys.append((key, tuple(value.shape), tuple(target_value.shape)))

        missing, unexpected = self.load_state_dict(filtered_state, strict=False)
        if strict and (missing or unexpected or unexpected_keys or skipped_keys):
            raise RuntimeError(
                "checkpoint load failed with strict=True: "
                f"missing={missing}, unexpected={unexpected + unexpected_keys}, skipped={skipped_keys}"
            )

        print(f"[info] loaded ckpt: {ckpt_path}")
        if adapted_keys:
            print(f"[info] adapted keys for no-camera-pose input: {adapted_keys}")
        if skipped_keys:
            print(f"[warn] skipped mismatched keys: {skipped_keys}")
        if missing or unexpected or unexpected_keys:
            print(
                "[warn] checkpoint key differences: "
                f"missing={len(missing)}, unexpected={len(unexpected) + len(unexpected_keys)}"
            )

        if optimizer is not None and "optimizer" in ckpt:
            if adapted_keys or skipped_keys or missing or unexpected or unexpected_keys:
                print("[warn] skipped optimizer state because model parameter set changed")
            else:
                optimizer.load_state_dict(ckpt["optimizer"])
                print("[info] loaded optimizer state")

        step = int(ckpt.get("step", 0))
        return step, ckpt


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
    rank=0,
    out_path=None,
    random_action_prob=0.2,
):
    del big_hw, nerf_iters, cap_max, max_splats_steps, max_points_per_frame
    agent_impl = agent.module if isinstance(agent, DDP) else agent

    actions = torch.zeros(E, T, device=device, dtype=torch.long)
    logprobs = torch.zeros(E, T, device=device, dtype=torch.float32)
    rewards = torch.zeros(E, T, device=device, dtype=torch.float32)
    dones = torch.zeros(E, T, device=device, dtype=torch.float32)
    values = torch.zeros(E, T, device=device, dtype=torch.float32)

    obs, _ = envs.reset(seed=seed, options={"start": True})
    rgb, k, c2w, depth = _base.obs_to_img_pose(obs, device)

    H0, W0 = rgb.shape[-2], rgb.shape[-1]
    rgb_ref = rgb.clone()
    k_ref = k.clone()
    c2w_ref = c2w.clone()

    imgs_all = torch.empty(E, T + 1, 3, H0, W0, device=device, dtype=torch.uint8)
    ks_all = torch.empty(E, T + 1, 4, device=device, dtype=torch.float32)
    c2ws_all = torch.empty(E, T + 1, 4, 4, device=device, dtype=torch.float32)
    imgs_all[:, 0].copy_(rgb)
    ks_all[:, 0].copy_(k)
    c2ws_all[:, 0].copy_(c2w)

    kv_caches = agent_impl.init_kv_cache(batch_size=E)
    prev_action = torch.zeros(E, device=device, dtype=torch.long)
    next_done = torch.zeros(E, device=device, dtype=torch.float32)

    frames = None
    if rank == 0 and out_path is not None:
        frames = []

    for t in range(T):
        dones[:, t] = next_done

        rgb_c = imgs_all[:, t]
        k_c = ks_all[:, t]
        c2w_c = c2ws_all[:, t]

        posed_input, policy_rgb = _build_policy_input_from_raw(
            pre,
            rgb_ref,
            k_ref,
            c2w_ref,
            rgb_c,
            k_c,
            c2w_c,
            prev_action.unsqueeze(1),
            action_T,
            device,
            amp_dtype,
            hw,
            zero_first_step=(t == 0),
        )

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                frame_tok = agent_impl._encode_frame(posed_input)
                logits, value = agent_impl._forward_step_from_frame_token(frame_tok, kv_caches=kv_caches)
                probs = Categorical(logits=logits)
                policy_action = probs.sample()
                random_action = torch.randint(0, logits.shape[-1], (E,), device=device, dtype=torch.long)
                use_random = torch.rand(E, device=device) < float(random_action_prob)
                action = torch.where(use_random, random_action, policy_action)
                logprob = probs.log_prob(action)

        values[:, t].copy_(value.reshape(-1).float())
        actions[:, t].copy_(action)
        logprobs[:, t].copy_(logprob.float())
        prev_action = action

        obs_next, _, term, trunc, infos = envs.step(action.detach().cpu().numpy())
        done_np = np.logical_or(term, trunc)
        next_done = torch.as_tensor(done_np, device=device, dtype=torch.float32)

        rgb1, k1, c2w1, depth1 = _base.obs_to_img_pose(obs_next, device)
        imgs_all[:, t + 1].copy_(rgb1)
        ks_all[:, t + 1].copy_(k1)
        c2ws_all[:, t + 1].copy_(c2w1)
        next_posed_input, _ = _build_policy_input_from_raw(
            pre,
            rgb_ref,
            k_ref,
            c2w_ref,
            rgb1,
            k1,
            c2w1,
            action.unsqueeze(1),
            action_T,
            device,
            amp_dtype,
            hw,
            zero_first_step=False,
        )
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                next_frame_tok = agent_impl._encode_frame(next_posed_input)
            bonus = agent_impl.icm.intrinsic_reward(frame_tok.float(), next_frame_tok.float(), action)
        rewards[:, t].copy_(bonus.float())

        if rank == 0:
            envs.call("update_meta", [{
                "rgb_gt": policy_rgb[0].permute(1, 2, 0).float().detach().cpu().numpy(),
                "policy_logits": logits[0].float().detach().cpu().numpy(),
                "mode": int(2 if bool(use_random[0].item()) else 3),
            }])
            envs.call("push_step_metrics", reward=float(rewards[0, t].item()))

        if rank == 0 and frames is not None:
            frame = np.asarray(envs.call("render")[0])
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            frames.append(frame)

    if rank == 0 and frames is not None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        imageio.mimwrite(out_path, frames, fps=12, codec="libx264", bitrate="4M")

    agent_impl.clear_kv_cache(kv_caches)
    torch.cuda.empty_cache()
    return actions, logprobs, rewards, dones, values, imgs_all, c2ws_all, ks_all, out_path


def ppo_update(
    agent,
    optimizer,
    pre,
    action_T,
    device,
    amp_dtype,
    hw,
    imgs_all,
    c2ws_all,
    ks_all,
    actions,
    logprobs,
    advantages,
    returns,
    values_old,
    update_epochs=5,
    clip_coef=0.3,
    vf_coef=0.1,
    max_grad_norm=1.0,
    ent_coef_start=0.05,
    ent_decay_rate=0.99,
    update_idx=1,
    random_action_prob=0.2,
    mbE=1,
    mbE_pre=None,
):
    agent_impl = agent.module if isinstance(agent, DDP) else agent
    agent_impl.train()

    E, T = actions.shape
    mbE = int(mbE)
    mbE_pre = int(mbE if mbE_pre is None else mbE_pre)
    denom = max(1, math.ceil(E / mbE))

    imgs_win_full = imgs_all[:, : T + 1]
    ks_win_full = ks_all[:, : T + 1]
    c2ws_win_full = c2ws_all[:, : T + 1]

    imgs_n_all = torch.empty((E, T + 1, 3, hw, hw), device=device, dtype=torch.float32)
    ks_n_all = torch.empty((E, T + 1, 4), device=device, dtype=torch.float32)
    c2ws_n_all = torch.empty((E, T + 1, 4, 4), device=device, dtype=torch.float32)

    with torch.no_grad():
        for e0 in range(0, E, mbE_pre):
            e1 = min(E, e0 + mbE_pre)
            imgs_n, ks_n, c2ws_n, _ = pre.process_window(
                imgs_win_full[e0:e1],
                ks_win_full[e0:e1],
                c2ws_win_full[e0:e1],
                device=device,
                amp_dtype=amp_dtype,
                outHW=hw,
            )
            imgs_n_all[e0:e1].copy_(imgs_n.float())
            ks_n_all[e0:e1].copy_(ks_n.float())
            c2ws_n_all[e0:e1].copy_(c2ws_n.float())

            del imgs_n, ks_n, c2ws_n
            torch.cuda.empty_cache()

    del imgs_win_full, ks_win_full, c2ws_win_full
    torch.cuda.empty_cache()

    pad = torch.zeros_like(actions[:, :1])
    actions_shifted = torch.cat([pad, actions], dim=1)[:, :T]

    clipfracs = []
    pg_loss_log = v_loss_log = entropy_log = approx_kl_log = old_kl_log = loss_log = None
    icm_loss_log = inv_loss_log = fwd_loss_log = None

    for epoch in range(int(update_epochs)):
        optimizer.zero_grad(set_to_none=True)

        loss_sum = pg_sum = v_sum = ent_sum = kl_sum = oldkl_sum = 0.0
        icm_sum = inv_sum = fwd_sum = 0.0
        n_mb = 0
        clipfracs_epoch = []

        for e0 in range(0, E, mbE):
            e1 = min(E, e0 + mbE)
            mb = e1 - e0

            imgs_n_full = imgs_n_all[e0:e1]
            imgs_n = imgs_n_full[:, :T]
            imgs_n_next = imgs_n_full[:, 1 : T + 1]
            ks_n = ks_n_all[e0:e1, :T]
            ks_n_next = ks_n_all[e0:e1, 1 : T + 1]
            c2ws_n = c2ws_n_all[e0:e1, :T]
            c2ws_n_next = c2ws_n_all[e0:e1, 1 : T + 1]

            posed_input = _build_policy_input_from_normalized(
                pre,
                imgs_n,
                ks_n,
                c2ws_n,
                actions_shifted[e0:e1],
                action_T,
                device,
                hw,
                zero_first_step=True,
            )
            next_posed_input = _build_policy_input_from_normalized(
                pre,
                imgs_n_next,
                ks_n_next,
                c2ws_n_next,
                actions[e0:e1],
                action_T,
                device,
                hw,
                zero_first_step=False,
            )

            act_mb = actions[e0:e1]
            oldlp_mb = logprobs[e0:e1]
            adv_mb = advantages[e0:e1]
            ret_mb = returns[e0:e1]
            vold_mb = values_old[e0:e1]

            with torch.cuda.amp.autocast(dtype=amp_dtype):
                frame_tok = agent_impl._encode_frames_batch(posed_input)
                logits, v_pred = agent_impl._forward_from_frame_tokens(frame_tok, last_only=False)
                next_frame_tok = agent_impl._encode_frames_batch(next_posed_input)
                logits_f = logits.reshape(mb * T, -1)
                v_pred_f = v_pred.reshape(mb * T).float()
                probs = Categorical(logits=logits_f)
                newlogp = probs.log_prob(act_mb.reshape(mb * T))
                entropy = probs.entropy().mean()

            icm_stats = agent_impl.icm.losses(
                frame_tok.reshape(mb * T, -1).float(),
                next_frame_tok.reshape(mb * T, -1).float(),
                act_mb.reshape(mb * T),
            )

            oldlogp_agent = oldlp_mb.reshape(mb * T)
            oldprob_agent = oldlogp_agent.exp().clamp_min(1e-8)
            uniform_prob = 1.0 / float(logits_f.shape[-1])
            oldprob_mix = float(random_action_prob) * uniform_prob + (1.0 - float(random_action_prob)) * oldprob_agent
            oldlogp = torch.log(oldprob_mix.clamp_min(1e-8))
            logratio = newlogp.float() - oldlogp
            ratio = logratio.exp()

            with torch.no_grad():
                old_approx_kl = (-logratio).mean()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1.0).abs() > float(clip_coef)).float().mean()
                clipfracs_epoch.append(float(clipfrac.item()))

            adv_f = adv_mb.reshape(mb * T)
            pg_loss1 = -adv_f * ratio
            pg_loss2 = -adv_f * torch.clamp(ratio, 1 - float(clip_coef), 1 + float(clip_coef))
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            v_target_f = ret_mb.reshape(mb * T).float()
            v_old_f = vold_mb.reshape(mb * T).float()
            v_loss_unclipped = (v_pred_f - v_target_f) ** 2
            v_clipped = v_old_f + torch.clamp(v_pred_f - v_old_f, -float(clip_coef), float(clip_coef))
            del v_clipped
            v_loss = 0.5 * v_loss_unclipped.mean()

            ent_coef = max(float(ent_coef_start) * (float(ent_decay_rate) ** int(update_idx)), 1e-3)
            loss = pg_loss + float(vf_coef) * v_loss - ent_coef * entropy + icm_stats["loss"]


            (loss / float(denom)).backward()

            loss_sum += float(loss.detach().item())
            pg_sum += float(pg_loss.detach().item())
            v_sum += float(v_loss.detach().item())
            ent_sum += float(entropy.detach().item())
            kl_sum += float(approx_kl.detach().item())
            oldkl_sum += float(old_approx_kl.detach().item())
            icm_sum += float(icm_stats["loss"].detach().item())
            inv_sum += float(icm_stats["inverse_loss"].detach().item())
            fwd_sum += float(icm_stats["forward_loss"].detach().item())
            n_mb += 1

            del posed_input, next_posed_input
            del logits, v_pred, logits_f, v_pred_f, probs, newlogp, entropy, logratio, ratio
            del frame_tok, next_frame_tok
            del adv_f, pg_loss1, pg_loss2, pg_loss, v_target_f, v_old_f, v_loss_unclipped, v_loss, loss
            del icm_stats
            torch.cuda.empty_cache()

        nn.utils.clip_grad_norm_(agent_impl.parameters(), float(max_grad_norm))
        optimizer.step()

        clipfracs.extend(clipfracs_epoch)
        loss_log = torch.tensor(loss_sum / max(n_mb, 1), device=device)
        pg_loss_log = torch.tensor(pg_sum / max(n_mb, 1), device=device)
        v_loss_log = torch.tensor(v_sum / max(n_mb, 1), device=device)
        entropy_log = torch.tensor(ent_sum / max(n_mb, 1), device=device)
        approx_kl_log = torch.tensor(kl_sum / max(n_mb, 1), device=device)
        old_kl_log = torch.tensor(oldkl_sum / max(n_mb, 1), device=device)
        icm_loss_log = torch.tensor(icm_sum / max(n_mb, 1), device=device)
        inv_loss_log = torch.tensor(inv_sum / max(n_mb, 1), device=device)
        fwd_loss_log = torch.tensor(fwd_sum / max(n_mb, 1), device=device)

    return {
        "loss": float(loss_log.item()) if loss_log is not None else None,
        "pg_loss": float(pg_loss_log.item()) if pg_loss_log is not None else None,
        "v_loss": float(v_loss_log.item()) if v_loss_log is not None else None,
        "entropy": float(entropy_log.item()) if entropy_log is not None else None,
        "approx_kl": float(approx_kl_log.item()) if approx_kl_log is not None else None,
        "old_approx_kl": float(old_kl_log.item()) if old_kl_log is not None else None,
        "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
        "icm_loss": float(icm_loss_log.item()) if icm_loss_log is not None else None,
        "icm_inverse_loss": float(inv_loss_log.item()) if inv_loss_log is not None else None,
        "icm_forward_loss": float(fwd_loss_log.item()) if fwd_loss_log is not None else None,
    }


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
    del big_hw, nerf_iters, cap_max, max_splats_steps, max_points_per_frame, optimize_gsplat, use_gsplat
    agent_impl = agent.module if isinstance(agent, DDP) else agent
    agent_impl.eval()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)

    env = _base.HabitatMP3DEnv(scene_list, max_steps=int(roll_length) + 1, render_mode="rgb_array")
    env._topdown_regen_interval = 20
    env._eval_panel_layout = True
    obs, _ = env.reset(seed=seed, options={"start": False})

    rgb, k, c2w, depth = _base.obs_to_img_pose(obs, device)
    H0, W0 = rgb.shape[-2], rgb.shape[-1]
    rgb_ref = rgb.clone()
    k_ref = k.clone()
    c2w_ref = c2w.clone()

    imgs_all = torch.empty(1, roll_length + 1, 3, H0, W0, device=device, dtype=torch.uint8)
    ks_all = torch.empty(1, roll_length + 1, 4, device=device, dtype=torch.float32)
    c2ws_all = torch.empty(1, roll_length + 1, 4, 4, device=device, dtype=torch.float32)
    imgs_all[:, 0].copy_(rgb)
    ks_all[:, 0].copy_(k)
    c2ws_all[:, 0].copy_(c2w)

    kv_caches = agent_impl.init_kv_cache(batch_size=1)
    prev_action = torch.zeros(1, device=device, dtype=torch.long)
    rewards = []
    video_writer = imageio.get_writer(out_path, fps=12, codec="libx264", bitrate="4M")

    try:
        for t in range(int(roll_length)):
            rgb_c = imgs_all[:, t]
            k_c = ks_all[:, t]
            c2w_c = c2ws_all[:, t]

            posed_input, policy_rgb = _build_policy_input_from_raw(
                pre,
                rgb_ref.unsqueeze(0),
                k_ref.unsqueeze(0),
                c2w_ref.unsqueeze(0),
                rgb_c,
                k_c,
                c2w_c,
                prev_action.unsqueeze(1),
                action_T,
                device,
                amp_dtype,
                hw,
                zero_first_step=(t == 0),
            )

            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    frame_tok = agent_impl._encode_frame(posed_input)
                    logits, _ = agent_impl._forward_step_from_frame_token(frame_tok, kv_caches=kv_caches)

            action = torch.argmax(logits, dim=-1) if greedy else Categorical(logits=logits).sample()
            prev_action = action

            obs_next, _, term, trunc, infos = env.step(int(action.item()))
            rgb, k, c2w, depth = _base.obs_to_img_pose(obs_next, device)
            rgb_b = rgb.unsqueeze(0)
            k_b = k.unsqueeze(0)
            c2w_b = c2w.unsqueeze(0)
            depth_b = depth.unsqueeze(0)
            imgs_all[:, t + 1].copy_(rgb_b)
            ks_all[:, t + 1].copy_(k_b)
            c2ws_all[:, t + 1].copy_(c2w_b)
            next_posed_input, _ = _build_policy_input_from_raw(
                pre,
                rgb_ref.unsqueeze(0),
                k_ref.unsqueeze(0),
                c2w_ref.unsqueeze(0),
                rgb_b,
                k_b,
                c2w_b,
                action.unsqueeze(1),
                action_T,
                device,
                amp_dtype,
                hw,
                zero_first_step=False,
            )
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    next_frame_tok = agent_impl._encode_frame(next_posed_input)
                reward = agent_impl.icm.intrinsic_reward(frame_tok.float(), next_frame_tok.float(), action)[0].item()
            rewards.append(reward)

            env.update_meta([{
                "rgb_gt": policy_rgb[0].permute(1, 2, 0).float().cpu().numpy(),
                "policy_logits": logits[0].float().detach().cpu().numpy(),
                "mode": 3,
            }])
            env.push_step_metrics(reward=reward)

            frame = np.asarray(env.render())
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            video_writer.append_data(frame)

            if term or trunc:
                break
    finally:
        video_writer.close()
        env.close()
        agent_impl.clear_kv_cache(kv_caches)
        torch.cuda.empty_cache()

    return out_path, float(np.sum(rewards)), float(np.mean(rewards) if rewards else 0.0)


PoseProcess = PoseProcessNoCameraPose
NavAgent = NavAgentNoCameraPoseICMFrameEncoder

init_distributed = _base.init_distributed
explained_variance = _base.explained_variance
obs_to_img_pose = _base.obs_to_img_pose
se3_from_translation_rotation = _base.se3_from_translation_rotation
action_pose_from_indices = _base.action_pose_from_indices
build_anchor_w2c = _base.build_anchor_w2c
preprocess_big_step = _base.preprocess_big_step
render_from_gsplat = _base.render_from_gsplat


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
    learning_rate=1.5e-5,
    hw=64,
    big_hw=128,
    gamma=0.995,
    gae_lambda=0.97,
    update_epochs=3,
    clip_coef=0.2,
    ent_coef_start=0.1,
    ent_decay_rate=0.985,
    vf_coef=0.5,
    max_grad_norm=1.0,
    nerf_iters=10,
    cap_max=750_000,
    attn_window=64,
    eval_every=10,
    checkpoint_path="checkpoints_saved/explore_transformer_icm.pt",
    random_action_prob=0.2,
    anneal_random_action=True,
    anneal_random_steps=2_500_000,
    anneal_random_duration=500_000,
):
    base_pose_process = _base.PoseProcess
    base_nav_agent = _base.NavAgent
    base_rollout_with_cache = _base.rollout_with_cache
    base_ppo_update = _base.ppo_update
    base_eval_rollout_video = _base.eval_rollout_video
    base_wandb_init = _base.wandb.init

    def _wandb_init_with_icm(*args, **kwargs):
        name = kwargs.get("name")
        if isinstance(name, str) and "nocam_icm_frameenc" not in name:
            kwargs["name"] = name.replace(
                "train_explore_base__",
                "train_explore_transformer_icm__",
                1,
            )
        return base_wandb_init(*args, **kwargs)

    _base.PoseProcess = PoseProcessNoCameraPose
    _base.NavAgent = NavAgentNoCameraPoseICMFrameEncoder
    _base.rollout_with_cache = rollout_with_cache
    _base.ppo_update = ppo_update
    _base.eval_rollout_video = eval_rollout_video
    _base.wandb.init = _wandb_init_with_icm
    try:
        return _base.main(
            logdir=logdir,
            save_eval_video=save_eval_video,
            train=train,
            base_ckpt=base_ckpt,
            eval_on_val=eval_on_val,
            eval_compile=eval_compile,
            greedy_eval=greedy_eval,
            num_envs=num_envs,
            roll_length=roll_length,
            eval_roll_length=eval_roll_length,
            learning_rate=learning_rate,
            hw=hw,
            big_hw=big_hw,
            gamma=gamma,
            gae_lambda=gae_lambda,
            update_epochs=update_epochs,
            clip_coef=clip_coef,
            ent_coef_start=ent_coef_start,
            ent_decay_rate=ent_decay_rate,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            nerf_iters=nerf_iters,
            cap_max=cap_max,
            attn_window=attn_window,
            eval_every=eval_every,
            checkpoint_path=checkpoint_path,
            random_action_prob=random_action_prob,
            anneal_random_action=anneal_random_action,
            anneal_random_steps=anneal_random_steps,
            anneal_random_duration=anneal_random_duration,
        )
    finally:
        _base.PoseProcess = base_pose_process
        _base.NavAgent = base_nav_agent
        _base.rollout_with_cache = base_rollout_with_cache
        _base.ppo_update = base_ppo_update
        _base.eval_rollout_video = base_eval_rollout_video
        _base.wandb.init = base_wandb_init


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="runs")
    parser.add_argument("--no_eval_video", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--base_ckpt", type=str, default=None)
    parser.add_argument("--eval_on_val", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval_roll_length", type=int, default=None)
    parser.add_argument("--eval_compile", action="store_true")
    parser.add_argument("--greedy_eval", action="store_true")
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--roll_length", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1.5e-5)
    parser.add_argument("--hw", type=int, default=64)
    parser.add_argument("--big_hw", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.97)
    parser.add_argument("--update_epochs", type=int, default=3)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--ent_coef_start", type=float, default=0.1)
    parser.add_argument("--ent_decay_rate", type=float, default=0.985)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--nerf_iters", type=int, default=10)
    parser.add_argument("--cap_max", type=int, default=750_000)
    parser.add_argument("--attn_window", type=int, default=64)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints_saved/explore_transformer_icm.pt",
    )
    parser.add_argument("--random_action_prob", type=float, default=0.2)
    parser.add_argument("--anneal_random_action", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--anneal_random_steps", type=int, default=2_500_000)
    parser.add_argument("--anneal_random_duration", type=int, default=750_000)
    args = parser.parse_args()

    main(
        logdir=args.logdir,
        save_eval_video=not args.no_eval_video,
        train=not args.eval_only,
        base_ckpt=args.base_ckpt,
        eval_on_val=args.eval_on_val,
        eval_compile=args.eval_compile,
        greedy_eval=args.greedy_eval,
        num_envs=args.num_envs,
        roll_length=args.roll_length,
        eval_roll_length=args.eval_roll_length,
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
    )
