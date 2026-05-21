"""Image-goal TTT PPO variant using the image-goal Habitat environment."""

from __future__ import annotations

import os
import sys

_ppo_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_ppo_dir))
if _root not in sys.path:
    sys.path.insert(0, _root)

import modules.ppo.train_ppo_apples as _base
from modules.agent.agent_image_goal import NavAgent as ImageGoalNavAgent
from modules.environment.env_image_goal import HabitatMP3DEnv


torch = _base.torch
np = _base.np
imageio = _base.imageio
DDP = _base.DDP
Categorical = _base.Categorical
ACT_LIST = _base.ACT_LIST


DEFAULT_REWARD_COEF = 10.0
FORCED_RANDOM_ACTION_PROB = 0.0
FORCED_ENTROPY_COEF_START = 0.0
FORCED_ENTROPY_DECAY_RATE = 1.0

PoseProcess = _base.PoseProcess
NavAgent = ImageGoalNavAgent

init_distributed = _base.init_distributed
explained_variance = _base.explained_variance
obs_to_img_pose = _base.obs_to_img_pose
se3_from_translation_rotation = _base.se3_from_translation_rotation
action_pose_from_indices = _base.action_pose_from_indices


def _goal_rgb_from_obs(obs, device, default_hw=None):
    goal = obs.get("goal_rgb") if isinstance(obs, dict) else None
    if goal is None:
        if default_hw is None:
            raise KeyError("Image-goal observation is missing 'goal_rgb'")
        h, w = default_hw
        return torch.zeros(1, 3, h, w, device=device, dtype=torch.uint8)

    goal_rgb = torch.as_tensor(goal, device=device, dtype=torch.uint8)
    if goal_rgb.ndim == 4:
        return goal_rgb.permute(0, 3, 1, 2).contiguous()
    if goal_rgb.ndim == 3:
        return goal_rgb.permute(2, 0, 1).unsqueeze(0).contiguous()
    raise ValueError(f"Unexpected goal_rgb shape: {tuple(goal_rgb.shape)}")


def _goal_rgb_to_policy_resolution(pre, goal_rgb_u8, amp_dtype, hw):
    return pre.preprocess_images(goal_rgb_u8[:, None], amp_dtype, outHW=hw)[:, 0]


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
    reward_coef=0.5,
):
    del big_hw, nerf_iters, cap_max, max_splats_steps, max_points_per_frame, random_action_prob

    agent_impl = agent.module if isinstance(agent, DDP) else agent

    actions = torch.zeros(E, T, device=device, dtype=torch.long)
    logprobs = torch.zeros(E, T, device=device, dtype=torch.float32)
    rewards = torch.zeros(E, T, device=device, dtype=torch.float32)
    dones = torch.zeros(E, T, device=device, dtype=torch.float32)
    values = torch.zeros(E, T, device=device, dtype=torch.float32)
    stop_action_idx = int(len(ACT_LIST) - 1)
    env_finished = np.zeros(E, dtype=bool)
    first_padded_step = np.full(E, T, dtype=np.int32)

    obs, _ = envs.reset(seed=seed, options={"start": True})

    rgb, k, c2w, depth = obs_to_img_pose(obs, device)

    h0, w0 = rgb.shape[-2], rgb.shape[-1]
    goal_rgb_u8 = _goal_rgb_from_obs(obs, device, default_hw=(h0, w0))
    goal_rgb_n = _goal_rgb_to_policy_resolution(pre, goal_rgb_u8, amp_dtype, hw)
    if hasattr(agent_impl, "set_episode_goal_rgb"):
        agent_impl.set_episode_goal_rgb(goal_rgb_n)

    rgb_ref = rgb.clone()
    k_ref = k.clone()
    c2w_ref = c2w.clone()

    imgs_all = torch.empty(E, T + 1, 3, h0, w0, device=device, dtype=torch.uint8)
    ks_all = torch.empty(E, T + 1, 4, device=device, dtype=torch.float32)
    c2ws_all = torch.empty(E, T + 1, 4, 4, device=device, dtype=torch.float32)
    imgs_all[:, 0].copy_(rgb)
    ks_all[:, 0].copy_(k)
    c2ws_all[:, 0].copy_(c2w)

    kv_caches = agent_impl.init_kv_cache(batch_size=E)
    global_state = None
    if hasattr(agent_impl, "init_global_state"):
        global_state = agent_impl.init_global_state(
            batch_size=E,
            device=device,
            dtype=amp_dtype,
        )
    prev_action = torch.zeros(E, device=device, dtype=torch.long)

    frames = None
    if rank == 0 and out_path is not None:
        frames = []

    for t in range(T):
        if env_finished.any():
            prev_action[torch.as_tensor(env_finished, device=device, dtype=torch.bool)] = 0

        rgb_c = imgs_all[:, t]
        k_c = ks_all[:, t]
        c2w_c = c2ws_all[:, t]

        rgb_win2 = torch.stack([rgb_ref, rgb_c], dim=1)
        k_win2 = torch.stack([k_ref, k_c], dim=1)
        c2w_win2 = torch.stack([c2w_ref, c2w_c], dim=1)

        rgb_n2, k_n2, c2w_n2, _ = pre.process_window(
            rgb_win2,
            k_win2,
            c2w_win2,
            device=device,
            amp_dtype=amp_dtype,
            outHW=hw,
        )

        rgb_n = rgb_n2[:, -1:]
        k_n = k_n2[:, -1:]
        c2w_n = c2w_n2[:, -1:]

        ro, rd = pre.compute_rays(c2w_n, k_n, h=hw, w=hw, device=device)
        posed_9 = pre.build_visual_input(rgb_n, ro, rd)

        act_pose = action_pose_from_indices(
            prev_action.unsqueeze(1),
            action_T,
            k_n,
            hw,
            device,
            pre,
            zero_first_step=(t == 0),
        )

        posed_15 = torch.cat([posed_9, act_pose], dim=2).squeeze(1)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=amp_dtype):
                fs_kwargs = {"goal_rgb": goal_rgb_n}
                if global_state is not None:
                    fs_kwargs["global_state"] = global_state
                logits, value = agent_impl.forward_step(
                    posed_15,
                    time_idx=t,
                    kv_caches=kv_caches,
                    **fs_kwargs,
                )
                probs = Categorical(logits=logits)
                action = probs.sample()
                logprob = probs.log_prob(action)

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
            envs.call(
                "update_meta",
                [
                    {
                        "policy_logits": logits[0].float().detach().cpu().numpy(),
                        "mode": 3,
                    }
                ],
            )

            envs.call("push_step_metrics", reward=float(rewards[0, t].item()))

        if rank == 0 and frames is not None:
            frame = envs.call("render")[0]
            frame = np.asarray(frame)
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            frames.append(frame)

    if rank == 0 and frames is not None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        imageio.mimwrite(out_path, frames, fps=12, codec="libx264", bitrate="4M")
    if hasattr(agent_impl, "set_episode_goal_rgb"):
        agent_impl.set_episode_goal_rgb(goal_rgb_n)
    agent_impl.clear_kv_cache(kv_caches)
    if global_state is not None and hasattr(agent_impl, "clear_global_state"):
        agent_impl.clear_global_state(global_state)
    torch.cuda.empty_cache()

    valid_mask = torch.ones(E, T, device=device, dtype=torch.float32)
    for e in range(E):
        if first_padded_step[e] < T:
            valid_mask[e, first_padded_step[e]:T] = 0.0

    return actions, logprobs, rewards, dones, valid_mask, values, imgs_all, c2ws_all, ks_all, out_path


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
    valid_mask,
    update_epochs=5,
    clip_coef=0.3,
    vf_coef=0.1,
    max_grad_norm=1.0,
    ent_coef_start=0.0,
    ent_decay_rate=0.99,
    update_idx=1,
    random_action_prob=0.2,
    mbE=1,
    mbE_pre=None,
    target_kl_early_stop=False,
    target_kl=0.02,
):
    agent_impl = agent.module if isinstance(agent, DDP) else agent
    agent_impl.train()
    del random_action_prob

    e_count, t_count = actions.shape
    mbE = int(mbE)
    mbE_pre = int(mbE if mbE_pre is None else mbE_pre)
    denom = max(1, int(np.ceil(e_count / mbE)))

    goal_rgb_env = None
    if hasattr(agent_impl, "_episode_goal_rgb") and isinstance(agent_impl._episode_goal_rgb, torch.Tensor):
        goal_rgb_env = agent_impl._episode_goal_rgb
        if goal_rgb_env.device != device:
            goal_rgb_env = goal_rgb_env.to(device=device)
        if int(goal_rgb_env.shape[0]) < int(e_count):
            goal_rgb_env = None

    imgs_win = imgs_all[:, :t_count]
    ks_win = ks_all[:, :t_count]
    c2ws_win = c2ws_all[:, :t_count]

    imgs_n_all = torch.empty((e_count, t_count, 3, hw, hw), device=device, dtype=amp_dtype)
    ks_n_all = torch.empty((e_count, t_count, 4), device=device, dtype=torch.float32)
    c2ws_n_all = torch.empty((e_count, t_count, 4, 4), device=device, dtype=torch.float32)

    with torch.no_grad():
        for e0 in range(0, e_count, mbE_pre):
            e1 = min(e_count, e0 + mbE_pre)

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

    del imgs_win, ks_win, c2ws_win
    torch.cuda.empty_cache()

    pad = torch.zeros_like(actions[:, :1])
    actions_shifted = torch.cat([pad, actions], dim=1)[:, :t_count]

    clipfracs = []
    pg_loss_log = v_loss_log = entropy_log = approx_kl_log = old_kl_log = loss_log = None
    epochs_ran = 0
    target_kl_triggered = False

    for epoch in range(int(update_epochs)):
        optimizer.zero_grad(set_to_none=True)

        loss_sum = pg_sum = v_sum = ent_sum = kl_sum = oldkl_sum = 0.0
        n_mb = 0
        clipfracs_epoch = []

        for e0 in range(0, e_count, mbE):
            e1 = min(e_count, e0 + mbE)
            mb = e1 - e0

            imgs_n = imgs_n_all[e0:e1]
            ks_n = ks_n_all[e0:e1]
            c2ws_n = c2ws_n_all[e0:e1]

            ro, rd = pre.compute_rays(c2ws_n, ks_n, h=hw, w=hw, device=device)
            posed_9 = pre.build_visual_input(imgs_n, ro, rd)

            act_pose = action_pose_from_indices(
                actions_shifted[e0:e1],
                action_T,
                ks_n,
                hw,
                device,
                pre,
                zero_first_step=True,
            )

            posed_15 = torch.cat([posed_9, act_pose], dim=2)

            act_mb = actions[e0:e1]
            oldlp_mb = logprobs[e0:e1]
            adv_mb = advantages[e0:e1]
            ret_mb = returns[e0:e1]
            vold_mb = values_old[e0:e1]
            valid_flat = valid_mask[e0:e1].reshape(mb * t_count)
            n_valid = valid_flat.sum().clamp(min=1.0)

            goal_mb = None
            if goal_rgb_env is not None:
                goal_mb = goal_rgb_env[e0:e1]

            with torch.cuda.amp.autocast(dtype=amp_dtype):
                logits, v_pred = agent_impl(posed_15, goal_rgb=goal_mb, last_only=False)

                logits_f = logits.reshape(mb * t_count, -1)
                v_pred_f = v_pred.reshape(mb * t_count).float()

                probs = Categorical(logits=logits_f)
                newlogp = probs.log_prob(act_mb.reshape(mb * t_count))
                entropy_per_step = probs.entropy()
                entropy = (entropy_per_step * valid_flat).sum() / n_valid

            oldlogp = oldlp_mb.reshape(mb * t_count)
            logratio = newlogp.float() - oldlogp
            ratio = logratio.exp()

            with torch.no_grad():
                old_approx_kl = ((-logratio) * valid_flat).sum() / n_valid
                approx_kl = (((ratio - 1) - logratio) * valid_flat).sum() / n_valid
                clipfrac = (((ratio - 1.0).abs() > float(clip_coef)).float() * valid_flat).sum() / n_valid
                clipfracs_epoch.append(float(clipfrac.item()))

            adv_f = adv_mb.reshape(mb * t_count)
            pg_loss1 = -adv_f * ratio
            pg_loss2 = -adv_f * torch.clamp(ratio, 1 - float(clip_coef), 1 + float(clip_coef))
            pg_loss = (torch.max(pg_loss1, pg_loss2) * valid_flat).sum() / n_valid

            v_target_f = ret_mb.reshape(mb * t_count).float()
            v_old_f = vold_mb.reshape(mb * t_count).float()
            v_loss_unclipped = (v_pred_f - v_target_f) ** 2
            v_clipped = v_old_f + torch.clamp(v_pred_f - v_old_f, -float(clip_coef), float(clip_coef))
            v_loss = 0.5 * (v_loss_unclipped * valid_flat).sum() / n_valid

            ent_coef = float(ent_coef_start) * (float(ent_decay_rate) ** int(update_idx))
            loss = 1.0 * pg_loss + float(vf_coef) * v_loss - ent_coef * entropy

            (loss / float(denom)).backward()

            loss_sum += float(loss.detach().item())
            pg_sum += float(pg_loss.detach().item())
            v_sum += float(v_loss.detach().item())
            ent_sum += float(entropy.detach().item())
            kl_sum += float(approx_kl.detach().item())
            oldkl_sum += float(old_approx_kl.detach().item())
            n_mb += 1

            del ro, rd, posed_9, act_pose, posed_15
            del logits, v_pred, logits_f, v_pred_f, probs, newlogp, entropy_per_step, entropy, logratio, ratio
            del adv_f, pg_loss1, pg_loss2, pg_loss, v_target_f, v_old_f, v_loss_unclipped, v_clipped, v_loss, loss
            del valid_flat, n_valid
            torch.cuda.empty_cache()

        _base.nn.utils.clip_grad_norm_(agent_impl.parameters(), float(max_grad_norm))
        optimizer.step()

        loss_log = torch.tensor(loss_sum / max(1, n_mb), device=device)
        pg_loss_log = torch.tensor(pg_sum / max(1, n_mb), device=device)
        v_loss_log = torch.tensor(v_sum / max(1, n_mb), device=device)
        entropy_log = torch.tensor(ent_sum / max(1, n_mb), device=device)
        approx_kl_log = torch.tensor(kl_sum / max(1, n_mb), device=device)
        old_kl_log = torch.tensor(oldkl_sum / max(1, n_mb), device=device)

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
    apple_collect_radius_m=None,
    **_unused_kwargs,
):
    del (
        big_hw,
        nerf_iters,
        cap_max,
        max_splats_steps,
        max_points_per_frame,
        apple_collect_radius_m,
        _unused_kwargs,
    )

    agent_impl = agent.module if isinstance(agent, DDP) else agent
    agent_impl.eval()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, out_name)

    env = HabitatMP3DEnv(scene_list, max_steps=int(roll_length) + 1, render_mode="rgb_array")
    obs, _ = env.reset(seed=seed, options={"start": False})

    rgb, k, c2w, depth = obs_to_img_pose(obs, device)
    h0, w0 = rgb.shape[-2], rgb.shape[-1]
    goal_rgb_u8 = _goal_rgb_from_obs(obs, device, default_hw=(h0, w0))
    goal_rgb_n = _goal_rgb_to_policy_resolution(pre, goal_rgb_u8, amp_dtype, hw)
    if hasattr(agent_impl, "set_episode_goal_rgb"):
        agent_impl.set_episode_goal_rgb(goal_rgb_n)

    rgb_ref = rgb.clone()
    k_ref = k.clone()
    c2w_ref = c2w.clone()

    imgs_all = torch.empty(1, roll_length + 1, 3, h0, w0, device=device, dtype=torch.uint8)
    ks_all = torch.empty(1, roll_length + 1, 4, device=device, dtype=torch.float32)
    c2ws_all = torch.empty(1, roll_length + 1, 4, 4, device=device, dtype=torch.float32)
    imgs_all[:, 0].copy_(rgb)
    ks_all[:, 0].copy_(k)
    c2ws_all[:, 0].copy_(c2w)

    kv_caches = agent_impl.init_kv_cache(batch_size=1)
    global_state = None
    if hasattr(agent_impl, "init_global_state"):
        global_state = agent_impl.init_global_state(
            batch_size=1,
            device=device,
            dtype=amp_dtype,
        )
    prev_action = torch.zeros(1, device=device, dtype=torch.long)

    rewards = []
    video_writer = imageio.get_writer(out_path, fps=12, codec="libx264", bitrate="4M")

    try:
        for t in range(int(roll_length)):
            _t0 = _base.time.perf_counter()

            rgb_c = imgs_all[:, t]
            k_c = ks_all[:, t]
            c2w_c = c2ws_all[:, t]

            rgb_win2 = torch.stack([rgb_ref.unsqueeze(0), rgb_c], dim=1)
            k_win2 = torch.stack([k_ref.unsqueeze(0), k_c], dim=1)
            c2w_win2 = torch.stack([c2w_ref.unsqueeze(0), c2w_c], dim=1)

            _t1 = _base.time.perf_counter()
            rgb_n2, k_n2, c2w_n2, _ = pre.process_window(
                rgb_win2,
                k_win2,
                c2w_win2,
                device=device,
                amp_dtype=amp_dtype,
                outHW=hw,
            )
            rgb_n = rgb_n2[:, -1:]
            k_n = k_n2[:, -1:]
            c2w_n = c2w_n2[:, -1:]

            ro, rd = pre.compute_rays(c2w_n, k_n, h=hw, w=hw, device=device)
            posed_9 = pre.build_visual_input(rgb_n, ro, rd)

            act_pose = action_pose_from_indices(
                prev_action.unsqueeze(1),
                action_T,
                k_n,
                hw,
                device,
                pre,
                zero_first_step=(t == 0),
            )
            posed_15 = torch.cat([posed_9, act_pose], dim=2).squeeze(1)

            _t2 = _base.time.perf_counter()
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    fs_kwargs = {"goal_rgb": goal_rgb_n}
                    if global_state is not None:
                        fs_kwargs["global_state"] = global_state
                    logits, _ = agent_impl.forward_step(
                        posed_15,
                        time_idx=t,
                        kv_caches=kv_caches,
                        **fs_kwargs,
                    )
            if greedy:
                action = torch.argmax(logits, dim=-1)
            else:
                action = Categorical(logits=logits).sample()
            prev_action = action

            _t3 = _base.time.perf_counter()
            obs_next, env_reward, term, trunc, infos = env.step(int(action.item()))
            rgb, k, c2w, depth = obs_to_img_pose(obs_next, device)

            imgs_all[:, t + 1].copy_(rgb)
            ks_all[:, t + 1].copy_(k)
            c2ws_all[:, t + 1].copy_(c2w)
            rew = float(reward_coef) * float(env_reward)
            rewards.append(rew)

            _t4 = _base.time.perf_counter()
            if rgb.dim() == 4:
                rgb_gt = rgb[0].permute(1, 2, 0).float().cpu().numpy() / 255.0
            else:
                rgb_gt = rgb.permute(1, 2, 0).float().cpu().numpy() / 255.0
            env.update_meta(
                [
                    {
                        "rgb_gt": rgb_gt,
                        "policy_logits": logits[0].float().detach().cpu().numpy(),
                        "mode": 3,
                    }
                ]
            )
            env.push_step_metrics(reward=rew)

            _t5 = _base.time.perf_counter()
            frame = np.asarray(env.render())
            if frame.shape[-1] == 4:
                frame = frame[..., :3]
            _t6 = _base.time.perf_counter()
            video_writer.append_data(frame)
            _t7 = _base.time.perf_counter()

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


def _patch_wandb_init_for_image_goal():
    base_wandb_init = _base.wandb.init

    def _wandb_init_with_image_goal(*args, **kwargs):
        name = kwargs.get("name")
        if isinstance(name, str):
            updated_name = name.replace(
                "train_apples__",
                "train_image_goal__",
                1,
            )
            if updated_name == name and "image_goal" not in updated_name:
                updated_name = f"{updated_name}_image_goal"
            kwargs["name"] = updated_name

        tags = kwargs.get("tags")
        if isinstance(tags, (list, tuple)):
            updated_tags = ["image_goal" if str(tag) == "apple" else tag for tag in tags]
            if "image_goal" not in updated_tags:
                updated_tags.append("image_goal")
            kwargs["tags"] = updated_tags

        return base_wandb_init(*args, **kwargs)

    return base_wandb_init, _wandb_init_with_image_goal


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
    reward_coef=DEFAULT_REWARD_COEF,
    max_grad_norm=1.0,
    nerf_iters=10,
    cap_max=750_000,
    attn_window=64,
    eval_every=10,
    checkpoint_path="checkpoints_saved/image_goal.pt",
    random_action_prob=0.0,
    anneal_random_action=False,
    anneal_random_steps=2_500_000,
    anneal_random_duration=500_000,
    wandb_name_suffix="",
    archive_checkpoint_interval=200_000,
    target_kl_early_stop=False,
    target_kl=0.02,
    reset_value_head_after_load=False,
):
    base_env_cls = _base.HabitatMP3DEnv
    base_nav_agent = _base.NavAgent
    base_rollout_with_cache = _base.rollout_with_cache
    base_ppo_update = _base.ppo_update
    base_eval_rollout_video = _base.eval_rollout_video
    base_wandb_init, image_goal_wandb_init = _patch_wandb_init_for_image_goal()

    old_reward_coef_env = os.environ.get("REWARD_COEF")
    image_goal_reward_override = os.environ.get("IMAGE_GOAL_REWARD_COEF", "").strip()
    if image_goal_reward_override:
        reward_coef = float(image_goal_reward_override)
    elif old_reward_coef_env is not None and old_reward_coef_env.strip():
        reward_coef = float(old_reward_coef_env)

    os.environ["REWARD_COEF"] = str(float(reward_coef))
    if abs(float(random_action_prob) - float(FORCED_RANDOM_ACTION_PROB)) > 1e-12:
        print(
            f"[image_goal] overriding random_action_prob={float(random_action_prob)} "
            f"-> {FORCED_RANDOM_ACTION_PROB}",
            flush=True,
        )
    if bool(anneal_random_action):
        print(
            "[image_goal] overriding anneal_random_action=True -> False",
            flush=True,
        )
    if abs(float(ent_coef_start) - float(FORCED_ENTROPY_COEF_START)) > 1e-12:
        print(
            f"[image_goal] overriding ent_coef_start={float(ent_coef_start)} "
            f"-> {FORCED_ENTROPY_COEF_START}",
            flush=True,
        )

    random_action_prob = float(FORCED_RANDOM_ACTION_PROB)
    anneal_random_action = False
    ent_coef_start = float(FORCED_ENTROPY_COEF_START)
    ent_decay_rate = float(FORCED_ENTROPY_DECAY_RATE)

    _base.HabitatMP3DEnv = HabitatMP3DEnv
    _base.NavAgent = ImageGoalNavAgent
    _base.rollout_with_cache = rollout_with_cache
    _base.ppo_update = ppo_update
    _base.eval_rollout_video = eval_rollout_video
    _base.wandb.init = image_goal_wandb_init
    try:
        return _base.main(
            logdir=logdir,
            save_eval_video=save_eval_video,
            train=train,
            base_ckpt=base_ckpt,
            weights_only_ckpt=weights_only_ckpt,
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
            reward_coef=reward_coef,
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
            wandb_name_suffix=wandb_name_suffix,
            archive_checkpoint_interval=archive_checkpoint_interval,
            target_kl_early_stop=target_kl_early_stop,
            target_kl=target_kl,
            reset_value_head_after_load=reset_value_head_after_load,
        )
    finally:
        _base.HabitatMP3DEnv = base_env_cls
        _base.NavAgent = base_nav_agent
        _base.rollout_with_cache = base_rollout_with_cache
        _base.ppo_update = base_ppo_update
        _base.eval_rollout_video = base_eval_rollout_video
        _base.wandb.init = base_wandb_init
        if old_reward_coef_env is None:
            os.environ.pop("REWARD_COEF", None)
        else:
            os.environ["REWARD_COEF"] = old_reward_coef_env


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="runs")
    parser.add_argument("--no_eval_video", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument(
        "--base_ckpt",
        type=str,
        default="checkpoints_saved/image_goal.pt",
    )
    parser.add_argument("--weights_only_ckpt", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--roll_length", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--hw", type=int, default=64)
    parser.add_argument("--big_hw", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.97)
    parser.add_argument("--update_epochs", type=int, default=3)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--ent_coef_start", type=float, default=0.0)
    parser.add_argument("--ent_decay_rate", type=float, default=0.99)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--reward_coef", type=float, default=DEFAULT_REWARD_COEF)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--nerf_iters", type=int, default=10)
    parser.add_argument("--cap_max", type=int, default=750_000)
    parser.add_argument("--attn_window", type=int, default=64)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints_saved/image_goal.pt",
    )
    parser.add_argument("--random_action_prob", type=float, default=0.0)
    parser.add_argument("--anneal_random_action", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--anneal_random_steps", type=int, default=2_500_000)
    parser.add_argument("--anneal_random_duration", type=int, default=750_000)
    parser.add_argument("--wandb_name_suffix", type=str, default="")
    parser.add_argument("--archive_checkpoint_interval", type=int, default=200_000)
    parser.add_argument("--target_kl_early_stop", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target_kl", type=float, default=0.02)
    parser.add_argument("--reset_value_head_after_load", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval_on_val", action="store_true")
    parser.add_argument("--eval_roll_length", type=int, default=None)
    parser.add_argument("--eval_compile", action="store_true")
    parser.add_argument("--greedy_eval", action="store_true")
    args = parser.parse_args()

    main(
        logdir=args.logdir,
        save_eval_video=(not args.no_eval_video),
        train=(not args.eval_only),
        base_ckpt=args.base_ckpt,
        weights_only_ckpt=args.weights_only_ckpt,
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
        reward_coef=args.reward_coef,
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
    )
