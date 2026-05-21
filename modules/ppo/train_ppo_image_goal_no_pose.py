"""Image-goal TTT PPO variant without camera-pose channels in policy input."""

from __future__ import annotations

import os
import sys

_ppo_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_ppo_dir))
if _root not in sys.path:
    sys.path.insert(0, _root)

import modules.ppo.train_ppo_apples as _apple_base
import modules.ppo.train_ppo_apples_no_pose as _base
from modules.environment.env_image_goal import HabitatMP3DEnv
from modules.ppo.train_ppo_image_goal import DEFAULT_REWARD_COEF

FORCED_RANDOM_ACTION_PROB = 0.0
FORCED_ENTROPY_COEF_START = 0.0
FORCED_ENTROPY_DECAY_RATE = 1.0


PoseProcess = _base.PoseProcess
NavAgent = _base.NavAgent

init_distributed = _base.init_distributed
explained_variance = _base.explained_variance
obs_to_img_pose = _base.obs_to_img_pose
se3_from_translation_rotation = _base.se3_from_translation_rotation
action_pose_from_indices = _base.action_pose_from_indices
rollout_with_cache = _base.rollout_with_cache
ppo_update = _base.ppo_update
eval_rollout_video = _base.eval_rollout_video


def main(
    logdir="runs",
    save_eval_video=True,
    train=True,
    base_ckpt=None,
    weights_only_ckpt=None,
    eval_on_val=False,
    eval_compile=False,
    greedy_eval=False,
    num_envs=72,
    roll_length=1024,
    eval_roll_length=None,
    learning_rate=None,
    hw=64,
    big_hw=128,
    gamma=0.995,
    gae_lambda=0.97,
    update_epochs=3,
    clip_coef=0.2,
    ent_coef_start=0.0,
    ent_decay_rate=None,
    vf_coef=0.5,
    reward_coef=DEFAULT_REWARD_COEF,
    max_grad_norm=1.0,
    nerf_iters=10,
    cap_max=750_000,
    attn_window=64,
    eval_every=10,
    checkpoint_path="checkpoints_saved/image_goal_no_pose.pt",
    random_action_prob=0.0,
    anneal_random_action=False,
    anneal_random_steps=2_500_000,
    anneal_random_duration=500_000,
    wandb_name_suffix="image_goal",
    archive_checkpoint_interval=200_000,
    target_kl_early_stop=False,
    target_kl=0.02,
):
    old_env_cls = _apple_base.HabitatMP3DEnv
    old_reward_coef_env = os.environ.get("REWARD_COEF")
    old_goal_terminate_env = os.environ.get("IMAGE_GOAL_TERMINATE_ON_SUCCESS")
    image_goal_reward_override = os.environ.get("IMAGE_GOAL_REWARD_COEF", "").strip()
    if image_goal_reward_override:
        reward_coef = float(image_goal_reward_override)
    elif old_reward_coef_env is not None and old_reward_coef_env.strip():
        reward_coef = float(old_reward_coef_env)

    os.environ["REWARD_COEF"] = str(float(reward_coef))
    # Image-goal PPO intentionally uses only environment rewards, no randmix exploration policy,
    # and no entropy regularization.
    if abs(float(random_action_prob) - float(FORCED_RANDOM_ACTION_PROB)) > 1e-12:
        print(
            f"[image_goal_nocam] overriding random_action_prob={float(random_action_prob)} "
            f"-> {FORCED_RANDOM_ACTION_PROB}",
            flush=True,
        )
    if bool(anneal_random_action):
        print(
            "[image_goal_nocam] overriding anneal_random_action=True -> False",
            flush=True,
        )
    if abs(float(ent_coef_start) - float(FORCED_ENTROPY_COEF_START)) > 1e-12:
        print(
            f"[image_goal_nocam] overriding ent_coef_start={float(ent_coef_start)} "
            f"-> {FORCED_ENTROPY_COEF_START}",
            flush=True,
        )

    random_action_prob = float(FORCED_RANDOM_ACTION_PROB)
    anneal_random_action = False
    ent_coef_start = float(FORCED_ENTROPY_COEF_START)
    ent_decay_rate = float(FORCED_ENTROPY_DECAY_RATE)

    _apple_base.HabitatMP3DEnv = HabitatMP3DEnv
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
        )
    finally:
        _apple_base.HabitatMP3DEnv = old_env_cls
        if old_reward_coef_env is None:
            os.environ.pop("REWARD_COEF", None)
        else:
            os.environ["REWARD_COEF"] = old_reward_coef_env
        if old_goal_terminate_env is None:
            os.environ.pop("IMAGE_GOAL_TERMINATE_ON_SUCCESS", None)
        else:
            os.environ["IMAGE_GOAL_TERMINATE_ON_SUCCESS"] = old_goal_terminate_env


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="runs")
    parser.add_argument("--no_eval_video", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument(
        "--base_ckpt",
        type=str,
        default="checkpoints_saved/image_goal_no_pose.pt",
    )
    parser.add_argument("--weights_only_ckpt", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=72)
    parser.add_argument("--roll_length", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--hw", type=int, default=64)
    parser.add_argument("--big_hw", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae_lambda", type=float, default=0.97)
    parser.add_argument("--update_epochs", type=int, default=3)
    parser.add_argument("--clip_coef", type=float, default=0.2)
    parser.add_argument("--ent_coef_start", type=float, default=0.0)
    parser.add_argument("--ent_decay_rate", type=float, default=None)
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
        default="checkpoints_saved/image_goal_no_pose.pt",
    )
    parser.add_argument("--random_action_prob", type=float, default=0.0)
    parser.add_argument("--anneal_random_action", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--anneal_random_steps", type=int, default=2_500_000)
    parser.add_argument("--anneal_random_duration", type=int, default=750_000)
    parser.add_argument("--wandb_name_suffix", type=str, default="image_goal")
    parser.add_argument("--archive_checkpoint_interval", type=int, default=200_000)
    parser.add_argument("--target_kl_early_stop", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--target_kl", type=float, default=0.02)
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
    )
