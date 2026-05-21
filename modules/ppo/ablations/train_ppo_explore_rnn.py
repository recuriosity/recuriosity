"""Version-1 curiosity ablation: no-camera-pose PPO with an LSTM policy backbone."""

import os
import sys

_ppo_dir = os.path.dirname(os.path.abspath(__file__))
_modules_dir = os.path.dirname(os.path.dirname(_ppo_dir))
_root = os.path.dirname(_modules_dir)
if _root not in sys.path:
    sys.path.insert(0, _root)

import modules.ppo.ablations.train_ppo_explore_base as _base
from modules.agent.ablations.agent_lstm import NavAgentLSTM
from modules.ppo.train_ppo_explore_no_pose import PoseProcessNoCameraPose


class NavAgentNoCameraPoseRNN(NavAgentLSTM):
    def __init__(self, *args, **kwargs):
        kwargs["in_channels"] = 9
        super().__init__(*args, **kwargs)


PoseProcess = PoseProcessNoCameraPose
NavAgent = NavAgentNoCameraPoseRNN

init_distributed = _base.init_distributed
explained_variance = _base.explained_variance
obs_to_img_pose = _base.obs_to_img_pose
se3_from_translation_rotation = _base.se3_from_translation_rotation
action_pose_from_indices = _base.action_pose_from_indices
build_anchor_w2c = _base.build_anchor_w2c
preprocess_big_step = _base.preprocess_big_step
render_from_gsplat = _base.render_from_gsplat
rollout_with_cache = _base.rollout_with_cache
ppo_update = _base.ppo_update
eval_rollout_video = _base.eval_rollout_video


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
    checkpoint_path="checkpoints_saved/explore_rnn.pt",
    random_action_prob=0.2,
    anneal_random_action=True,
    anneal_random_steps=2_500_000,
    anneal_random_duration=500_000,
):
    base_pose_process = _base.PoseProcess
    base_nav_agent = _base.NavAgent
    base_wandb_init = _base.wandb.init

    def _wandb_init_with_ablation(*args, **kwargs):
        name = kwargs.get("name")
        if isinstance(name, str) and "nocam_rnn" not in name:
            kwargs["name"] = name.replace(
                "train_explore_base__",
                "train_explore_rnn__",
                1,
            )
        return base_wandb_init(*args, **kwargs)

    _base.PoseProcess = PoseProcessNoCameraPose
    _base.NavAgent = NavAgentNoCameraPoseRNN
    _base.wandb.init = _wandb_init_with_ablation
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
        default="checkpoints_saved/explore_rnn.pt",
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
