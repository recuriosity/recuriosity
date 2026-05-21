"""Apple TTT PPO variant without camera-pose plucker channels in policy input."""

import os
import sys

import torch

_ppo_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_ppo_dir))
if _root not in sys.path:
    sys.path.insert(0, _root)

import modules.ppo.train_ppo_apples as _base


_BasePoseProcess = _base.PoseProcess
_BaseNavAgent = _base.NavAgent


def _adapt_patch_embed_weight(
    source_weight: torch.Tensor,
    target_weight: torch.Tensor,
) -> torch.Tensor | None:
    if source_weight.ndim != 2 or target_weight.ndim != 2:
        return None
    if source_weight.shape[0] != target_weight.shape[0]:
        return None
    if source_weight.shape[1] % 15 != 0 or target_weight.shape[1] % 9 != 0:
        return None

    patch_area_src = source_weight.shape[1] // 15
    patch_area_tgt = target_weight.shape[1] // 9
    if patch_area_src != patch_area_tgt:
        return None

    keep_channels = torch.tensor(
        [0, 1, 2, 9, 10, 11, 12, 13, 14],
        device=source_weight.device,
        dtype=torch.long,
    )
    adapted = source_weight.reshape(source_weight.shape[0], patch_area_src, 15)
    adapted = adapted.index_select(2, keep_channels)
    return adapted.reshape_as(target_weight)


class PoseProcessNoCameraPose(_BasePoseProcess):
    def build_visual_input(self, images=None, ray_o=None, ray_d=None, method="default_plucker"):
        if images is None:
            return super().build_visual_input(images=None, ray_o=ray_o, ray_d=ray_d, method=method)
        return images * 2.0 - 1.0


class NavAgentNoCameraPose(_BaseNavAgent):
    def __init__(self, *args, **kwargs):
        kwargs["in_channels"] = 9
        super().__init__(*args, **kwargs)

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
            if adapted_keys or skipped_keys:
                print("[warn] skipped optimizer state because model parameter shapes changed")
            else:
                optimizer.load_state_dict(ckpt["optimizer"])
                print("[info] loaded optimizer state")

        step = int(ckpt.get("step", 0))
        return step, ckpt


PoseProcess = PoseProcessNoCameraPose
NavAgent = NavAgentNoCameraPose

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
    base_learning_rate=1e-5,
    base_num_envs=32,
    base_ent_decay_rate=0.99,
    hw=64,
    big_hw=128,
    gamma=0.995,
    gae_lambda=0.97,
    update_epochs=3,
    clip_coef=0.2,
    ent_coef_start=0.0,
    ent_decay_rate=None,
    reward_coef=0.5,
    num_apples=5,
    vf_coef=0.5,
    max_grad_norm=1.0,
    nerf_iters=10,
    cap_max=750_000,
    attn_window=64,
    eval_every=10,
    checkpoint_path="checkpoints_saved/apples_no_pose.pt",
    random_action_prob=0.0,
    anneal_random_action=False,
    anneal_random_steps=2_500_000,
    anneal_random_duration=500_000,
    wandb_name_suffix="",
    archive_checkpoint_interval=200_000,
    target_kl_early_stop=False,
    target_kl=0.02,
):
    if learning_rate is None:
        learning_rate = base_learning_rate * (float(num_envs) / float(base_num_envs))
    if ent_decay_rate is None:
        ent_decay_rate = float(base_ent_decay_rate ** (float(num_envs) / float(base_num_envs)))
    base_ckpt_path = str(base_ckpt or os.environ.get("BASE_CKPT", "")).strip()
    weights_only_ckpt_path = str(
        weights_only_ckpt or os.environ.get("WEIGHTS_ONLY_CKPT", "")
    ).strip()
    run_origin_suffix = "from_curiosity" if (base_ckpt_path or weights_only_ckpt_path) else "vanilla"
    base_pose_process = _base.PoseProcess
    base_nav_agent = _base.NavAgent
    base_wandb_init = _base.wandb.init

    def _wandb_init_with_nocam(*args, **kwargs):
        name = kwargs.get("name")
        if isinstance(name, str):
            updated_name = name
            # Determine correct task prefix based on suffix hints.
            if "image_goal" in (wandb_name_suffix or ""):
                base_prefix = "train_image_goal_no_pose__"
                updated_name = updated_name.replace("train_apples__", base_prefix, 1)
                # strip the now-redundant suffix that was already baked in
                updated_name = updated_name.replace("_image_goal", "", 1)
            elif "nocam" not in updated_name:
                updated_name = updated_name.replace(
                    "train_apples__",
                    "train_apples_no_pose__",
                    1,
                )
            if run_origin_suffix not in updated_name:
                updated_name = f"{updated_name}_{run_origin_suffix}"
            kwargs["name"] = updated_name
        return base_wandb_init(*args, **kwargs)

    _base.PoseProcess = PoseProcessNoCameraPose
    _base.NavAgent = NavAgentNoCameraPose
    _base.wandb.init = _wandb_init_with_nocam
    try:
        return _base.main(
            logdir=logdir,
            save_eval_video=save_eval_video,
            train=train,
            base_ckpt=base_ckpt,
            weights_only_ckpt=weights_only_ckpt,
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
            reward_coef=reward_coef,
            num_apples=num_apples,
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
            eval_on_val=eval_on_val,
            eval_compile=eval_compile,
            greedy_eval=greedy_eval,
            wandb_name_suffix=wandb_name_suffix,
            archive_checkpoint_interval=archive_checkpoint_interval,
            target_kl_early_stop=target_kl_early_stop,
            target_kl=target_kl,
        )
    finally:
        _base.PoseProcess = base_pose_process
        _base.NavAgent = base_nav_agent
        _base.wandb.init = base_wandb_init


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--logdir", type=str, default="runs")
    p.add_argument("--no_eval_video", action="store_true")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--base_ckpt", type=str, default="checkpoints_saved/apples_no_pose.pt")
    p.add_argument("--weights_only_ckpt", type=str, default=None)
    p.add_argument("--num_envs", type=int, default=72)
    p.add_argument("--roll_length", type=int, default=1024)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--hw", type=int, default=64)
    p.add_argument("--big_hw", type=int, default=128)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae_lambda", type=float, default=0.97)
    p.add_argument("--update_epochs", type=int, default=3)
    p.add_argument("--clip_coef", type=float, default=0.2)
    p.add_argument("--ent_coef_start", type=float, default=0.0)
    p.add_argument("--ent_decay_rate", type=float, default=None)
    p.add_argument("--reward_coef", type=float, default=0.5)
    p.add_argument("--num_apples", type=int, default=5)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--nerf_iters", type=int, default=10)
    p.add_argument("--cap_max", type=int, default=750_000)
    p.add_argument("--attn_window", type=int, default=64)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints_saved/apples_no_pose.pt",
    )
    p.add_argument("--random_action_prob", type=float, default=0.0)
    p.add_argument("--anneal_random_action", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--anneal_random_steps", type=int, default=2_500_000)
    p.add_argument("--anneal_random_duration", type=int, default=750_000)
    p.add_argument("--wandb_name_suffix", type=str, default="")
    p.add_argument("--archive_checkpoint_interval", type=int, default=200_000)
    p.add_argument("--target_kl_early_stop", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--target_kl", type=float, default=0.02)
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
        reward_coef=args.reward_coef,
        num_apples=args.num_apples,
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
        wandb_name_suffix=args.wandb_name_suffix,
        archive_checkpoint_interval=args.archive_checkpoint_interval,
        target_kl_early_stop=args.target_kl_early_stop,
        target_kl=args.target_kl,
    )
