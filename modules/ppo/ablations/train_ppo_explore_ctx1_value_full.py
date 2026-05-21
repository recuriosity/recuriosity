"""No-camera-pose randmix PPO ablation: actor context=1, critic full memory."""

import os
import sys

import torch

_ppo_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_ppo_dir))
if _root not in sys.path:
    sys.path.insert(0, _root)

import modules.ppo.ablations.train_ppo_explore_base as _base


_BasePoseProcess = _base.PoseProcess
_BaseNavAgent = _base.NavAgent
DEFAULT_ATTN_WINDOW = 64


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
    def _temporal_encode_full(self, frame_tok: torch.Tensor) -> torch.Tensor:
        x_tm = self.temporal_in_ln(frame_tok)
        x_tm = self._run_blocks(x_tm, self.temporal_blocks, attn_bias=None, kv_caches=None)
        return self.head_ln(x_tm)

    def _temporal_encode_ctx1(self, frame_tok: torch.Tensor) -> torch.Tensor:
        bsz, vlen, d_model = frame_tok.shape
        x_tm = self.temporal_in_ln(frame_tok.reshape(bsz * vlen, 1, d_model))
        x_tm = self._run_blocks(x_tm, self.temporal_blocks, attn_bias=None, kv_caches=None)
        return self.head_ln(x_tm).reshape(bsz, vlen, d_model)

    def _temporal_step_full(self, frame_tok: torch.Tensor, kv_caches) -> torch.Tensor:
        assert kv_caches is not None and len(kv_caches) == self.n_layer_temporal
        x_tm = self.temporal_in_ln(frame_tok.unsqueeze(1))
        for i, blk in enumerate(self.temporal_blocks):
            x_tm = blk(x_tm, attn_bias=None, kv_cache=kv_caches[i])
        return self.head_ln(x_tm).squeeze(1)

    def _temporal_step_ctx1(self, frame_tok: torch.Tensor) -> torch.Tensor:
        x_tm = self.temporal_in_ln(frame_tok.unsqueeze(1))
        x_tm = self._run_blocks(x_tm, self.temporal_blocks, attn_bias=None, kv_caches=None)
        return self.head_ln(x_tm).squeeze(1)

    def forward(self, posed_images, last_only=False):
        _bsz, vlen, channels, height, width = posed_images.shape
        assert vlen <= self.max_v
        assert channels == self.in_channels
        assert height == self.image_size and width == self.image_size

        frame_tok = self._encode_frames_batch(posed_images)
        actor_z = self._temporal_encode_ctx1(frame_tok)
        critic_z = self._temporal_encode_full(frame_tok)
        logits = self.pi_head(actor_z)
        values = self.v_head(critic_z).squeeze(-1)
        if last_only:
            return logits[:, -1], values[:, -1]
        return logits, values

    def forward_step(self, posed_image, time_idx: int, kv_caches):
        frame_tok = self._encode_frame(posed_image)
        actor_z = self._temporal_step_ctx1(frame_tok)
        critic_z = self._temporal_step_full(frame_tok, kv_caches)
        logits = self.pi_head(actor_z)
        value = self.v_head(critic_z).squeeze(-1)
        return logits, value

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
    num_envs=72,
    roll_length=1024,
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
    ent_coef_start=0.1,
    ent_decay_rate=None,
    vf_coef=0.5,
    max_grad_norm=1.0,
    nerf_iters=10,
    cap_max=750_000,
    attn_window=DEFAULT_ATTN_WINDOW,
    eval_every=10,
    checkpoint_path="checkpoints_saved/explore_ctx1_value_full.pt",
    random_action_prob=0.2,
    anneal_random_action=True,
    anneal_random_steps=2_500_000,
    anneal_random_duration=500_000,
    eval_on_val=False,
):
    if learning_rate is None:
        learning_rate = base_learning_rate * (float(num_envs) / float(base_num_envs))
    if ent_decay_rate is None:
        ent_decay_rate = float(base_ent_decay_rate ** (float(num_envs) / float(base_num_envs)))
    base_pose_process = _base.PoseProcess
    base_nav_agent = _base.NavAgent
    base_wandb_init = _base.wandb.init

    def _wandb_init_with_nocam(*args, **kwargs):
        name = kwargs.get("name")
        if isinstance(name, str):
            variant_tag = "actor_ctx1_value_full"
            if variant_tag not in name:
                if "train_explore_base__" in name and "nocam" not in name:
                    name = name.replace(
                        "train_explore_base__",
                        f"train_explore_{variant_tag}__",
                        1,
                    )
                else:
                    name = f"{name}_{variant_tag}"
            kwargs["name"] = name
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
            num_envs=num_envs,
            roll_length=roll_length,
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
            eval_on_val=eval_on_val,
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
    p.add_argument("--base_ckpt", type=str, default=None)
    p.add_argument("--num_envs", type=int, default=72)
    p.add_argument("--roll_length", type=int, default=1024)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--hw", type=int, default=64)
    p.add_argument("--big_hw", type=int, default=128)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--gae_lambda", type=float, default=0.97)
    p.add_argument("--update_epochs", type=int, default=3)
    p.add_argument("--clip_coef", type=float, default=0.2)
    p.add_argument("--ent_coef_start", type=float, default=0.1)
    p.add_argument("--ent_decay_rate", type=float, default=None)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--nerf_iters", type=int, default=10)
    p.add_argument("--cap_max", type=int, default=750_000)
    p.add_argument("--attn_window", type=int, default=DEFAULT_ATTN_WINDOW)
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints_saved/explore_ctx1_value_full.pt",
    )
    p.add_argument("--random_action_prob", type=float, default=0.2)
    p.add_argument("--anneal_random_action", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--anneal_random_steps", type=int, default=2_500_000)
    p.add_argument("--anneal_random_duration", type=int, default=750_000)
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
    )
