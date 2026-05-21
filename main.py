"""
Entrypoint for training and evaluation.

Usage (single GPU):
    python main.py --script explore_no_pose --data_root /path/to/hm3d [script args...]

Usage (multi-GPU):
    torchrun --standalone --nproc_per_node=NUM_GPUS main.py --script explore_no_pose \
        --data_root /path/to/hm3d [script args...]

The --script flag selects which module to run; all remaining args are forwarded
to that module's argparse unchanged. Scripts can also be run directly:
    torchrun --standalone --nproc_per_node=N modules/ppo/train_ppo_explore_no_pose.py [args...]
"""

import argparse
import os
import runpy
import sys

# Ensure the project root is on the path regardless of how this file is invoked.
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Script registry
# ---------------------------------------------------------------------------

SCRIPTS: dict[str, str] = {
    # --- exploration (main) ---
    "explore":            "modules/ppo/train_ppo_explore.py",
    "explore_no_pose":    "modules/ppo/train_ppo_explore_no_pose.py",
    # --- downstream: apple picking ---
    "apples":             "modules/ppo/train_ppo_apples.py",
    "apples_no_pose":     "modules/ppo/train_ppo_apples_no_pose.py",
    # --- downstream: image-goal navigation ---
    "image_goal":         "modules/ppo/train_ppo_image_goal.py",
    "image_goal_no_pose": "modules/ppo/train_ppo_image_goal_no_pose.py",
    # --- evaluation ---
    "eval":               "modules/eval/ppo_checkpoint_eval.py",
    "eval_apples":        "modules/eval/apples_checkpoint_eval.py",
    "eval_image_goal":    "modules/eval/image_goal_checkpoint_eval.py",
    # --- ablations ---
    "ablation_base":        "modules/ppo/ablations/train_ppo_explore_base.py",
    "ablation_rnn":         "modules/ppo/ablations/train_ppo_explore_rnn.py",
    "ablation_rnn_icm":     "modules/ppo/ablations/train_ppo_explore_rnn_icm.py",
    "ablation_icm":         "modules/ppo/ablations/train_ppo_explore_transformer_icm.py",
    "ablation_ctx16":       "modules/ppo/ablations/train_ppo_explore_ctx16.py",
    "ablation_ctx1_vfull":  "modules/ppo/ablations/train_ppo_explore_ctx1_value_full.py",
    "ablation_full_vctx1":  "modules/ppo/ablations/train_ppo_explore_full_value_ctx1.py",
    "ablation_forgetful":   "modules/ppo/ablations/train_ppo_explore_forgetful_gsplat.py",
}

_EVAL_SCRIPTS = {"eval", "eval_apples", "eval_image_goal"}

# ---------------------------------------------------------------------------
# Env-var → CLI arg translation for eval scripts (cluster / container mode)
# ---------------------------------------------------------------------------

def _eval_args_from_env() -> list[str]:
    """Build eval CLI args from EVAL_* env vars (used when running without CLI args)."""
    args: list[str] = []
    e = os.environ.get

    if e("EVAL_CHECKPOINT_PATH"):
        args += ["--checkpoint-path", e("EVAL_CHECKPOINT_PATH")]
    if e("EVAL_PPO_MODULE"):
        args += ["--ppo-module", e("EVAL_PPO_MODULE")]
    if e("EVAL_EPISODES_JSON"):
        args += ["--episodes-json", e("EVAL_EPISODES_JSON")]
    if e("EVAL_HOLE_FIX", "").lower() in ("true", "1"):
        args += ["--eval-hole-fix"]
    if e("EVAL_HOLE_FIX_FORCE_WHITE", "").lower() in ("true", "1"):
        args += ["--eval-hole-fix-force-white"]
    if e("EVAL_OUTPUT_DIR"):
        args += ["--output-dir", e("EVAL_OUTPUT_DIR")]
    if e("EVAL_WANDB_PROJECT"):
        args += ["--wandb-project", e("EVAL_WANDB_PROJECT")]
    if e("EVAL_WANDB_ENTITY"):
        args += ["--wandb-entity", e("EVAL_WANDB_ENTITY")]
    if e("EVAL_RUN_NAME"):
        args += ["--run-name", e("EVAL_RUN_NAME")]
    if e("EVAL_MAX_STEPS"):
        args += ["--max-steps", e("EVAL_MAX_STEPS")]
    if e("EVAL_SEED"):
        args += ["--seed", e("EVAL_SEED")]
    if e("EVAL_NN_BACKEND"):
        args += ["--torch-nn-backend", e("EVAL_NN_BACKEND")]
    if e("EVAL_GREEDY", "").lower() in ("true", "1"):
        args += ["--greedy"]
    return args


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _main() -> None:
    p = argparse.ArgumentParser(
        description="recuriosity — training and evaluation dispatcher",
        add_help=False,
    )
    p.add_argument(
        "--script",
        default=os.environ.get("SCRIPT", "explore_no_pose"),
        choices=sorted(SCRIPTS),
        help=(
            "Which module to run. Default: explore_no_pose. "
            "All remaining args are forwarded to the chosen script."
        ),
    )
    p.add_argument(
        "--data_root",
        default=os.environ.get("HM3D_DATA_ROOT", "/workspace/data"),
        help="Path to the directory containing hm3d/ and hm3d-val/ (sets HM3D_DATA_ROOT).",
    )
    p.add_argument(
        "--gibson_root",
        default=os.environ.get("GIBSON_SCENE_ROOT", ""),
        help="Path to the Gibson scene directory (sets GIBSON_SCENE_ROOT).",
    )
    p.add_argument("--list-scripts", action="store_true", help="List available scripts and exit.")
    args, remaining = p.parse_known_args()

    # Set data roots before any training/env module is imported.
    os.environ["HM3D_DATA_ROOT"] = args.data_root
    if args.gibson_root:
        os.environ["GIBSON_SCENE_ROOT"] = args.gibson_root

    if args.list_scripts:
        print("Available --script values:")
        for name, path in sorted(SCRIPTS.items()):
            print(f"  {name:<22}  {path}")
        sys.exit(0)

    # For eval scripts running in container/cluster mode (no CLI args passed),
    # translate EVAL_* env vars into the equivalent CLI args.
    if args.script in _EVAL_SCRIPTS and not remaining:
        remaining = _eval_args_from_env()

    script_path = os.path.join(_ROOT, SCRIPTS[args.script])
    if not os.path.isfile(script_path):
        print(f"[main] Script not found: {script_path}", file=sys.stderr)
        sys.exit(1)

    # Replace sys.argv so the chosen script's argparse sees only its own args.
    sys.argv = [script_path] + remaining

    runpy.run_path(script_path, run_name="__main__")


if __name__ == "__main__":
    _main()
