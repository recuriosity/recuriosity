# Remember to be Curious

Official code for:

> **Remember to be Curious: Episodic Context and Persistent Worlds for 3D Exploration**  
> Lily Goli, Justin Kerr, Daniele Reda, Alec Jacobson, Andrea Tagliasacchi, Angjoo Kanazawa  
> *University of Toronto · UC Berkeley · Wayve · Vector Institute · Simon Fraser University*  
> [[Project page]](https://recuriosity.github.io) &nbsp;|&nbsp; [[arXiv]](https://arxiv.org/abs/2605.22814)

A reinforcement learning agent that learns to explore 3D indoor scenes using only an RGB camera. The agent uses a sliding-window transformer policy (DINO visual encoder + KV-cached transformer backbone) and is trained with PPO, using real-time Gaussian Splatting (GSplat) reconstruction as its intrinsic reward signal.

---

## Overview

- **Task**: Camera exploration in Habitat-Matterport 3D (HM3D) and Gibson scenes
- **Policy**: DINO ViT-B/8 + sliding-window transformer with KV cache
- **Reward**: GSplat reconstruction quality (MSE between predicted and ground-truth novel views)
- **Downstream tasks**: Apple picking, image-goal navigation (finetuned from the explore checkpoint)

### Repository layout

```
modules/
  agent/          Policy and transformer architecture
  environment/    Habitat environment wrappers, GSplat, rendering
  eval/           Checkpoint evaluation scripts
  ppo/            PPO training scripts
    ablations/    Ablation variants (RNN, ICM, context-window ablations, etc.)
scripts/          Data download and split generation utilities
data/
  splits/         Validation episode splits (HM3D, Gibson)
main.py           Dispatcher entrypoint
environment.yml   Conda environment
requirements.txt  Python dependencies
```

---

## Installation

### Conda

#### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate recuriosity
```

#### 2. Install PyTorch

Install the wheel matching your CUDA driver. Example for CUDA 12.4:

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --extra-index-url https://download.pytorch.org/whl/cu124
```

See https://pytorch.org/get-started/locally/ for other driver versions.

#### 3. Install fused-ssim (required before gsplat)

```bash
CUDA_HOME=$CONDA_PREFIX CUDA_ARCHITECTURES="75;80;89;90" \
pip install --no-build-isolation \
    "git+https://github.com/rahul-goel/fused-ssim"
```

#### 4. Install remaining dependencies (includes gsplat build)

```bash
CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST="7.5;8.0;8.9;9.0" MAX_JOBS=$(nproc) \
pip install --no-build-isolation -r requirements.txt
```

#### 5. Install habitat-sim from source

habitat-sim must be built from source with CUDA and headless rendering support.

```bash
pip install scikit-build-core pybind11

CUDA_HOME=$CONDA_PREFIX \
HABITAT_BUILD_GUI_VIEWERS=OFF \
HABITAT_WITH_BULLET=ON \
HABITAT_WITH_CUDA=ON \
MAX_JOBS=$(nproc) \
pip install --no-build-isolation -v \
    "habitat-sim @ git+https://github.com/facebookresearch/habitat-sim.git"
```

This takes 10–60 minutes. See https://github.com/facebookresearch/habitat-sim/blob/main/BUILD_FROM_SOURCE.md for details.

Verify:
```bash
python -c "import habitat_sim; print('habitat_sim ok')"
```

---

## Data

### HM3D (Habitat-Matterport 3D)

1. Register at https://aihabitat.org/datasets/hm3d/
2. Download the HM3D train and validation splits (GLB + navmesh)
3. Extract to a local directory.

Pass the path via `--data_root` in all training and eval commands. Expected layout:

```
<data_root>/hm3d/hm3d_glb/<scene_id>/<scene>.glb
<data_root>/hm3d/hm3d_nav/<scene_id>/<scene>.navmesh
<data_root>/hm3d-val/hm3d_glb/<scene_id>/<scene>.glb
<data_root>/hm3d-val/hm3d_nav/<scene_id>/<scene>.navmesh
```

### Gibson (optional — for cross-dataset generalization eval)

1. Request access and download the Gibson dataset (Habitat-compatible GLBs) via the [Gibson Dataset request form](https://docs.google.com/forms/d/e/1FAIpQLScWlx5Z1DM1M-wTSXaa6zV8lTFkPmTHW1LqMsoCBDWsTDjBkQ/viewform).
2. Extract to a local directory and pass the path via `--gibson_root` in eval commands.

Gibson scenes follow the same GLB format as HM3D. The expected directory layout (matching the episode split paths) is:

```
$GIBSON_SCENE_ROOT/
  Adrian.glb
  Annawan.glb
  ...
```

### Validation episode splits

Pre-generated splits are included under `data/splits/`:

| File | Task | Episodes |
|------|------|----------|
| `data/splits/hm3d/val/val.json.gz` | Exploration | 200 (100 scenes × 2 starts) |
| `data/splits/hm3d/val/val_apples.json` | Apple picking | 200 |
| `data/splits/hm3d/val/val_image_goal.json` | Image-goal navigation | 200 |
| `data/splits/gibson/val/val_gibson_bigisland.json.gz` | Exploration (Gibson) | 86 (86 scenes × 1 start) |

---

## Pretrained Checkpoints

Download the pretrained checkpoints from [HuggingFace](https://huggingface.co/lily-goli/recuriosity):

| File | Description |
|------|-------------|
| [`explorer.pt`](https://huggingface.co/lily-goli/recuriosity/resolve/main/explorer.pt) | Main exploration policy, trained on HM3D |
| [`apple_finetuned.pt`](https://huggingface.co/lily-goli/recuriosity/resolve/main/apple_finetuned.pt) | Apple-picking fine-tune |
| [`image_goal_finetuned.pt`](https://huggingface.co/lily-goli/recuriosity/resolve/main/image_goal_finetuned.pt) | Image-goal navigation fine-tune |

Pass the checkpoint path via `--checkpoint-path` (eval) or `--base_ckpt` / `--weights_only_ckpt` (fine-tuning). The commands below use `checkpoints/explore.pt` as an example path.

---

## Training

All training scripts are invoked via `main.py`. Multi-GPU training uses `torchrun`.

Pass the HM3D data directory via `--data_root` (the directory containing `hm3d/` and `hm3d-val/`). When using Docker via `make`, this is handled automatically by the Makefile.

### W&B logging

```bash
export WANDB_API_KEY=your_key_here
# To disable: export WANDB_MODE=disabled
```

### Exploration (main model)

To train run the following command:

```bash
# Single GPU
python main.py --script explore_no_pose \
    --data_root /path/to/hm3d \
    --num_envs 4 \
    --logdir runs \
    --checkpoint_path checkpoints/explore.pt

# Multi-GPU (8× H100)
torchrun --standalone --nproc_per_node=8 main.py --script explore_no_pose \
    --data_root /path/to/hm3d \
    --num_envs 72 \
    --logdir runs \
    --checkpoint_path checkpoints/explore.pt
```

Training runs on 8× H100s; with `--num_envs 72` it takes ~ 3 days on 80 GB GPUs. For a lower GPU memory usage, set `--num_envs 32` (~6 days; this is the configuration used for the released checkpoint).

Resume from a checkpoint: `--base_ckpt checkpoints/explore.pt`

Key hyperparameters (all have sensible defaults):

| Flag | Default | Description |
|------|---------|-------------|
| `--num_envs` | 4 | Number of parallel environments (use 32 or 72 for multi-GPU) |
| `--roll_length` | 1024 | Steps per rollout |
| `--learning_rate` | 1e-5 | Adam learning rate |
| `--attn_window` | 64 | Sliding attention window size (frames) |
| `--nerf_iters` | 10 | GSplat optimization steps per rollout step |
| `--ent_coef_start` | 0.1 | Initial entropy coefficient |

### Ablations

```bash
# RNN backbone
torchrun --standalone --nproc_per_node=8 main.py --script ablation_rnn ...

# RNN + ICM
torchrun --standalone --nproc_per_node=8 main.py --script ablation_rnn_icm ...

# Transformer + ICM
torchrun --standalone --nproc_per_node=8 main.py --script ablation_icm ...

# Context length ablation: --context_window caps how many past frame tokens the agent receives as input (1, 4, or 16)
torchrun --standalone --nproc_per_node=8 main.py --script ablation_ctx16 --context_window 1 ...

# Forgetful GSplat (sliding scene window)
torchrun --standalone --nproc_per_node=8 main.py --script ablation_forgetful ...
```

### Apple-picking finetune

Finetune from the pretrained exploration checkpoint:

```bash
torchrun --standalone --nproc_per_node=8 main.py --script apples_no_pose \
    --data_root /path/to/hm3d \
    --weights_only_ckpt checkpoints/explore.pt \
    --num_envs 72 \
    --logdir runs \
    --checkpoint_path checkpoints/apples.pt
```

### Image-goal navigation finetune

```bash
torchrun --standalone --nproc_per_node=8 main.py --script image_goal_no_pose \
    --data_root /path/to/hm3d \
    --weights_only_ckpt checkpoints/explore.pt \
    --learning_rate 1e-6 \
    --num_envs 72 \
    --logdir runs \
    --checkpoint_path checkpoints/image_goal.pt
```

---

## Evaluation

Evaluation runs the policy on fixed pre-generated episodes and computes surface coverage completeness at 0.05 m threshold. Results are written to `eval_outputs/` and logged to W&B.

The `--eval-hole-fix` and `--eval-hole-fix-force-white` flags patch scene meshes by adding a white material to backfaces in areas where the mesh has holes. Without this, Habitat disables backface rendering in those regions, leading to mismatched RGB rendering and collision detection in hole areas during evaluation.

### Exploration on HM3D

```bash
python main.py --script eval --data_root /path/to/hm3d \
    --ppo-module modules.ppo.train_ppo_explore_no_pose \
    --checkpoint-path checkpoints/explore.pt \
    --episodes-json data/splits/hm3d/val/val.json.gz \
    --eval-hole-fix --eval-hole-fix-force-white \
    --output-dir eval_outputs/
```

Multi-GPU (episodes are distributed across GPUs):

```bash
torchrun --standalone --nproc_per_node=8 main.py --script eval --data_root /path/to/hm3d \
    --ppo-module modules.ppo.train_ppo_explore_no_pose \
    --checkpoint-path checkpoints/explore.pt \
    --episodes-json data/splits/hm3d/val/val.json.gz \
    --eval-hole-fix --eval-hole-fix-force-white \
    --output-dir eval_outputs/
```

### Exploration on Gibson

```bash
python main.py --script eval \
    --data_root /path/to/hm3d \
    --gibson_root /path/to/gibson \
    --ppo-module modules.ppo.train_ppo_explore_no_pose \
    --checkpoint-path checkpoints/explore.pt \
    --episodes-json data/splits/gibson/val/val_gibson_bigisland.json.gz \
    --eval-hole-fix --eval-hole-fix-force-white \
    --output-dir eval_outputs/
```

### Apple picking

```bash
python main.py --script eval_apples --data_root /path/to/hm3d \
    --ppo-module modules.ppo.train_ppo_apples_no_pose \
    --checkpoint-path checkpoints/apples.pt \
    --episodes-json data/splits/hm3d/val/val_apples.json \
    --eval-hole-fix --eval-hole-fix-force-white \
    --output-dir eval_outputs/
```

### Image-goal navigation

```bash
python main.py --script eval_image_goal --data_root /path/to/hm3d \
    --ppo-module modules.ppo.train_ppo_image_goal_no_pose \
    --checkpoint-path checkpoints/image_goal.pt \
    --episodes-json data/splits/hm3d/val/val_image_goal.json \
    --eval-hole-fix --eval-hole-fix-force-white \
    --output-dir eval_outputs/
```

---

## Active Mapping Baselines

Code and instructions for running the active mapping baselines (ANS-RGB, ANS-Depth, OccAnt-RGB, OccAnt-RGBD) will be released in a future update.

---

## Fix for Common Issues

| Issue | Cause / Fix |
|-------|-------------|
| `FileNotFoundError: No GLBs found` | HM3D data missing or `--data_root` path incorrect |
| `ModuleNotFoundError: habitat_sim` | habitat-sim not installed; build from source (see above) |
| `Unable to create windowless context` / `unable to find CUDA device 0 among EGL devices` | Container needs `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`. The Makefile sets this automatically. Also requires `--gpus all` and the nvidia-container-toolkit EGL setup on the host. |
| `torch.hub` download failure (DINO) | DINOv2 weights are downloaded on first run; ensure internet access or pre-cache with `torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')` |
| gsplat build fails | Ensure `TORCH_CUDA_ARCH_LIST` is set and matches your GPU |
| habitat-sim build fails | Ensure `cmake >= 3.14`, `libegl1-mesa-dev`, `libgl1-mesa-dev` are installed |
| OOM with many environments | Reduce `--num_envs` (32 is safe for 4× H100 80GB) |

---

## Citation

```bibtex
@article{goli2026recuriosity,
  title   = {Remember to be Curious: Episodic Context and Persistent Worlds for 3D Exploration},
  author  = {Goli, Lily and Kerr, Justin and Reda, Daniele and Jacobson, Alec and Tagliasacchi, Andrea and Kanazawa, Angjoo},
  journal = {arXiv preprint arXiv:2605.22814},
  year    = {2026},
  url     = {https://arxiv.org/abs/2605.22814},
}
```
