#!/usr/bin/env python3
"""
Generate Gibson val split with the current HM3D eval-start selection logic.

This mirrors scripts/generate_hm3d_split.py eval behavior:
1) sample random navigable candidates
2) apply robust start checks (ceiling, depth validity, local same-floor support)
3) rank and choose fixed starts per scene for reproducible eval

make generate-gibson-split-test ARGS="--include-train-scenes --output-filename val_gibson_bigisland.json.gz --min-island-radius 3.0 --min-start-enclosure-frac 0.60 --starts-per-scene 1 --no-prefer-distinct-islands"
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

# Add curious_camera to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.generate_hm3d_split import (
    LOCAL_SAME_FLOOR_Y_TOL_M,
    MIN_CEILING_CLEARANCE,
    MIN_ISLAND_RADIUS,
    MIN_LOCAL_SAME_FLOOR_COVERAGE,
    MIN_START_CLEAR_DEPTH_M,
    MIN_START_ENCLOSURE_FRAC,
    MIN_START_SEPARATION_M,
    MIN_START_VALID_DEPTH_FRAC,
    _sample_navmesh_starts,
    _warn,
)

_DEFAULT_GIBSON_SCENE_ROOT = ""
_GIBSON_SCENE_ID_PREFIX = "data/scene_datasets/gibson/"


def _index_glb_files(root: str) -> dict[str, tuple[str, str | None]]:
    """Map scene_name -> (glb_path, navmesh_path|None) by recursively scanning root."""
    index: dict[str, tuple[str, str | None]] = {}
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if not fname.endswith(".glb"):
                continue
            scene_name = os.path.splitext(fname)[0]
            glb_path = os.path.abspath(os.path.join(dirpath, fname))
            navmesh_path: str | None = None
            for nav_fname in (
                f"{scene_name}.navmesh",
                f"{scene_name}.glb.navmesh",
                f"{scene_name}.basis.navmesh",
            ):
                candidate = os.path.join(dirpath, nav_fname)
                if os.path.isfile(candidate):
                    navmesh_path = os.path.abspath(candidate)
                    break
            index[scene_name] = (glb_path, navmesh_path)
    return index


def _load_gibson_val_scene_names(dataset_root: str) -> list[str]:
    val_json_path = os.path.join(dataset_root, "val", "val.json.gz")
    if not os.path.isfile(val_json_path):
        raise FileNotFoundError(
            f"Could not find Gibson val split file: {val_json_path}. "
            "Expected an existing val/val.json.gz to define the val scene set."
        )

    with gzip.open(val_json_path, "rt", encoding="utf-8") as f:
        payload = json.load(f)
    episodes = payload.get("episodes", [])
    if not episodes:
        raise RuntimeError(f"No episodes found in {val_json_path}")

    scene_names: set[str] = set()
    for ep in episodes:
        scene_id = str(ep.get("scene_id", ""))
        basename = os.path.basename(scene_id)
        stem, _ = os.path.splitext(basename)
        if stem:
            scene_names.add(stem)

    if not scene_names:
        raise RuntimeError(f"Could not infer any scene names from {val_json_path}")
    return sorted(scene_names)


def _load_gibson_train_scene_names(dataset_root: str) -> list[str]:
    train_dir = os.path.join(dataset_root, "train")
    content_dir = os.path.join(train_dir, "content")
    if not os.path.isdir(content_dir):
        raise FileNotFoundError(
            f"Could not find Gibson train content directory: {content_dir}"
        )

    scene_names: set[str] = set()
    for fname in sorted(os.listdir(content_dir)):
        if not fname.endswith(".json.gz"):
            continue
        stem = os.path.splitext(os.path.splitext(fname)[0])[0]
        if stem:
            scene_names.add(stem)

    if not scene_names:
        raise RuntimeError(f"No train scene names found in {content_dir}")
    return sorted(scene_names)


def _resolve_scene_root(requested_root: str | None) -> str:
    if requested_root:
        return os.path.abspath(requested_root)
    env_root = os.environ.get("GIBSON_SCENE_ROOT", "").strip()
    if env_root:
        return os.path.abspath(env_root)
    return _DEFAULT_GIBSON_SCENE_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=os.path.join(_repo_root, "data", "splits", "gibson"),
        help="Root directory containing gibson/{val/train}/...",
    )
    parser.add_argument(
        "--scene-root",
        type=str,
        default=None,
        help="Directory containing Gibson scene .glb/.navmesh files. "
        "Defaults to GIBSON_SCENE_ROOT env var.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for Gibson splits (default: same as --dataset-root)",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="val_gibson.json.gz",
        help="Output filename under <output-dir>/val (default: val_gibson.json.gz)",
    )
    parser.add_argument(
        "--starts-per-scene",
        type=int,
        default=None,
        help=(
            "Fixed start positions per scene. "
            "Default is 2 for val-only generation, or 1 when --include-train-scenes is enabled."
        ),
    )
    parser.add_argument("--val-seed", type=int, default=42, help="Seed for val navmesh sampling")
    parser.add_argument(
        "--min-ceiling-clearance",
        type=float,
        default=MIN_CEILING_CLEARANCE,
        help=f"Minimum metres above camera (default {MIN_CEILING_CLEARANCE})",
    )
    parser.add_argument(
        "--min-island-radius",
        type=float,
        default=MIN_ISLAND_RADIUS,
        help=f"Minimum island radius in metres (default {MIN_ISLAND_RADIUS})",
    )
    parser.add_argument(
        "--min-start-separation",
        type=float,
        default=MIN_START_SEPARATION_M,
        help=f"Minimum distance between fixed starts in a scene (default {MIN_START_SEPARATION_M})",
    )
    parser.add_argument(
        "--min-start-valid-depth-frac",
        type=float,
        default=MIN_START_VALID_DEPTH_FRAC,
        help=f"Minimum valid-depth fraction at episode start (default {MIN_START_VALID_DEPTH_FRAC})",
    )
    parser.add_argument(
        "--min-start-clear-depth",
        type=float,
        default=MIN_START_CLEAR_DEPTH_M,
        help=f"Minimum closest valid depth at episode start (default {MIN_START_CLEAR_DEPTH_M})",
    )
    parser.add_argument(
        "--min-local-same-floor-coverage",
        type=float,
        default=MIN_LOCAL_SAME_FLOOR_COVERAGE,
        help=(
            "Minimum local same-floor support for candidate starts "
            f"(default {MIN_LOCAL_SAME_FLOOR_COVERAGE})"
        ),
    )
    parser.add_argument(
        "--min-start-enclosure-frac",
        type=float,
        default=MIN_START_ENCLOSURE_FRAC,
        help=f"Minimum enclosure fraction at episode start (default {MIN_START_ENCLOSURE_FRAC})",
    )
    parser.add_argument(
        "--local-same-floor-y-tol",
        type=float,
        default=LOCAL_SAME_FLOOR_Y_TOL_M,
        help=f"Max Y-height delta for local same-floor support (default {LOCAL_SAME_FLOOR_Y_TOL_M})",
    )
    parser.add_argument(
        "--prefer-distinct-islands",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Prefer selecting starts from distinct islands when possible",
    )
    parser.add_argument(
        "--floor-plan-dir",
        type=str,
        default=None,
        help="Deprecated and ignored; HM3D-style split generation does not emit floor-plan PNGs.",
    )
    parser.add_argument(
        "--include-train-scenes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include train scene set in addition to val scenes for generated output",
    )
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    out_dir_base = os.path.abspath(args.output_dir) if args.output_dir else dataset_root
    scene_root = _resolve_scene_root(args.scene_root)
    if args.floor_plan_dir:
        _warn(
            "--floor-plan-dir is deprecated and ignored; "
            "Gibson split generation now mirrors current HM3D split behavior."
        )

    val_scene_names = _load_gibson_val_scene_names(dataset_root)
    if args.include_train_scenes:
        train_scene_names = _load_gibson_train_scene_names(dataset_root)
        scene_names = sorted(set(val_scene_names) | set(train_scene_names))
        print(
            "[generate_gibson_split] Found "
            f"{len(val_scene_names)} val + {len(train_scene_names)} train scenes "
            f"({len(scene_names)} unique total) in {dataset_root}",
            flush=True,
        )
    else:
        scene_names = val_scene_names
        print(
            f"[generate_gibson_split] Found {len(scene_names)} Gibson val scenes in {dataset_root}",
            flush=True,
        )

    glb_index = _index_glb_files(scene_root)
    if not glb_index:
        raise FileNotFoundError(f"No Gibson .glb files found under scene root: {scene_root}")

    default_n_starts = 1 if args.include_train_scenes else 2
    requested_n_starts = default_n_starts if args.starts_per_scene is None else int(args.starts_per_scene)
    n_starts = max(1, min(requested_n_starts, 5))
    print(f"[generate_gibson_split] Using {n_starts} start(s) per scene", flush=True)

    episodes: list[dict] = []
    skipped_scenes = 0
    missing_scenes = 0

    for scene_idx, scene_name in enumerate(scene_names):
        scene_paths = glb_index.get(scene_name)
        if scene_paths is None:
            missing_scenes += 1
            _warn(f"Skipping val scene {scene_name}: .glb not found under {scene_root}")
            continue
        glb_path, navmesh_path = scene_paths

        starts = _sample_navmesh_starts(
            glb_path,
            navmesh_path,
            n_starts,
            int(args.val_seed) + scene_idx,
            min_ceiling_clearance=float(args.min_ceiling_clearance),
            min_island_radius=float(args.min_island_radius),
            min_start_separation_m=float(args.min_start_separation),
            min_start_valid_depth_frac=float(args.min_start_valid_depth_frac),
            min_start_clear_depth_m=float(args.min_start_clear_depth),
            min_local_same_floor_coverage=float(args.min_local_same_floor_coverage),
            min_start_enclosure_frac=float(args.min_start_enclosure_frac),
            local_same_floor_y_tol_m=float(args.local_same_floor_y_tol),
            prefer_distinct_islands=bool(args.prefer_distinct_islands),
        )
        if len(starts) < n_starts:
            skipped_scenes += 1
            _warn(
                f"Skipping val scene {scene_name}: found {len(starts)}/{n_starts} "
                "valid start positions after sampling"
            )
            continue

        for start_idx, start in enumerate(starts):
            episode = {
                "episode_id": f"val_{scene_idx}_{start_idx}",
                "scene_id": f"{_GIBSON_SCENE_ID_PREFIX}{scene_name}.glb",
                "scene_name": scene_name,
                "start_position": start["start_position"],
                "start_rotation": start["start_rotation"],
                "island_index": start["island_index"],
            }
            episodes.append(episode)

    out_dir = os.path.join(out_dir_base, "val")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.output_filename)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump({"episodes": episodes}, f, indent=2)

    print(f"[generate_gibson_split] Wrote {len(episodes)} episodes to {out_path}", flush=True)
    if missing_scenes:
        _warn(f"Missing scene assets for {missing_scenes} val scenes")
    if skipped_scenes:
        _warn(f"Skipped {skipped_scenes} val scenes due to insufficient valid start positions")


if __name__ == "__main__":
    main()
