import io
import os
import threading
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, Response, jsonify, request, send_file
import torch
import torch.nn.functional as F
from gsplat.strategy import MCMCStrategy

from .gsplat import (
    depth_points_colors_batch,
    grow_splats_with_new_points,
    k_to_intr,
    optimize_splat,
    render_img,
)
from .env import HabitatMP3DEnv, list_scene_glbs


KEYMAP = {
    "w": 0,
    "q": 1,
    "e": 2,
    "s": 3,
    "1": 0,
    "2": 1,
    "3": 2,
    "4": 3,
}
ACTION_NAMES = ["forward", "look left", "look right", "stop"]


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "off", "no"}


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _float_env(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _optional_int_env(name: str, default: int | None) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    if value == "" or value.lower() == "none":
        return None
    parsed = int(value)
    return None if parsed <= 0 else parsed


PORT = _int_env("ENV_GSPLAT_VIEWER_PORT", 8084)
GPU_ID = _int_env("ENV_GSPLAT_VIEWER_GPU_ID", 0)
MAX_STEPS = _int_env("ENV_GSPLAT_VIEWER_MAX_STEPS", 256)
BIG_HW = _int_env("ENV_GSPLAT_VIEWER_BIG_HW", 128)
CAP_MAX = _int_env("ENV_GSPLAT_VIEWER_CAP_MAX", 1_000_000)
MAX_POINTS_PER_FRAME = _int_env("ENV_GSPLAT_VIEWER_MAX_POINTS_PER_FRAME", 5_000)
OPTIMIZE_EVERY = _int_env("ENV_GSPLAT_VIEWER_OPTIMIZE_EVERY", 16)
ITERS_PER_OPT = _int_env("ENV_GSPLAT_VIEWER_ITERS_PER_OPT", 10)
BATCH_CAMS = _int_env("ENV_GSPLAT_VIEWER_BATCH_CAMS", 10)
MAX_SPLAT_STEPS = _int_env("ENV_GSPLAT_VIEWER_MAX_SPLAT_STEPS", 10**9)
SCENE_INDEX = _int_env("ENV_GSPLAT_VIEWER_SCENE_INDEX", 0)
SCENE_COUNT = _int_env("ENV_GSPLAT_VIEWER_SCENE_COUNT", 0)
WINDOW = _optional_int_env("ENV_GSPLAT_VIEWER_WINDOW", None)
NEWEST_PROB = _float_env("ENV_GSPLAT_VIEWER_NEWEST_PROB", 0.1)
TEST_SPLIT = _bool_env("ENV_GSPLAT_VIEWER_TEST_SPLIT", False)
DEVICE = os.environ.get(
    "ENV_GSPLAT_VIEWER_DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu",
)


app = Flask(__name__)
lock = threading.Lock()
state: dict[str, Any] = {
    "env_rgb": None,
    "gsplat_rgb": None,
    "stats": {},
}


def _select_scene_list() -> list[dict[str, str | None]]:
    scenes = list_scene_glbs(test=TEST_SPLIT)
    if SCENE_COUNT <= 0:
        return scenes
    start = max(0, min(SCENE_INDEX, len(scenes) - 1))
    end = min(len(scenes), start + max(1, SCENE_COUNT))
    return scenes[start:end]


scene_list = _select_scene_list()
env = HabitatMP3DEnv(
    scene_list=scene_list,
    max_steps=MAX_STEPS,
    render_mode="rgb_array",
    gpu_id=GPU_ID,
)
env._eval_panel_layout = True


def _empty_track() -> dict[str, Any]:
    return dict(
        splats=None,
        optimizers=None,
        schedulers=None,
        strategy_state=None,
        frames_seen=0,
    )


gs_state: dict[str, Any] = {
    "anchor_w2c": None,
    "track": _empty_track(),
    "strategy": MCMCStrategy(
        cap_max=CAP_MAX,
        noise_lr=5e5,
        refine_start_iter=0,
        refine_stop_iter=10**9,
        refine_every=3,
        min_opacity=0.001,
        verbose=True,
    ),
    "history": dict(images=[], depths=[], c2ws=[], Ks=[]),
}


def _default_stats() -> dict[str, Any]:
    return {
        "scene_name": None,
        "step": 0,
        "frames_seen": 0,
        "terminated": False,
        "truncated": False,
        "last_action": None,
        "moved": None,
        "Nsplats": 0,
        "num_new": 0,
        "num_pruned": 0,
        "num_pruned_ingest": 0,
        "num_pruned_opt": 0,
        "hit_cap": False,
        "optimize_ran": False,
        "window": WINDOW,
        "cap_max": CAP_MAX,
        "max_points_per_frame": MAX_POINTS_PER_FRAME,
    }


state["stats"] = _default_stats()


def _build_anchor_w2c(c2w: torch.Tensor) -> torch.Tensor:
    c2w = c2w.float()
    R = c2w[:3, :3]
    t = c2w[:3, 3:4]
    w2c = torch.eye(4, device=c2w.device, dtype=torch.float32)
    w2c[:3, :3] = R.transpose(0, 1)
    w2c[:3, 3] = (-w2c[:3, :3] @ t)[:, 0]
    return w2c


def _manual_logits(last_action: int | None) -> np.ndarray:
    if last_action is None:
        return np.ones((4,), dtype=np.float32)
    logits = np.zeros((4,), dtype=np.float32)
    logits[int(last_action)] = 10.0
    return logits


def _png_bytes(rgb_uint8: np.ndarray) -> bytes:
    image = Image.fromarray(rgb_uint8)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _blank_gsplat_frame(message: str, stats: dict[str, Any]) -> np.ndarray:
    image = Image.new("RGB", (BIG_HW, BIG_HW), (18, 18, 18))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    draw.text((14, BIG_HW // 2 - 12), message, fill=(220, 220, 220), font=font)
    return np.asarray(image, dtype=np.uint8)


def _reset_gs_state() -> None:
    gs_state["anchor_w2c"] = None
    gs_state["track"] = _empty_track()
    gs_state["history"] = dict(images=[], depths=[], c2ws=[], Ks=[])


def _preprocess_obs(obs: dict[str, Any]) -> dict[str, Any]:
    rgb_u8 = torch.as_tensor(obs["rgb"], device=DEVICE, dtype=torch.uint8)
    depth = torch.as_tensor(obs["depth"], device=DEVICE, dtype=torch.float32)
    k_vec = torch.as_tensor(obs["fxfycxcy"], device=DEVICE, dtype=torch.float32)
    c2w = torch.as_tensor(obs["c2w"], device=DEVICE, dtype=torch.float32)

    H0, W0 = rgb_u8.shape[0], rgb_u8.shape[1]
    rgb_cf = rgb_u8.permute(2, 0, 1).unsqueeze(0).float() / 255.0
    rgb_big = F.interpolate(
        rgb_cf,
        size=(BIG_HW, BIG_HW),
        mode="bilinear",
        align_corners=False,
    )[0].permute(1, 2, 0).contiguous()

    depth_big = F.interpolate(
        depth.unsqueeze(0).unsqueeze(0),
        size=(BIG_HW, BIG_HW),
        mode="nearest",
    )[0, 0].contiguous()

    sx = float(BIG_HW) / float(W0)
    sy = float(BIG_HW) / float(H0)
    k_big_vec = k_vec.clone()
    k_big_vec[0] *= sx
    k_big_vec[1] *= sy
    k_big_vec[2] *= sx
    k_big_vec[3] *= sy
    K_big = k_to_intr(k_big_vec.unsqueeze(0))[0].contiguous()

    if gs_state["anchor_w2c"] is None:
        gs_state["anchor_w2c"] = _build_anchor_w2c(c2w)
    c2w_canon = (gs_state["anchor_w2c"] @ c2w).contiguous()

    return {
        "rgb_sensor": np.asarray(obs["rgb"], dtype=np.uint8),
        "rgb_big": rgb_big,
        "depth_big": depth_big,
        "K_big": K_big,
        "c2w_big": c2w_canon,
    }


def _maybe_init_opt_state(track: dict[str, Any]) -> None:
    if track["splats"] is None or track["optimizers"] is None:
        return
    if track["strategy_state"] is None:
        track["strategy_state"] = gs_state["strategy"].initialize_state()
    if track["schedulers"] is None:
        track["schedulers"] = [
            torch.optim.lr_scheduler.ExponentialLR(
                track["optimizers"]["means"],
                gamma=0.01 ** (1.0 / MAX_SPLAT_STEPS),
            )
        ]


def _optimize_track_if_needed(step_idx: int) -> tuple[bool, int]:
    if OPTIMIZE_EVERY <= 0 or step_idx % OPTIMIZE_EVERY != 0:
        return False, 0

    track = gs_state["track"]
    if track["splats"] is None:
        return False, 0

    _maybe_init_opt_state(track)
    history = gs_state["history"]
    start = 0 if WINDOW is None else max(0, len(history["images"]) - int(WINDOW))
    images = torch.stack(history["images"][start:], dim=0)
    depths = torch.stack(history["depths"][start:], dim=0)
    c2ws = torch.stack(history["c2ws"][start:], dim=0)
    Ks = torch.stack(history["Ks"][start:], dim=0)

    before = int(track["splats"]["means"].shape[0])
    (
        track["splats"],
        track["optimizers"],
        _,
        track["strategy_state"],
        track["schedulers"],
    ) = optimize_splat(
        track["splats"],
        track["optimizers"],
        gs_state["strategy"],
        track["strategy_state"],
        track["schedulers"],
        images,
        depths,
        c2ws,
        Ks,
        BIG_HW,
        BIG_HW,
        iters=ITERS_PER_OPT,
        batch_cams=BATCH_CAMS,
        newest_prob=NEWEST_PROB,
        device=DEVICE,
    )
    after = int(track["splats"]["means"].shape[0])
    if DEVICE.startswith("cuda"):
        torch.cuda.empty_cache()
    return True, max(0, before - after)


def _render_current_gsplat(c2w_big: torch.Tensor, K_big: torch.Tensor, stats: dict[str, Any]) -> np.ndarray:
    track = gs_state["track"]
    if track["splats"] is None:
        return _blank_gsplat_frame("No splats yet", stats)

    with torch.no_grad():
        pred_rgb, _pred_depth = render_img(track["splats"], c2w_big, K_big, BIG_HW, BIG_HW)
    pred_rgb_u8 = (
        pred_rgb.clamp(0.0, 1.0).mul(255.0).to(torch.uint8).cpu().numpy()
    )
    return pred_rgb_u8


def _update_env_panel(rgb_sensor: np.ndarray, last_action: int | None) -> np.ndarray:
    env.update_meta(
        [
            {
                "rgb_gt": rgb_sensor.astype(np.float32) / 255.0,
                "policy_logits": _manual_logits(last_action),
                "mode": 3,
            }
        ]
    )
    return np.asarray(env.render(), dtype=np.uint8)


def _refresh_from_obs(
    obs: dict[str, Any],
    *,
    last_action: int | None,
    moved: bool | None,
) -> None:
    prep = _preprocess_obs(obs)
    rgb_big = prep["rgb_big"]
    depth_big = prep["depth_big"]
    K_big = prep["K_big"]
    c2w_big = prep["c2w_big"]
    rgb_sensor = prep["rgb_sensor"]

    history = gs_state["history"]
    history["images"].append(rgb_big)
    history["depths"].append(depth_big)
    history["c2ws"].append(c2w_big)
    history["Ks"].append(K_big)

    track = gs_state["track"]
    points_list, rgbs_list, depth_list = depth_points_colors_batch(
        rgb_big.unsqueeze(0),
        depth_big.unsqueeze(0),
        K_big.unsqueeze(0),
        c2w_big.unsqueeze(0),
        stride=1,
        max_points_per_frame=MAX_POINTS_PER_FRAME,
    )
    points_new = points_list[0]
    rgbs_new = rgbs_list[0]
    depths_new = depth_list[0]

    prev_n = 0 if track["splats"] is None else int(track["splats"]["means"].shape[0])
    num_new = int(points_new.shape[0])

    if not (track["splats"] is None and num_new == 0):
        track["splats"], track["optimizers"] = grow_splats_with_new_points(
            track["splats"],
            track["optimizers"],
            points_new,
            rgbs_new,
            d_new=depths_new,
            K_new=K_big,
            stride=1,
            device=DEVICE,
            cap_max=CAP_MAX,
        )

    track["frames_seen"] += 1
    after_ingest_n = 0 if track["splats"] is None else int(track["splats"]["means"].shape[0])
    num_pruned_ingest = max(0, prev_n + num_new - after_ingest_n)
    optimize_ran, num_pruned_opt = _optimize_track_if_needed(track["frames_seen"])
    nsplats = 0 if track["splats"] is None else int(track["splats"]["means"].shape[0])

    stats = {
        "scene_name": getattr(env, "scene_name", None),
        "step": int(getattr(env, "_t", 0)),
        "frames_seen": int(track["frames_seen"]),
        "last_action": None if last_action is None else ACTION_NAMES[int(last_action)],
        "moved": moved,
        "Nsplats": nsplats,
        "num_new": num_new,
        "num_pruned": int(num_pruned_ingest + num_pruned_opt),
        "num_pruned_ingest": int(num_pruned_ingest),
        "num_pruned_opt": int(num_pruned_opt),
        "hit_cap": bool(nsplats >= CAP_MAX),
        "optimize_ran": bool(optimize_ran),
        "window": WINDOW,
        "cap_max": CAP_MAX,
        "max_points_per_frame": MAX_POINTS_PER_FRAME,
    }

    state["env_rgb"] = _update_env_panel(rgb_sensor, last_action)
    state["gsplat_rgb"] = _render_current_gsplat(c2w_big, K_big, stats)
    state["stats"] = stats


@app.get("/")
def index() -> Response:
    return Response(
        """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Env + GSplat Viewer</title>
  <style>
    body {
      margin: 0;
      background: #101214;
      color: #f1f3f5;
      font-family: "IBM Plex Sans", "Helvetica Neue", sans-serif;
    }
    .wrap {
      display: flex;
      flex-direction: column;
      gap: 12px;
      padding: 16px;
    }
    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }
    button {
      padding: 10px 14px;
      border: 1px solid #3b434b;
      background: #171b20;
      color: #f1f3f5;
      border-radius: 10px;
      cursor: pointer;
    }
    button:hover {
      background: #20262c;
    }
    .panes {
      display: grid;
      grid-template-columns: minmax(0, 2fr) minmax(360px, 1fr);
      gap: 12px;
      align-items: start;
    }
    .pane {
      background: #161a1f;
      border: 1px solid #2d343c;
      border-radius: 14px;
      padding: 12px;
    }
    .pane h2 {
      margin: 0 0 10px 0;
      font-size: 15px;
      font-weight: 600;
      letter-spacing: 0.02em;
    }
    img {
      display: block;
      width: 100%;
      height: auto;
      border-radius: 10px;
      background: #0f1113;
    }
    pre {
      margin: 10px 0 0 0;
      padding: 10px 12px;
      background: #0f1113;
      border-radius: 10px;
      border: 1px solid #2d343c;
      overflow: auto;
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 13px;
      line-height: 1.4;
    }
    .hint {
      color: #aab4be;
      font-size: 13px;
    }
    code {
      background: #0f1113;
      padding: 2px 6px;
      border-radius: 6px;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="controls">
      <button onclick="doReset()">Reset</button>
      <button onclick="sendKey('w')">Forward (W)</button>
      <button onclick="sendKey('q')">Look left (Q)</button>
      <button onclick="sendKey('e')">Look right (E)</button>
      <button onclick="sendKey('s')">Stop (S)</button>
      <button onclick="refreshPanels()">Refresh</button>
    </div>

    <div class="hint">
      Click the page, then use <code>W</code> forward, <code>Q/E</code> yaw, <code>S</code> stop, <code>R</code> reset.
    </div>

    <div class="panes">
      <div class="pane">
        <h2>Environment</h2>
        <img id="env" src="/env_frame?ts=0" />
      </div>

      <div class="pane">
        <h2>GSplat</h2>
        <img id="gsplat" src="/gsplat_frame?ts=0" />
        <pre id="stats">loading...</pre>
      </div>
    </div>
  </div>

<script>
  const envImg = document.getElementById("env");
  const gsplatImg = document.getElementById("gsplat");
  const statsEl = document.getElementById("stats");

  async function refreshPanels() {
    const ts = Date.now();
    envImg.src = "/env_frame?ts=" + ts;
    gsplatImg.src = "/gsplat_frame?ts=" + ts;
    const res = await fetch("/stats?ts=" + ts);
    const stats = await res.json();
    statsEl.textContent = JSON.stringify(stats, null, 2);
  }

  async function sendKey(k) {
    await fetch("/act", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key: k }),
    });
    await refreshPanels();
  }

  async function doReset() {
    await fetch("/reset", { method: "POST" });
    await refreshPanels();
  }

  window.addEventListener("keydown", async (ev) => {
    const k = ev.key.toLowerCase();
    if (k === "r") {
      await doReset();
      return;
    }
    if (["w", "q", "e", "s"].includes(k)) {
      await sendKey(k);
    }
  });

  window.addEventListener("load", refreshPanels);
</script>
</body>
</html>
        """,
        mimetype="text/html",
    )


@app.get("/env_frame")
def env_frame():
    with lock:
        rgb = state["env_rgb"]
        if rgb is None:
            rgb = np.zeros((360, 640, 3), dtype=np.uint8)
        png = _png_bytes(rgb)
    return send_file(io.BytesIO(png), mimetype="image/png")


@app.get("/gsplat_frame")
def gsplat_frame():
    with lock:
        rgb = state["gsplat_rgb"]
        if rgb is None:
            rgb = _blank_gsplat_frame("No splats yet", state["stats"])
        png = _png_bytes(rgb)
    return send_file(io.BytesIO(png), mimetype="image/png")


@app.get("/stats")
def stats():
    with lock:
        return jsonify(state["stats"])


@app.post("/act")
def act():
    data = request.get_json(force=True, silent=True) or {}
    key = str(data.get("key", "")).lower()
    if key not in KEYMAP:
        return ("", 204)

    action = int(KEYMAP[key])
    with lock:
        obs, _reward, terminated, truncated, info = env.step(action)
        _refresh_from_obs(obs, last_action=action, moved=info.get("moved"))
        state["stats"]["terminated"] = bool(terminated)
        state["stats"]["truncated"] = bool(truncated)

    return ("", 204)


@app.post("/reset")
def reset():
    with lock:
        _reset_gs_state()
        obs, _info = env.reset()
        _refresh_from_obs(obs, last_action=None, moved=None)
        state["stats"]["terminated"] = False
        state["stats"]["truncated"] = False
    return ("", 204)


if __name__ == "__main__":
    with lock:
        obs, _info = env.reset()
        _refresh_from_obs(obs, last_action=None, moved=None)
        state["stats"]["terminated"] = False
        state["stats"]["truncated"] = False

    print(f"Open: http://localhost:{PORT}")
    app.run(
        host="0.0.0.0",
        port=PORT,
        debug=False,
        threaded=False,
        processes=1,
        use_reloader=False,
    )
