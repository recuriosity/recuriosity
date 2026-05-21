import io
import os
import threading

import numpy as np
from flask import Flask, Response, request, send_file
from PIL import Image

from .env import HEIGHT, WIDTH
from .env_apples import HabitatMP3DEnv, list_scene_glbs


KEYMAP = {
    "w": 0,  # forward
    "q": 1,  # look left
    "e": 2,  # look right
    "s": 3,  # stop
    "1": 0,
    "2": 1,
    "3": 2,
    "4": 3,
}

app = Flask(__name__)

lock = threading.Lock()
state = {"LAST_RGB": None, "LAST_ACTION": None}
env = None
env_init_error: str | None = None
env_settings: dict[str, object] = {}


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name, "").strip()
    if not val:
        return int(default)
    try:
        return int(val)
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name, "").strip()
    if not val:
        return float(default)
    try:
        return float(val)
    except ValueError:
        return float(default)


def populate_dummy_meta_for_testing(env, last_action=None):
    """
    For debugging rendering without a trainer:
    - uses current sensor rgb as both gt/pred
    - creates a one-hot-ish policy logits
    - pushes a dummy reward so metric strip shows arrows
    """
    obs = env._observe()
    rgb = obs["rgb"].astype(np.float32) / 255.0
    env._gt_rgb = rgb
    env._pred_rgb = rgb

    if last_action is not None:
        logits = np.zeros(4, dtype=np.float32)
        logits[last_action] = 10.0
    else:
        logits = np.ones(4, dtype=np.float32)
    env._last_logits = logits

    env.set_rollout_metrics(rewards=[1.0])


def get_png_bytes(rgb_uint8: np.ndarray) -> bytes:
    im = Image.fromarray(rgb_uint8)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _ensure_env_ready() -> bool:
    global env
    global env_init_error

    if env is not None:
        return True
    if env_init_error is not None:
        return False

    try:
        print("[apple_viewer] initializing environment...", flush=True)
        scene_list = list_scene_glbs()[:1]
        env = HabitatMP3DEnv(
            scene_list=scene_list,
            max_steps=int(env_settings.get("max_steps", 256)),
            render_mode="rgb_array",
            gpu_id=int(env_settings.get("gpu_id", 0)),
            num_apples=int(env_settings.get("num_apples", 5)),
            apple_asset_path=str(env_settings.get("apple_asset_path", "data/Apple.glb")),
            apple_collect_radius_m=float(env_settings.get("collect_radius_m", 0.80)),
            apple_diameter_m=float(env_settings.get("apple_diameter_m", 0.60)),
            apple_spawn_min_separation_m=float(env_settings.get("min_separation_m", 0.8)),
            apple_height_offset_m=float(env_settings.get("apple_height_offset_m", -0.15)),
        )
        _obs, _info = env.reset()
        populate_dummy_meta_for_testing(env, last_action=None)
        state["LAST_RGB"] = env.render()
        print("[apple_viewer] environment ready", flush=True)
        return True
    except Exception as exc:
        env_init_error = f"{type(exc).__name__}: {exc}"
        print(f"[apple_viewer] init failed: {env_init_error}", flush=True)
        return False


@app.get("/")
def index():
    return Response(
        """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Habitat Apple Env Debug Viewer</title>
  <style>
    body { margin:0; background:#111; color:#eee; font-family: ui-sans-serif, system-ui; }
    .wrap { display:flex; flex-direction:column; align-items:center; gap:12px; padding:16px; }
    img { max-width:98vw; height:auto; border:1px solid #444; }
    code { background:#222; padding:2px 6px; border-radius:6px; }
    .row { display:flex; gap:10px; flex-wrap:wrap; justify-content:center; }
    button { padding:10px 14px; border-radius:10px; border:1px solid #444; background:#1b1b1b; color:#eee; cursor:pointer; }
    button:hover { background:#242424; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="row">
      <button onclick="doReset()">Reset</button>
      <button onclick="sendKey('w')">Forward (W)</button>
      <button onclick="sendKey('q')">Look left (Q)</button>
      <button onclick="sendKey('e')">Look right (E)</button>
      <button onclick="sendKey('s')">Stop (S)</button>
    </div>

    <img id="view" src="/frame?ts=0" />

    <div>
      Keys:
      <code>W</code> forward,
      <code>Q/E</code> yaw,
      <code>S</code> stop,
      <code>R</code> reset
    </div>
  </div>

<script>
  const img = document.getElementById("view");
  function refresh() { img.src = "/frame?ts=" + Date.now(); }

  async function sendKey(k) {
    await fetch("/act", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key: k })
    });
    refresh();
  }

  async function doReset() {
    await fetch("/reset", { method: "POST" });
    refresh();
  }

  window.addEventListener("keydown", async (ev) => {
    const k = ev.key.toLowerCase();
    if (k === "r") { await doReset(); return; }
    await sendKey(k);
  });
</script>
</body>
</html>
        """,
        mimetype="text/html",
    )


@app.get("/frame")
def frame():
    with lock:
        if not _ensure_env_ready():
            rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            png = get_png_bytes(rgb)
            return send_file(io.BytesIO(png), mimetype="image/png")
        rgb = state["LAST_RGB"]
        if rgb is None:
            rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        png = get_png_bytes(rgb)
    return send_file(io.BytesIO(png), mimetype="image/png")


@app.post("/act")
def act():
    data = request.get_json(force=True, silent=True) or {}
    key = str(data.get("key", "")).lower()
    if key not in KEYMAP:
        return ("", 204)

    action = int(KEYMAP[key])

    with lock:
        if not _ensure_env_ready():
            return (env_init_error or "env init failed", 500)
        _obs, reward, _terminated, _truncated, info = env.step(action)
        populate_dummy_meta_for_testing(env, last_action=action)
        state["LAST_RGB"] = env.render()
        state["LAST_ACTION"] = action

    print(
        f"[apple_viewer] action={action} reward={reward} "
        f"visible={info.get('apple_visible')} "
        f"remaining={info.get('apples_remaining')} "
        f"collected_step={info.get('apples_collected_step')}",
        flush=True,
    )

    return ("", 204)


@app.post("/reset")
def reset():
    with lock:
        if not _ensure_env_ready():
            return (env_init_error or "env init failed", 500)
        _obs, info = env.reset()
        populate_dummy_meta_for_testing(env, last_action=None)
        state["LAST_RGB"] = env.render()
        state["LAST_ACTION"] = None

    print(
        f"[apple_viewer] reset visible={info.get('apple_visible')} "
        f"remaining={info.get('apples_remaining')} "
        f"total={info.get('num_apples_total')}",
        flush=True,
    )
    return ("", 204)


if __name__ == "__main__":
    port = _env_int("APPLE_VIEWER_PORT", 8083)
    num_apples = _env_int("APPLE_VIEWER_NUM_APPLES", 5)
    apple_asset_path = os.environ.get("APPLE_VIEWER_ASSET_PATH", "data/Apple.glb")
    collect_radius_m = _env_float("APPLE_VIEWER_COLLECT_RADIUS_M", 1.5)
    apple_diameter_m = _env_float("APPLE_VIEWER_DIAMETER_M", 0.40)
    min_separation_m = _env_float("APPLE_VIEWER_MIN_SEPARATION_M", 0.8)
    apple_height_offset_m = _env_float("APPLE_VIEWER_HEIGHT_OFFSET_M", -0.15)
    env_settings.update(
        {
            "max_steps": 256,
            "gpu_id": 0,
            "num_apples": num_apples,
            "apple_asset_path": apple_asset_path,
            "collect_radius_m": collect_radius_m,
            "apple_diameter_m": apple_diameter_m,
            "min_separation_m": min_separation_m,
            "apple_height_offset_m": apple_height_offset_m,
        }
    )

    print(f"Open: http://localhost:{port}")
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
        threaded=False,
        processes=1,
        use_reloader=False,
    )
