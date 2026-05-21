import os
import math
from collections import deque
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.cm as cm
import torch
import torch.nn.functional as F


def _safe_int_attr(env, name, default, min_value=1):
    try:
        v = int(getattr(env, name, default))
    except Exception:
        v = int(default)
    return max(min_value, v)


def _render_cache(env):
    cache = getattr(env, "_render_cache", None)
    if cache is None:
        cache = {}
        env._render_cache = cache
    return cache


def _resize_filter(env):
    # Fast-by-default. Set env._render_use_lanczos = True for higher quality.
    use_lanczos = bool(getattr(env, "_render_use_lanczos", False))
    return Image.LANCZOS if use_lanczos else Image.BILINEAR


def compute_topdown_camera(
    trajectory_positions,
    map_size=512,
    margin_m=4.0,
    fov_deg=90.0,
    camera_height_m=1.25,
    max_height_above_floor=1.6,
    min_height_above_floor=0.6,
):
    """
    trajectory_positions: list of (3,) CAMERA world positions [x,y,z] (Habitat world: Y up)
    Returns:
      c2w:        (4,4) torch.float32   camera->world (OpenCV camera axes: x right, y down, z forward)
      intrinsics: (4,)  torch.float32   [fx, fy, cx, cy] in pixels for a map_size x map_size render
      metadata:   dict  compatible with world_to_map_pixel() using X-Z plane bounds

    Conventions:
      - World: +Y up
      - OpenCV camera frame: +x right, +y down, +z forward
      - For topdown: camera looks down => z_cam_world = (0, -1, 0)
      - We want image +x to align with world +X, and image +y to align with world +Z
        so that world_to_map_pixel(x,z) overlays correctly.
    """

    # ---------------- handle empty ----------------
    if trajectory_positions is None or len(trajectory_positions) == 0:
        size = 10.0
        half = 0.5 * size
        cx_w, cz_w = 0.0, 0.0

        # If we have no trajectory, assume floor at 0
        floor_y = 0.0

        fov = math.radians(float(fov_deg))
        # geometric height to fit half-extent at given FOV
        height_geom = float(half / max(1e-6, math.tan(0.5 * fov)))
        height = float(np.clip(height_geom, min_height_above_floor, max_height_above_floor))
        cam_y = float(floor_y + height)

        # camera axes in world
        x_cam_w = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)   # world +X
        y_cam_w = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)   # world +Z  (image down)
        z_cam_w = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32)  # world -Y  (look down)

        R = torch.stack([x_cam_w, y_cam_w, z_cam_w], dim=1)  # columns are camera axes in world

        c2w = torch.eye(4, dtype=torch.float32)
        c2w[:3, :3] = R
        c2w[:3, 3]  = torch.tensor([cx_w, cam_y, cz_w], dtype=torch.float32)

        cx_img = cy_img = (map_size - 1) * 0.5
        fx = fy = float((map_size * 0.5) / math.tan(0.5 * fov))
        intr = torch.tensor([fx, fy, cx_img, cy_img], dtype=torch.float32)

        meta = dict(
            min_x=-half, max_x=half,
            min_z=-half, max_z=half,
            floor_y=float(floor_y),
            range=float(size),
            pixels_per_meter=float(map_size / size),
        )
        return c2w, intr, meta

    traj = np.asarray(trajectory_positions, dtype=np.float32)  # (N,3) camera positions

    # ---------------- square bounds in X-Z ----------------
    min_x = float(traj[:, 0].min()); max_x = float(traj[:, 0].max())
    min_z = float(traj[:, 2].min()); max_z = float(traj[:, 2].max())

    min_x -= float(margin_m); max_x += float(margin_m)
    min_z -= float(margin_m); max_z += float(margin_m)

    cx_w = 0.5 * (min_x + max_x)
    cz_w = 0.5 * (min_z + max_z)

    size_x = max_x - min_x
    size_z = max_z - min_z
    size = float(max(size_x, size_z))
    size = max(size, 1e-3)
    half = 0.5 * size

    # force square crop around center (stable metadata)
    min_x = cx_w - half; max_x = cx_w + half
    min_z = cz_w - half; max_z = cz_w + half
    ppm = float(map_size / size)

    # ---------------- IMPORTANT FIX: estimate floor_y from camera y ----------------
    # traj[:,1] is camera height. Floor is camera_y - camera_height_m.
    cam_y_med = float(np.median(traj[:, 1]))
    floor_y = float(cam_y_med - float(camera_height_m))

    # ---------------- camera height to fit square given FOV ----------------
    fov = math.radians(float(fov_deg))
    height_geom = float(half / max(1e-6, math.tan(0.5 * fov)))

    # clamp so we don't end up above the ceiling
    height = float(np.clip(height_geom, min_height_above_floor, max_height_above_floor))
    cam_y = float(floor_y + height)

    # ---------------- OpenCV camera->world rotation ----------------
    # Keep orientation consistent with world_to_map_pixel:
    #  - image x axis aligns with world +X
    #  - image y axis aligns with world +Z
    #  - camera forward (z) points down world -Y
    x_cam_w = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)   # right
    y_cam_w = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)   # down in image
    z_cam_w = torch.tensor([0.0, -1.0, 0.0], dtype=torch.float32)  # forward (down)

    # normalize (safe)
    x_cam_w = x_cam_w / (x_cam_w.norm() + 1e-8)
    y_cam_w = y_cam_w / (y_cam_w.norm() + 1e-8)
    z_cam_w = z_cam_w / (z_cam_w.norm() + 1e-8)

    R = torch.stack([x_cam_w, y_cam_w, z_cam_w], dim=1)

    c2w = torch.eye(4, dtype=torch.float32)
    c2w[:3, :3] = R
    c2w[:3, 3]  = torch.tensor([cx_w, cam_y, cz_w], dtype=torch.float32)

    # ---------------- intrinsics ----------------
    cx_img = cy_img = (map_size - 1) * 0.5
    fx = fy = float((map_size * 0.5) / math.tan(0.5 * fov))
    intrinsics = torch.tensor([fx, fy, cx_img, cy_img], dtype=torch.float32)

    metadata = dict(
        min_x=float(min_x), max_x=float(max_x),
        min_z=float(min_z), max_z=float(max_z),
        floor_y=float(floor_y),
        range=float(size),
        pixels_per_meter=float(ppm),
        # useful debug extras (won't break your code)
        cam_y=float(cam_y),
        height=float(height),
        height_geom=float(height_geom),
        cam_y_median=float(cam_y_med),
    )

    return c2w, intrinsics, metadata



    
# --------------------------- error/reward overlay helpers ---------------------------

def gaussian_blur2d(x, sigma=1.0, k=None):
    # x: (B,C,H,W)
    if k is None:
        k = int(2 * round(3 * sigma) + 1)
    if k <= 1:
        return x
    device, dtype = x.device, x.dtype
    t = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2
    g = torch.exp(-(t**2) / (2 * sigma**2))
    g = g / g.sum()
    gx = g.view(1, 1, 1, k).repeat(x.shape[1], 1, 1, 1)
    gy = g.view(1, 1, k, 1).repeat(x.shape[1], 1, 1, 1)
    x = F.conv2d(x, gx, padding=(0, k//2), groups=x.shape[1])
    x = F.conv2d(x, gy, padding=(k//2, 0), groups=x.shape[1])
    return x


def mse_lowfreq_blur_then_downsample(gt, pred, factor=4, sigma=None):
    single = (gt.ndim == 3)
    if single:
        gt = gt.unsqueeze(0)
        pred = pred.unsqueeze(0)

    H, W = gt.shape[-2], gt.shape[-1]
    outH, outW = max(1, H // factor), max(1, W // factor)
    if sigma is None:
        sigma = 1.0 * factor

    gt_lp   = gaussian_blur2d(gt,   sigma=sigma)
    pred_lp = gaussian_blur2d(pred, sigma=sigma)

    gt_ds   = F.interpolate(gt_lp,   size=(outH, outW), mode="area")
    pred_ds = F.interpolate(pred_lp, size=(outH, outW), mode="area")

    mse = (gt_ds - pred_ds).pow(2).mean(dim=(1,2,3))
    return mse[0] if single else mse


MSE_REWARD_THRESHOLD = 0.0099

def reward_from_mse_threshold(mse):
    return torch.where(mse > MSE_REWARD_THRESHOLD, 0.5, -0.0002)


# --------------------------- small helpers ---------------------------

def _to_uint8_img(x, H, W):
    """Accepts None or float [0,1] or uint8; returns uint8 HxWx3."""
    if x is None:
        return np.full((H, W, 3), 255, np.uint8)
    x = np.asarray(x)
    if x.dtype != np.uint8:
        x = np.clip(x, 0.0, 1.0)
        x = (x * 255.0).astype(np.uint8)
    if x.shape[-1] == 4:
        x = x[..., :3]
    if x.shape[:2] != (H, W):
        x = np.array(Image.fromarray(x).resize((W, H), Image.BILINEAR))
    return np.ascontiguousarray(x, dtype=np.uint8)


def _softmax_np(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-8)


def _text_size(draw_obj, text, font_obj):
    if hasattr(draw_obj, "textbbox"):
        l, t, r, b = draw_obj.textbbox((0, 0), text, font=font_obj)
        return r - l, b - t
    if font_obj is not None and hasattr(font_obj, "getbbox"):
        l, t, r, b = font_obj.getbbox(text)
        return r - l, b - t
    return max(1, 6 * len(text)), 12


def _render_status_text_panel(H, W, title, lines):
    img = Image.new("RGB", (W, H), (22, 22, 22))
    draw = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype("DejaVuSans.ttf", 18)
        font_body = ImageFont.truetype("DejaVuSansMono.ttf", 14)
    except Exception:
        font_title = ImageFont.load_default()
        font_body = ImageFont.load_default()

    x = 10
    y = 8
    draw.text((x, y), str(title), fill=(235, 235, 235), font=font_title)
    y += 26
    draw.line([(x, y), (W - 10, y)], fill=(80, 80, 80), width=1)
    y += 8

    for line in lines:
        if y > H - 18:
            draw.text((x, H - 18), "...", fill=(180, 180, 180), font=font_body)
            break
        line_s = str(line)
        col = (215, 215, 215)
        if "collect=T" in line_s:
            col = (120, 255, 120)
        elif "vis=F" in line_s and "collect=" in line_s:
            col = (255, 190, 120)
        elif "collected" in line_s:
            col = (130, 180, 255)
        draw.text((x, y), line_s, fill=col, font=font_body)
        y += 16

    return np.asarray(img, dtype=np.uint8)



# --------------------------- policy panel ---------------------------

def _gray_from_p_global(p_all, idx):
    pmax = float(p_all.max()) + 1e-8
    v = float(p_all[idx]) / pmax
    return int(80 + 175 * v)


def _draw_tri_box(drw, x, y, s, p_all, idxs, labels, font):
    vt = _gray_from_p_global(p_all, idxs[0])
    vl = _gray_from_p_global(p_all, idxs[1])
    vr = _gray_from_p_global(p_all, idxs[2])
    vb = _gray_from_p_global(p_all, idxs[3])

    cx, cy = x + s // 2, y + s // 2

    drw.polygon([(x, y), (x + s, y), (cx, cy)], fill=(vt, vt, vt))
    drw.polygon([(x, y), (x, y + s), (cx, cy)], fill=(vl, vl, vl))
    drw.polygon([(x + s, y), (x + s, y + s), (cx, cy)], fill=(vr, vr, vr))
    drw.polygon([(x, y + s), (x + s, y + s), (cx, cy)], fill=(vb, vb, vb))

    edge = (210, 210, 210)
    drw.rectangle([x, y, x + s, y + s], outline=edge, width=2)
    drw.line([x, y, x + s, y + s], fill=edge, width=2)
    drw.line([x + s, y, x, y + s], fill=edge, width=2)

    def txt_col(v):
        return (0, 0, 0) if v > 140 else (255, 255, 255)

    def _draw_centered(tri_points, label, fill):
        tx, ty = np.mean(np.asarray(tri_points, dtype=np.float32), axis=0)
        tw, th = _text_size(drw, str(label), font)
        x0 = int(round(float(tx) - 0.5 * float(tw)))
        y0 = int(round(float(ty) - 0.5 * float(th)))
        drw.text((x0, y0), str(label), fill=fill, font=font)

    top_tri = [(x, y), (x + s, y), (cx, cy)]
    left_tri = [(x, y), (x, y + s), (cx, cy)]
    right_tri = [(x + s, y), (x + s, y + s), (cx, cy)]
    bottom_tri = [(x, y + s), (x + s, y + s), (cx, cy)]

    _draw_centered(top_tri, labels[0], txt_col(vt))
    _draw_centered(left_tri, labels[1], txt_col(vl))
    _draw_centered(right_tri, labels[2], txt_col(vr))
    _draw_centered(bottom_tri, labels[3], txt_col(vb))


def _short_action_label(label):
    label_norm = str(label).strip().lower()
    if label_norm in {"f", "l", "r", "s"}:
        return label_norm.upper()
    if label_norm.startswith("forward"):
        return "F"
    if label_norm.startswith("left"):
        return "L"
    if label_norm.startswith("right"):
        return "R"
    if label_norm.startswith("stop") or label_norm.startswith("stay") or label_norm.startswith("noop"):
        return "S"
    return label_norm[:1].upper() if label_norm else "?"


def render_policy_tripanel(env, Hs):
    Wp = 260
    pad = 8
    info_h = 22
    s = max(8, min(Wp - 2 * pad, Hs - (2 * pad + info_h)))
    x0 = (Wp - s) // 2
    y0 = pad

    im = Image.new("RGB", (Wp, Hs), (255, 255, 255))
    d  = ImageDraw.Draw(im)
    try:
        font_s = ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:
        font_s = ImageFont.load_default()

    lg = getattr(env, "_last_logits", None)
    if lg is None:
        d.text((8, 8), "Policy: n/a", fill=(20, 20, 20), font=font_s)
        return np.asarray(im, np.uint8)

    p = _softmax_np(np.asarray(lg, np.float32).reshape(-1))
    # Allow env to override which 4 indices to show and their labels.
    # GibsonExplorationEnv stores reordered 4-element logits so these defaults work for both envs.
    idxs   = getattr(env, "_policy_viz_idxs",   [0, 1, 2, 3])
    labels_raw = getattr(env, "_policy_viz_labels", ["F", "L", "R", "S"])
    labels = [_short_action_label(label) for label in labels_raw][:4]
    if len(labels) < 4:
        labels = labels + ["?"] * (4 - len(labels))
    _draw_tri_box(d, x0, y0, s, p, idxs=idxs, labels=labels, font=font_s)
    ent = -float(np.sum(p * np.log(p + 1e-8)))
    entropy_text = f"Entropy: {ent:.2f}"
    tw, th = _text_size(d, entropy_text, font_s)
    d.text(((Wp - tw) // 2, Hs - th - pad), entropy_text, fill=(20, 20, 20), font=font_s)
    return np.asarray(im, np.uint8)


def render_eval_side_panel(env, Hs):
    custom_panel = getattr(env, "_policy_panel_image", None)
    if custom_panel is None:
        return render_policy_tripanel(env, Hs)

    panel = np.asarray(custom_panel)
    if panel.ndim == 2:
        panel = np.repeat(panel[..., None], 3, axis=2)
    if panel.shape[-1] == 4:
        panel = panel[..., :3]
    if panel.dtype != np.uint8:
        if np.issubdtype(panel.dtype, np.floating):
            finite_panel = panel[np.isfinite(panel)]
            max_val = float(np.max(finite_panel)) if finite_panel.size > 0 else 0.0
            scale = 255.0 if max_val <= 1.0 else 1.0
            panel = np.clip(panel * scale, 0.0, 255.0).astype(np.uint8)
        else:
            panel = np.clip(panel, 0, 255).astype(np.uint8)
    if panel.shape[0] != Hs:
        resize_mode = _resize_filter(env)
        target_w = max(1, int(round(float(panel.shape[1]) * float(Hs) / float(panel.shape[0]))))
        panel = np.array(Image.fromarray(panel).resize((target_w, Hs), resize_mode))
    return panel


# --------------------------- topdown map rendering ---------------------------

def world_to_map_pixel(env, world_pos, map_info):
    x, _, z = world_pos
    min_x = float(map_info["min_x"])
    min_z = float(map_info["min_z"])
    max_x = float(map_info.get("max_x", min_x + 1.0))
    max_z = float(map_info.get("max_z", min_z + 1.0))
    default_hw = int(getattr(env, "_topdown_height", 512))
    W = int(map_info.get("map_width", default_hw))
    H = int(map_info.get("map_height", default_hw))
    W = max(1, W)
    H = max(1, H)
    ppm_default_x = float(W) / max(max_x - min_x, 1e-6)
    ppm_default_z = float(H) / max(max_z - min_z, 1e-6)
    ppm_x = float(
        map_info.get(
            "pixels_per_meter_x",
            map_info.get("pixels_per_meter", ppm_default_x),
        )
    )
    ppm_z = float(
        map_info.get(
            "pixels_per_meter_z",
            map_info.get("pixels_per_meter", ppm_default_z),
        )
    )
    # Older metadata may carry stale ppm after map resize. Prefer geometry-consistent ppm in that case.
    if "pixels_per_meter_x" not in map_info and abs(ppm_x - ppm_default_x) > (0.25 * ppm_default_x):
        ppm_x = ppm_default_x
    if "pixels_per_meter_z" not in map_info and abs(ppm_z - ppm_default_z) > (0.25 * ppm_default_z):
        ppm_z = ppm_default_z

    px = int((float(x) - min_x) * ppm_x)
    py = int((float(z) - min_z) * ppm_z)
    px = max(0, min(W - 1, px))
    py = max(0, min(H - 1, py))
    return px, py


def get_current_floor_bounds(env, y_ref: float, floor_thickness: float = 1.2):
    pf = env.sim.pathfinder
    lower, upper = pf.get_bounds()
    y_min, y_max = y_ref - floor_thickness, y_ref + floor_thickness

    step = 0.25
    xs = np.arange(lower[0], upper[0], step)
    zs = np.arange(lower[2], upper[2], step)

    pts = []
    for x in xs:
        for z in zs:
            p = np.array([x, y_ref, z], np.float32)
            if pf.is_navigable(p):
                sp = pf.snap_point(p)
                if y_min <= sp[1] <= y_max:
                    pts.append(sp)

    if len(pts) == 0:
        return float(lower[0]), float(upper[0]), float(lower[2]), float(upper[2]), float(y_ref)

    pts = np.asarray(pts)
    min_x, max_x = float(pts[:, 0].min()), float(pts[:, 0].max())
    min_z, max_z = float(pts[:, 2].min()), float(pts[:, 2].max())
    floor_y = float(np.median(pts[:, 1]))

    pad = 4.0
    return min_x - pad, max_x + pad, min_z - pad, max_z + pad, floor_y



def _estimate_floor_y(env, min_x, max_x, min_z, max_z, cast_y, max_dist,
                      n_samples=256, y_clip=None):
    ys = []
    rng = np.random.default_rng(0)

    for _ in range(n_samples):
        x = float(rng.uniform(min_x, max_x))
        z = float(rng.uniform(min_z, max_z))
        origin = env._mn_vec3(x, float(cast_y), z)
        ray = env._hab_ray(origin, env._mn_vec3(0.0, -1.0, 0.0))
        hit = env.sim.cast_ray(ray, max_distance=max_dist)
        if not hit.has_hits():
            continue

        # take the *lowest* hit among returned hits as a floor candidate
        hy_all = [float(h.point.y) for h in hit.hits]
        hy = min(hy_all)

        if y_clip is not None:
            lo, hi = y_clip
            if not (lo <= hy <= hi):
                continue
        ys.append(hy)

    if len(ys) < max(16, n_samples // 8):
        return None
    return float(np.median(np.asarray(ys, np.float32)))


def render_topdown_map(env, y_ref: float):
    min_x, max_x, min_z, max_z, floor_y = get_current_floor_bounds(env, y_ref)

    if len(env._trajectory_positions) > 0:
        traj = np.asarray(env._trajectory_positions, np.float32)
        margin = 2.0
        min_x = min(min_x, float(traj[:, 0].min()) - margin)
        max_x = max(max_x, float(traj[:, 0].max()) + margin)
        min_z = min(min_z, float(traj[:, 2].min()) - margin)
        max_z = max(max_z, float(traj[:, 2].max()) + margin)

    # square crop
    cx, cz = 0.5 * (min_x + max_x), 0.5 * (min_z + max_z)
    rng = max(max_x - min_x, max_z - min_z)
    min_x, max_x = cx - rng / 2, cx + rng / 2
    min_z, max_z = cz - rng / 2, cz + rng / 2

    H = W = env._topdown_height
    ppm = W / max(rng, 1e-6)

    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (20, 20, 20)

    max_dist = 25.0
    cast_y = float(floor_y + 6.0)   # start higher, reduces “miss floor” cases

    stride = max(1, int(getattr(env, "_topdown_ray_stride", 2)))

    # Robust floor estimate (fallback to provided floor_y)
    floor_y_est = _estimate_floor_y(
        env, min_x, max_x, min_z, max_z,
        cast_y=cast_y, max_dist=max_dist,
        n_samples=256,
        y_clip=(floor_y - 6.0, floor_y + 6.0)  # loose sanity window
    )
    if floor_y_est is None:
        floor_y_est = float(floor_y)

    # slightly wider acceptance band
    y_min = floor_y_est - 2.5
    y_max = floor_y_est + 2.5

    down = env._mn_vec3(0.0, -1.0, 0.0)

    for py in range(0, H, stride):
        z = min_z + (py + 0.5) / ppm
        y2 = min(py + stride, H)
        for px in range(0, W, stride):
            x = min_x + (px + 0.5) / ppm
            x2 = min(px + stride, W)

            origin = env._mn_vec3(float(x), float(cast_y), float(z))
            ray = env._hab_ray(origin, down)
            hit = env.sim.cast_ray(ray, max_distance=max_dist)

            if not hit.has_hits():
                continue

            # pick the hit closest to floor_y_est
            best_hy = None
            best_abs = 1e9
            for h in hit.hits:
                hy = float(h.point.y)
                a = abs(hy - floor_y_est)
                if a < best_abs:
                    best_abs = a
                    best_hy = hy

            if best_hy is not None and (y_min <= best_hy <= y_max):
                img[py:y2, px:x2] = (200, 200, 200)

    info = dict(
        min_x=float(min_x), max_x=float(max_x),
        min_z=float(min_z), max_z=float(max_z),
        floor_y=float(floor_y_est),
        range=float(rng),
        pixels_per_meter=float(ppm),
        pixels_per_meter_x=float(ppm),
        pixels_per_meter_z=float(ppm),
        map_width=int(W),
        map_height=int(H),
    )
    return img, info


def _get_scene_name_for_topdown(env):
    for name in ("scene_name", "_current_scene_name"):
        value = getattr(env, name, None)
        if isinstance(value, str) and value:
            return value
    scene_dict = getattr(env, "_scene_dict", None)
    if isinstance(scene_dict, dict):
        for key in ("scene_name", "scene", "glb_path"):
            value = scene_dict.get(key, None)
            if isinstance(value, str) and value:
                return value
    return "unknown_scene"


def _point_to_xyz(point):
    try:
        return float(point[0]), float(point[1]), float(point[2])
    except Exception:
        return float(point.x), float(point.y), float(point.z)


def _binary_close_mask(mask: np.ndarray, radius_cells: int, iterations: int = 1) -> np.ndarray:
    radius = max(0, int(radius_cells))
    iters = max(1, int(iterations))
    if radius <= 0:
        return mask.astype(bool, copy=True)
    kernel = int(2 * radius + 1)
    x = torch.from_numpy(mask.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
    for _ in range(iters):
        x = F.max_pool2d(x, kernel_size=kernel, stride=1, padding=radius)  # dilation
        x = 1.0 - F.max_pool2d(1.0 - x, kernel_size=kernel, stride=1, padding=radius)  # erosion
    return (x[0, 0].cpu().numpy() > 0.5)


def _binary_dilate_mask(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    radius = max(0, int(radius_cells))
    if radius <= 0:
        return mask.astype(bool, copy=True)
    kernel = int(2 * radius + 1)
    x = torch.from_numpy(mask.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
    x = F.max_pool2d(x, kernel_size=kernel, stride=1, padding=radius)
    return (x[0, 0].cpu().numpy() > 0.5)


def _convex_hull_xy(points_xy: np.ndarray) -> np.ndarray:
    if points_xy.size == 0:
        return np.zeros((0, 2), dtype=np.int32)

    pts = np.unique(points_xy.astype(np.int32, copy=False), axis=0)
    if pts.shape[0] <= 2:
        return pts

    pts_list = [tuple(p) for p in pts.tolist()]
    pts_list.sort()

    def _cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts_list:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts_list):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return np.asarray(hull, dtype=np.int32)


def _polygon_to_mask(height: int, width: int, poly_xy: np.ndarray) -> np.ndarray:
    if poly_xy.shape[0] < 3:
        return np.zeros((height, width), dtype=bool)
    img = Image.new("L", (int(width), int(height)), 0)
    draw = ImageDraw.Draw(img)
    draw.polygon([tuple(map(int, p)) for p in poly_xy.tolist()], outline=1, fill=1)
    return np.asarray(img, dtype=np.uint8) > 0


def _fill_between_hit_pairs(hit_indices: np.ndarray, length: int) -> np.ndarray:
    line = np.zeros(int(length), dtype=bool)
    if int(length) <= 0:
        return line
    if hit_indices.size < 2:
        return line
    hits = np.unique(np.clip(hit_indices.astype(np.int32, copy=False), 0, int(length) - 1))
    pair_count = int((hits.size // 2) * 2)
    for i in range(0, pair_count, 2):
        a = int(hits[i])
        b = int(hits[i + 1])
        lo = min(a, b)
        hi = max(a, b)
        if hi - lo <= 1:
            continue
        line[lo + 1 : hi] = True
    return line


def _flood_fill_reachable(free_mask: np.ndarray, start_row: int, start_col: int) -> np.ndarray:
    h, w = free_mask.shape
    if h <= 0 or w <= 0:
        return np.zeros((h, w), dtype=bool)
    sr = int(np.clip(start_row, 0, h - 1))
    sc = int(np.clip(start_col, 0, w - 1))
    if not free_mask[sr, sc]:
        free_rows, free_cols = np.nonzero(free_mask)
        if len(free_rows) == 0:
            return np.zeros_like(free_mask, dtype=bool)
        d2 = (free_rows - sr) ** 2 + (free_cols - sc) ** 2
        idx = int(np.argmin(d2))
        sr = int(free_rows[idx])
        sc = int(free_cols[idx])

    reachable = np.zeros_like(free_mask, dtype=bool)
    q = deque([(sr, sc)])
    reachable[sr, sc] = True
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                continue
            if reachable[nr, nc] or (not free_mask[nr, nc]):
                continue
            reachable[nr, nc] = True
            q.append((nr, nc))
    return reachable


def _render_sweep_topdown_map(env, y_ref: float):
    if env.sim is None:
        return None, None
    if not hasattr(env, "_mn_vec3") or not hasattr(env, "_hab_ray"):
        return None, None
    pathfinder = getattr(env.sim, "pathfinder", None)
    if pathfinder is None:
        return None, None

    try:
        lower, upper = pathfinder.get_bounds()
    except Exception:
        return None, None

    pad_m = float(getattr(env, "_topdown_sweep_bounds_pad_m", 1.0))
    min_x = float(lower[0]) - pad_m
    max_x = float(upper[0]) + pad_m
    min_z = float(lower[2]) - pad_m
    max_z = float(upper[2]) + pad_m

    cx = 0.5 * (min_x + max_x)
    cz = 0.5 * (min_z + max_z)
    span = max(max_x - min_x, max_z - min_z, 1e-3)
    min_x, max_x = cx - 0.5 * span, cx + 0.5 * span
    min_z, max_z = cz - 0.5 * span, cz + 0.5 * span

    map_scale_m = float(getattr(env, "_topdown_sweep_scale_m", 0.05))
    map_scale_m = max(0.02, map_scale_m)
    grid_size = int(math.ceil(span / map_scale_m)) + 1
    grid_size = max(16, grid_size)

    sweep_offsets = tuple(
        float(v)
        for v in getattr(env, "_topdown_sweep_height_offsets_m", (-0.10, 0.0, 0.10))
    )
    if len(sweep_offsets) == 0:
        sweep_offsets = (0.0,)
    collision_idx = int(np.argmin(np.abs(np.asarray(sweep_offsets, dtype=np.float32))))
    # Default to convex-hull outside masking; parity/winding-style fill can be re-enabled per-env.
    use_parity_inside_mask = bool(getattr(env, "_topdown_sweep_use_parity_inside_mask", False))
    use_hull_outside_mask = bool(getattr(env, "_topdown_sweep_use_convex_hull_outside_mask", True))
    parity_close_radius = int(getattr(env, "_topdown_sweep_parity_close_radius_cells", 1))
    parity_pad = int(getattr(env, "_topdown_sweep_parity_pad_cells", 1))
    hull_pad = int(getattr(env, "_topdown_sweep_hull_pad_cells", 2))

    floor_quant_m = max(
        float(getattr(env, "_topdown_sweep_floor_quantization_m", 0.5)),
        1e-3,
    )
    floor_bin = int(round(float(y_ref) / floor_quant_m))
    cache_key = (
        _get_scene_name_for_topdown(env),
        floor_bin,
        round(map_scale_m, 4),
        sweep_offsets,
        bool(use_parity_inside_mask),
        int(max(0, parity_close_radius)),
        int(max(0, parity_pad)),
        bool(use_hull_outside_mask),
        int(max(0, hull_pad)),
    )
    cache = _render_cache(env)
    cached = cache.get("sweep_topdown", None)
    if isinstance(cached, dict) and cached.get("key") == cache_key:
        return cached.get("map"), cached.get("info")

    occupied_mask = np.zeros((grid_size, grid_size), dtype=bool)
    collision_mask = np.zeros((grid_size, grid_size), dtype=bool)
    row_inside_mask = np.zeros((grid_size, grid_size), dtype=bool)
    col_inside_mask = np.zeros((grid_size, grid_size), dtype=bool)
    parity_rows_used = 0
    parity_cols_used = 0

    x_centers = min_x + (np.arange(grid_size, dtype=np.float64) + 0.5) * map_scale_m
    z_centers = min_z + (np.arange(grid_size, dtype=np.float64) + 0.5) * map_scale_m
    x_origin = min_x - 1.0
    z_origin = min_z - 1.0
    max_x_dist = (max_x - min_x) + 2.0
    max_z_dist = (max_z - min_z) + 2.0

    def _mark_hits(ray_hits, mark_collision: bool):
        hit_rows: list[int] = []
        hit_cols: list[int] = []
        if not ray_hits.has_hits():
            return hit_rows, hit_cols
        for hit in ray_hits.hits:
            try:
                hx, _, hz = _point_to_xyz(hit.point)
            except Exception:
                continue
            col = int((hx - min_x) / map_scale_m)
            row = int((hz - min_z) / map_scale_m)
            if 0 <= row < grid_size and 0 <= col < grid_size:
                occupied_mask[row, col] = True
                hit_rows.append(row)
                hit_cols.append(col)
                if mark_collision:
                    collision_mask[row, col] = True
        return hit_rows, hit_cols

    for idx, offset in enumerate(sweep_offsets):
        sweep_y = float(y_ref) + float(offset)
        mark_collision = idx == collision_idx

        for row_idx, z in enumerate(z_centers):
            ray = env._hab_ray(
                env._mn_vec3(float(x_origin), float(sweep_y), float(z)),
                env._mn_vec3(1.0, 0.0, 0.0),
            )
            try:
                hits = env.sim.cast_ray(ray, max_distance=float(max_x_dist))
            except Exception:
                continue
            _, hit_cols = _mark_hits(hits, mark_collision=mark_collision)
            if mark_collision and use_parity_inside_mask and len(hit_cols) >= 2:
                line_inside = _fill_between_hit_pairs(
                    np.asarray(hit_cols, dtype=np.int32),
                    length=grid_size,
                )
                if np.any(line_inside):
                    row_inside_mask[row_idx, :] = np.logical_or(row_inside_mask[row_idx, :], line_inside)
                    parity_rows_used += 1

        for col_idx, x in enumerate(x_centers):
            ray = env._hab_ray(
                env._mn_vec3(float(x), float(sweep_y), float(z_origin)),
                env._mn_vec3(0.0, 0.0, 1.0),
            )
            try:
                hits = env.sim.cast_ray(ray, max_distance=float(max_z_dist))
            except Exception:
                continue
            hit_rows, _ = _mark_hits(hits, mark_collision=mark_collision)
            if mark_collision and use_parity_inside_mask and len(hit_rows) >= 2:
                line_inside = _fill_between_hit_pairs(
                    np.asarray(hit_rows, dtype=np.int32),
                    length=grid_size,
                )
                if np.any(line_inside):
                    col_inside_mask[:, col_idx] = np.logical_or(col_inside_mask[:, col_idx], line_inside)
                    parity_cols_used += 1

    wall_mask = np.logical_or(occupied_mask, collision_mask)
    close_radius = int(getattr(env, "_topdown_sweep_close_radius_cells", 2))
    close_iters = int(getattr(env, "_topdown_sweep_close_iterations", 2))
    wall_mask = _binary_close_mask(wall_mask, radius_cells=close_radius, iterations=close_iters)

    inside_mask = None
    if use_parity_inside_mask:
        parity_inside_mask = np.logical_and(row_inside_mask, col_inside_mask)
        if np.any(parity_inside_mask):
            if parity_close_radius > 0:
                parity_inside_mask = _binary_close_mask(
                    parity_inside_mask,
                    radius_cells=parity_close_radius,
                    iterations=1,
                )
            if parity_pad > 0:
                parity_inside_mask = _binary_dilate_mask(
                    parity_inside_mask,
                    radius_cells=parity_pad,
                )
            inside_mask = parity_inside_mask

    hull_mask = None
    if inside_mask is None and use_hull_outside_mask:
        wall_rows, wall_cols = np.nonzero(wall_mask)
        if wall_rows.size >= 3:
            hull_xy = _convex_hull_xy(np.stack([wall_cols, wall_rows], axis=1))
            hull_mask = _polygon_to_mask(grid_size, grid_size, hull_xy)
            if hull_pad > 0:
                hull_mask = _binary_dilate_mask(hull_mask, radius_cells=hull_pad)

    _, _, _, cam_pos = env._get_cam_basis_and_pos()
    agent_x = float(cam_pos[0])
    agent_z = float(cam_pos[2])
    if len(getattr(env, "_trajectory_positions", [])) > 0:
        try:
            last = np.asarray(env._trajectory_positions[-1], dtype=np.float32).reshape(3)
            agent_x = float(last[0])
            agent_z = float(last[2])
        except Exception:
            pass

    start_col = int((agent_x - min_x) / map_scale_m)
    start_row = int((agent_z - min_z) / map_scale_m)
    free_mask = ~wall_mask
    if inside_mask is not None:
        free_mask = np.logical_and(free_mask, inside_mask)
        if not np.any(free_mask):
            free_mask = ~wall_mask
    elif hull_mask is not None:
        free_mask = np.logical_and(free_mask, hull_mask)
        if not np.any(free_mask):
            free_mask = ~wall_mask
    navigable_mask = _flood_fill_reachable(free_mask, start_row=start_row, start_col=start_col)

    map_rgb = np.full((grid_size, grid_size, 3), 255, dtype=np.uint8)
    map_rgb[navigable_mask] = np.asarray([205, 205, 205], dtype=np.uint8)
    map_rgb[wall_mask] = np.asarray([0, 0, 0], dtype=np.uint8)

    out_hw = _safe_int_attr(env, "_topdown_height", 512, min_value=64)
    if map_rgb.shape[0] != out_hw or map_rgb.shape[1] != out_hw:
        row_idx = np.linspace(0, map_rgb.shape[0] - 1, num=out_hw, dtype=np.int64)
        col_idx = np.linspace(0, map_rgb.shape[1] - 1, num=out_hw, dtype=np.int64)
        map_rgb = map_rgb[row_idx[:, None], col_idx[None, :]]

    ppm = float(out_hw) / max(float(span), 1e-6)
    map_info = dict(
        min_x=float(min_x),
        max_x=float(max_x),
        min_z=float(min_z),
        max_z=float(max_z),
        floor_y=float(y_ref),
        range=float(span),
        pixels_per_meter=float(ppm),
        pixels_per_meter_x=float(ppm),
        pixels_per_meter_z=float(ppm),
        map_width=int(out_hw),
        map_height=int(out_hw),
        topdown_mode="sweep",
        collision_slice_y=float(y_ref + sweep_offsets[collision_idx]),
        sweep_wall_cell_count=int(np.count_nonzero(wall_mask)),
        sweep_navigable_cell_count=int(np.count_nonzero(navigable_mask)),
        sweep_inside_constraint_mode=(
            "parity" if inside_mask is not None else ("convex_hull" if hull_mask is not None else "none")
        ),
        sweep_parity_row_scans_used=int(parity_rows_used),
        sweep_parity_col_scans_used=int(parity_cols_used),
        sweep_parity_inside_cell_count=int(np.count_nonzero(inside_mask)) if inside_mask is not None else 0,
    )
    cache["sweep_topdown"] = {
        "key": cache_key,
        "map": map_rgb,
        "info": map_info,
    }
    return map_rgb, map_info


def draw_trajectory_and_agent(env, map_rgb, map_info):
    if len(env._trajectory_positions) == 0:
        return map_rgb

    img = Image.fromarray(map_rgb)
    draw = ImageDraw.Draw(img)

    pixel_positions = []
    for pos in env._trajectory_positions:
        pixel_positions.append(world_to_map_pixel(env, pos, map_info))

    # trajectory colored by time (plasma)
    if len(pixel_positions) > 1:
        n = len(pixel_positions) - 1
        denom = max(1, n - 1)
        cmap = cm.get_cmap("plasma")
        for i in range(n):
            t01 = float(i) / float(denom)          # 0..1 over time
            r, g, b, _ = cmap(t01)
            col = (int(255*r), int(255*g), int(255*b))
            draw.line([pixel_positions[i], pixel_positions[i + 1]], fill=col, width=4)

    curr_px, curr_py = pixel_positions[-1]
    hide_heading_beam = bool(getattr(env, "_hide_heading_beam", False))
    if not hide_heading_beam:
        # heading arrow (uses env._get_cam_basis_and_pos())
        _, fwd, _, _ = env._get_cam_basis_and_pos()
        fx, fz = float(fwd[0]), float(fwd[2])
        mag = math.hypot(fx, fz)
        if mag > 1e-6:
            fx, fz = fx / mag, fz / mag
        else:
            fx, fz = 0.0, 1.0
        end_x = curr_px + fx * 20
        end_y = curr_py + fz * 20
        draw.line([(curr_px, curr_py), (end_x, end_y)], fill=(255, 0, 0), width=4)

    # camera position dot
    r = 5
    draw.ellipse([curr_px - r, curr_py - r, curr_px + r, curr_py + r],
                 fill=(255, 255, 0), outline=(0, 0, 0), width=2)

    # Optional overlay points (e.g., apples) exposed by env.
    overlay_points = None
    getter = getattr(env, "get_topdown_overlay_world_points", None)
    if callable(getter):
        try:
            overlay_points = getter()
        except Exception:
            overlay_points = None
    if overlay_points is None:
        overlay_points = getattr(env, "_topdown_overlay_world_points", None)

    if overlay_points is not None:
        overlay_radius_px = max(1, int(getattr(env, "_topdown_overlay_point_radius_px", 4)))
        for p in overlay_points:
            try:
                px, py = world_to_map_pixel(env, p, map_info)
            except Exception:
                continue
            rr = overlay_radius_px
            draw.ellipse(
                [px - rr, py - rr, px + rr, py + rr],
                fill=(220, 25, 25),
                outline=(0, 0, 0),
                width=1,
            )

    return np.array(img, dtype=np.uint8)


def render_topdown_trajectory_layer(env, *, include_camera_marker: bool = False) -> np.ndarray:
    """Render only the topdown trajectory as a transparent RGBA layer."""
    default_hw = max(1, int(getattr(env, "_topdown_height", 512)))
    if env.sim is None:
        return np.zeros((default_hw, default_hw, 4), dtype=np.uint8)

    if env._topdown_map is None or env._topdown_map_info is None:
        _ = render_topdown(env)

    map_info = env._topdown_map_info
    if not isinstance(map_info, dict):
        return np.zeros((default_hw, default_hw, 4), dtype=np.uint8)

    W = max(1, int(map_info.get("map_width", default_hw)))
    H = max(1, int(map_info.get("map_height", default_hw)))
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    pixel_positions = []
    for pos in getattr(env, "_trajectory_positions", []):
        try:
            pixel_positions.append(world_to_map_pixel(env, pos, map_info))
        except Exception:
            continue

    if len(pixel_positions) > 1:
        n = len(pixel_positions) - 1
        denom = max(1, n - 1)
        cmap = cm.get_cmap("plasma")
        traj_width = max(4, int(getattr(env, "_topdown_traj_line_width_px", 4)))
        for i in range(n):
            t01 = float(i) / float(denom)
            r, g, b, _ = cmap(t01)
            col = (int(255 * r), int(255 * g), int(255 * b), 255)
            draw.line([pixel_positions[i], pixel_positions[i + 1]], fill=col, width=traj_width)

    if include_camera_marker and len(pixel_positions) > 0:
        curr_px, curr_py = pixel_positions[-1]
        r = 5
        draw.ellipse(
            [curr_px - r, curr_py - r, curr_px + r, curr_py + r],
            fill=(255, 255, 0, 255),
            outline=(0, 0, 0, 255),
            width=2,
        )

    return np.asarray(layer, dtype=np.uint8)


def _render_point_cloud_topdown_progress_with_trajectory(env):
    progress = getattr(env, "_point_cloud_topdown_progress", None)
    if not isinstance(progress, dict):
        return None

    points = np.asarray(progress.get("points", []), dtype=np.float32)
    min_distances = np.asarray(progress.get("min_distances_m", []), dtype=np.float32).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        return None
    if len(min_distances) != len(points):
        return None

    threshold_m = float(progress.get("threshold_m", 0.05))
    step_value = int(progress.get("step", 0))
    max_step = max(int(progress.get("max_step", step_value)), 1)

    observed_points = progress.get("observed_points", None)
    if observed_points is not None:
        observed_points = np.asarray(observed_points, dtype=np.float32)
        if observed_points.ndim != 2 or observed_points.shape[1] != 3:
            observed_points = None
        elif len(observed_points) > 20000:
            sample_idx = np.linspace(0, len(observed_points) - 1, num=20000, dtype=np.int64)
            observed_points = observed_points[sample_idx]

    # Render this panel at higher source resolution to preserve point detail
    # after the final video row is resized down to eval frame height.
    image_size = max(
        1024,
        _safe_int_attr(env, "_topdown_height", 512, min_value=64),
    )
    pad = 24

    # Keep a stable camera/bounds within an episode. Re-fitting every frame makes the map "swim".
    view_state = getattr(env, "_point_cloud_topdown_view_state", None)
    recompute_view = view_state is None or int(step_value) == 0
    if not recompute_view:
        try:
            cached_image_size = int(view_state.get("image_size", image_size))
            cached_center = np.asarray(view_state.get("center", []), dtype=np.float32).reshape(2)
            cached_scale = float(view_state.get("scale", 0.0))
            if (
                cached_image_size != int(image_size)
                or not np.isfinite(cached_scale)
                or cached_scale <= 0.0
            ):
                recompute_view = True
        except Exception:
            recompute_view = True

    if recompute_view:
        bounds_sources = [points[:, [0, 2]]]
        if observed_points is not None and len(observed_points) > 0:
            bounds_sources.append(observed_points[:, [0, 2]])
        traj_positions = np.asarray(getattr(env, "_trajectory_positions", []), dtype=np.float32)
        if traj_positions.ndim == 2 and traj_positions.shape[1] == 3 and len(traj_positions) > 0:
            bounds_sources.append(traj_positions[:, [0, 2]])

        all_xz = np.concatenate(bounds_sources, axis=0)
        mins = all_xz.min(axis=0)
        maxs = all_xz.max(axis=0)
        center = 0.5 * (mins + maxs)
        # Small pad keeps future motion inside frame without per-frame rescaling.
        span = float(max(maxs[0] - mins[0], maxs[1] - mins[1], 1e-6)) * 1.10
        scale = float((image_size - 2 * pad - 1) / span)
        if not np.isfinite(scale) or scale <= 0.0:
            return None
        env._point_cloud_topdown_view_state = {
            "center": center.astype(np.float32, copy=False),
            "scale": float(scale),
            "image_size": int(image_size),
        }
    else:
        center = np.asarray(view_state["center"], dtype=np.float32).reshape(2)
        scale = float(view_state["scale"])

    point_radius_px = 2
    observed_point_radius_px = 2
    camera_radius_px = 4
    camera_color = np.asarray([66, 156, 255], dtype=np.uint8)

    def _project_points(points_xyz):
        pts_xz = np.asarray(points_xyz, dtype=np.float32)[:, [0, 2]]
        coords = (pts_xz - center[None, :]) * scale
        px = np.clip(np.rint(coords[:, 0] + image_size / 2.0), 0, image_size - 1).astype(np.int32)
        py = np.clip(np.rint(image_size / 2.0 + coords[:, 1]), 0, image_size - 1).astype(np.int32)
        return px, py

    def _blend_points(frame, xs, ys, color, *, alpha=0.55, radius_px=1):
        if len(xs) == 0:
            return
        color_arr = np.asarray(color, dtype=np.float32).reshape(1, 3)
        alpha_f = float(np.clip(alpha, 0.0, 1.0))
        if alpha_f <= 0.0:
            return
        for dy in range(-int(radius_px), int(radius_px) + 1):
            for dx in range(-int(radius_px), int(radius_px) + 1):
                yy = np.clip(ys + dy, 0, image_size - 1)
                xx = np.clip(xs + dx, 0, image_size - 1)
                base = frame[yy, xx].astype(np.float32, copy=False)
                blended = np.rint((1.0 - alpha_f) * base + alpha_f * color_arr)
                frame[yy, xx] = np.clip(blended, 0.0, 255.0).astype(np.uint8, copy=False)

    gt_px, gt_py = _project_points(points)
    completed_mask = min_distances < threshold_m

    frame = np.empty((image_size, image_size, 3), dtype=np.uint8)
    frame[...] = np.asarray([255, 255, 255], dtype=np.uint8)
    incompletion_color = np.asarray([140, 140, 140], dtype=np.uint8)
    completion_color = np.asarray([46, 139, 87], dtype=np.uint8)
    observed_color = np.asarray([232, 73, 64], dtype=np.uint8)
    _blend_points(
        frame,
        gt_px[~completed_mask],
        gt_py[~completed_mask],
        incompletion_color,
        alpha=0.70,
        radius_px=point_radius_px,
    )
    _blend_points(
        frame,
        gt_px[completed_mask],
        gt_py[completed_mask],
        completion_color,
        alpha=0.70,
        radius_px=point_radius_px,
    )

    hide_progress_observed_points = bool(getattr(env, "_hide_progress_observed_points", False))
    if (not hide_progress_observed_points) and observed_points is not None and len(observed_points) > 0:
        obs_px, obs_py = _project_points(observed_points)
        _blend_points(
            frame,
            obs_px,
            obs_py,
            observed_color,
            alpha=0.70,
            radius_px=observed_point_radius_px,
        )

    progress = float(step_value) / float(max_step)
    bar_x0 = pad
    bar_x1 = image_size - pad
    bar_y0 = image_size - 14
    bar_y1 = image_size - 8
    frame[bar_y0:bar_y1, bar_x0:bar_x1] = np.asarray([230, 230, 230], dtype=np.uint8)
    fill_x1 = bar_x0 + int(round((bar_x1 - bar_x0) * np.clip(progress, 0.0, 1.0)))
    frame[bar_y0:bar_y1, bar_x0:fill_x1] = completion_color

    traj_positions = np.asarray(getattr(env, "_trajectory_positions", []), dtype=np.float32)
    if traj_positions.ndim == 2 and traj_positions.shape[1] == 3 and len(traj_positions) > 0:
        # Match floorplan trajectory coloring (time-ordered plasma gradient).
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        traj_px, traj_py = _project_points(traj_positions)
        if len(traj_px) > 1:
            n = len(traj_px) - 1
            denom = max(1, n - 1)
            cmap = cm.get_cmap("plasma")
            for i in range(n):
                t01 = float(i) / float(denom)
                r, g, b, _ = cmap(t01)
                col = (int(255 * r), int(255 * g), int(255 * b))
                draw.line(
                    [(int(traj_px[i]), int(traj_py[i])), (int(traj_px[i + 1]), int(traj_py[i + 1]))],
                    fill=col,
                    width=max(8, image_size // 128),
                )
        hide_progress_camera_marker = bool(getattr(env, "_hide_progress_camera_marker", False))
        if not hide_progress_camera_marker:
            cam_px, cam_py = _project_points(traj_positions[-1].reshape(1, 3))
            cam_x, cam_y = int(cam_px[0]), int(cam_py[0])
            draw.ellipse(
                [
                    cam_x - camera_radius_px,
                    cam_y - camera_radius_px,
                    cam_x + camera_radius_px,
                    cam_y + camera_radius_px,
                ],
                fill=tuple(int(value) for value in camera_color.tolist()),
            )

        overlay_points = None
        getter = getattr(env, "get_topdown_overlay_world_points", None)
        if callable(getter):
            try:
                overlay_points = getter()
            except Exception:
                overlay_points = None
        if overlay_points is None:
            overlay_points = getattr(env, "_topdown_overlay_world_points", None)
        if overlay_points is not None:
            overlay_radius_px = max(
                1,
                int(
                    getattr(
                        env,
                        "_progress_overlay_point_radius_px",
                        getattr(env, "_topdown_overlay_point_radius_px", 4),
                    )
                ),
            )
            for p in overlay_points:
                try:
                    p_arr = np.asarray(p, dtype=np.float32).reshape(1, 3)
                    pxx, pyy = _project_points(p_arr)
                    px, py = int(pxx[0]), int(pyy[0])
                except Exception:
                    continue
                rr = overlay_radius_px
                draw.ellipse(
                    [px - rr, py - rr, px + rr, py + rr],
                    fill=(220, 25, 25),
                    outline=(0, 0, 0),
                    width=1,
                )
        frame = np.array(img, dtype=np.uint8, copy=True)

    return frame


def render_topdown(env):
    if env.sim is None:
        H = env._topdown_height
        return np.zeros((H, H, 3), dtype=np.uint8)

    _, _, _, cam_pos = env._get_cam_basis_and_pos()
    y_ref = float(cam_pos[1])

    use_sweep_map = bool(getattr(env, "_prefer_sweep_topdown_map", True))
    if use_sweep_map:
        sweep_map, sweep_info = _render_sweep_topdown_map(env, y_ref)
        if sweep_map is not None and sweep_info is not None:
            env._topdown_map = sweep_map
            env._topdown_map_info = sweep_info
            return draw_trajectory_and_agent(env, sweep_map.copy(), sweep_info)

    need_regen = False
    force_regen = False
    if env._topdown_map is None or env._topdown_map_info is None:
        need_regen = True
        force_regen = True
    else:
        if abs(y_ref - env._topdown_map_info["floor_y"]) > 2.0:
            need_regen = True
        elif len(env._trajectory_positions) > 0:
            info = env._topdown_map_info
            curr = env._trajectory_positions[-1]
            margin = 1.0
            if (curr[0] < info["min_x"] + margin or curr[0] > info["max_x"] - margin or
                curr[2] < info["min_z"] + margin or curr[2] > info["max_z"] - margin):
                need_regen = True

    if need_regen:
        call_idx = int(getattr(env, "_render_call_idx", 0))
        last_regen = int(getattr(env, "_topdown_last_regen_call", -10**9))
        regen_interval = _safe_int_attr(env, "_topdown_regen_interval", 8, min_value=1)
        if force_regen or (call_idx - last_regen >= regen_interval):
            env._topdown_map, env._topdown_map_info = render_topdown_map(env, y_ref)
            env._topdown_last_regen_call = call_idx

    return draw_trajectory_and_agent(env, env._topdown_map.copy(), env._topdown_map_info)


# --------------------------- metric strip rendering ---------------------------

def metric_to_rgb(v, vmin, vmax, cmap_name="Reds"):
    cmap = cm.get_cmap(cmap_name)
    if vmax <= vmin + 1e-12:
        t = 0.0
    else:
        t = (float(v) - float(vmin)) / (float(vmax) - float(vmin))
    t = float(np.clip(t, 0.0, 1.0))
    r, g, b, _ = cmap(t)
    return (int(255*r), int(255*g), int(255*b))


def draw_arrow(draw, p0, p1, color, width=3, head_len=8, head_w=6):
    x0, y0 = p0
    x1, y1 = p1
    draw.line([(x0, y0), (x1, y1)], fill=color, width=width)

    dx, dy = x1 - x0, y1 - y0
    n = (dx*dx + dy*dy) ** 0.5
    if n < 1e-6:
        return
    ux, uy = dx / n, dy / n
    bx, by = x1 - ux * head_len, y1 - uy * head_len
    px, py = -uy, ux
    p2 = (bx + px * head_w * 0.5, by + py * head_w * 0.5)
    p3 = (bx - px * head_w * 0.5, by - py * head_w * 0.5)
    draw.polygon([(x1, y1), p2, p3], fill=color)


def make_colorbar(vmin, vmax, H, bar_w=24, text_w=70, cmap_name="Reds"):
    cmap = cm.get_cmap(cmap_name)
    grad = np.linspace(1.0, 0.0, H).reshape(H, 1)
    rgb = (cmap(grad)[:, :, :3] * 255).astype(np.uint8)
    bar = np.repeat(rgb, bar_w, axis=1)

    cbar = Image.fromarray(bar)
    out = Image.new("RGB", (bar_w + text_w, H), (255, 255, 255))
    out.paste(cbar, (0, 0))
    d = ImageDraw.Draw(out)
    d.text((bar_w + 6, 2), f"{float(vmax):.2f}", fill=(0, 0, 0))
    d.text((bar_w + 6, H - 14), f"{float(vmin):.2f}", fill=(0, 0, 0))
    return out


def render_metric_panel(env, base_map_rgb, map_info, metric, vmin, vmax, title, cmap_name="Reds", alpha=120):
    H, W = base_map_rgb.shape[:2]
    base_img = Image.fromarray(base_map_rgb.copy()).convert("RGBA")
    overlay  = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    draw     = ImageDraw.Draw(overlay)

    draw.rectangle([0, 0, W, 26], fill=(0, 0, 0, 255))
    draw.text((8, 5), title, fill=(255, 255, 255, 255))

    turn_eps_px  = 2.0
    arrow_len_px = 14

    T = len(metric)
    for t in range(T):
        p0 = world_to_map_pixel(env, env._trajectory_positions[t], map_info)
        p1 = world_to_map_pixel(env, env._trajectory_positions[t + 1], map_info)

        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        dist = (dx * dx + dy * dy) ** 0.5

        col_rgb = metric_to_rgb(float(metric[t]), float(vmin), float(vmax), cmap_name=cmap_name)
        col = (col_rgb[0], col_rgb[1], col_rgb[2], int(alpha))

        if dist >= turn_eps_px:
            ux, uy = dx / dist, dy / dist
            p1s = (p1[0] - ux * 2.0, p1[1] - uy * 2.0)
            draw_arrow(draw, p0, p1s, col, width=3)
        else:
            yaw = float(env._traj_yaws[t + 1]) if len(env._traj_yaws) > t + 1 else 0.0
            ox, oy = math.cos(yaw), math.sin(yaw)
            base_pt = (p1[0] + ox * 3.0, p1[1] + oy * 3.0)
            tip = (base_pt[0] + ox * arrow_len_px, base_pt[1] + oy * arrow_len_px)
            draw_arrow(draw, base_pt, tip, col, width=2)

    cbar = make_colorbar(vmin, vmax, H, cmap_name=cmap_name)

    base_img = Image.alpha_composite(base_img, overlay).convert("RGB")
    out = Image.new("RGB", (W + 8 + cbar.size[0], H), (255, 255, 255))
    out.paste(base_img, (0, 0))
    out.paste(cbar, (W + 8, 0))
    return np.array(out, dtype=np.uint8)


def render_blank_metric_panel(base_map_rgb, title):
    H, W = base_map_rgb.shape[:2]
    img = Image.fromarray(base_map_rgb.copy())
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, 26], fill=(0, 0, 0))
    draw.text((8, 5), title, fill=(255, 255, 255))
    # light overlay so it’s obvious it’s “disabled”
    draw.rectangle([0, 26, W, H], fill=(245, 245, 245))
    return np.array(img, dtype=np.uint8)


def render_topdown_metrics_strip(env):
    cache = _render_cache(env)
    call_idx = int(getattr(env, "_render_call_idx", 0))
    strip_every = _safe_int_attr(env, "_metric_strip_every", 4, min_value=1)

    if cache.get("metric_strip") is not None:
        last = int(cache.get("metric_strip_call", -10**9))
        if call_idx - last < strip_every:
            return cache["metric_strip"]

    if env._topdown_map is None or env._topdown_map_info is None:
        _ = render_topdown(env)

    base_map = env._topdown_map.copy()
    info = env._topdown_map_info

    # how many steps we can draw online (trajectory is growing)
    T = min(len(env._step_rewards), max(0, len(env._trajectory_positions) - 1))
    if T <= 0:
        H = base_map.shape[0]
        blank = np.full((H, 3 * H, 3), 255, np.uint8)
        cache["metric_strip"] = blank
        cache["metric_strip_call"] = call_idx
        return blank

    rewards = env._step_rewards[:T]
    rmin, rmax = env._reward_vmin, env._reward_vmax

    # --- Entropy (computed online from logits in update_meta) ---
    step_entropies = getattr(env, "_step_entropies", [])

    # --- POST metrics (only exist after GAE) ---
    post_ready = bool(getattr(env, "_post_metrics_ready", False))
    post_returns = getattr(env, "_post_returns", None)
    post_advs    = getattr(env, "_post_advs", None)

    # Temporarily slice positions/yaws to T+1 for consistent drawing
    pos_save = env._trajectory_positions
    yaw_save = env._traj_yaws
    env._trajectory_positions = pos_save[:T + 1]
    env._traj_yaws = yaw_save[:T + 1]

    try:
        p_r = render_metric_panel(env, base_map, info, rewards, rmin, rmax, "Reward", cmap_name="Reds", alpha=110)

        # Entropy panel: per-episode normalization for maximum color contrast
        T_ent = min(len(step_entropies), T)
        if T_ent > 0:
            entropies = step_entropies[:T_ent]
            ent_min_obs = float(np.min(entropies))
            ent_max_obs = float(np.max(entropies))
            # pad by 5% so extremes aren't clipped to the very edge of the colormap
            pad = max(0.02, 0.05 * (ent_max_obs - ent_min_obs))
            ent_vmin = max(0.0, ent_min_obs - pad)
            ent_vmax = ent_max_obs + pad
            p_ent = render_metric_panel(env, base_map, info, entropies, ent_vmin, ent_vmax,
                                        "Entropy (H)", cmap_name="plasma", alpha=190)
        else:
            p_ent = render_blank_metric_panel(base_map, "Entropy (H)")

        if post_ready and post_returns is not None and len(post_returns) >= T:
            returns = post_returns[:T]
            ret_min, ret_max = float(np.min(returns)), float(np.max(returns))
            p_ret = render_metric_panel(env, base_map, info, returns, ret_min, ret_max, "Return", cmap_name="Reds", alpha=110)
        else:
            p_ret = render_blank_metric_panel(base_map, "Return (post)")

        if post_ready and post_advs is not None and len(post_advs) >= T:
            advs = post_advs[:T]
            adv_min, adv_max = float(np.min(advs)), float(np.max(advs))
            p_adv = render_metric_panel(env, base_map, info, advs, adv_min, adv_max, "Adv", cmap_name="Reds", alpha=110)
        else:
            p_adv = render_blank_metric_panel(base_map, "Adv (post)")

    finally:
        env._trajectory_positions = pos_save
        env._traj_yaws = yaw_save

    gap = np.full((p_r.shape[0], 10, 3), 255, np.uint8)
    strip = np.concatenate([p_r, gap, p_ent, gap, p_ret, gap, p_adv], axis=1)
    cache["metric_strip"] = strip
    cache["metric_strip_call"] = call_idx
    return strip


# --------------------------- base panel rendering + final frame assembly ---------------------------

def _render_eval_panel_gt(env, Hs, Ws):
    """GT image panel for eval layout."""
    return _to_uint8_img(env._gt_rgb, Hs, Ws)


def _build_base_panels(env, HEIGHT, WIDTH):
    Hs, Ws = HEIGHT, WIDTH

    gt = _to_uint8_img(env._gt_rgb, Hs, Ws)
    pred_missing = getattr(env, "_pred_rgb", None) is None
    pr = _to_uint8_img(env._pred_rgb, Hs, Ws)
    if pred_missing:
        getter = getattr(env, "get_aux_status_panel_lines", None)
        if callable(getter):
            try:
                lines = list(getter())
            except Exception:
                lines = []
            if lines:
                title = str(getattr(env, "_aux_status_panel_title", "Status"))
                pr = _render_status_text_panel(Hs, Ws, title=title, lines=lines)

    # uncertainty (optional)
    unc_value = getattr(env, "_unc", None)
    if unc_value is not None:
        unc_scalar = float(np.clip(np.squeeze(unc_value) * 10.0, 0.0, 1.0))
        cmapu = cm.get_cmap("bwr")
        unc_color = cmapu(unc_scalar)[..., :3]
        unc = _to_uint8_img(unc_color, Hs, Ws)
    else:
        unc = np.full((Hs, Ws, 3), 255, np.uint8)

    disable_curiosity_panel = bool(getattr(env, "_disable_curiosity_reward_panel", False))

    # error map with MSE text (cached + throttled because it's expensive)
    if disable_curiosity_panel:
        lines = None
        if bool(getattr(env, "_prefer_aux_status_panel", False)):
            getter = getattr(env, "get_aux_status_panel_lines", None)
            if callable(getter):
                try:
                    lines = list(getter())
                except Exception:
                    lines = None
        if not lines:
            reason = getattr(env, "_disable_curiosity_reward_panel_reason", "")
            lines = ["Disabled for this task."]
            if isinstance(reason, str) and reason.strip():
                lines.append(reason.strip())
        title = str(getattr(env, "_aux_status_panel_title", "Curiosity Reward"))
        err_map = _render_status_text_panel(
            Hs,
            Ws,
            title=title,
            lines=lines,
        )
    elif (env._gt_rgb is not None) and (env._pred_rgb is not None):
        cache = _render_cache(env)
        call_idx = int(getattr(env, "_render_call_idx", 0))
        mse_every = _safe_int_attr(env, "_render_mse_every", 4, min_value=1)
        err_key = (id(env._gt_rgb), id(env._pred_rgb), Hs, Ws)
        last_key = cache.get("err_key")
        recompute = (cache.get("err_map") is None) or (err_key != last_key) or ((call_idx % mse_every) == 0)

        if recompute:
            gt_t = torch.from_numpy(gt).permute(2, 0, 1).float() / 255.0
            pr_t = torch.from_numpy(pr).permute(2, 0, 1).float() / 255.0
            mse_val = float(mse_lowfreq_blur_then_downsample(gt_t, pr_t, factor=16).item())

            blur_factor = 16
            blur_sigma = float(blur_factor)
            err_h, err_w = max(1, Hs // blur_factor), max(1, Ws // blur_factor)
            gt_lp = gaussian_blur2d(gt_t.unsqueeze(0), sigma=blur_sigma)
            pr_lp = gaussian_blur2d(pr_t.unsqueeze(0), sigma=blur_sigma)
            gt_ds = F.interpolate(gt_lp, size=(err_h, err_w), mode="area")
            pr_ds = F.interpolate(pr_lp, size=(err_h, err_w), mode="area")

            err_lf = (gt_ds - pr_ds).pow(2).mean(dim=1).squeeze(0).clamp(0.0, 1.0)
            err_vmax = 0.5
            err_lf_np = (err_lf / err_vmax).clamp(0.0, 1.0).cpu().numpy()
            err_rgb = (cm.get_cmap("coolwarm")(err_lf_np)[..., :3] * 255).astype(np.uint8)
            err_map = np.array(Image.fromarray(err_rgb).resize((Ws, Hs), Image.NEAREST))

            em_img = Image.fromarray(err_map)
            draw = ImageDraw.Draw(em_img)
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 24)
            except Exception:
                font = ImageFont.load_default()

            reward1 = reward_from_mse_threshold(torch.tensor(mse_val)).item() * 100
            reward_overlay_mode = str(getattr(env, "_reward_overlay_mode", "full")).strip().lower()
            if reward_overlay_mode == "sign_only":
                text = "reward: +" if reward1 > 0.0 else "reward: -"
                text_color = (20, 140, 20) if reward1 > 0.0 else (196, 35, 35)
                tw, th = _text_size(draw, text, font)
                x = Ws - tw - 8
                y = Hs - th - 8
                draw.rectangle([x - 5, y - 3, x + tw + 5, y + th + 3], fill=(255, 255, 255))
                draw.text((x, y), text, fill=text_color, font=font)
            else:
                text = f"MSE: {mse_val:.3f}, reward: {reward1:.3f}"
                tw, th = _text_size(draw, text, font)
                x = Ws - tw - 8
                y = Hs - th - 8
                draw.rectangle([x - 3, y - 2, x + tw + 3, y + th + 2], fill=(0, 0, 0))
                draw.text((x, y), text, fill=(255, 255, 255), font=font)
            err_map = np.asarray(em_img, dtype=np.uint8)

            cache["err_map"] = err_map
            cache["err_key"] = err_key
        else:
            err_map = cache["err_map"]
    else:
        err_map = np.full((Hs, Ws, 3), 255, np.uint8)

    policy_panel = render_eval_side_panel(env, Hs)
    return gt, pr, err_map, policy_panel


def render_base_panel(env, HEIGHT, WIDTH):
    Hs, _ = HEIGHT, WIDTH
    gt, pr, err_map, policy_panel = _build_base_panels(env, HEIGHT, WIDTH)
    gap = np.full((Hs, 8, 3), 255, np.uint8)
    # Default ordering: gt | pred | err | policy
    return np.concatenate([gt, gap, pr, gap, err_map, gap, policy_panel], axis=1)


def render_frame(env, HEIGHT, WIDTH):
    """
    Used by env.render(). This function returns the full final frame.
    When env._eval_panel_layout is True: GT | floorplan | progress | policy.
    """
    env._render_call_idx = int(getattr(env, "_render_call_idx", 0)) + 1
    resize_mode = _resize_filter(env)
    eval_layout = bool(getattr(env, "_eval_panel_layout", False))

    if eval_layout:
        Hs, Ws = HEIGHT, WIDTH
        gt = _render_eval_panel_gt(env, Hs, Ws)
        floorplan = render_topdown(env)
        floorplan_resized = np.array(
            Image.fromarray(floorplan).resize(
                (int(Hs * (floorplan.shape[1] / floorplan.shape[0])), Hs),
                resize_mode,
            )
        )

        progress_panel = _render_point_cloud_topdown_progress_with_trajectory(env)
        progress_resize_mode = Image.NEAREST
        if progress_panel is None:
            progress_panel = floorplan
            progress_resize_mode = resize_mode
        progress_resized = np.array(
            Image.fromarray(progress_panel).resize(
                (int(Hs * (progress_panel.shape[1] / progress_panel.shape[0])), Hs),
                progress_resize_mode,
            )
        )

        policy_panel = render_eval_side_panel(env, Hs)
        gap = np.full((Hs, 8, 3), 255, np.uint8)
        # Layout: observed RGB | floorplan | point cloud progress | action panel.
        final = np.concatenate(
            [gt, gap, floorplan_resized, gap, progress_resized, gap, policy_panel],
            axis=1,
        )
    else:
        use_image_goal_order = bool(getattr(env, "_render_layout_goal_order", False))
        if use_image_goal_order:
            gt, pred, status_panel, action_panel = _build_base_panels(env, HEIGHT, WIDTH)
            topdown = render_topdown(env)
            Hs = gt.shape[0]
            topdown_resized = np.array(
                Image.fromarray(topdown).resize(
                    (int(Hs * (topdown.shape[1] / topdown.shape[0])), Hs),
                    resize_mode
                )
            )
            gap = np.full((Hs, 8, 3), 255, np.uint8)
            # Image-goal ordering: observation | target | topdown | status | action
            row1 = np.concatenate(
                [gt, gap, pred, gap, topdown_resized, gap, status_panel, gap, action_panel],
                axis=1,
            )
        else:
            gt, pred, err_map, action_panel = _build_base_panels(env, HEIGHT, WIDTH)
            base_render = np.concatenate(
                [gt, np.full((gt.shape[0], 8, 3), 255, np.uint8), pred, np.full((gt.shape[0], 8, 3), 255, np.uint8), err_map],
                axis=1,
            )
            topdown = render_topdown(env)
            Hs = base_render.shape[0]
            topdown_resized = np.array(
                Image.fromarray(topdown).resize(
                    (int(Hs * (topdown.shape[1] / topdown.shape[0])), Hs),
                    resize_mode
                )
            )
            gap = np.full((Hs, 8, 3), 255, np.uint8)
            # Default ordering: observation | prediction | error | floorplan | action panel.
            row1 = np.concatenate([base_render, gap, topdown_resized, gap, action_panel], axis=1)

        if bool(getattr(env, "_disable_bottom_metric_strip", False)):
            final = row1
        else:
            strip = render_topdown_metrics_strip(env)
            strip_resized = np.array(Image.fromarray(strip).resize((row1.shape[1], row1.shape[0]), resize_mode))
            vgap = np.full((10, row1.shape[1], 3), 255, np.uint8)
            final = np.concatenate([row1, vgap, strip_resized], axis=0)
    return final

# ============================================
# Training visualization helpers (used by train_ppo_explore.py)
# ============================================

def _derive_train_completion_media_paths(train_video_path: str) -> dict[str, str]:
    stem, _ = os.path.splitext(train_video_path)
    return {
        "progress_video": f"{stem}_point_cloud_topdown_progress.mp4",
    }


def _derive_env_train_panel_video_path(train_video_path: str, env_index: int) -> str:
    if int(env_index) == 0:
        return train_video_path
    stem, ext = os.path.splitext(train_video_path)
    suffix = ext if ext else ".mp4"
    return f"{stem}_env{int(env_index):02d}_panel{suffix}"


def _write_observation_video(
    imgs_all: torch.Tensor,
    output_path: str,
    *,
    env_index: int = 0,
    fps: int = 12,
    upscale: int = 1,
    bitrate: str = "12M",
) -> None:
    """Write rank-0 env RGB observations as an MP4.

    imgs_all is expected to be (E, T+1, 3, H, W) uint8.
    """
    if imgs_all.ndim != 5:
        raise ValueError(f"Expected imgs_all ndim=5, got {imgs_all.ndim}")
    if imgs_all.shape[2] != 3:
        raise ValueError(f"Expected channel-first RGB tensor, got shape={tuple(imgs_all.shape)}")
    env_idx = int(env_index)
    if env_idx < 0 or env_idx >= int(imgs_all.shape[0]):
        raise ValueError(
            f"env_index={env_idx} out of range for imgs_all with {int(imgs_all.shape[0])} envs"
        )

    frames_u8 = imgs_all[env_idx].detach().to(device="cpu", dtype=torch.uint8)
    up = max(1, int(upscale))
    ffmpeg_params = ["-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p"]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with imageio.get_writer(
        output_path,
        fps=int(fps),
        codec="libx264",
        bitrate=str(bitrate),
        macro_block_size=1,
        ffmpeg_params=ffmpeg_params,
    ) as writer:
        for frame_chw in frames_u8:
            frame_hwc = frame_chw.permute(1, 2, 0).contiguous().numpy()
            if up > 1:
                h, w = int(frame_hwc.shape[0]), int(frame_hwc.shape[1])
                frame_hwc = np.asarray(
                    Image.fromarray(frame_hwc).resize((w * up, h * up), Image.LANCZOS),
                    dtype=np.uint8,
                )
            writer.append_data(frame_hwc)


def _write_train_panel_video(
    frames: list[np.ndarray],
    output_path: str,
    *,
    fps: int = 12,
    bitrate: str = "12M",
) -> None:
    """Write the composite train/video panel stream at native frame size."""
    if not frames:
        return

    ffmpeg_params = ["-crf", "16", "-preset", "slow", "-pix_fmt", "yuv420p"]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with imageio.get_writer(
        output_path,
        fps=int(fps),
        codec="libx264",
        bitrate=str(bitrate),
        macro_block_size=1,
        ffmpeg_params=ffmpeg_params,
    ) as writer:
        for frame in frames:
            frame_u8 = np.asarray(frame, dtype=np.uint8)
            if frame_u8.ndim == 2:
                frame_u8 = np.repeat(frame_u8[..., None], 3, axis=2)
            if frame_u8.shape[-1] == 4:
                frame_u8 = frame_u8[..., :3]
            writer.append_data(frame_u8)


def _scene_info_from_state(state: dict, env_index: int) -> dict[str, str | int]:
    scene_payload = state.get("scene", {}) if isinstance(state, dict) else {}
    scene_name = str(scene_payload.get("scene_name", ""))
    mesh_path = str(scene_payload.get("glb_path", ""))
    mesh_name = os.path.basename(mesh_path) if mesh_path else ""
    scene_id = scene_name or (os.path.splitext(mesh_name)[0] if mesh_name else "")
    return {
        "env_index": int(env_index),
        "scene_id": scene_id,
        "scene_name": scene_name,
        "scene_mesh_name": mesh_name,
        "scene_mesh_path": mesh_path,
    }


def _clip_middle(text: str, max_chars: int = 140) -> str:
    if len(text) <= int(max_chars):
        return text
    keep = max(8, int(max_chars) - 3)
    left = keep // 2
    right = keep - left
    return f"{text[:left]}...{text[-right:]}"


def _render_scene_info_panel(scene_info: dict[str, str | int]) -> np.ndarray:
    width, height = 1600, 160
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
        font_body = ImageFont.truetype("DejaVuSans.ttf", 21)
    except Exception:
        font_title = ImageFont.load_default()
        font_body = ImageFont.load_default()

    scene_id = str(scene_info.get("scene_id", ""))
    scene_name = str(scene_info.get("scene_name", ""))
    scene_mesh_name = str(scene_info.get("scene_mesh_name", ""))
    scene_mesh_path = str(scene_info.get("scene_mesh_path", ""))
    env_index = int(scene_info.get("env_index", 0))

    y = 12
    draw.text((16, y), f"Training Scene (env {env_index})", fill=(20, 20, 20), font=font_title)
    y += 38
    draw.text((16, y), f"scene_id: {scene_id}", fill=(20, 20, 20), font=font_body)
    y += 30
    draw.text((16, y), f"scene_name: {scene_name}", fill=(20, 20, 20), font=font_body)
    y += 30
    mesh_label = f"mesh: {scene_mesh_name} | path: {_clip_middle(scene_mesh_path)}"
    draw.text((16, y), mesh_label, fill=(20, 20, 20), font=font_body)
    return np.asarray(img, dtype=np.uint8)
