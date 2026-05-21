import os
import math
import random
import glob
import time
import json
from typing import Optional, Dict, Any, List
import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces
import habitat_sim
import torch
import torch.nn.functional as F
import magnum as mn
from einops import rearrange
from habitat_sim.utils.common import quat_from_angle_axis, quat_to_coeffs

from .render_log_utils import render_frame


# ---------------- Paths / constants ----------------
# HM3D_DATA_ROOT is set by main.py via scripts.dl_hm3d_data.resolve_hm3d_root()
_HM3D_BASE = os.environ.get("HM3D_DATA_ROOT", "/workspace/data")
HM3D_GLB_ROOT = f"{_HM3D_BASE}/hm3d/hm3d_glb"
HM3D_NAV_ROOT = f"{_HM3D_BASE}/hm3d/hm3d_nav"
HM3D_GLB_VAL_ROOT = f"{_HM3D_BASE}/hm3d-val/hm3d_glb"
HM3D_NAV_VAL_ROOT = f"{_HM3D_BASE}/hm3d-val/hm3d_nav"

WIDTH, HEIGHT   = 640, 360
HFOV_DEG        = 90.0

AGENT_HEIGHT    = 1.25 #1.5
AGENT_RADIUS    = 0.10
STEP_METERS     = 0.25
COLLISION_RADIUS = 0.08

YAW_DEG         = 15.0
PITCH_DEG = 5.0
PITCH_LIMIT_DEG = 30.0

ACT_LIST = ["forward", "look left", "look right", "stop"] # "stop" not really useful

TOPDOWN_HEIGHT_PX = 512
TOPDOWN_RAY_STRIDE = 2

# Optional: quieter logs
try:
    habitat_sim.logging.set_level(habitat_sim.logging.Level.Error)
except Exception:
    pass


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


def _env_is_true(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}
    
# ---------------- Utility helpers ----------------
def intrinsics_from_hfov(width: int, height: int, hfov_deg: float) -> np.ndarray:
    hfov = math.radians(hfov_deg)
    fx = (width / 2.0) / math.tan(hfov / 2.0)
    vfov = 2.0 * math.atan((height / width) * math.tan(hfov / 2.0))
    fy = (height / 2.0) / math.tan(vfov / 2.0)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return np.array([fx, fy, cx, cy], dtype=np.float32)


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    # q = [x,y,z,w]
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float32)


def list_scene_glbs(test=False) -> List[Dict[str, Optional[str]]]:
    root = HM3D_GLB_VAL_ROOT if test else HM3D_GLB_ROOT
    nav_root = HM3D_NAV_VAL_ROOT if test else HM3D_NAV_ROOT
    # Retry: NFS/network mounts can report dir exists but listdir empty until mount is ready
    # NFS/network mounts can be slow to populate; retry with long backoff
    max_retries = 30
    retry_delay = 15
    glbs = []
    for attempt in range(max_retries):
        glbs = sorted(glob.glob(os.path.join(root, "*", "*.glb")))
        if glbs:
            break
        if attempt < max_retries - 1:
            print(f"[HM3D] No GLBs yet (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...", flush=True)
            time.sleep(retry_delay)
    out = []
    for glb in glbs:
        x = os.path.basename(os.path.dirname(glb))
        y = os.path.splitext(os.path.basename(glb))[0]
        navmesh_path = os.path.join(nav_root, x, f"{y}.basis.navmesh")
        navmesh = os.path.abspath(navmesh_path) if os.path.isfile(navmesh_path) else None
        out.append({"scene_name": y, "glb_path": os.path.abspath(glb), "navmesh": navmesh})
    if not out:
        base = os.environ.get("HM3D_DATA_ROOT", "/workspace/data")
        for probe in [base]:
            try:
                if os.path.isdir(probe):
                    top = sorted(os.listdir(probe))[:20]
                    print(f"[HM3D] {probe} exists, contents (first 20): {top}", flush=True)
                else:
                    print(f"[HM3D] {probe} does not exist or is not a directory", flush=True)
            except Exception as e:
                print(f"[HM3D] Error probing {probe}: {e}", flush=True)
        raise FileNotFoundError(
            f"No GLBs found under {root}. "
            f"Set HM3D_DATA_ROOT env var to the directory containing the HM3D dataset. "
            f"Current base={base!r}"
        )
    return out


# ---------------- Env ----------------
class HabitatMP3DEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        scene_list: List[Dict[str, Optional[str]]],
        max_steps: int = 128,
        render_mode: Optional[str] = None,
        gpu_id=0,
        check_holes: bool = False,
    ):
        super().__init__()
        self.scene_list = scene_list
        self.max_steps  = int(max_steps)
        self.render_mode = render_mode
        self.sim: Optional[habitat_sim.Simulator] = None
        self.scene_name: Optional[str] = None
        self.gpu_id = int(gpu_id)
        self.check_holes = bool(check_holes)

        self.k_vec = intrinsics_from_hfov(WIDTH, HEIGHT, HFOV_DEG)

        # render caches (producer: trainer via update_meta)
        self._pred_rgb = None
        self._gt_rgb   = None
        self._last_logits = None
        self._unc = None
        self._last_mode = None
        self._point_cloud_topdown_progress = None

        self.action_space = spaces.Discrete(len(ACT_LIST))
        BIG = 1e9
        self.observation_space = spaces.Dict({
            "rgb": spaces.Box(low=0, high=255, shape=(HEIGHT, WIDTH, 3), dtype=np.uint8),
            "depth": spaces.Box(low=0, high=BIG, shape=(HEIGHT, WIDTH), dtype=np.float32),
            "fxfycxcy": spaces.Box(low=-BIG, high=BIG, shape=(4,), dtype=np.float32),
            "c2w": spaces.Box(low=-BIG, high=BIG, shape=(4,4), dtype=np.float32),
        })

        self._t = 0

        # topdown/metrics state (used by utils)
        self._topdown_height = TOPDOWN_HEIGHT_PX
        self._topdown_ray_stride = TOPDOWN_RAY_STRIDE

        self._topdown_map = None
        self._topdown_map_info = None

        self._trajectory_positions = []  # camera world positions
        self._traj_yaws = []             # yaw per position

        self._step_rewards = []
        self._step_returns = []
        self._step_advs = []
        self._step_entropies = []

        self._reward_vmin, self._reward_vmax = -0.1, 60.0
        self._reward_display_mode = "threshold"
        self._reward_overlay_mode = "sign_only" if _env_is_true("TRAIN_VIS_REWARD_SIGN_ONLY") else "full"
        self._ret_vmin = self._ret_vmax = None
        self._adv_vmin = self._adv_vmax = None
        self._metric_bounds_mode = "per_episode"
        self._post_metrics_ready = False
        self._post_returns = None
        self._post_advs = None
        self._traj_overlay = None
        self._traj_overlay_len = 0
        self._disable_bottom_metric_strip = bool(_env_is_true("TRAIN_VIS_FIRST_ROW_ONLY"))
        self._render_use_lanczos = bool(_env_is_true("TRAIN_VIS_VIDEO_USE_LANCZOS"))
        self._sensor_min_interval_s = 0.003
        self._last_sensor_read_t = 0.0
        debug_mode = os.environ.get("CURIOUS_CAMERA_HABITAT_DEBUG", "0").strip().lower()
        self._debug_habitat = debug_mode not in {"", "0", "false", "off"}
        self._debug_habitat_verbose = debug_mode in {"2", "verbose", "full"}
        self._debug_rank = os.environ.get("RANK", "0")
        self._last_action_name: Optional[str] = None
        self._last_action_idx: Optional[int] = None
        self._debug_event_handle = None
        self._debug_snapshot_handle = None
        self._debug_sensor_seq = 0
        if self._debug_habitat:
            session_dir = os.environ.get("TRAINING_SESSION_DIR", "")
            if session_dir:
                debug_dir = os.path.join(session_dir, "debug_habitat")
                os.makedirs(debug_dir, exist_ok=True)
                try:
                    self._debug_snapshot_handle = open(
                        os.path.join(
                            debug_dir,
                            f"rank{self._debug_rank}_pid{os.getpid()}_last_sensor.json",
                        ),
                        "w+",
                        encoding="utf-8",
                        buffering=1,
                    )
                except OSError:
                    self._debug_snapshot_handle = None
                if self._debug_habitat_verbose:
                    try:
                        self._debug_event_handle = open(
                            os.path.join(
                                debug_dir,
                                f"rank{self._debug_rank}_pid{os.getpid()}.log",
                            ),
                            "a",
                            encoding="utf-8",
                            buffering=1,
                        )
                    except OSError:
                        self._debug_event_handle = None

        self._camera_pose_trace_enabled = bool(_env_is_true("TRAIN_VIS_CAMERA_POSE_JSON"))
        self._camera_pose_trace_dir = ""
        self._camera_pose_trace_episode_index = 0
        self._camera_pose_trace_entries: list[dict[str, Any]] = []
        self._camera_pose_trace_scene: dict[str, Any] | None = None
        self._camera_pose_trace_seed: int | None = None
        self._camera_pose_trace_started_at = 0.0
        self._camera_pose_trace_written = False
        if self._camera_pose_trace_enabled:
            session_dir = os.environ.get("TRAINING_SESSION_DIR", "").strip()
            default_dir = (
                os.path.join(session_dir, "camera_pose_traces")
                if session_dir
                else "camera_pose_traces"
            )
            self._camera_pose_trace_dir = (
                os.environ.get("TRAIN_VIS_CAMERA_POSE_DIR", default_dir).strip() or default_dir
            )
            os.makedirs(self._camera_pose_trace_dir, exist_ok=True)

        # tiny indirections so utils can raycast without importing mn/habitat_sim
        self._mn_vec3 = mn.Vector3
        self._hab_ray = habitat_sim.geo.Ray
        self._forced_scene_query_raw = os.environ.get("TRAIN_FORCE_SCENE_ID", "").strip()
        self._forced_scene_dict = None
        if self._forced_scene_query_raw:
            self._forced_scene_dict = self._resolve_forced_scene(self._forced_scene_query_raw)
            if self._forced_scene_dict is None:
                raise ValueError(
                    "TRAIN_FORCE_SCENE_ID did not match any loaded scene: "
                    f"{self._forced_scene_query_raw!r}"
                )
        

    def update_scenes(self, scene_list):
        self.scene_list = scene_list
        if self._forced_scene_query_raw:
            self._forced_scene_dict = self._resolve_forced_scene(self._forced_scene_query_raw)
            if self._forced_scene_dict is None:
                raise ValueError(
                    "TRAIN_FORCE_SCENE_ID no longer matches the active scene list: "
                    f"{self._forced_scene_query_raw!r}"
                )

    def get_num_scenes(self):
        return len(getattr(self, "scene_list", []))

    def _scene_aliases(self, scene_dict: Dict[str, Optional[str]]) -> set[str]:
        aliases: set[str] = set()
        scene_name = str(scene_dict.get("scene_name") or "").strip()
        glb_path = str(scene_dict.get("glb_path") or "").strip()
        if scene_name:
            aliases.add(scene_name.lower())
        if glb_path:
            glb_path_norm = os.path.abspath(glb_path).lower()
            glb_base = os.path.basename(glb_path).lower()
            glb_stem = os.path.splitext(glb_base)[0]
            aliases.add(glb_path_norm)
            aliases.add(glb_base)
            aliases.add(glb_stem)
            folder_base = os.path.basename(os.path.dirname(glb_path)).lower()
            if folder_base:
                aliases.add(folder_base)
                if "-" in folder_base:
                    aliases.add(folder_base.split("-", 1)[-1])
        return aliases

    def _resolve_forced_scene(self, query: str) -> Optional[Dict[str, Optional[str]]]:
        q = query.strip().lower()
        if not q:
            return None
        q_base = os.path.basename(q)
        q_stem = os.path.splitext(q_base)[0]
        candidates = {q, q_base, q_stem}
        if "-" in q_stem:
            candidates.add(q_stem.split("-", 1)[-1])
        for scene_dict in self.scene_list:
            aliases = self._scene_aliases(scene_dict)
            if candidates.intersection(aliases):
                return scene_dict
        return None

    def _debug_serialize(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            return value.item()
        if isinstance(value, (list, tuple)):
            return [self._debug_serialize(v) for v in value]
        if isinstance(value, dict):
            return {str(k): self._debug_serialize(v) for k, v in value.items()}
        return value

    def _build_debug_payload(self, event: str, **kwargs: Any) -> dict[str, Any]:
        payload = {
            "event": event,
            "pid": os.getpid(),
            "rank": self._debug_rank,
            "scene": getattr(self, "scene_name", None),
            "env_step": self._t,
            "action_idx": self._last_action_idx,
            "action_name": self._last_action_name,
            "timestamp": time.time(),
        }
        payload.update({k: self._debug_serialize(v) for k, v in kwargs.items()})
        return payload

    def _debug_log(self, event: str, force: bool = False, **kwargs: Any) -> None:
        if not self._debug_habitat:
            return
        if not (force or self._debug_habitat_verbose):
            return
        payload = self._build_debug_payload(event, **kwargs)
        line = json.dumps(payload, sort_keys=True)
        print(f"[habitat_debug] {line}", flush=force)
        if self._debug_event_handle is not None:
            try:
                self._debug_event_handle.write(line + "\n")
                if force:
                    self._debug_event_handle.flush()
            except OSError:
                pass

    def _debug_capture_sensor_context(
        self,
        caller: str,
        debug_context: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._debug_snapshot_handle is None:
            return
        self._debug_sensor_seq += 1
        line = json.dumps(
            self._build_debug_payload(
                "last_sensor_context",
                sensor_seq=self._debug_sensor_seq,
                caller=caller,
                debug_context=debug_context or {},
                **self._current_agent_debug_state(),
            ),
            sort_keys=True,
        )
        try:
            self._debug_snapshot_handle.seek(0)
            self._debug_snapshot_handle.truncate()
            self._debug_snapshot_handle.write(line + "\n")
            self._debug_snapshot_handle.flush()
        except OSError:
            pass

    def _current_agent_debug_state(self) -> dict[str, Any]:
        if self.sim is None:
            return {}
        agent = self.sim.get_agent(0)
        st = agent.get_state()
        pos = np.array(st.position, dtype=np.float32)
        rot = np.array(quat_to_coeffs(st.rotation), dtype=np.float32)
        return {
            "agent_position": pos,
            "agent_rotation": rot,
        }

    def _current_camera_pose_state(self) -> dict[str, Any]:
        if self.sim is None:
            return {}
        agent = self.sim.get_agent(0)
        st = agent.get_state()
        agent_pos = np.array(st.position, dtype=np.float32)
        agent_rot = np.array(quat_to_coeffs(st.rotation), dtype=np.float32)
        _, _, _, cam_pos = self._get_cam_basis_and_pos()
        return {
            "agent_position": agent_pos,
            "agent_rotation": agent_rot,
            "camera_position": np.asarray(cam_pos, dtype=np.float32),
            "agent_yaw_radians": float(self._get_agent_yaw()),
        }

    def _append_camera_pose_trace(self, event: str, **kwargs: Any) -> None:
        if not self._camera_pose_trace_enabled or self._camera_pose_trace_written:
            return
        payload_raw = {
            "event": str(event),
            "env_step": int(self._t),
            "timestamp": time.time(),
        }
        payload_raw.update(self._current_camera_pose_state())
        payload_raw.update(kwargs)
        payload = {k: self._debug_serialize(v) for k, v in payload_raw.items()}
        self._camera_pose_trace_entries.append(payload)

    def _start_camera_pose_trace(
        self,
        seed: int | None,
        *,
        restore_state: bool,
        fixed_start_pose: bool,
    ) -> None:
        if not self._camera_pose_trace_enabled:
            return
        self._camera_pose_trace_episode_index += 1
        self._camera_pose_trace_entries = []
        scene_dict = getattr(self, "_scene_dict", None)
        if isinstance(scene_dict, dict):
            self._camera_pose_trace_scene = dict(scene_dict)
        else:
            self._camera_pose_trace_scene = None
        self._camera_pose_trace_seed = int(seed) if seed is not None else None
        self._camera_pose_trace_started_at = time.time()
        self._camera_pose_trace_written = False
        self._append_camera_pose_trace(
            "reset",
            restore_state=bool(restore_state),
            fixed_start_pose=bool(fixed_start_pose),
            scene=self._camera_pose_trace_scene,
            seed=self._camera_pose_trace_seed,
        )

    def _write_camera_pose_trace(self, reason: str) -> None:
        if not self._camera_pose_trace_enabled or self._camera_pose_trace_written:
            return
        if not self._camera_pose_trace_entries:
            return
        out = {
            "episode_index": int(self._camera_pose_trace_episode_index),
            "rank": self._debug_rank,
            "pid": os.getpid(),
            "scene": self._debug_serialize(self._camera_pose_trace_scene),
            "seed": self._camera_pose_trace_seed,
            "started_at": float(self._camera_pose_trace_started_at),
            "ended_at": time.time(),
            "end_reason": str(reason),
            "num_entries": len(self._camera_pose_trace_entries),
            "poses": self._camera_pose_trace_entries,
        }
        out_path = os.path.join(
            self._camera_pose_trace_dir,
            (
                f"rank{self._debug_rank}_pid{os.getpid()}_"
                f"episode_{self._camera_pose_trace_episode_index:06d}.json"
            ),
        )
        try:
            with open(out_path, "w", encoding="utf-8") as handle:
                json.dump(out, handle, indent=2, sort_keys=True)
        except OSError as exc:
            print(
                f"[camera_pose_trace] failed to write '{out_path}': {exc}",
                flush=True,
            )
            return
        self._camera_pose_trace_written = True
        self._camera_pose_trace_entries = []

    # ---------- Gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        self._pred_rgb = None
        self._unc = None
        options = options or {}
        if self._camera_pose_trace_entries and not self._camera_pose_trace_written:
            self._write_camera_pose_trace(reason="reset_before_new_episode")

        self._trajectory_positions = []
        self._traj_yaws = []
        self._topdown_map = None
        self._topdown_map_info = None
        self._point_cloud_topdown_progress = None

        self._step_rewards = []
        self._step_returns = []
        self._step_advs = []
        self._step_entropies = []
        self._ret_vmin = self._ret_vmax = None
        self._adv_vmin = self._adv_vmax = None
        self._post_metrics_ready = False
        # _reward_display_mode persists (set by PPO, not reset)
        self._post_returns = None
        self._post_advs = None
        self._traj_overlay = None
        self._traj_overlay_len = 0
        self._last_sensor_read_t = 0.0


        if seed is not None:
            # Ensure seed is in valid range for numpy [0, 2^32 - 1]
            seed = int(seed) % (2**32)
            random.seed(seed)
            np.random.seed(seed)

        self._last_action_name = None
        self._last_action_idx = None
            
       
        restore = options.get("restore_state", None)
        scene_dict = options.get("scene", None) if restore is None else restore.get("scene")
        if scene_dict is None:
            if self._forced_scene_dict is not None:
                scene_dict = self._forced_scene_dict
            else:
                scene_dict = random.choice(self.scene_list)
        self._scene_dict = scene_dict

        new_scene_name = scene_dict["scene_name"]
        if self.sim is None or self.scene_name != new_scene_name:
            if self.sim is not None:
                self.sim.close()
                self.sim = None
                # Give NVIDIA driver time to release GL context before recreating (habitat-sim #1176:
                # rapid destroy/reinit triggers EGL/GL crashes / SIGSEGV)
                time.sleep(0.5)
            self.sim = self._make_sim(scene_dict["glb_path"], scene_dict.get("navmesh"))
            self.scene_name = new_scene_name

        # When fixed_start_pose: use deterministic pose (same position + yaw every time)
        # for reproducible test/eval runs. Otherwise use seed for randomization.
        fixed_start_pose = options.get("fixed_start_pose", False)
        pose_seed = 0 if fixed_start_pose else seed

        # seed the pathfinder if sim exists
        if self.sim is not None:
            self.sim.pathfinder.seed(pose_seed if pose_seed is not None else 0)

        if fixed_start_pose:
            random.seed(0)
            np.random.seed(0)

        if restore is not None:
            agent = self.sim.get_agent(0)
            st = agent.get_state()
            st.position = mn.Vector3(
                float(restore["position"][0]),
                float(restore["position"][1]),
                float(restore["position"][2]),
            )
            # set_state expects rotation as coeffs (b,c,d,a), not magnum Quaternion
            st.rotation = np.array(restore["rotation"], dtype=np.float32)
            agent.set_state(st)
            obs = self._observe()
            _, _, _, cam_pos = self._get_cam_basis_and_pos()
            self._trajectory_positions.append(cam_pos.copy())
            self._traj_yaws.append(self._get_agent_yaw())
            self._start_camera_pose_trace(
                seed=seed,
                restore_state=True,
                fixed_start_pose=bool(fixed_start_pose),
            )
            if self._debug_habitat_verbose:
                self._debug_log(
                    "reset_restore_complete",
                    seed=seed,
                    options=options,
                    restore_state=True,
                    **self._current_agent_debug_state(),
                )
            return obs, {}

        for _ in range(256):
            st = self._randomize_start(fixed_start_pose=fixed_start_pose)
            if st is not None:
                obs = self._observe()

                # record initial camera pos once
                _, _, _, cam_pos = self._get_cam_basis_and_pos()
                self._trajectory_positions.append(cam_pos.copy())
                self._traj_yaws.append(self._get_agent_yaw())
                self._start_camera_pose_trace(
                    seed=seed,
                    restore_state=False,
                    fixed_start_pose=bool(fixed_start_pose),
                )
                if self._debug_habitat_verbose:
                    self._debug_log(
                        "reset_complete",
                        seed=seed,
                        options=options,
                        restore_state=False,
                        **self._current_agent_debug_state(),
                    )

                return obs, {}

        raise RuntimeError("Failed to sample a valid start after many tries")

    def step(self, action: int):
        self._t += 1
        self._last_action_idx = int(action)
        self._last_action_name = ACT_LIST[action] if 0 <= action < len(ACT_LIST) else f"unknown_{action}"
        if self._debug_habitat_verbose:
            self._debug_log("step_start", **self._current_agent_debug_state())

        if   action == 0: moved = self._try_move_local(dz= STEP_METERS)          # forward
        elif action == 1: moved = self._apply_rotation(delta_yaw_deg= YAW_DEG)   # look left
        elif action == 2: moved = self._apply_rotation(delta_yaw_deg=-YAW_DEG)  # look right
        elif action == 3: moved = False                                         # stop
        else: raise ValueError(f"Unknown action: {action}")

        obs = self._observe()
        reward = 0.0
        terminated = False
        truncated = (self._t >= self.max_steps)
        info = {"scene_name": self.scene_name, "moved": moved}

        # record exactly once per step
        _, _, _, cam_pos = self._get_cam_basis_and_pos()
        self._trajectory_positions.append(cam_pos.copy())
        self._traj_yaws.append(self._get_agent_yaw())
        if self._debug_habitat_verbose:
            self._debug_log(
                "step_complete",
                moved=bool(moved),
                truncated=bool(truncated),
                **self._current_agent_debug_state(),
            )
        self._append_camera_pose_trace(
            "step",
            action_idx=int(action),
            action_name=self._last_action_name,
            moved=bool(moved),
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        if terminated or truncated:
            self._write_camera_pose_trace(reason="episode_done")

        return obs, reward, terminated, truncated, info

    # ---------- metrics API (trainer calls these) ----------
    def push_step_metrics(self, reward=None):
        if reward is not None:
            self._step_rewards.append(float(reward) * 100)
        
    def set_reward_display_mode(self, mode: str):
        """Set reward display for visualization: 'sigmoid' or 'continuous'."""
        self._reward_display_mode = mode

    def set_rollout_metrics(self, rewards=None, returns=None, advs=None):
        # rewards are online; returns/advs are post
        if rewards is not None:
            self._step_rewards = list(np.asarray(rewards, dtype=np.float32) * 100)
    
        if returns is not None:
            self._post_returns = list(returns)
            self._post_metrics_ready = True
    
        if advs is not None:
            self._post_advs = list(advs)
            self._post_metrics_ready = True

    # ---------------- render (ONLY this stays here) ----------------
    def render(self):
        if self.render_mode not in (None, "rgb_array"):
            raise NotImplementedError
        return render_frame(self, HEIGHT, WIDTH)

    # ---------- trainer hooks ----------
    def update_meta(self, meta: dict):
        m = meta[0] if isinstance(meta, (list, tuple)) else meta

        if m.get("rgb_pred", None) is not None:
            self._pred_rgb = np.asarray(m["rgb_pred"])
        if m.get("rgb_gt", None) is not None:
            self._gt_rgb = np.asarray(m["rgb_gt"])
            
        md = m.get("mode", None)
        if md is not None:
            self._last_mode = int(md)    

        lg = m.get("policy_logits", None)
        if lg is not None:
            self._last_logits = np.asarray(lg, dtype=np.float32).reshape(-1)
            logits = self._last_logits
            logits_shifted = logits - np.max(logits)
            p = np.exp(logits_shifted)
            p = p / (p.sum() + 1e-8)
            ent = -float(np.sum(p * np.log(p + 1e-8)))
            self._step_entropies.append(ent)

        if m.get("unc", None) is not None:
            self._unc = m["unc"]

        point_cloud_progress = m.get("point_cloud_topdown_progress", None)
        if point_cloud_progress is not None:
            points = np.asarray(point_cloud_progress.get("points", []), dtype=np.float32)
            min_distances = np.asarray(
                point_cloud_progress.get("min_distances_m", []), dtype=np.float32
            ).reshape(-1)
            observed_points = point_cloud_progress.get("observed_points", None)
            if observed_points is not None:
                observed_points = np.asarray(observed_points, dtype=np.float32)
            self._point_cloud_topdown_progress = {
                "points": points,
                "min_distances_m": min_distances,
                "threshold_m": float(point_cloud_progress.get("threshold_m", 0.05)),
                "step": int(point_cloud_progress.get("step", 0)),
                "max_step": int(point_cloud_progress.get("max_step", max(self._t, 1))),
                "observed_points": observed_points,
            }

    # ---------------- topdown helpers needed by utils ----------------
    def _get_agent_yaw(self) -> float:
        agent = self.sim.get_agent(0)
        st = agent.get_state()
        q = np.array(quat_to_coeffs(st.rotation), np.float32)
        R = quat_to_rotation_matrix(q)
        fwd = -R[:, 2]  # world forward
        return float(np.arctan2(fwd[2], fwd[0]))

    def get_trajectory_positions(self):
        """Called by trainer to get current trajectory."""
        return self._trajectory_positions

    def get_scene_and_agent_state(self):
        """
        Return (scene_dict, position, rotation) for replay/reset to same pose.
        Used when resampling truncated rollouts from same start.
        """
        agent = self.sim.get_agent(0)
        st = agent.get_state()
        pos = np.array(st.position, dtype=np.float32)
        rot = np.array(quat_to_coeffs(st.rotation), dtype=np.float32)
        return {"scene": self._scene_dict, "position": pos, "rotation": rot}

    def update_topdown_from_splat(self, data):
        """
        data: dict with 'rgb' (H,W,3) array and 'metadata' dict
        """
        if isinstance(data, list):
            data = data[0]
        
        self._splat_topdown_rgb = np.asarray(data['rgb'])
        self._splat_topdown_meta = data['metadata']    

    # ---------- Internals ----------
    def _make_sim(self, glb_path: str, navmesh_path: Optional[str] = None) -> habitat_sim.Simulator:
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.enable_physics = True
        sim_cfg.scene_id = glb_path
        sim_cfg.gpu_device_id = self.gpu_id
        sim_cfg.load_semantic_mesh = False  # HM3D GLBs do not include semantic mesh files

        rgb_spec = habitat_sim.CameraSensorSpec()
        rgb_spec.uuid = "color"
        rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
        rgb_spec.resolution = [HEIGHT, WIDTH]
        rgb_spec.position = [0.0, AGENT_HEIGHT, 0.0]
        rgb_spec.hfov = HFOV_DEG

        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = [HEIGHT, WIDTH]
        depth_spec.position = [0.0, AGENT_HEIGHT, 0.0]
        depth_spec.hfov = HFOV_DEG

        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_spec, depth_spec]
        agent_cfg.height = AGENT_HEIGHT
        agent_cfg.radius = AGENT_RADIUS

        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        sim = habitat_sim.Simulator(cfg)
        if navmesh_path and os.path.isfile(navmesh_path):
            if not sim.pathfinder.load_nav_mesh(navmesh_path):
                raise RuntimeError(f"Failed to load navmesh: {navmesh_path}")
        else:
            settings = habitat_sim.nav.NavMeshSettings()
            settings.cell_size = 0.05
            settings.cell_height = 0.2
            settings.agent_radius = AGENT_RADIUS
            settings.agent_height = AGENT_HEIGHT
            settings.agent_max_climb = 0.2
            settings.agent_max_slope = 45.0
            ok = sim.recompute_navmesh(sim.pathfinder, settings, include_static_objects=True)
            if not ok:
                raise RuntimeError("Navmesh recompute failed.")

        if not sim.pathfinder.is_loaded:
            raise RuntimeError("Navmesh not loaded/recomputed.")

        sim.pathfinder.seed(random.randint(0, 2**31 - 1))  
        
        return sim

    def _randomize_start(self, max_tries: int = 128, fixed_start_pose: bool = False):
        agent = self.sim.get_agent(0)
        pf = self.sim.pathfinder

        for _ in range(max_tries):
            p = pf.get_random_navigable_point()

            if np.any(np.isnan(p)) or np.any(np.isinf(p)):
                continue

            snapped = pf.snap_point(p)
            if np.any(np.isnan(snapped)):
                continue

            cam_pos = np.array([p[0], p[1], p[2]], dtype=np.float32)

            if fixed_start_pose:
                yaw = 0.0  # deterministic
            else:
                yaw = random.uniform(-math.pi, math.pi)
            pitch = 0.0
            q_yaw = quat_from_angle_axis(yaw, np.array([0, 1, 0], np.float32))
            q_pitch = quat_from_angle_axis(pitch, np.array([1, 0, 0], np.float32))
            q_new = q_yaw * q_pitch

            agent_pos = cam_pos.astype(np.float32)

            if not self._all_directions_clear(agent_pos, q_new, radius=COLLISION_RADIUS, uuid="depth"):
                continue

            st = agent.get_state()
            st.position = agent_pos
            st.rotation = q_new
            agent.set_state(st)
            return st

        return None

    def _get_cam_basis_and_pos(self):
        agent = self.sim.get_agent(0)
        st = agent.get_state()
        q = np.array(quat_to_coeffs(st.rotation), np.float32)
        R = quat_to_rotation_matrix(q)  # cols: right, up, back
        right, fwd = R[:, 0], -R[:, 2]
        cam_pos = np.array(st.position, np.float32) + R @ np.array([0, AGENT_HEIGHT, 0], np.float32)
        return right, fwd, R, cam_pos

    def _check_sphere_collision(self, center: np.ndarray, radius: float, fwd, right) -> bool:
        origin = mn.Vector3(float(center[0]), float(center[1]), float(center[2]))

        fwd_np = np.asarray(fwd, dtype=np.float32)
        right_np = np.asarray(right, dtype=np.float32)
        up_np = np.cross(right_np, fwd_np)

        coeffs = []
        for cx in (-1, 0, 1):
            for cy in (-1, 0, 1):
                for cz in (-1, 0, 1):
                    if cx == 0 and cy == 0 and cz == 0:
                        continue
                    coeffs.append((cx, cy, cz))

        for cx, cy, cz in coeffs:
            d_np = cx * right_np + cy * up_np + cz * fwd_np
            n = np.linalg.norm(d_np)
            if n < 1e-6:
                continue
            d_unit = d_np / n
            d_vec = mn.Vector3(float(d_unit[0]), float(d_unit[1]), float(d_unit[2]))

            ray_out = habitat_sim.geo.Ray(origin, d_vec)
            hit_out = self.sim.cast_ray(ray_out, max_distance=radius * 2)
            if hit_out.has_hits():
                return True

            start_in = origin + d_vec * radius
            ray_in = habitat_sim.geo.Ray(start_in, -d_vec)
            hit_in = self.sim.cast_ray(ray_in, max_distance=radius * 2) # !!!
            if hit_in.has_hits():
                return True

        return False

    def _sensor_pos_inside_geometry(self, agent_pos: np.ndarray) -> bool:
        """Raycast pre-check: detect if the sensor position is inside mesh geometry.

        Casts rays in 6 cardinal directions from the sensor mount point. If any
        ray hits geometry within MIN_CLEARANCE the position is almost certainly
        inside a wall/floor/ceiling and would crash the renderer.
        """
        sensor_pos = agent_pos.astype(np.float32) + np.array(
            [0, AGENT_HEIGHT, 0], dtype=np.float32
        )
        if np.any(np.isnan(sensor_pos)) or np.any(np.isinf(sensor_pos)):
            return True

        origin = mn.Vector3(
            float(sensor_pos[0]), float(sensor_pos[1]), float(sensor_pos[2])
        )
        min_clearance = 0.02
        for d in (
            mn.Vector3(1, 0, 0),
            mn.Vector3(-1, 0, 0),
            mn.Vector3(0, 1, 0),
            mn.Vector3(0, -1, 0),
            mn.Vector3(0, 0, 1),
            mn.Vector3(0, 0, -1),
        ):
            ray = habitat_sim.geo.Ray(origin, d)
            hit = self.sim.cast_ray(ray, max_distance=min_clearance)
            if hit.has_hits():
                return True
        return False

    ########### new #############
    def _hole_mismatch_check(self, depth, sensor_pos, q_rot):
        """Compare rendered depth vs Bullet raycast. If Bullet hits something
        much closer than what the renderer shows, we're looking through a
        backface-culled hole. Returns True if hole detected."""
        H, W = depth.shape[:2]
        fx, fy, cx, cy = self.k_vec

        q_arr = np.array(quat_to_coeffs(q_rot), np.float32)
        R_check = quat_to_rotation_matrix(q_arr)
        RS = R_check @ np.diag([1, -1, -1]).astype(np.float32)

        origin = mn.Vector3(float(sensor_pos[0]), float(sensor_pos[1]), float(sensor_pos[2]))

        mismatch = 0
        total = 0

        for vi in np.linspace(H // 4, 3 * H // 4, 4, dtype=int):
            for ui in np.linspace(W // 4, 3 * W // 4, 5, dtype=int):
                rz = depth[vi, ui]
                if not (np.isfinite(rz) and 0.3 < rz < 5.0):
                    continue

                d_cam = np.array([(ui - cx) / fx, (vi - cy) / fy, 1.0], np.float32)
                d_world = RS @ d_cam
                d_world /= np.linalg.norm(d_world) + 1e-8

                d_vec = mn.Vector3(float(d_world[0]), float(d_world[1]), float(d_world[2]))
                ray = habitat_sim.geo.Ray(origin, d_vec)
                hit = self.sim.cast_ray(ray, max_distance=rz + 0.5)

                total += 1
                if hit.has_hits() and hit.hits[0].ray_distance < rz - 0.3:
                    mismatch += 1

        # if total >= 3 and (mismatch / total) > 0.15:
        if total >= 1 and mismatch >= 1:
            return True
        return False
    #############################

    def _all_directions_clear(
        self,
        cam_world_pos,
        q_new,
        radius=0.5,
        uuid="depth",
        check_holes: Optional[bool] = None,
        debug_context: Optional[Dict[str, Any]] = None,
    ):
        agent = self.sim.get_agent(0)
        st_orig = agent.get_state()
        check_holes = self.check_holes if check_holes is None else bool(check_holes)
        debug_context = dict(debug_context or {})
        debug_context.update(
            {
                "cam_world_pos": np.asarray(cam_world_pos, dtype=np.float32),
                "radius": float(radius),
                "sensor_uuid": uuid,
            }
        )
        if q_new is not None:
            debug_context["proposed_rotation"] = np.asarray(quat_to_coeffs(q_new), dtype=np.float32)

        if self._sensor_pos_inside_geometry(cam_world_pos):
            self._debug_log(
                "clearance_inside_geometry",
                force=True,
                **debug_context,
                **self._current_agent_debug_state(),
            )
            return False

        st = agent.get_state()
        st.position = cam_world_pos.astype(np.float32)
        if q_new is not None:
            st.rotation = q_new

        try:
            agent.set_state(st)
            obs = self._safe_get_sensor_observations(
                caller="_all_directions_clear",
                debug_context=debug_context,
            )
            depth = obs[uuid]
        except Exception as exc:
            self._debug_log(
                "clearance_sensor_exception",
                force=True,
                exception_type=type(exc).__name__,
                exception_message=str(exc),
                **debug_context,
                **self._current_agent_debug_state(),
            )
            agent.set_state(st_orig)
            return False

        valid = np.isfinite(depth) & (depth > 0)
        if valid.sum() < 0.6 * depth.size:
            self._debug_log(
                "clearance_invalid_depth_fraction",
                force=True,
                valid_count=int(valid.sum()),
                total_count=int(depth.size),
                **debug_context,
            )
            agent.set_state(st_orig)
            return False

        ok = float(depth[valid].min()) >= radius
        
        # Hole mismatch check: reject positions where renderer sees through
        # a backface-culled hole but Bullet raycasts hit it.
        if ok and check_holes:
            q_check = q_new if q_new is not None else st_orig.rotation
            R_q = quat_to_rotation_matrix(np.array(quat_to_coeffs(q_check), np.float32))
            sensor_pos = np.array(cam_world_pos, np.float32) + \
                R_q @ np.array([0, AGENT_HEIGHT, 0], np.float32)
            if self._hole_mismatch_check(depth, sensor_pos, q_check):
                ok = False

        if self._debug_habitat_verbose:
            self._debug_log(
                "clearance_result",
                min_depth=float(depth[valid].min()),
                valid_count=int(valid.sum()),
                result=bool(ok),
                **debug_context,
            )
        agent.set_state(st_orig)
        return ok

    def _try_move_local(self, dx=0.0, dy=0.0, dz=0.0) -> bool:
        agent = self.sim.get_agent(0)
        st0 = agent.get_state()

        right, fwd, R, cam_pos = self._get_cam_basis_and_pos()
        delta = dx * right + dz * fwd + dy * np.array([0, 1, 0], np.float32)

        if np.linalg.norm(delta) < 1e-8:
            return False

        step_size = 0.05
        num_steps = max(5, int(np.linalg.norm(delta) / step_size))

        for i in range(1, num_steps + 1):
            t = i / num_steps
            interp_cam = cam_pos + delta * t
            if self._check_sphere_collision(interp_cam, COLLISION_RADIUS, fwd, right):
                return False

            check_cam = (interp_cam - (R @ np.array([0, AGENT_HEIGHT, 0], np.float32))).astype(np.float32)
            if not self._all_directions_clear(
                check_cam,
                None,
                radius=COLLISION_RADIUS,
                uuid="depth",
                check_holes=self.check_holes and (i == num_steps),
                debug_context={
                    "source": "_try_move_local",
                    "interp_index": i,
                    "interp_fraction": float(t),
                    "interp_cam": interp_cam.astype(np.float32),
                    "move_delta": delta.astype(np.float32),
                    "num_interp_steps": int(num_steps),
                },
            ):
                return False

        end_cam = cam_pos + delta

        st1 = agent.get_state()
        st1.position = (end_cam - (R @ np.array([0, AGENT_HEIGHT, 0], np.float32))).astype(np.float32)
        agent.set_state(st1)
        return True

    def _apply_rotation(self, delta_yaw_deg=0.0, delta_pitch_deg=0.0):
        agent = self.sim.get_agent(0)
        st0 = agent.get_state()

        right, fwd, R, cam_pos = self._get_cam_basis_and_pos()

        q_yaw = quat_from_angle_axis(
            math.radians(delta_yaw_deg),
            np.array([0.0, 1.0, 0.0], np.float32)
        )
        q_tmp = q_yaw * st0.rotation

        R_tmp = quat_to_rotation_matrix(np.array(quat_to_coeffs(q_tmp), np.float32))
        fwd_new = -R_tmp[:, 2]
        pitch_now = math.atan2(float(fwd_new[1]), math.hypot(float(fwd_new[0]), float(fwd_new[2])))
        pitch_target = np.clip(
            pitch_now + math.radians(delta_pitch_deg),
            -math.radians(PITCH_LIMIT_DEG),
            +math.radians(PITCH_LIMIT_DEG),
        )
        dp_apply = pitch_target - pitch_now

        if abs(dp_apply) > 1e-8:
            right_world = R_tmp[:, 0]
            q_pitch = quat_from_angle_axis(dp_apply, right_world.astype(np.float32))
            q_new = q_pitch * q_tmp
        else:
            q_new = q_tmp

        st1 = agent.get_state()
        st1.rotation = q_new

        R_new = quat_to_rotation_matrix(np.array(quat_to_coeffs(q_new), np.float32))
        agent_to_cam_offset = R_new @ np.array([0, AGENT_HEIGHT, 0], np.float32)
        st1.position = (cam_pos - agent_to_cam_offset).astype(np.float32)

        check_cam = (cam_pos - agent_to_cam_offset).astype(np.float32)
        if not self._all_directions_clear(
            check_cam,
            q_new,
            radius=COLLISION_RADIUS,
            uuid="depth",
            debug_context={
                "source": "_apply_rotation",
                "delta_yaw_deg": float(delta_yaw_deg),
                "delta_pitch_deg": float(delta_pitch_deg),
                "check_cam": check_cam,
            },
        ):
            return False

        agent.set_state(st1)
        return True

    def _get_c2w(self) -> np.ndarray:
        agent = self.sim.get_agent(0)
        st = agent.get_state()
        q = np.array(quat_to_coeffs(st.rotation), np.float32)
        R_hab = quat_to_rotation_matrix(q)
        t_agent = np.array(st.position, np.float32)
        t_cam = t_agent + R_hab @ np.array([0.0, AGENT_HEIGHT, 0.0], np.float32)
        c2w_hab = np.eye(4, dtype=np.float32)
        c2w_hab[:3, :3] = R_hab
        c2w_hab[:3, 3] = t_cam
        S = np.diag([1, -1, -1, 1]).astype(np.float32)  # Habitat→OpenCV
        return (c2w_hab @ S).astype(np.float32)

    def _observe(self):
        sim_obs = self._safe_get_sensor_observations(caller="_observe")
        return self._format_sim_obs(sim_obs)

    def _safe_get_sensor_observations(self, caller: str = "_observe", debug_context: Optional[Dict[str, Any]] = None):
        debug_context = debug_context or {}
        self._debug_capture_sensor_context(caller=caller, debug_context=debug_context)
        if self._debug_habitat_verbose:
            self._debug_log(
                "sensor_read_start",
                caller=caller,
                debug_context=debug_context,
                **self._current_agent_debug_state(),
            )
        sim_obs = self.sim.get_sensor_observations()
        if self._debug_habitat_verbose:
            depth = sim_obs.get("depth")
            depth_summary: Dict[str, Any] = {}
            if depth is not None:
                depth_np = np.asarray(depth)
                finite = np.isfinite(depth_np)
                depth_summary = {
                    "shape": list(depth_np.shape),
                    "finite_count": int(finite.sum()),
                    "total_count": int(depth_np.size),
                }
                if finite.any():
                    depth_summary["min"] = float(depth_np[finite].min())
                    depth_summary["max"] = float(depth_np[finite].max())
            self._debug_log(
                "sensor_read_done",
                caller=caller,
                debug_context=debug_context,
                depth_summary=depth_summary,
            )
        return sim_obs

    def _format_sim_obs(self, sim_obs):
        rgb = sim_obs["color"]
        depth = sim_obs["depth"]

        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]   

        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        depth = np.ascontiguousarray(depth, dtype=np.float32)
   
        return {
            "rgb": rgb,
            "depth": depth,
            "fxfycxcy": self.k_vec.astype(np.float32, copy=True),
            "c2w": self._get_c2w(),
        }

    def close(self):
        self._write_camera_pose_trace(reason="close")
        for handle_name in ("_debug_event_handle", "_debug_snapshot_handle"):
            handle = getattr(self, handle_name, None)
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
                setattr(self, handle_name, None)
        if self.sim is not None:
            self.sim.close()
            self.sim = None
            # Give driver time to release GL context (habitat-sim #1176: rapid destroy can corrupt GPU state)
            time.sleep(2.0)
        super().close()


# ---------------- PoseProcess (unchanged from your file) ----------------
class PoseProcess:
    def __init__(self, out_hw: int = 256, scene_scale_factor: float = 1.35):
        self.out_hw = int(out_hw)
        self.scene_scale_factor = float(scene_scale_factor)

    @torch.no_grad()
    def pose_canon(self, c2w_win, device):
        E = c2w_win.shape[0]
        c2w = c2w_win.to(torch.float32)

        anchor = c2w[:, 0]
        R = anchor[:, :3, :3]
        t = anchor[:, :3, 3:4]

        avg_w2c = torch.eye(4, device=c2w.device, dtype=torch.float32).expand(E, 4, 4).clone()
        avg_w2c[:, :3, :3] = R.transpose(1, 2)
        avg_w2c[:, :3, 3] = (-avg_w2c[:, :3, :3] @ t)[..., 0]

        c2w_aligned = avg_w2c.unsqueeze(1) @ c2w
        c2w_win_norm = c2w_aligned.clone()
        return c2w_win_norm.to(device)

    @torch.no_grad()
    def preprocess_images(self, imgs_win, amp_dtype, interp="bilinear", outHW=None):
        x = imgs_win
        outHW = self.out_hw if outHW is None else outHW
        if x.dtype == torch.uint8:
            x = x.to(torch.float32) / 255.0
        else:
            x = x.to(torch.float32)
        xx = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        xx = F.interpolate(xx, size=(outHW, outHW), mode=interp)
        c = imgs_win.shape[2]
        imgs_win_norm = xx.view(x.shape[0], x.shape[1], c, outHW, outHW).to(amp_dtype, non_blocking=True)
        return imgs_win_norm

    @torch.no_grad()
    def process_window(
        self,
        imgs_win: torch.Tensor,
        k_win:    torch.Tensor,
        c2w_win:  torch.Tensor,
        device:   torch.device,
        amp_dtype: torch.dtype,
        depths_win: torch.Tensor = None,
        outHW = None
    ):
        E, T, _, H, W = imgs_win.shape
        outH = outW = self.out_hw if outHW is None else outHW

        imgs_win_norm = self.preprocess_images(imgs_win, amp_dtype, outHW=outHW).clamp_(0, 1)
        if depths_win is not None:
            depths_win_norm = self.preprocess_images(depths_win.unsqueeze(2), amp_dtype, interp="nearest", outHW=outHW).squeeze(2)
        else:
            depths_win_norm = None

        k = k_win.to(torch.float32)
        sx = float(outW) / float(W)
        sy = float(outH) / float(H)
        fx, fy, cx, cy = k[..., 0], k[..., 1], k[..., 2], k[..., 3]

        fx = fx * sx; fy = fy * sy; cx = cx * sx; cy = cy * sy
        k_win_norm = torch.stack([fx, fy, cx, cy], dim=-1).to(torch.float32)

        c2w_win_norm = self.pose_canon(c2w_win, device)
        return imgs_win_norm.to(device, non_blocking=True), k_win_norm.to(device), c2w_win_norm, depths_win_norm

    @torch.no_grad()
    def compute_rays(self, c2w, fxfycxcy, h=None, w=None, device="cuda"):
        b, v = c2w.size()[:2]
        c2w = c2w.reshape(b * v, 4, 4)

        fx, fy, cx, cy = fxfycxcy[:,:, 0], fxfycxcy[:,:,  1], fxfycxcy[:,:,  2], fxfycxcy[:,:,  3]
        h_orig = int(2 * cy.max().item()) + 1
        w_orig = int(2 * cx.max().item()) + 1

        if h is None or w is None:
            h, w = h_orig, w_orig

        if h_orig != h or w_orig != w:
            fx = fx * w / w_orig
            fy = fy * h / h_orig
            cx = cx * w / w_orig
            cy = cy * h / h_orig

        fxfycxcy = fxfycxcy.reshape(b * v, 4)
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        y, x = y.to(device), x.to(device)
        x = x[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        y = y[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
        x = (x + 0.5 - fxfycxcy[:, 2:3]) / fxfycxcy[:, 0:1]
        y = (y + 0.5 - fxfycxcy[:, 3:4]) / fxfycxcy[:, 1:2]
        z = torch.ones_like(x)
        ray_d = torch.stack([x, y, z], dim=2)
        ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))
        ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True)
        ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)

        ray_o = rearrange(ray_o, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)
        ray_d = rearrange(ray_d, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)
        return ray_o, ray_d

    def build_visual_input(self, images=None, ray_o=None, ray_d=None, method="default_plucker"):
        if method == "custom_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            pose_cond = torch.cat([ray_d, nearest_pts], dim=2)

        elif method == "aug_plucker":
            o_dot_d = torch.sum(-ray_o * ray_d, dim=2, keepdim=True)
            nearest_pts = ray_o + o_dot_d * ray_d
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d, nearest_pts], dim=2)

        else:
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            pose_cond = torch.cat([o_cross_d, ray_d], dim=2)

        if images is None:
            return pose_cond
        else:
            return torch.cat([images * 2.0 - 1.0, pose_cond], dim=2)
