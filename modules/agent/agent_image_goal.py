# Copyright (c) 2026.
# Sliding-window Space->Time transformer for image-goal PPO camera agent with DINO conditioning.
#
# This variant extends the linear-memory sliding agent with image-goal conditioning:
#   - A target RGB image is encoded into a target token.
#   - The temporal stack keeps the usual causal sliding-window processing for frame tokens.
#   - At each temporal layer, the target token reads from the sequence and the sequence
#     is conditioned by the updated target token.
#   - The target-token output is discarded at the head; only frame-token outputs are used
#     for policy/value predictions.
#

import os
import warnings
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from typing import Optional, List

from .transformer import QK_Norm_TransformerBlock, KVCache, init_weights


# =============================================================================
# Cross-attention block (unchanged from base NavAgent)
# =============================================================================
class CrossAttentionBlock(nn.Module):
    """Cross-attention block: queries attend to context"""
    def __init__(self, d_model, d_head, use_qk_norm=True):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = d_model // d_head
        assert d_model % d_head == 0, "d_model must be divisible by d_head"

        self.norm_q = nn.LayerNorm(d_model, bias=False)
        self.norm_kv = nn.LayerNorm(d_model, bias=False)

        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_kv = nn.Linear(d_model, 2 * d_model, bias=False)

        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = nn.LayerNorm(d_head, bias=False)
            self.k_norm = nn.LayerNorm(d_head, bias=False)

        self.proj = nn.Linear(d_model, d_model, bias=False)

        self.norm_ffn = nn.LayerNorm(d_model, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )

    def forward(self, x, context):
        B, N_q, D = x.shape
        N_kv = context.shape[1]

        q = self.to_q(self.norm_q(x))
        kv = self.to_kv(self.norm_kv(context))
        k, v = kv.chunk(2, dim=-1)

        q = q.reshape(B, N_q, self.n_heads, self.d_head).transpose(1, 2)
        k = k.reshape(B, N_kv, self.n_heads, self.d_head).transpose(1, 2)
        v = v.reshape(B, N_kv, self.n_heads, self.d_head).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        attn = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N_q, D)
        out = self.proj(out)
        x = x + out

        x = x + self.ffn(self.norm_ffn(x))
        return x


# =============================================================================
# Linear attention global memory
# =============================================================================
class LinearMemState:
    """Rollout state for linear-attention global memory.

    Holds one running outer-product state S and a running normalizer z per memory
    layer. Duck-types minimally so it can be created/cleared like a KVCache.

    Shapes per layer:
        S:  [B, d_k, d_v]
        z:  [B, d_k]
    """
    def __init__(self, num_layers: int, batch_size: int, d_k: int, d_v: int,
                 device=None, dtype=None):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.d_k = d_k
        self.d_v = d_v
        self.device = device
        self.dtype = dtype
        self.S: List[Optional[torch.Tensor]] = [None] * num_layers
        self.z: List[Optional[torch.Tensor]] = [None] * num_layers

    def clear(self):
        self.S = [None] * self.num_layers
        self.z = [None] * self.num_layers


class LinearAttentionMemory(nn.Module):
    """Linear-attention global memory layer.

    Maintains S_t = sum_{i<=t} phi(k_i) v_i^T and reads with phi(q_t).
    Uses "read BEFORE update" semantics so a token can't attend to itself.

    Training (parallel path): uses cumsum to compute S_{t-1} for every t in O(V) work
    and O(log V) span. Fully differentiable.

    Rollout (incremental path): maintains S as a LinearMemState and does one step.
    """
    def __init__(self, d_model: int, d_k: int = 64, d_v: int = 64,
                 eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.eps = eps

        self.to_q = nn.Linear(d_model, d_k, bias=False)
        self.to_k = nn.Linear(d_model, d_k, bias=False)
        self.to_v = nn.Linear(d_model, d_v, bias=False)
        self.readout = nn.Linear(d_v, d_model, bias=False)

        self.in_norm = nn.LayerNorm(d_model, bias=False)

        # Zero-init the readout so the layer starts as an identity (no-op) contribution.
        # This is important so that adding this module to a pretrained temporal stack
        # doesn't immediately destabilize training.
        nn.init.normal_(self.to_q.weight, std=0.02)
        nn.init.normal_(self.to_k.weight, std=0.02)
        nn.init.normal_(self.to_v.weight, std=0.02)
        nn.init.zeros_(self.readout.weight)

    @staticmethod
    def _phi(x):
        # Positive feature map (Performer-style but cheap). ELU+1 is standard for
        # linear attention and keeps the normalizer strictly positive.
        return torch.nn.functional.elu(x) + 1.0

    def forward_parallel(self, x: torch.Tensor) -> torch.Tensor:
        """Training path: process a full [B, V, D] sequence in parallel.

        Returns a residual [B, V, D] to be added to the input by the caller.
        Implements read-before-update via an exclusive prefix sum (S_prev = shift(cumsum)).
        """
        B, V, D = x.shape
        h = self.in_norm(x)
        q = self._phi(self.to_q(h))  # [B, V, d_k]
        k = self._phi(self.to_k(h))  # [B, V, d_k]
        v = self.to_v(h)             # [B, V, d_v]

        # Per-step contributions to S and z.
        # outer[b,t] = k[b,t].unsqueeze(-1) * v[b,t].unsqueeze(-2)  -> [d_k, d_v]
        outer = k.unsqueeze(-1) * v.unsqueeze(-2)  # [B, V, d_k, d_v]
        S_incl = outer.cumsum(dim=1)               # inclusive cumsum: S_incl[:,t] = sum_{i<=t}
        z_incl = k.cumsum(dim=1)                   # [B, V, d_k]

        # Exclusive: shift by one so we "read before update".
        zero_S = torch.zeros_like(S_incl[:, :1])
        zero_z = torch.zeros_like(z_incl[:, :1])
        S_prev = torch.cat([zero_S, S_incl[:, :-1]], dim=1)  # [B, V, d_k, d_v]
        z_prev = torch.cat([zero_z, z_incl[:, :-1]], dim=1)  # [B, V, d_k]

        # Readout m_t = (q_t^T S_prev_t) / (q_t^T z_prev_t + eps)
        num = torch.einsum('bvk, bvkd -> bvd', q, S_prev)       # [B, V, d_v]
        den = torch.einsum('bvk, bvk  -> bv',  q, z_prev)       # [B, V]
        m = num / (den.unsqueeze(-1) + self.eps)                # [B, V, d_v]

        return self.readout(m)  # [B, V, D]

    def forward_step(self, x_t: torch.Tensor, state: LinearMemState,
                     layer_idx: int) -> torch.Tensor:
        """Rollout path: process one frame [B, D] with incremental state.

        Reads global memory BEFORE updating it with the current token (matches the
        parallel path's exclusive-prefix-sum semantics).

        Returns a residual [B, D] to be added to the input by the caller.
        """
        B, D = x_t.shape
        h = self.in_norm(x_t)
        q = self._phi(self.to_q(h))  # [B, d_k]
        k = self._phi(self.to_k(h))  # [B, d_k]
        v = self.to_v(h)             # [B, d_v]

        S = state.S[layer_idx]  # [B, d_k, d_v] or None
        z = state.z[layer_idx]  # [B, d_k]     or None

        # READ (from S_{t-1}, z_{t-1}) -- before update.
        if S is None:
            m = torch.zeros(B, self.d_v, device=x_t.device, dtype=x_t.dtype)
        else:
            num = torch.einsum('bk, bkd -> bd', q, S)       # [B, d_v]
            den = torch.einsum('bk, bk  -> b',  q, z)       # [B]
            m = num / (den.unsqueeze(-1) + self.eps)        # [B, d_v]

        # UPDATE state with current token: S_t = S_{t-1} + k v^T ; z_t = z_{t-1} + k
        new_outer = k.unsqueeze(-1) * v.unsqueeze(-2)  # [B, d_k, d_v]
        if S is None:
            state.S[layer_idx] = new_outer
            state.z[layer_idx] = k
        else:
            state.S[layer_idx] = S + new_outer
            state.z[layer_idx] = z + k

        return self.readout(m)  # [B, D]


# =============================================================================
# NavAgent with linear-attention global memory
# =============================================================================
class NavAgent(nn.Module):
    def __init__(
        self,
        image_size=64,
        patch_size=32,
        in_channels=15,
        d_model=768,
        d_head=64,
        n_layer_spatial_cross=4,
        n_layer_temporal=16,
        use_qk_norm=True,
        n_act=4,
        max_v=512,
        attn_window=64,
        checkpoint_every=0,
        pos_init_std=0.02,
        dino_model_name="dinov2_vitb14",
        dino_freeze=True,
        frame_token_mode="learnable_query",
        # ----- NEW: global memory kwargs (all optional) -----
        use_global_memory=True,
        global_mem_layers=None,   # list of temporal layer indices that get a mem block; defaults to every 4th
        global_mem_d_k=64,
        global_mem_d_v=64,
        goal_rgb_channels=3,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model
        self.d_head = d_head
        self.n_layer_spatial_cross = n_layer_spatial_cross
        self.n_layer_temporal = n_layer_temporal
        self.use_qk_norm = use_qk_norm
        self.n_act = n_act
        self.max_v = max_v
        self.attn_window = int(attn_window) if attn_window is not None else None
        self.checkpoint_every = int(checkpoint_every) if checkpoint_every else 0
        self.dino_freeze = dino_freeze
        self.frame_token_mode = frame_token_mode

        self.use_global_memory = bool(use_global_memory)
        if global_mem_layers is None:
            # Default: insert a memory block every 4 temporal layers (4 blocks for n=16).
            global_mem_layers = list(range(3, n_layer_temporal, 4))
        self.global_mem_layers = list(global_mem_layers) if self.use_global_memory else []
        self.global_mem_d_k = int(global_mem_d_k)
        self.global_mem_d_v = int(global_mem_d_v)
        self.goal_rgb_channels = int(goal_rgb_channels)
        self._episode_goal_rgb: Optional[torch.Tensor] = None
        self._episode_goal_cursor: int = 0

        # Patch grid
        hh = self.image_size // self.patch_size
        ww = self.image_size // self.patch_size
        assert hh * self.patch_size == self.image_size and ww * self.patch_size == self.image_size
        self.hh = hh
        self.ww = ww
        self.n_patches = hh * ww

        # ============================================
        # DINO Feature Extractor
        # ============================================
        # Pin explicit ref + skip validation so cached hub loads do not probe GitHub
        # avoids probing GitHub on every rank at startup.
        self.dino = torch.hub.load(
            "facebookresearch/dinov2:main",
            dino_model_name,
            skip_validation=True,
            trust_repo=True,
        )
        self.dino_d_model = self.dino.embed_dim

        if dino_freeze:
            for param in self.dino.parameters():
                param.requires_grad = False
            self.dino.eval()

        if self.dino_d_model != d_model:
            self.dino_proj = nn.Linear(self.dino_d_model, d_model, bias=False)
            self.dino_proj.apply(init_weights)
        else:
            self.dino_proj = nn.Identity()

        # ============================================
        # Patch embeddings
        # ============================================
        self.patch_embed = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> (b v) (hh ww) (ph pw c)",
                ph=patch_size,
                pw=patch_size,
            ),
            nn.Linear(in_channels * (patch_size ** 2), d_model, bias=False),
        )
        self.patch_embed.apply(init_weights)

        self.spatial_pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.normal_(self.spatial_pos, std=pos_init_std)

        # ============================================
        # Frame Token Generation (mode-dependent) -- unchanged
        # ============================================
        if frame_token_mode == "learnable_query":
            self.frame_query = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.frame_query, std=pos_init_std)
            self.spatial_in_ln = nn.LayerNorm(d_model, bias=False)
            self.spatial_cross_blocks = nn.ModuleList([
                CrossAttentionBlock(d_model, d_head, use_qk_norm=use_qk_norm)
                for _ in range(n_layer_spatial_cross)
            ])
            for blk in self.spatial_cross_blocks:
                blk.apply(init_weights)
        elif frame_token_mode == "dino_cls_only":
            self.spatial_in_ln = nn.LayerNorm(d_model, bias=False)
            self.spatial_cross_blocks = nn.ModuleList([
                CrossAttentionBlock(d_model, d_head, use_qk_norm=use_qk_norm)
                for _ in range(n_layer_spatial_cross)
            ])
            for blk in self.spatial_cross_blocks:
                blk.apply(init_weights)
            self.frame_pool = nn.Sequential(
                nn.LayerNorm(d_model, bias=False),
                nn.Linear(d_model, d_model, bias=False),
            )
            self.frame_pool.apply(init_weights)
        else:  # mean_pool
            self.spatial_in_ln = nn.LayerNorm(d_model, bias=False)
            self.spatial_cross_blocks = nn.ModuleList([
                CrossAttentionBlock(d_model, d_head, use_qk_norm=use_qk_norm)
                for _ in range(n_layer_spatial_cross)
            ])
            for blk in self.spatial_cross_blocks:
                blk.apply(init_weights)
            self.frame_pool = nn.Sequential(
                nn.LayerNorm(d_model, bias=False),
                nn.Linear(d_model, d_model, bias=False),
            )
            self.frame_pool.apply(init_weights)

        # ============================================
        # Temporal Transformer Blocks
        # ============================================
        self.temporal_in_ln = nn.LayerNorm(d_model, bias=False)
        self.temporal_blocks = nn.ModuleList([
            QK_Norm_TransformerBlock(
                d_model,
                d_head,
                use_qk_norm=use_qk_norm,
                causal=True,
                sliding_window_size=self.attn_window,
            )
            for _ in range(n_layer_temporal)
        ])
        for blk in self.temporal_blocks:
            blk.apply(init_weights)

        # ============================================
        # Global memory modules (one per entry in global_mem_layers)
        # ============================================
        # We use a ModuleDict keyed by string layer index so that load_state_dict
        # works naturally when global_mem_layers is a subset of layers.
        self.global_mem = nn.ModuleDict()
        if self.use_global_memory:
            for idx in self.global_mem_layers:
                assert 0 <= idx < self.n_layer_temporal, f"bad global_mem_layers entry {idx}"
                self.global_mem[str(idx)] = LinearAttentionMemory(
                    d_model=d_model, d_k=self.global_mem_d_k, d_v=self.global_mem_d_v
                )

        # ============================================
        # PPO Heads
        # ============================================
        self.head_ln = nn.LayerNorm(d_model, bias=False)
        self.pi_head = nn.Linear(d_model, n_act, bias=True)
        self.v_head = nn.Linear(d_model, 1, bias=True)

        nn.init.normal_(self.pi_head.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.pi_head.bias)
        nn.init.normal_(self.v_head.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.v_head.bias)

    # -------------------------
    # KV Cache Management (unchanged signatures)
    # -------------------------
    def init_kv_cache(self, batch_size: int = 1) -> List[KVCache]:
        # Cache only main-sequence tokens. During rollout we temporarily append goal as an
        # ephemeral token for attention, then remove it from cache immediately.
        base_len = self.attn_window if (self.attn_window is not None and self.attn_window > 0) else self.max_v
        max_len = max(2, int(base_len) + 2)
        return [KVCache(max_seq_len=max_len) for _ in range(self.n_layer_temporal)]

    def clear_kv_cache(self, caches: List[KVCache]):
        for cache in caches:
            cache.clear()

    # -------------------------
    # Global memory state management (NEW)
    # -------------------------
    def init_global_state(self, batch_size: int = 1, device=None, dtype=None) -> LinearMemState:
        """Create a fresh global-memory rollout state. Call once per rollout, alongside
        init_kv_cache(), and pass the returned object into forward_step(global_state=...).
        """
        return LinearMemState(
            num_layers=self.n_layer_temporal,
            batch_size=batch_size,
            d_k=self.global_mem_d_k,
            d_v=self.global_mem_d_v,
            device=device,
            dtype=dtype,
        )

    def clear_global_state(self, state: LinearMemState):
        if state is not None:
            state.clear()

    def set_episode_goal_rgb(self, goal_rgb: torch.Tensor):
        self._episode_goal_rgb = goal_rgb.detach()
        self._episode_goal_cursor = 0

    def clear_episode_goal_rgb(self):
        self._episode_goal_rgb = None
        self._episode_goal_cursor = 0

    def _consume_episode_goal_rgb(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if self._episode_goal_rgb is None:
            return None
        goals = self._episode_goal_rgb
        if goals.device != device:
            goals = goals.to(device=device)
        if goals.dtype != dtype:
            goals = goals.to(dtype=dtype)
        total = int(goals.shape[0])
        if total == batch_size:
            return goals
        if total > batch_size:
            start = int(self._episode_goal_cursor)
            end = start + int(batch_size)
            if end > total:
                start = 0
                end = int(batch_size)
            self._episode_goal_cursor = int(end)
            return goals[start:end]
        reps = (int(batch_size) + total - 1) // total
        tiled = goals.repeat(reps, 1, 1, 1)
        return tiled[:batch_size]

    # -------------------------
    # DINO Feature Extraction (unchanged)
    # -------------------------
    def _extract_dino_features(self, rgb_images):
        B, C, H, W = rgb_images.shape
        target_size = 224
        if H != target_size or W != target_size:
            rgb_resized = torch.nn.functional.interpolate(
                rgb_images, size=(target_size, target_size),
                mode='bilinear', align_corners=False
            )
        else:
            rgb_resized = rgb_images

        if self.dino_freeze:
            with torch.no_grad():
                features = self.dino.forward_features(rgb_resized)
        else:
            features = self.dino.forward_features(rgb_resized)

        if isinstance(features, dict):
            patch_tokens = features['x_norm_patchtokens']
            cls_token = features['x_norm_clstoken']
        else:
            cls_token = features[:, 0, :]
            patch_tokens = features[:, 1:, :]

        dino_cls = self.dino_proj(cls_token)
        dino_patches = self.dino_proj(patch_tokens)
        return dino_cls, dino_patches

    # -------------------------
    # Spatial Encoding (unchanged)
    # -------------------------
    def _encode_frame(self, posed_image):
        B = posed_image.shape[0]
        rgb = posed_image[:, :3, :, :]
        dino_cls, dino_patches = self._extract_dino_features(rgb)

        x = posed_image.unsqueeze(1)
        patch_tok = self.patch_embed(x)
        patch_tok = patch_tok + self.spatial_pos[:, :patch_tok.shape[1], :]

        if self.frame_token_mode == "learnable_query":
            context = torch.cat([
                dino_cls.unsqueeze(1),
                dino_patches,
                patch_tok,
            ], dim=1)
            query = self.frame_query.expand(B, 1, self.d_model)
            query = self.spatial_in_ln(query)
            for blk in self.spatial_cross_blocks:
                query = blk(query, context=context)
            patch_summary = patch_tok.mean(dim=1)
            frame_tok = query.squeeze(1) + 0.5 * patch_summary
        elif self.frame_token_mode == "dino_cls_only":
            context = dino_cls.unsqueeze(1)
            patch_tok = self.spatial_in_ln(patch_tok)
            for blk in self.spatial_cross_blocks:
                patch_tok = blk(patch_tok, context=context)
            frame_tok = patch_tok.mean(dim=1)
            frame_tok = self.frame_pool(frame_tok)
        else:  # mean_pool
            context = torch.cat([
                dino_cls.unsqueeze(1),
                dino_patches,
            ], dim=1)
            patch_tok = self.spatial_in_ln(patch_tok)
            for blk in self.spatial_cross_blocks:
                patch_tok = blk(patch_tok, context=context)
            frame_tok = patch_tok.mean(dim=1)
            frame_tok = self.frame_pool(frame_tok)

        return frame_tok

    def _encode_goal_token(self, goal_rgb):
        B, C, H, W = goal_rgb.shape
        assert C == self.goal_rgb_channels
        assert H == self.image_size and W == self.image_size

        dino_cls, dino_patches = self._extract_dino_features(goal_rgb)
        # Reuse the original pose patch embed by padding goal RGB with zero dummy
        # channels for the missing plucker inputs.
        posed_goal = torch.zeros(
            B,
            1,
            self.in_channels,
            H,
            W,
            device=goal_rgb.device,
            dtype=goal_rgb.dtype,
        )
        posed_goal[:, 0, : self.goal_rgb_channels, :, :] = goal_rgb
        patch_tok = self.patch_embed(posed_goal)
        patch_tok = patch_tok + self.spatial_pos[:, :patch_tok.shape[1], :]

        if self.frame_token_mode == "learnable_query":
            context = torch.cat(
                [
                    dino_cls.unsqueeze(1),
                    dino_patches,
                    patch_tok,
                ],
                dim=1,
            )
            query = self.frame_query.expand(B, 1, self.d_model)
            query = self.spatial_in_ln(query)
            for blk in self.spatial_cross_blocks:
                query = blk(query, context=context)
            patch_summary = patch_tok.mean(dim=1)
            goal_tok = query.squeeze(1) + 0.5 * patch_summary
        elif self.frame_token_mode == "dino_cls_only":
            context = dino_cls.unsqueeze(1)
            patch_tok = self.spatial_in_ln(patch_tok)
            for blk in self.spatial_cross_blocks:
                patch_tok = blk(patch_tok, context=context)
            goal_tok = patch_tok.mean(dim=1)
            goal_tok = self.frame_pool(goal_tok)
        else:
            context = torch.cat(
                [
                    dino_cls.unsqueeze(1),
                    dino_patches,
                ],
                dim=1,
            )
            patch_tok = self.spatial_in_ln(patch_tok)
            for blk in self.spatial_cross_blocks:
                patch_tok = blk(patch_tok, context=context)
            goal_tok = patch_tok.mean(dim=1)
            goal_tok = self.frame_pool(goal_tok)

        return goal_tok

    def _build_goal_augmented_attn_bias(
        self,
        num_main: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build [1, L, L] float bias for [main_0..main_{V-1}, goal] tokens.

        Main queries keep causal sliding-window attention over main keys, and can always
        attend to the goal key. The goal query attends to all keys.
        """
        total = int(num_main) + 1
        bias = torch.full((1, total, total), float("-inf"), device=device, dtype=torch.float32)
        goal_idx = int(num_main)

        for q in range(int(num_main)):
            if self.attn_window is None or int(self.attn_window) <= 0:
                k_start = 0
            else:
                k_start = max(0, q - int(self.attn_window) + 1)
            bias[:, q, k_start : q + 1] = 0.0
            bias[:, q, goal_idx] = 0.0

        bias[:, goal_idx, :] = 0.0
        return bias

    def _build_rollout_pair_attn_bias(
        self,
        past_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build [1, 2, past_len+2] bias for rollout pair queries [frame, goal].

        Frame query attends to a sliding subset of cached keys plus current frame+goal.
        Goal query attends to all keys.
        """
        q_len = 2
        kv_len = int(past_len) + q_len
        bias = torch.full((1, q_len, kv_len), float("-inf"), device=device, dtype=torch.float32)

        if self.attn_window is None or int(self.attn_window) <= 0:
            k_start = 0
        else:
            k_start = max(0, int(past_len) - int(self.attn_window) + 1)

        # frame query (row 0): selected cache + current frame + current goal
        bias[:, 0, k_start:kv_len] = 0.0
        # goal query (row 1): full visibility
        bias[:, 1, :] = 0.0
        return bias

    # -------------------------
    # Temporal runner -- training (parallel) path
    # -------------------------
    def _run_temporal_parallel(self, x, goal_tok: Optional[torch.Tensor] = None):
        """Run the n_layer_temporal stack over x: [B, V, D] in parallel (training).

        If goal_tok is provided, run over [main, goal] with a custom mask:
          - main tokens: causal sliding over main + always-visible goal key
          - goal token: full attention
        Only main-token outputs are returned.
        """
        if goal_tok is not None:
            num_main = x.shape[1]
            x_full = torch.cat([x, goal_tok.unsqueeze(1)], dim=1)
            attn_bias = self._build_goal_augmented_attn_bias(
                num_main=num_main,
                device=x_full.device,
            )

            if not self.training or self.checkpoint_every <= 0:
                for i, blk in enumerate(self.temporal_blocks):
                    x_full = blk(x_full, attn_bias=attn_bias, kv_cache=None)
                    if self.use_global_memory and str(i) in self.global_mem:
                        x_main = x_full[:, :num_main, :]
                        x_main = x_main + self.global_mem[str(i)].forward_parallel(x_main)
                        x_full = torch.cat([x_main, x_full[:, num_main:, :]], dim=1)
                return x_full[:, :num_main, :]

            num_layers = len(self.temporal_blocks)
            every = self.checkpoint_every

            def _group(inp, s, e):
                y = inp
                for li in range(s, e):
                    y = self.temporal_blocks[li](y, attn_bias=attn_bias, kv_cache=None)
                    if self.use_global_memory and str(li) in self.global_mem:
                        y_main = y[:, :num_main, :]
                        y_main = y_main + self.global_mem[str(li)].forward_parallel(y_main)
                        y = torch.cat([y_main, y[:, num_main:, :]], dim=1)
                return y

            for s in range(0, num_layers, every):
                e = min(s + every, num_layers)
                x_full = torch.utils.checkpoint.checkpoint(
                    lambda inp, s=s, e=e: _group(inp, s, e), x_full, use_reentrant=False
                )
            return x_full[:, :num_main, :]

        if not self.training or self.checkpoint_every <= 0:
            for i, blk in enumerate(self.temporal_blocks):
                x = blk(x, attn_bias=None, kv_cache=None)
                if self.use_global_memory and str(i) in self.global_mem:
                    x = x + self.global_mem[str(i)].forward_parallel(x)
            return x

        # Checkpointed path: group layers and inject global memory at boundaries
        # that fall inside each group. We keep group boundaries aligned so that the
        # global memory modules participate in checkpointing too.
        num_layers = len(self.temporal_blocks)
        every = self.checkpoint_every

        def _group(inp, s, e):
            y = inp
            for i in range(s, e):
                y = self.temporal_blocks[i](y, attn_bias=None, kv_cache=None)
                if self.use_global_memory and str(i) in self.global_mem:
                    y = y + self.global_mem[str(i)].forward_parallel(y)
            return y

        for s in range(0, num_layers, every):
            e = min(s + every, num_layers)
            x = torch.utils.checkpoint.checkpoint(
                lambda inp, s=s, e=e: _group(inp, s, e), x, use_reentrant=False
            )
        return x

    # -------------------------
    # Forward: Training Mode
    # -------------------------
    def forward(self, posed_images, goal_rgb=None, last_only=False):
        """
        Args:
            posed_images: [B, V, C=15, H, W]
            goal_rgb: [B, 3, H, W], optional
            last_only: If True, only return outputs for last timestep
        Returns:
            logits: [B, V, n_act] (or [B, n_act] if last_only)
            values: [B, V]        (or [B]        if last_only)
        """
        B, V, C, H, W = posed_images.shape
        assert V <= self.max_v
        assert C == self.in_channels
        assert H == self.image_size and W == self.image_size
        if goal_rgb is None:
            goal_rgb = self._consume_episode_goal_rgb(
                batch_size=B,
                device=posed_images.device,
                dtype=posed_images.dtype,
            )
        if goal_rgb is None:
            goal_rgb = torch.zeros(
                B,
                self.goal_rgb_channels,
                H,
                W,
                device=posed_images.device,
                dtype=posed_images.dtype,
            )
        goal_tok = self._encode_goal_token(goal_rgb)

        posed_flat = posed_images.reshape(B * V, C, H, W)
        rgb_flat = posed_flat[:, :3, :, :]

        dino_cls, dino_patches = self._extract_dino_features(rgb_flat)

        patch_tok = self.patch_embed(posed_images)
        patch_tok = patch_tok + self.spatial_pos[:, :patch_tok.shape[1], :]

        if self.frame_token_mode == "learnable_query":
            context = torch.cat([
                dino_cls.unsqueeze(1),
                dino_patches,
                patch_tok,
            ], dim=1)
            query = self.frame_query.expand(B * V, 1, self.d_model)
            query = self.spatial_in_ln(query)
            for blk in self.spatial_cross_blocks:
                query = blk(query, context=context)
            patch_summary = patch_tok.mean(dim=1)
            frame_tok = (query.squeeze(1) + 0.5 * patch_summary).reshape(B, V, self.d_model)
        elif self.frame_token_mode == "dino_cls_only":
            context = dino_cls.unsqueeze(1)
            patch_tok = self.spatial_in_ln(patch_tok)
            for blk in self.spatial_cross_blocks:
                patch_tok = blk(patch_tok, context=context)
            frame_tok = patch_tok.mean(dim=1)
            frame_tok = self.frame_pool(frame_tok)
            frame_tok = frame_tok.reshape(B, V, self.d_model)
        else:  # mean_pool
            context = torch.cat([
                dino_cls.unsqueeze(1),
                dino_patches,
            ], dim=1)
            patch_tok = self.spatial_in_ln(patch_tok)
            for blk in self.spatial_cross_blocks:
                patch_tok = blk(patch_tok, context=context)
            frame_tok = patch_tok.mean(dim=1)
            frame_tok = self.frame_pool(frame_tok)
            frame_tok = frame_tok.reshape(B, V, self.d_model)

        x_tm = self.temporal_in_ln(frame_tok)
        x_tm = self._run_temporal_parallel(x_tm, goal_tok=goal_tok)

        z = self.head_ln(x_tm)
        logits = self.pi_head(z)
        values = self.v_head(z).squeeze(-1)

        if last_only:
            return logits[:, -1], values[:, -1]
        return logits, values

    # -------------------------
    # Forward: Rollout Mode
    # -------------------------
    @torch.no_grad()
    def forward_step(
        self,
        posed_image,
        time_idx: int,
        kv_caches: List[KVCache],
        global_state: Optional[LinearMemState] = None,
        goal_rgb: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            posed_image: [B, C=15, H, W]
            time_idx: Current timestep
            kv_caches: List of KVCache objects (length n_layer_temporal)
            global_state: Optional LinearMemState for long-horizon global memory.
                          If None and use_global_memory=True, a fresh per-step state
                          is used (effectively disabling global memory) and a warning
                          is emitted once.
            goal_rgb: Optional [B,3,H,W] image-goal RGB.

        Returns:
            logits: [B, n_act]
            value: [B]
        """
        B, C, H, W = posed_image.shape
        assert C == self.in_channels
        assert H == self.image_size and W == self.image_size
        assert len(kv_caches) == self.n_layer_temporal
        del time_idx

        if goal_rgb is None:
            goal_rgb = torch.zeros(
                B,
                self.goal_rgb_channels,
                H,
                W,
                device=posed_image.device,
                dtype=posed_image.dtype,
            )
        goal_tok = self._encode_goal_token(goal_rgb)
        goal_seq = goal_tok.unsqueeze(1)

        if self.use_global_memory and global_state is None:
            if not getattr(self, "_warned_no_global_state", False):
                warnings.warn(
                    "NavAgent.forward_step called with global_state=None but "
                    "use_global_memory=True. Global memory will be effectively "
                    "disabled during rollout. Call init_global_state() and pass it in.",
                    stacklevel=2,
                )
                self._warned_no_global_state = True
            global_state = self.init_global_state(batch_size=B,
                                                   device=posed_image.device,
                                                   dtype=posed_image.dtype)

        # Spatial encoding
        frame_tok = self._encode_frame(posed_image)  # [B, D]
        frame_tok = frame_tok.unsqueeze(1)            # [B, 1, D]

        # Temporal with caching + goal-token augmentation + global-memory injection.
        x_tm = self.temporal_in_ln(frame_tok)
        main_cache_len = self.attn_window if (self.attn_window is not None and self.attn_window > 0) else self.max_v
        main_cache_len = max(1, int(main_cache_len))
        for i, blk in enumerate(self.temporal_blocks):
            cache = kv_caches[i]
            prev_len = cache.get_seq_len()
            pair_in = torch.cat([x_tm, goal_seq], dim=1)  # [B, 2, D]
            pair_bias = self._build_rollout_pair_attn_bias(
                past_len=prev_len,
                device=pair_in.device,
            )
            pair_out = blk(pair_in, attn_bias=pair_bias, kv_cache=cache)
            x_tm = pair_out[:, :1, :]
            goal_seq = pair_out[:, 1:, :]
            # Cache should keep only main tokens. The goal token is per-step ephemeral
            # conditioning and must not accumulate in cache.
            if cache.k_cache is not None and cache.v_cache is not None:
                cache.k_cache = cache.k_cache[:, :-1, :, :].contiguous()
                cache.v_cache = cache.v_cache[:, :-1, :, :].contiguous()
                if cache.k_cache.shape[1] > main_cache_len:
                    cache.k_cache = cache.k_cache[:, -main_cache_len:, :, :].contiguous()
                    cache.v_cache = cache.v_cache[:, -main_cache_len:, :, :].contiguous()
            if self.use_global_memory and str(i) in self.global_mem:
                mem_residual = self.global_mem[str(i)].forward_step(
                    x_tm.squeeze(1), global_state, layer_idx=i
                )
                x_tm = x_tm + mem_residual.unsqueeze(1)

        z = self.head_ln(x_tm).squeeze(1)
        logits = self.pi_head(z)
        value = self.v_head(z).squeeze(-1)
        return logits, value

    # -------------------------
    # Checkpoint I/O (unchanged)
    # -------------------------
    @torch.no_grad()
    def load_ckpt(self, load_path, optimizer=None, strict=False):
        if os.path.isdir(load_path):
            ckpt_names = sorted([fn for fn in os.listdir(load_path) if fn.endswith(".pt")])
            assert len(ckpt_names) > 0
            ckpt_path = os.path.join(load_path, ckpt_names[-1])
        else:
            ckpt_path = load_path

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        missing, unexpected = self.load_state_dict(ckpt["model"], strict=strict)
        print(f"[info] loaded ckpt: {ckpt_path}")
        if not strict:
            if len(missing) or len(unexpected):
                print(f"[warn] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

        if optimizer is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            print("[info] loaded optimizer state")

        step = int(ckpt.get("step", 0))
        return step, ckpt

    def save_ckpt(self, save_path, optimizer=None, step=None, extra=None):
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        payload = {"model": self.state_dict()}
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        if step is not None:
            payload["step"] = int(step)
        if extra is not None:
            payload.update(extra)
        torch.save(payload, save_path)
        print(f"[info] Saved checkpoint to {save_path}")
