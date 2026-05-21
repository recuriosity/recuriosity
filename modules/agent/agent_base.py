# Copyright (c) 2025.
# Sliding-window Space→Time transformer for PPO camera agent with DINO conditioning.
#
# Design:
#   1) Extract DINO features per frame (frozen or finetuned)
#   2) Spatial encoder: learnable query token cross-attends to DINO features + input patches
#   3) Temporal encoder: causal sliding-window attention over frame tokens
#   4) Heads per timestep: logits/value
#
# Input per frame: C = 3 (rgb) + 6 (plucker rays for pose) + 6 (plucker rays for prev action) = 15

import os
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from typing import Optional, List

from .transformer import QK_Norm_TransformerBlock, KVCache, init_weights


class CrossAttentionBlock(nn.Module):
    """Cross-attention block: queries attend to context"""
    def __init__(self, d_model, d_head, use_qk_norm=True):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.n_heads = d_model // d_head
        assert d_model % d_head == 0, "d_model must be divisible by d_head"
        
        # Q from queries, K/V from context
        self.norm_q = nn.LayerNorm(d_model, bias=False)
        self.norm_kv = nn.LayerNorm(d_model, bias=False)
        
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_kv = nn.Linear(d_model, 2 * d_model, bias=False)
        
        # QK normalization
        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = nn.LayerNorm(d_head, bias=False)
            self.k_norm = nn.LayerNorm(d_head, bias=False)
        
        self.proj = nn.Linear(d_model, d_model, bias=False)
        
        # FFN
        self.norm_ffn = nn.LayerNorm(d_model, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )
    
    def forward(self, x, context):
        """
        Args:
            x: [B, N_q, D] - queries
            context: [B, N_kv, D] - keys/values
        Returns:
            x: [B, N_q, D]
        """
        B, N_q, D = x.shape
        N_kv = context.shape[1]
        
        # Cross-attention
        q = self.to_q(self.norm_q(x))
        kv = self.to_kv(self.norm_kv(context))
        k, v = kv.chunk(2, dim=-1)
        
        # Reshape for multi-head: [B, H, N, d_head]
        q = q.reshape(B, N_q, self.n_heads, self.d_head).transpose(1, 2)
        k = k.reshape(B, N_kv, self.n_heads, self.d_head).transpose(1, 2)
        v = v.reshape(B, N_kv, self.n_heads, self.d_head).transpose(1, 2)
        
        # QK normalization
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = attn.softmax(dim=-1)
        
        # Output
        out = (attn @ v).transpose(1, 2).reshape(B, N_q, D)
        out = self.proj(out)
        x = x + out
        
        # FFN
        x = x + self.ffn(self.norm_ffn(x))
        
        return x


class NavAgent(nn.Module):
    def __init__(
        self,
        image_size=64,
        patch_size=32,
        in_channels=15,
        d_model=768,
        d_head=64,
        n_layer_spatial_cross=4,  # Cross-attention layers
        n_layer_temporal=16,
        use_qk_norm=True,
        n_act=4,
        max_v=512,
        attn_window=64,
        checkpoint_every=0,
        pos_init_std=0.02,
        dino_model_name="dinov2_vitb14",
        dino_freeze=True,
        # NEW: Aggregation strategy for frame token
        frame_token_mode="learnable_query",  # Options: "learnable_query", "dino_cls_only", "mean_pool"
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
        self.dino = torch.hub.load('facebookresearch/dinov2', dino_model_name)
        self.dino_d_model = self.dino.embed_dim
        
        if dino_freeze:
            for param in self.dino.parameters():
                param.requires_grad = False
            self.dino.eval()
        
        # Project DINO features to our d_model
        if self.dino_d_model != d_model:
            self.dino_proj = nn.Linear(self.dino_d_model, d_model, bias=False)
            self.dino_proj.apply(init_weights)
        else:
            self.dino_proj = nn.Identity()

        # ============================================
        # Patch Embeddings (RGB + Plucker coords)
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

        # Spatial positions for patches
        self.spatial_pos = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        nn.init.normal_(self.spatial_pos, std=pos_init_std)

        # ============================================
        # Frame Token Generation (mode-dependent)
        # ============================================
        if frame_token_mode == "learnable_query":
            # Learnable query token that cross-attends to everything
            self.frame_query = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.frame_query, std=pos_init_std)
            
            # Cross-attention: query attends to [DINO features, input patches]
            self.spatial_in_ln = nn.LayerNorm(d_model, bias=False)
            self.spatial_cross_blocks = nn.ModuleList([
                CrossAttentionBlock(d_model, d_head, use_qk_norm=use_qk_norm)
                for _ in range(n_layer_spatial_cross)
            ])
            for blk in self.spatial_cross_blocks:
                blk.apply(init_weights)
        
        elif frame_token_mode == "dino_cls_only":
            # Input patches cross-attend to DINO CLS only
            self.spatial_in_ln = nn.LayerNorm(d_model, bias=False)
            self.spatial_cross_blocks = nn.ModuleList([
                CrossAttentionBlock(d_model, d_head, use_qk_norm=use_qk_norm)
                for _ in range(n_layer_spatial_cross)
            ])
            for blk in self.spatial_cross_blocks:
                blk.apply(init_weights)
            
            # Pool patches to single token
            self.frame_pool = nn.Sequential(
                nn.LayerNorm(d_model, bias=False),
                nn.Linear(d_model, d_model, bias=False),
            )
            self.frame_pool.apply(init_weights)
        
        else:  # mean_pool
            # Input patches cross-attend to all DINO features, then mean pool
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
    # KV Cache Management
    # -------------------------
    def init_kv_cache(self, batch_size: int = 1) -> List[KVCache]:
        max_len = self.attn_window if (self.attn_window is not None and self.attn_window > 0) else self.max_v
        return [KVCache(max_seq_len=max_len) for _ in range(self.n_layer_temporal)]
    
    def clear_kv_cache(self, caches: List[KVCache]):
        for cache in caches:
            cache.clear()

    # -------------------------
    # DINO Feature Extraction
    # -------------------------
    def _extract_dino_features(self, rgb_images):
        """
        Extract DINO features from RGB images.
        
        Args:
            rgb_images: [B, 3, H, W] - can be any size
        
        Returns:
            dino_cls: [B, D] - CLS token
            dino_patches: [B, N_patches, D] - Patch tokens
        """
        # Resize to nearest DINO-compatible size (multiple of 14)
        # Common sizes: 224, 336, 448, 518
        B, C, H, W = rgb_images.shape
        
        # Choose target size (224 is standard for DINO)
        target_size = 224
        if H != target_size or W != target_size:
            rgb_resized = torch.nn.functional.interpolate(
                rgb_images, 
                size=(target_size, target_size), 
                mode='bilinear', 
                align_corners=False
            )
        else:
            rgb_resized = rgb_images
        
        if self.dino_freeze:
            with torch.no_grad():
                features = self.dino.forward_features(rgb_resized)
        else:
            features = self.dino.forward_features(rgb_resized)
        
        # Extract CLS and patch tokens
        if isinstance(features, dict):
            patch_tokens = features['x_norm_patchtokens']
            cls_token = features['x_norm_clstoken']
        else:
            cls_token = features[:, 0, :]
            patch_tokens = features[:, 1:, :]
        
        # Project to d_model
        dino_cls = self.dino_proj(cls_token)
        dino_patches = self.dino_proj(patch_tokens)
        
        return dino_cls, dino_patches

    # -------------------------
    # Spatial Encoding with DINO Cross-Attention
    # -------------------------
    def _encode_frames_batch(self, posed_images):
        """
        Encode a batch of frame sequences into per-frame tokens.

        Args:
            posed_images: [B, V, C, H, W]

        Returns:
            frame_tok: [B, V, D]
        """
        B, V, C, H, W = posed_images.shape
        assert C == self.in_channels
        assert H == self.image_size and W == self.image_size

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
            frame_tok = query.squeeze(1) + 0.5 * patch_summary

        elif self.frame_token_mode == "dino_cls_only":
            context = dino_cls.unsqueeze(1)

            patch_tok = self.spatial_in_ln(patch_tok)
            for blk in self.spatial_cross_blocks:
                patch_tok = blk(patch_tok, context=context)

            frame_tok = patch_tok.mean(dim=1)
            frame_tok = self.frame_pool(frame_tok)

        else:
            context = torch.cat([
                dino_cls.unsqueeze(1),
                dino_patches,
            ], dim=1)

            patch_tok = self.spatial_in_ln(patch_tok)
            for blk in self.spatial_cross_blocks:
                patch_tok = blk(patch_tok, context=context)

            frame_tok = patch_tok.mean(dim=1)
            frame_tok = self.frame_pool(frame_tok)

        return frame_tok.reshape(B, V, self.d_model)

    def _encode_frame(self, posed_image):
        """
        Encode a single frame using DINO-conditioned cross-attention.
        
        Args:
            posed_image: [B, C=15, H, W]
        
        Returns:
            frame_token: [B, D]
        """
        B = posed_image.shape[0]
        
        # Split RGB from plucker coordinates
        rgb = posed_image[:, :3, :, :]  # [B, 3, H, W]
        
        # Extract DINO features
        dino_cls, dino_patches = self._extract_dino_features(rgb)
        
        # Patch embeddings from full input
        x = posed_image.unsqueeze(1)  # [B, 1, C, H, W]
        patch_tok = self.patch_embed(x)  # [B, P, D]
        patch_tok = patch_tok + self.spatial_pos[:, :patch_tok.shape[1], :]
        
        # ---- Mode-specific processing ----
        if self.frame_token_mode == "learnable_query":
            # Learnable query attends to [DINO CLS, DINO patches, input patches]
            # Concatenate all context
            context = torch.cat([
                dino_cls.unsqueeze(1),  # [B, 1, D]
                dino_patches,           # [B, N_dino, D]
                patch_tok,              # [B, P, D]
            ], dim=1)  # [B, 1+N_dino+P, D]
            
            # Query token cross-attends
            query = self.frame_query.expand(B, 1, self.d_model)  # [B, 1, D]
            query = self.spatial_in_ln(query)
            
            for blk in self.spatial_cross_blocks:
                query = blk(query, context=context)

            patch_summary = patch_tok.mean(dim=1)  # [B, D]
            frame_tok = query.squeeze(1) + 0.5 * patch_summary 
        
        elif self.frame_token_mode == "dino_cls_only":
            # Input patches cross-attend to DINO CLS only
            context = dino_cls.unsqueeze(1)  # [B, 1, D]
            
            patch_tok = self.spatial_in_ln(patch_tok)
            for blk in self.spatial_cross_blocks:
                patch_tok = blk(patch_tok, context=context)
            
            # Pool patches
            frame_tok = patch_tok.mean(dim=1)  # [B, D]
            frame_tok = self.frame_pool(frame_tok)
        
        else:  # mean_pool
            # Input patches cross-attend to all DINO features
            context = torch.cat([
                dino_cls.unsqueeze(1),  # [B, 1, D]
                dino_patches,           # [B, N_dino, D]
            ], dim=1)
            
            patch_tok = self.spatial_in_ln(patch_tok)
            for blk in self.spatial_cross_blocks:
                patch_tok = blk(patch_tok, context=context)
            
            # Mean pool
            frame_tok = patch_tok.mean(dim=1)  # [B, D]
            frame_tok = self.frame_pool(frame_tok)
    
        
        return frame_tok

    # -------------------------
    # Block runner (optional checkpointing)
    # -------------------------
    def _run_blocks(self, x, blocks, attn_bias=None, kv_caches: Optional[List[KVCache]] = None):
        if not self.training or self.checkpoint_every <= 0:
            for i, blk in enumerate(blocks):
                cache = kv_caches[i] if kv_caches is not None else None
                x = blk(x, attn_bias=attn_bias, kv_cache=cache)
            return x

        assert kv_caches is None, "Checkpointing not compatible with KV caching"
        num_layers = len(blocks)
        every = self.checkpoint_every
        for s in range(0, num_layers, every):
            e = min(s + every, num_layers)
            def _group(inp, s=s, e=e, bias=attn_bias):
                y = inp
                for i in range(s, e):
                    y = blocks[i](y, attn_bias=bias, kv_cache=None)
                return y
            x = torch.utils.checkpoint.checkpoint(_group, x, use_reentrant=False)
        return x

    def _forward_from_frame_tokens(self, frame_tok, last_only=False):
        x_tm = self.temporal_in_ln(frame_tok)
        x_tm = self._run_blocks(x_tm, self.temporal_blocks, attn_bias=None, kv_caches=None)

        z = self.head_ln(x_tm)
        logits = self.pi_head(z)
        values = self.v_head(z).squeeze(-1)

        if last_only:
            return logits[:, -1], values[:, -1]
        return logits, values

    def _forward_step_from_frame_token(self, frame_tok, kv_caches: List[KVCache]):
        x_tm = self.temporal_in_ln(frame_tok.unsqueeze(1))
        for i, blk in enumerate(self.temporal_blocks):
            x_tm = blk(x_tm, attn_bias=None, kv_cache=kv_caches[i])

        z = self.head_ln(x_tm).squeeze(1)
        logits = self.pi_head(z)
        value = self.v_head(z).squeeze(-1)
        return logits, value

    # -------------------------
    # Forward: Training Mode (batch)
    # -------------------------
    def forward(self, posed_images, last_only=False):
        """
        Training mode: process entire trajectories at once.
        
        Args:
            posed_images: [B, V, C=15, H, W]
            last_only: If True, only return outputs for last timestep
        
        Returns:
            logits: [B, V, n_act] (or [B, n_act] if last_only)
            values: [B, V]        (or [B]        if last_only)
        """
        B, V, C, H, W = posed_images.shape
        assert V <= self.max_v
        assert C == self.in_channels
        assert H == self.image_size and W == self.image_size

        frame_tok = self._encode_frames_batch(posed_images)
        return self._forward_from_frame_tokens(frame_tok, last_only=last_only)

    # -------------------------
    # Forward: Rollout Mode (incremental with caching)
    # -------------------------
    @torch.no_grad()
    def forward_step(self, posed_image, time_idx: int, kv_caches: List[KVCache]):
        """
        Rollout mode: process one frame at a time with KV caching.
        
        Args:
            posed_image: [B, C=15, H, W]
            time_idx: Current timestep
            kv_caches: List of KVCache objects
        
        Returns:
            logits: [B, n_act]
            value: [B]
        """
        B, C, H, W = posed_image.shape
        assert C == self.in_channels
        assert H == self.image_size and W == self.image_size
        assert len(kv_caches) == self.n_layer_temporal

        frame_tok = self._encode_frame(posed_image)
        return self._forward_step_from_frame_token(frame_tok, kv_caches=kv_caches)

    # -------------------------
    # Checkpoint I/O
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
