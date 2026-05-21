"""LSTM ablation of the sliding RGB policy agent.

This keeps the current per-frame encoder intact and swaps the temporal
transformer for a stacked LSTM so PPO integration stays unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ..agent_base import NavAgent as TransformerNavAgent
from ..transformer import init_weights


def _count_trainable_params(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def _init_lstm_weights(module: nn.LSTM) -> None:
    for name, param in module.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(param)
        elif "weight_hh" in name:
            nn.init.orthogonal_(param)
        elif "bias" in name:
            nn.init.zeros_(param)


def _lstm_param_count(input_size: int, hidden_size: int, num_layers: int) -> int:
    total = 4 * hidden_size * input_size + 4 * hidden_size * hidden_size + 8 * hidden_size
    total += max(0, num_layers - 1) * (8 * hidden_size * hidden_size + 8 * hidden_size)
    return total


def _projection_param_count(hidden_size: int, d_model: int) -> int:
    if hidden_size == d_model:
        return 0
    return hidden_size * d_model


def _adapter_param_count(d_model: int, adapter_width: int) -> int:
    if adapter_width <= 0:
        return 0
    return d_model + adapter_width * d_model + adapter_width + adapter_width * d_model + d_model


@dataclass(frozen=True)
class LSTMTemporalConfig:
    num_layers: int
    hidden_size: int
    adapter_width: int
    param_count: int
    param_gap: int


def _select_temporal_config(
    *,
    target_params: int,
    d_model: int,
    preferred_layers: tuple[int, ...] = (4, 2, 3, 1),
    hidden_size_multiple: int = 64,
    max_hidden_size: int = 6144,
    max_adapter_width: int = 2048,
    tolerance: int = 1024,
) -> LSTMTemporalConfig:
    best_within_tolerance: tuple[tuple[int, int, int, int], LSTMTemporalConfig] | None = None
    best_overall: tuple[tuple[int, int, int, int], LSTMTemporalConfig] | None = None

    for layer_rank, num_layers in enumerate(preferred_layers):
        for hidden_size in range(max(d_model, hidden_size_multiple), max_hidden_size + 1, hidden_size_multiple):
            base_param_count = _lstm_param_count(d_model, hidden_size, num_layers)
            base_param_count += _projection_param_count(hidden_size, d_model)
            if base_param_count - target_params > tolerance:
                break

            remaining = target_params - base_param_count
            candidate_adapter_widths = {0}
            if remaining > 0:
                approx_width = max(0, int(round((remaining - 2 * d_model) / (2 * d_model + 1))))
                for adapter_width in range(max(0, approx_width - 2), min(max_adapter_width, approx_width + 2) + 1):
                    candidate_adapter_widths.add(adapter_width)
                if remaining >= 2 * d_model:
                    candidate_adapter_widths.add(min(max_adapter_width, remaining // max(1, 2 * d_model + 1)))

            for adapter_width in sorted(candidate_adapter_widths):
                param_count = base_param_count + _adapter_param_count(d_model, adapter_width)
                param_gap = abs(target_params - param_count)
                cfg = LSTMTemporalConfig(
                    num_layers=num_layers,
                    hidden_size=hidden_size,
                    adapter_width=adapter_width,
                    param_count=param_count,
                    param_gap=param_gap,
                )

                overall_key = (param_gap, layer_rank, hidden_size, adapter_width)
                if best_overall is None or overall_key < best_overall[0]:
                    best_overall = (overall_key, cfg)

                if param_gap <= tolerance:
                    within_key = (layer_rank, hidden_size, param_gap, adapter_width)
                    if best_within_tolerance is None or within_key < best_within_tolerance[0]:
                        best_within_tolerance = (within_key, cfg)

    chosen = best_within_tolerance[1] if best_within_tolerance is not None else best_overall[1]
    return chosen


class NavAgentLSTM(TransformerNavAgent):
    """Transformer frame encoder + LSTM temporal backbone."""

    def __init__(
        self,
        *args,
        temporal_target_tolerance: int = 1024,
        preferred_lstm_layers: tuple[int, ...] = (4, 2, 3, 1),
        hidden_size_multiple: int = 64,
        max_hidden_size: int = 6144,
        max_adapter_width: int = 2048,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        reference_trainable_params = _count_trainable_params(self)
        target_temporal_params = _count_trainable_params(self.temporal_blocks)
        temporal_cfg = _select_temporal_config(
            target_params=target_temporal_params,
            d_model=self.d_model,
            preferred_layers=preferred_lstm_layers,
            hidden_size_multiple=hidden_size_multiple,
            max_hidden_size=max_hidden_size,
            max_adapter_width=max_adapter_width,
            tolerance=temporal_target_tolerance,
        )

        self.temporal_backbone = "lstm"
        self.lstm_hidden_size = temporal_cfg.hidden_size
        self.lstm_num_layers = temporal_cfg.num_layers
        self.temporal_adapter_width = temporal_cfg.adapter_width
        self.reference_trainable_params = reference_trainable_params
        self.reference_temporal_params = target_temporal_params

        del self.temporal_blocks

        self.temporal_rnn = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
        )
        _init_lstm_weights(self.temporal_rnn)

        if self.lstm_hidden_size == self.d_model:
            self.temporal_proj: nn.Module = nn.Identity()
        else:
            self.temporal_proj = nn.Linear(self.lstm_hidden_size, self.d_model, bias=False)
            self.temporal_proj.apply(init_weights)

        if self.temporal_adapter_width > 0:
            self.temporal_adapter_ln: nn.Module = nn.LayerNorm(self.d_model, bias=False)
            self.temporal_adapter: nn.Module = nn.Sequential(
                nn.Linear(self.d_model, self.temporal_adapter_width, bias=True),
                nn.GELU(),
                nn.Linear(self.temporal_adapter_width, self.d_model, bias=True),
            )
            self.temporal_adapter.apply(init_weights)
        else:
            self.temporal_adapter_ln = nn.Identity()
            self.temporal_adapter = nn.Identity()

        self.trainable_param_count = _count_trainable_params(self)
        self.trainable_param_gap = self.reference_trainable_params - self.trainable_param_count
        self.temporal_param_gap = target_temporal_params - temporal_cfg.param_count

        print(
            "[info] configured LSTM temporal backbone: "
            f"layers={self.lstm_num_layers}, hidden={self.lstm_hidden_size}, "
            f"adapter={self.temporal_adapter_width}, "
            f"trainable_params={self.trainable_param_count}, "
            f"reference_trainable_params={self.reference_trainable_params}, "
            f"gap={self.trainable_param_gap}"
        )

    def init_kv_cache(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        device = self.head_ln.weight.device
        h = torch.zeros(self.lstm_num_layers, batch_size, self.lstm_hidden_size, device=device)
        c = torch.zeros_like(h)
        return {"h": h, "c": c}

    def clear_kv_cache(self, caches: dict[str, torch.Tensor]) -> None:
        caches["h"].zero_()
        caches["c"].zero_()

    def _encode_frames_batch(self, posed_images: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, channels, height, width = posed_images.shape
        assert seq_len <= self.max_v
        assert channels == self.in_channels
        assert height == self.image_size and width == self.image_size

        posed_flat = posed_images.reshape(bsz * seq_len, channels, height, width)
        rgb_flat = posed_flat[:, :3]

        dino_cls, dino_patches = self._extract_dino_features(rgb_flat)

        patch_tok = self.patch_embed(posed_images)
        patch_tok = patch_tok + self.spatial_pos[:, : patch_tok.shape[1]]

        if self.frame_token_mode == "learnable_query":
            context = torch.cat(
                [
                    dino_cls.unsqueeze(1),
                    dino_patches,
                    patch_tok,
                ],
                dim=1,
            )
            query = self.frame_query.expand(bsz * seq_len, 1, self.d_model)
            query = self.spatial_in_ln(query)
            for block in self.spatial_cross_blocks:
                query = block(query, context=context)
            patch_summary = patch_tok.mean(dim=1)
            frame_tok = query.squeeze(1) + 0.5 * patch_summary
        elif self.frame_token_mode == "dino_cls_only":
            context = dino_cls.unsqueeze(1)
            patch_tok = self.spatial_in_ln(patch_tok)
            for block in self.spatial_cross_blocks:
                patch_tok = block(patch_tok, context=context)
            frame_tok = self.frame_pool(patch_tok.mean(dim=1))
        else:
            context = torch.cat(
                [
                    dino_cls.unsqueeze(1),
                    dino_patches,
                ],
                dim=1,
            )
            patch_tok = self.spatial_in_ln(patch_tok)
            for block in self.spatial_cross_blocks:
                patch_tok = block(patch_tok, context=context)
            frame_tok = self.frame_pool(patch_tok.mean(dim=1))

        return frame_tok.reshape(bsz, seq_len, self.d_model)

    def _run_lstm(
        self,
        frame_tokens: torch.Tensor,
        state: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = self.temporal_in_ln(frame_tokens)
        x, next_state = self.temporal_rnn(x, state)
        x = self.temporal_proj(x)
        x = x + self.temporal_adapter(self.temporal_adapter_ln(x))
        return x, next_state

    def _forward_from_frame_tokens(self, frame_tok: torch.Tensor, last_only: bool = False):
        x_tm, _ = self._run_lstm(frame_tok)

        z = self.head_ln(x_tm)
        logits = self.pi_head(z)
        values = self.v_head(z).squeeze(-1)

        if last_only:
            return logits[:, -1], values[:, -1]
        return logits, values

    def _forward_step_from_frame_token(self, frame_tok: torch.Tensor, kv_caches: dict[str, torch.Tensor]):
        x_tm, next_state = self._run_lstm(frame_tok.unsqueeze(1), state=(kv_caches["h"], kv_caches["c"]))
        kv_caches["h"] = next_state[0].detach()
        kv_caches["c"] = next_state[1].detach()

        z = self.head_ln(x_tm).squeeze(1)
        logits = self.pi_head(z)
        value = self.v_head(z).squeeze(-1)
        return logits, value

    def forward(self, posed_images: torch.Tensor, last_only: bool = False):
        frame_tok = self._encode_frames_batch(posed_images)
        return self._forward_from_frame_tokens(frame_tok, last_only=last_only)

    @torch.no_grad()
    def forward_step(self, posed_image, time_idx: int, kv_caches: dict[str, torch.Tensor]):
        del time_idx
        bsz, channels, height, width = posed_image.shape
        assert channels == self.in_channels
        assert height == self.image_size and width == self.image_size

        frame_tok = self._encode_frame(posed_image).unsqueeze(1)
        return self._forward_step_from_frame_token(frame_tok.squeeze(1), kv_caches=kv_caches)
