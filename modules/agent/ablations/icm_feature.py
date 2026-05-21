"""Feature-space intrinsic curiosity module for shared-encoder ablations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..transformer import init_weights


class FeatureICM(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        forward_loss_weight: float = 0.2,
        reward_scale: float = 0.01,
        loss_scale: float = 10.0,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.forward_loss_weight = float(forward_loss_weight)
        self.reward_scale = float(reward_scale)
        self.loss_scale = float(loss_scale)

        self.inverse_head = nn.Sequential(
            nn.Linear(2 * self.feature_dim, self.hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.action_dim, bias=True),
        )
        self.forward_head = nn.Sequential(
            nn.Linear(self.feature_dim + self.action_dim, self.hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.feature_dim, bias=True),
        )

        self.inverse_head.apply(init_weights)
        self.forward_head.apply(init_weights)

    def _one_hot_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim == 2 and actions.shape[-1] == self.action_dim:
            return actions.float()
        return F.one_hot(actions.long(), num_classes=self.action_dim).float()

    def _forward_error_per_sample(
        self,
        features: torch.Tensor,
        next_features: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        actions_one_hot = self._one_hot_actions(actions).to(features.dtype)
        pred_next_features = self.forward_head(torch.cat([features, actions_one_hot], dim=-1))
        return 0.5 * (pred_next_features - next_features).pow(2).mean(dim=-1) * self.feature_dim

    @torch.no_grad()
    def intrinsic_reward(
        self,
        features: torch.Tensor,
        next_features: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        forward_error = self._forward_error_per_sample(features, next_features, actions)
        return self.reward_scale * forward_error

    def losses(
        self,
        features: torch.Tensor,
        next_features: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        actions_long = actions.long().reshape(-1)
        actions_one_hot = F.one_hot(actions_long, num_classes=self.action_dim).to(features.dtype)

        inverse_logits = self.inverse_head(torch.cat([features, next_features], dim=-1))
        inverse_loss = F.cross_entropy(inverse_logits, actions_long)

        pred_next_features = self.forward_head(torch.cat([features, actions_one_hot], dim=-1))
        forward_error = 0.5 * (pred_next_features - next_features).pow(2).mean(dim=-1) * self.feature_dim
        forward_loss = forward_error.mean()

        total_loss = self.loss_scale * (
            (1.0 - self.forward_loss_weight) * inverse_loss
            + self.forward_loss_weight * forward_loss
        )
        return {
            "loss": total_loss,
            "inverse_loss": inverse_loss,
            "forward_loss": forward_loss,
            "forward_error": forward_error,
        }
