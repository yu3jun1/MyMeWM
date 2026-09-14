from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    latent_dim: int
    action_dim: int
    max_horizon: int = 5
    hidden_dim: int = 128
    action_embed_dim: int = 32
    time_embed_dim: int = 16
    horizon_embed_dim: int = 16
    ensemble_size: int = 5
    delta_scale_days: float = 365.0

    def to_dict(self) -> dict:
        return asdict(self)


class ActionSequenceEncoder(nn.Module):
    def __init__(
        self,
        action_dim: int,
        action_embed_dim: int,
        time_embed_dim: int,
        hidden_dim: int,
        delta_scale_days: float,
    ) -> None:
        super().__init__()
        self.delta_scale_days = delta_scale_days
        self.action_projection = nn.Sequential(
            nn.Linear(action_dim, action_embed_dim),
            nn.LayerNorm(action_embed_dim),
            nn.SiLU(),
        )
        self.time_projection = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.gru = nn.GRU(action_embed_dim + time_embed_dim, hidden_dim, batch_first=True)

    def forward(
        self,
        actions: torch.Tensor,
        delta_days: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        action_embedding = self.action_projection(actions)
        scaled_delta = torch.log1p(delta_days.clamp_min(0.0)) / math.log1p(self.delta_scale_days)
        time_embedding = self.time_projection(scaled_delta.unsqueeze(-1))
        sequence = torch.cat([action_embedding, time_embedding], dim=-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            sequence,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return hidden[-1]


class HorizonEncoder(nn.Module):
    def __init__(self, max_horizon: int, horizon_embed_dim: int, delta_scale_days: float) -> None:
        super().__init__()
        self.delta_scale_days = delta_scale_days
        self.embedding = nn.Embedding(max_horizon + 1, horizon_embed_dim)
        self.elapsed_projection = nn.Sequential(
            nn.Linear(1, horizon_embed_dim),
            nn.SiLU(),
            nn.Linear(horizon_embed_dim, horizon_embed_dim),
        )
        self.output = nn.Sequential(
            nn.Linear(horizon_embed_dim * 2, horizon_embed_dim),
            nn.LayerNorm(horizon_embed_dim),
            nn.SiLU(),
        )

    def forward(self, horizons: torch.Tensor, delta_days: torch.Tensor) -> torch.Tensor:
        mask = torch.arange(delta_days.shape[1], device=delta_days.device)[None, :] < horizons[:, None]
        total_days = (delta_days * mask).sum(dim=1, keepdim=True)
        total_days = torch.log1p(total_days) / math.log1p(self.delta_scale_days)
        return self.output(torch.cat([self.embedding(horizons), self.elapsed_projection(total_days)], dim=-1))


class DynamicsHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        nn.init.normal_(self.network[-1].weight, std=0.01)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, z_start: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        residual = self.network(torch.cat([z_start, context], dim=-1))
        return z_start + residual


class EnsembleDynamics(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.ensemble_size < 1:
            raise ValueError("ensemble_size must be positive")
        self.config = config
        self.action_encoder = ActionSequenceEncoder(
            action_dim=config.action_dim,
            action_embed_dim=config.action_embed_dim,
            time_embed_dim=config.time_embed_dim,
            hidden_dim=config.hidden_dim,
            delta_scale_days=config.delta_scale_days,
        )
        self.horizon_encoder = HorizonEncoder(
            config.max_horizon,
            config.horizon_embed_dim,
            config.delta_scale_days,
        )
        context_dim = config.hidden_dim + config.horizon_embed_dim
        self.heads = nn.ModuleList(
            DynamicsHead(config.latent_dim + context_dim, config.hidden_dim, config.latent_dim)
            for _ in range(config.ensemble_size)
        )

    @property
    def ensemble_size(self) -> int:
        return len(self.heads)

    def encode_context(
        self,
        actions: torch.Tensor,
        delta_days: torch.Tensor,
        horizons: torch.Tensor,
    ) -> torch.Tensor:
        action_context = self.action_encoder(actions, delta_days, horizons)
        horizon_context = self.horizon_encoder(horizons, delta_days)
        return torch.cat([action_context, horizon_context], dim=-1)

    def forward(
        self,
        z_start: torch.Tensor,
        actions: torch.Tensor,
        delta_days: torch.Tensor,
        horizons: torch.Tensor,
    ) -> torch.Tensor:
        """Return member predictions with shape [M,B,D]."""
        context = self.encode_context(actions, delta_days, horizons)
        return torch.stack([head(z_start, context) for head in self.heads], dim=0)

    def forward_memberwise(
        self,
        member_states: torch.Tensor,
        actions: torch.Tensor,
        delta_days: torch.Tensor,
    ) -> torch.Tensor:
        """Advance ensemble particles; member_states is [M,D] and actions is [1,1,A]."""
        if member_states.shape[0] != self.ensemble_size:
            raise ValueError("member_states does not match ensemble size")
        horizons = torch.ones(1, dtype=torch.long, device=member_states.device)
        context = self.encode_context(actions, delta_days, horizons).expand(self.ensemble_size, -1)
        return torch.stack(
            [head(member_states[index : index + 1], context[index : index + 1]).squeeze(0)
             for index, head in enumerate(self.heads)],
            dim=0,
        )


def ensemble_mean_and_uncertainty(predictions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = predictions.mean(dim=0)
    if predictions.shape[0] == 1:
        uncertainty = torch.zeros(predictions.shape[1], device=predictions.device, dtype=predictions.dtype)
    else:
        uncertainty = predictions.var(dim=0, unbiased=True).mean(dim=-1)
    return mean, uncertainty

