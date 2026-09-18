from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    latent_dim: int
    action_dim: int
    hidden_dim: int = 128
    action_embed_dim: int = 32
    time_embed_dim: int = 16
    ensemble_size: int = 5
    delta_scale_days: float = 365.0

    def to_dict(self) -> dict:
        return asdict(self)


class OneStepDynamics(nn.Module):
    """The same residual dynamics architecture is applied once per transition."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.delta_scale_days = config.delta_scale_days
        self.action_projection = nn.Sequential(
            nn.Linear(config.action_dim, config.action_embed_dim),
            nn.LayerNorm(config.action_embed_dim),
            nn.SiLU(),
        )
        self.time_projection = nn.Sequential(
            nn.Linear(1, config.time_embed_dim),
            nn.SiLU(),
            nn.Linear(config.time_embed_dim, config.time_embed_dim),
        )
        self.gru = nn.GRU(config.action_embed_dim + config.time_embed_dim, config.hidden_dim, batch_first=True)
        context_dim = config.hidden_dim
        self.network = nn.Sequential(
            nn.LayerNorm(config.latent_dim + context_dim),
            nn.Linear(config.latent_dim + context_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.latent_dim),
        )
        nn.init.normal_(self.network[-1].weight, std=0.01)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, z: torch.Tensor, action: torch.Tensor, delta_days: torch.Tensor) -> torch.Tensor:
        # Inputs are [B,D], [B,A], [B]. The GRU sees exactly one action.
        scaled = torch.log1p(delta_days.clamp_min(0.0)) / math.log1p(self.delta_scale_days)
        sequence = torch.cat((self.action_projection(action), self.time_projection(scaled[:, None])), dim=-1)
        _, hidden = self.gru(sequence[:, None, :])
        return z + self.network(torch.cat((z, hidden[-1]), dim=-1))


class EnsembleDynamics(nn.Module):
    """Independent dynamics members, each maintaining its own rollout state."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.ensemble_size < 1:
            raise ValueError("ensemble_size must be positive")
        self.config = config
        self.members = nn.ModuleList(OneStepDynamics(config) for _ in range(config.ensemble_size))

    @property
    def ensemble_size(self) -> int:
        return len(self.members)

    def forward_memberwise(self, member_states: torch.Tensor, actions: torch.Tensor, delta_days: torch.Tensor) -> torch.Tensor:
        """Advance [M,B,D] states using a common [B,A] action and [B] elapsed days."""
        if member_states.ndim != 3 or member_states.shape[0] != self.ensemble_size:
            raise ValueError("member_states must be [M,B,D]")
        return torch.stack([
            member(member_states[index], actions, delta_days)
            for index, member in enumerate(self.members)
        ])

    def forward(self, z_start: torch.Tensor, actions: torch.Tensor, delta_days: torch.Tensor, horizons: torch.Tensor) -> torch.Tensor:
        """Recursive terminal predictions [M,B,D], with no teacher forcing."""
        if torch.any(horizons < 1) or torch.any(horizons > actions.shape[1]):
            raise ValueError("horizons must be within the action sequence")
        states = z_start.unsqueeze(0).expand(self.ensemble_size, -1, -1)
        for step in range(int(horizons.max().item())):
            advanced = self.forward_memberwise(states, actions[:, step], delta_days[:, step])
            states = torch.where((horizons > step)[None, :, None], advanced, states)
        return states


def ensemble_mean_and_disagreement(predictions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = predictions.mean(dim=0)
    # Definition in the protocol is 1/M, including M=1 where disagreement is zero.
    disagreement = (predictions - mean.unsqueeze(0)).square().mean(dim=0).mean(dim=-1)
    return mean, disagreement
