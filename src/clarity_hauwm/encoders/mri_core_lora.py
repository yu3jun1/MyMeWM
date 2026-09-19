from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from ..mri_core_extract import load_mri_core_image_encoder


class FusedQKVLoRA(nn.Module):
    """Frozen fused QKV projection with trainable low-rank Q and V updates."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0,
                 dropout: float = 0.05) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear) or base.out_features != 3 * base.in_features:
            raise ValueError("MRI-CORE attention must use a fused [Q,K,V] Linear")
        if rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid Q/V LoRA rank, alpha, or dropout")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        width = base.in_features
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / rank
        self.lora_dropout = nn.Dropout(dropout)
        self.q_down = nn.Linear(width, rank, bias=False)
        self.q_up = nn.Linear(rank, width, bias=False)
        self.v_down = nn.Linear(width, rank, bias=False)
        self.v_up = nn.Linear(rank, width, bias=False)
        for down, up in ((self.q_down, self.q_up), (self.v_down, self.v_up)):
            nn.init.kaiming_uniform_(down.weight, a=math.sqrt(5))
            nn.init.zeros_(up.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        projected = self.base(value.to(dtype=self.base.weight.dtype))
        # Keep the small trainable matrices in FP32 even when the frozen backbone is BF16.
        dropped = self.lora_dropout(value.float())
        query = self.q_up(self.q_down(dropped)) * self.scaling
        value_update = self.v_up(self.v_down(dropped)) * self.scaling
        update = torch.cat((query, torch.zeros_like(query), value_update), dim=-1)
        return projected + update.to(dtype=projected.dtype)


class MRICoreLoRAEncoder(nn.Module):
    """The frozen MRI-CORE image encoder with Q/V LoRA in every ViT block."""

    def __init__(self, image_encoder: nn.Module, rank: int = 8,
                 alpha: float = 16.0, dropout: float = 0.05) -> None:
        super().__init__()
        for parameter in image_encoder.parameters():
            parameter.requires_grad_(False)
        blocks = getattr(image_encoder, "blocks", None)
        if blocks is None or len(blocks) == 0:
            raise ValueError("MRI-CORE image encoder has no Transformer blocks")
        for block in blocks:
            attention = getattr(block, "attn", None)
            projection = getattr(attention, "qkv", None)
            if not isinstance(projection, nn.Linear):
                raise ValueError("Every MRI-CORE block must have a fused attn.qkv Linear")
            attention.qkv = FusedQKVLoRA(projection, rank, alpha, dropout)
        self.image_encoder = image_encoder
        self.rank = rank
        self.alpha = float(alpha)
        self.dropout = float(dropout)

    @classmethod
    def from_pretrained(cls, repository: str | Path, checkpoint: str | Path,
                        sam_checkpoint: str | Path, image_size: int = 1024,
                        rank: int = 8, alpha: float = 16.0,
                        dropout: float = 0.05) -> "MRICoreLoRAEncoder":
        base = load_mri_core_image_encoder(
            repository, checkpoint, sam_checkpoint, image_size)
        return cls(base, rank=rank, alpha=alpha, dropout=dropout)

    def forward(self, mri_slices: torch.Tensor) -> torch.Tensor:
        feature_maps = self.image_encoder(mri_slices)
        if feature_maps.ndim != 4 or feature_maps.shape[1] != 256:
            raise ValueError(f"Expected MRI-CORE [B,256,H,W], got {tuple(feature_maps.shape)}")
        return feature_maps.mean(dim=(-2, -1))


    def set_compute_dtype(self, dtype: torch.dtype) -> "MRICoreLoRAEncoder":
        if dtype not in (torch.float32, torch.bfloat16):
            raise ValueError(f"Unsupported MRI-CORE compute dtype: {dtype}")
        self.image_encoder.to(dtype=dtype)
        # AdamW moments and LoRA updates remain FP32 for numerical stability.
        for module in self.modules():
            if isinstance(module, FusedQKVLoRA):
                module.q_down.float()
                module.q_up.float()
                module.v_down.float()
                module.v_up.float()
        return self

    @property
    def compute_dtype(self) -> torch.dtype:
        return self.image_encoder.blocks[0].attn.qkv.base.weight.dtype

    def adapter_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def load_adapter_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        expected = {
            name: parameter for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        if state.keys() != expected.keys():
            raise ValueError(
                f"LoRA state keys differ: missing={sorted(expected.keys() - state.keys())}, "
                f"unexpected={sorted(state.keys() - expected.keys())}"
            )
        for name, parameter in expected.items():
            value = state[name]
            if value.shape != parameter.shape:
                raise ValueError(f"LoRA tensor shape mismatch for {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))

    def parameter_counts(self) -> dict[str, float | int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters()
                        if parameter.requires_grad)
        return {
            "encoder_total_parameters": total,
            "encoder_trainable_parameters": trainable,
            "encoder_trainable_ratio": trainable / total,
            "lora_blocks": len(self.image_encoder.blocks),
        }


def load_lora_checkpoint(path: str | Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema_version") != "mri_core_lora_1":
        raise ValueError(f"Unsupported MRI-CORE LoRA checkpoint: {path}")
    return checkpoint
