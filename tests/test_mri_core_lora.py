import numpy as np
import pytest
import torch
from torch import nn

from clarity_hauwm.data import Trajectory, save_dataset
from clarity_hauwm.encoders.mri_core_lora import FusedQKVLoRA, MRICoreLoRAEncoder
from clarity_hauwm.lora_adaptation import LoRAAdaptationConfig, _train_group
from clarity_hauwm.lora_data import FrozenMRICoreSource, Transition, lora_output_root
from clarity_hauwm.lora_reporting import _patient_macro
from clarity_hauwm.model import ModelConfig, OneStepDynamics


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(3, 9)


class TinyImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock()])
        self.projection = nn.Linear(6, 256)

    def forward(self, images):
        pooled = images.mean(dim=(-2, -1))
        qkv = self.blocks[0].attn.qkv(pooled)
        value = self.projection(torch.cat((qkv[:, :3], qkv[:, 6:]), dim=-1))
        return value[:, :, None, None]


def test_qv_lora_preserves_frozen_projection_and_trains_only_qv():
    torch.manual_seed(3)
    base = nn.Linear(3, 9)
    wrapper = FusedQKVLoRA(base, rank=2, alpha=4, dropout=0)
    inputs = torch.randn(5, 3)
    assert torch.allclose(wrapper(inputs), base(inputs))
    with torch.no_grad():
        wrapper.q_up.weight.fill_(0.2)
        wrapper.v_up.weight.fill_(-0.1)
    changed = wrapper(inputs)
    original = base(inputs)
    assert torch.allclose(changed[:, 3:6], original[:, 3:6])
    assert not torch.allclose(changed[:, :3], original[:, :3])
    changed.square().sum().backward()
    assert base.weight.grad is None
    assert wrapper.q_up.weight.grad is not None
    encoder = MRICoreLoRAEncoder(TinyImageEncoder(), rank=2, alpha=4, dropout=0)
    names = [name for name, parameter in encoder.named_parameters()
             if parameter.requires_grad]
    assert len(names) == 4
    assert all(any(part in name for part in ("q_down", "q_up", "v_down", "v_up"))
               for name in names)
    state = encoder.adapter_state_dict()
    copied = MRICoreLoRAEncoder(TinyImageEncoder(), rank=2, alpha=4, dropout=0)
    copied.load_adapter_state_dict(state)
    assert all(torch.equal(copied.adapter_state_dict()[key], value)
               for key, value in state.items())


def test_lora_train_group_replays_slice_gradients_without_base_updates():
    torch.manual_seed(4)
    encoder = MRICoreLoRAEncoder(TinyImageEncoder(), rank=2, alpha=4, dropout=0.2)
    dynamics = OneStepDynamics(ModelConfig(
        latent_dim=256, action_dim=2, hidden_dim=8,
        action_embed_dim=4, time_embed_dim=4))
    config = LoRAAdaptationConfig(
        physical_batch_size=1, gradient_accumulation_steps=1,
        slice_batch_size=1, image_size=4, epochs=1)

    class Source:
        def iter_preprocessed_slices(self, patient_id, timepoint, *args):
            level = 0.2 if timepoint == "T1" else 0.6
            yield torch.full((1, 3, 4, 4), level)
            yield torch.full((1, 3, 4, 4), level + 0.1)

    record = Transition(
        "p1", "T1", "T2", np.array([1.0, 0.0], dtype=np.float32),
        90.0, np.ones(256, dtype=np.float32))
    optimizer = torch.optim.AdamW([
        {"params": [p for p in encoder.parameters() if p.requires_grad]},
        {"params": dynamics.parameters()},
    ], lr=1e-3)
    lora = encoder.image_encoder.blocks[0].attn.qkv
    original_base = lora.base.weight.detach().clone()
    original_adapter = lora.q_up.weight.detach().clone()
    loss, dyn_loss = _train_group(
        [record], encoder, dynamics, Source(), config,
        torch.device("cpu"), optimizer)
    assert np.isfinite(loss) and np.isfinite(dyn_loss)
    assert torch.equal(lora.base.weight, original_base)
    assert not torch.equal(lora.q_up.weight, original_adapter)


def test_lora_source_uses_only_train_patients_and_checks_preprocessing(tmp_path):
    latent_dir = tmp_path / "frozen_latents"
    latent_dir.mkdir()
    mri_root = tmp_path / "mri"
    mri_root.mkdir()
    repository = tmp_path / "mri_foundation"
    repository.mkdir()
    checkpoint = tmp_path / "mri.pth"
    checkpoint.touch()
    sam = tmp_path / "sam.pth"
    sam.touch()
    timeline = tmp_path / "timeline.json"
    timeline.write_text("{}")
    trajectories = []
    for index in range(8):
        patient_id = f"p{index}"
        latents = np.ones((3, 256), dtype=np.float32) * (index + 1)
        for timepoint in range(1, 4):
            np.save(latent_dir / f"{patient_id}_Timepoint_{timepoint}.npy",
                    latents[timepoint - 1])
        trajectories.append(Trajectory(
            patient_id, latents,
            np.ones((2, 2), dtype=np.float32),
            np.array([30.0, 60.0], dtype=np.float32),
            np.array(["T1", "T2", "T3"])))
    data_dir = tmp_path / "frozen_data"
    save_dataset(data_dir, trajectories, ["a", "b"], extra_metadata={
        "kind": "clarity", "action_anchor": "source",
        "latent_dir": str(latent_dir), "timeline": str(timeline),
        "min_token_count": 1,
        "latent_extraction": {
            "encoder": "mri_core", "adapter": None,
            "output_kind": "mean", "extracted": 24,
            "mri_root": str(mri_root), "encoder_repository": str(repository),
            "encoder_checkpoint": str(checkpoint), "sam_checkpoint": str(sam),
            "image_size": 1024, "normalization": "minmax",
            "slice_policy": "all", "slices_per_modality": None,
        },
    })
    source = FrozenMRICoreSource(data_dir)
    records = source.transitions("train")
    assert {row.patient_id for row in records} == set(source.split["train"])
    assert len(records) == 2 * len(source.split["train"])
    assert len(source.frozen_latent_files()) == 24
    source.assert_preprocessing(1024, "minmax", "all", 16)
    with pytest.raises(ValueError, match="preprocessing differs"):
        source.assert_preprocessing(1024, "sam", "all", 16)
    with pytest.raises(ValueError, match="distinct"):
        lora_output_root(tmp_path / "mri_core")


def test_patient_macro_weights_patients_equally():
    records = [
        {"patient_id": "a", "horizon": 1, "mse": 1.0},
        {"patient_id": "a", "horizon": 1, "mse": 3.0},
        {"patient_id": "b", "horizon": 1, "mse": 9.0},
        {"patient_id": "a", "horizon": 2, "mse": 2.0},
        {"patient_id": "b", "horizon": 2, "mse": 4.0},
        {"patient_id": "a", "horizon": 3, "mse": 3.0},
        {"patient_id": "b", "horizon": 3, "mse": 5.0},
    ]
    result = _patient_macro(records)
    assert result["by_horizon"]["1"]["mse"] == 5.5
    assert result["long_mse"] == 3.5


def test_slice_gradient_replay_matches_direct_autograd():
    import copy
    import torch.nn.functional as F

    torch.manual_seed(11)
    direct_encoder = MRICoreLoRAEncoder(TinyImageEncoder(), rank=2, alpha=4, dropout=0.35)
    direct_dynamics = OneStepDynamics(ModelConfig(
        latent_dim=256, action_dim=2, hidden_dim=8,
        action_embed_dim=4, time_embed_dim=4))
    replay_encoder = copy.deepcopy(direct_encoder)
    replay_dynamics = copy.deepcopy(direct_dynamics)
    config = LoRAAdaptationConfig(
        physical_batch_size=1, gradient_accumulation_steps=1,
        slice_batch_size=1, image_size=4, epochs=1)

    class Source:
        def iter_preprocessed_slices(self, patient_id, timepoint, *args):
            level = 0.2 if timepoint == "T1" else 0.6
            yield torch.full((1, 3, 4, 4), level)
            yield torch.full((1, 3, 4, 4), level + 0.1)

    record = Transition(
        "p1", "T1", "T2", np.array([1.0, 0.0], dtype=np.float32),
        90.0, np.ones(256, dtype=np.float32))

    def optimizer(encoder, dynamics):
        return torch.optim.AdamW([
            {"params": [p for p in encoder.parameters() if p.requires_grad]},
            {"params": dynamics.parameters()},
        ], lr=1e-3)

    torch.manual_seed(25)
    direct_optimizer = optimizer(direct_encoder, direct_dynamics)
    direct_encoder.train()
    source_latent = torch.stack([
        direct_encoder(batch)[0]
        for batch in Source().iter_preprocessed_slices("p1", "T1")
    ]).mean(dim=0, keepdim=True)
    direct_encoder.eval()
    with torch.no_grad():
        target_latent = torch.stack([
            direct_encoder(batch)[0]
            for batch in Source().iter_preprocessed_slices("p1", "T2")
        ]).mean(dim=0, keepdim=True)
    predicted = direct_dynamics(
        source_latent, torch.from_numpy(record.action)[None],
        torch.tensor([record.delta_days]))
    loss = (predicted - target_latent).square().mean() + config.anchor_weight * (
        1 - F.cosine_similarity(
            source_latent, torch.from_numpy(record.frozen_source)[None]).mean())
    loss.backward()
    parameters = [p for p in direct_encoder.parameters() if p.requires_grad]
    parameters.extend(direct_dynamics.parameters())
    torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip_norm)
    direct_optimizer.step()

    torch.manual_seed(25)
    _train_group(
        [record], replay_encoder, replay_dynamics, Source(), config,
        torch.device("cpu"), optimizer(replay_encoder, replay_dynamics))
    for direct, replay in zip(
            direct_encoder.parameters(), replay_encoder.parameters(), strict=True):
        assert torch.allclose(direct, replay, rtol=1e-5, atol=1e-7)
    for direct, replay in zip(
            direct_dynamics.parameters(), replay_dynamics.parameters(), strict=True):
        assert torch.allclose(direct, replay, rtol=1e-5, atol=1e-7)
