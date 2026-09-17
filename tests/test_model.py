import torch
from torch import nn

from clarity_hauwm.model import EnsembleDynamics, ModelConfig, ensemble_mean_and_uncertainty


class AddAction(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, z, action, delta_days):
        return z + self.scale * action


def test_recursive_rollout_uses_each_members_own_state_and_population_disagreement():
    model = EnsembleDynamics(ModelConfig(latent_dim=1, action_dim=1, ensemble_size=2))
    model.members = nn.ModuleList([AddAction(1), AddAction(2)])
    start = torch.zeros(2, 1)
    actions = torch.ones(2, 3, 1)
    deltas = torch.ones(2, 3)
    horizons = torch.tensor([1, 3])
    predictions = model(start, actions, deltas, horizons)
    assert torch.allclose(predictions[:, 0, 0], torch.tensor([1.0, 2.0]))
    assert torch.allclose(predictions[:, 1, 0], torch.tensor([3.0, 6.0]))
    mean, uncertainty = ensemble_mean_and_uncertainty(predictions)
    assert torch.allclose(mean[:, 0], torch.tensor([1.5, 4.5]))
    assert torch.allclose(uncertainty, torch.tensor([0.25, 2.25]))
    predictions[:, 1].mean().backward()
    assert model.members[0].scale.grad is not None
    assert model.members[1].scale.grad is not None
