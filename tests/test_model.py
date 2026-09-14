import torch

from clarity_hauwm.model import EnsembleDynamics, ModelConfig, ensemble_mean_and_uncertainty


def test_ensemble_shapes_and_memberwise_rollout():
    config = ModelConfig(latent_dim=6, action_dim=3, max_horizon=4, hidden_dim=12, ensemble_size=3)
    model = EnsembleDynamics(config)
    z_start = torch.randn(5, 6)
    actions = torch.zeros(5, 4, 3)
    delta_days = torch.ones(5, 4) * 30
    horizons = torch.tensor([1, 2, 3, 4, 2])
    predictions = model(z_start, actions, delta_days, horizons)
    mean, uncertainty = ensemble_mean_and_uncertainty(predictions)
    assert predictions.shape == (3, 5, 6)
    assert mean.shape == (5, 6)
    assert uncertainty.shape == (5,)
    member_states = z_start[0].repeat(3, 1)
    next_states = model.forward_memberwise(member_states, actions[:1, :1], delta_days[:1, :1])
    assert next_states.shape == (3, 6)

