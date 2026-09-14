import json

from clarity_hauwm.evaluation import evaluate_checkpoint
from clarity_hauwm.synthetic import generate_synthetic_dataset
from clarity_hauwm.training import TrainingConfig, train_model


def test_tiny_train_and_evaluate(tmp_path):
    data_dir = tmp_path / "data"
    generate_synthetic_dataset(data_dir, patients=12, max_timepoints=4, seed=5)
    config = TrainingConfig(
        max_horizon=2,
        batch_size=8,
        epochs=1,
        hidden_dim=16,
        action_embed_dim=8,
        time_embed_dim=4,
        horizon_embed_dim=4,
        ensemble_size=2,
        early_stopping_patience=0,
        device="cpu",
    )
    run_dir = tmp_path / "run"
    checkpoint = train_model(data_dir, run_dir, config, 7, True, True, "hs_ensemble")
    report = evaluate_checkpoint(data_dir, checkpoint, run_dir, "cpu")
    assert checkpoint.exists()
    assert report["direct"]["n"] > 0
    assert report["rollout"]["n"] > 0
    assert (run_dir / "direct_records.csv").exists()
    json.loads((run_dir / "evaluation.json").read_text())

