import json
from dataclasses import asdict

import torch

from clarity_hauwm.ablation import evaluate_stage1, train_stage1
from clarity_hauwm.evaluation import matched_h3_slopes, recursive_metrics, uncertainty_metrics
from clarity_hauwm.reporting import summarize_stage1
from clarity_hauwm.training import TrainingConfig


def test_matched_h3_slope_uses_only_complete_starts():
    records = [
        {"patient_id": "a", "start": 0, "horizon": horizon, "mse": float(horizon)}
        for horizon in (1, 2, 3)
    ]
    records.append({"patient_id": "a", "start": 1, "horizon": 1, "mse": 100.0})
    slopes = matched_h3_slopes(records)
    assert abs(slopes["a"] - 1) < 1e-8
    report = recursive_metrics([{**row, "cosine_distance": 0.1} for row in records], "baseline", 7)
    assert report["matched_h3_patients"] == 1
    assert report["long_horizon_mse"] == 2.5


def test_within_horizon_reliability_is_not_pooled_horizon_correlation():
    records = []
    for horizon in (1, 2, 3):
        for patient in range(4):
            records.append({"patient_id": f"p{patient}", "start": 0, "horizon": horizon,
                            "uncertainty": 10 * horizon + patient,
                            "mse": 10 * horizon + 4 - patient})
    report = uncertainty_metrics(records, "ensemble", 7)
    assert all(row["uncertainty_error_spearman"] == -1 for row in report["by_horizon"])
    assert report["macro_spearman"] == -1
    assert report["uncertainty_horizon_spearman"] > 0


def test_stage1_pipeline_writes_separate_reports(tmp_path, dataset_factory):
    torch.set_num_threads(1)
    data_dir = tmp_path / "data"
    dataset_factory(data_dir, patients=12, latent_dim=4, action_dim=2,
                    min_timepoints=5, max_timepoints=5, seed=5)
    config = TrainingConfig(batch_size=16, epochs=1, hidden_dim=8, action_embed_dim=4,
                            time_embed_dim=4, horizon_embed_dim=4, ensemble_size=2,
                            early_stopping_patience=0, device="cpu", experiment_seeds=[7],
                            bootstrap_samples=20)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(config)))
    output = tmp_path / "stage1"
    checkpoints = train_stage1(data_dir, config_path, output)
    assert len(checkpoints) == 4
    recursive = evaluate_stage1(output, "recursive", device="cpu")
    uncertainty = evaluate_stage1(output, "uncertainty", device="cpu")
    assert len(recursive) == 4
    assert len(uncertainty) == 2
    summary = summarize_stage1(output, bootstrap_samples=20)
    assert summary["protocol_complete"] is False
    assert summary["stage1_pass"] is False
    run = output / "seed_7" / "rhrt_ensemble"
    training = json.loads((run / "training.json").read_text())
    assert sum(training["horizon_sampling_counts"].values()) == (
        training["num_training_windows"] * 2
    )
    assert all(training["horizon_sampling_counts"][str(k)] > 0 for k in (1, 2, 3))
    assert (run / "recursive_metrics.json").exists()
    assert (run / "uncertainty_metrics.json").exists()
    assert not list(output.rglob("*.csv"))
    for name in ("rhrt_summary", "rhrt_bootstrap", "ensemble_summary",
                 "ensemble_bootstrap", "stage1_summary"):
        assert (output / "reports" / f"{name}.json").exists()
