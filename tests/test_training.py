import json
from dataclasses import asdict

import pytest
import torch

from clarity_hauwm.ablation import evaluate_stage1, train_horizon_ablation, train_split_robustness, train_stage1
from clarity_hauwm.evaluation import recursive_metrics, selective_risk_metrics, uncertainty_metrics
from clarity_hauwm.reporting import _prediction_comparison, summarize_stage1
from clarity_hauwm.training import TrainingConfig


def test_recursive_and_selective_risk_metrics():
    records = []
    for horizon in (1, 2, 3):
        for index in range(5):
            records.append({"patient_id": f"p{index}", "start": 0, "horizon": horizon,
                            "mse": float(index + horizon), "cosine_distance": 0.1,
                            "disagreement": float(index)})
    prediction = recursive_metrics(records, "baseline", 7)
    assert prediction["long_horizon_mse"] == 4.5
    assert "matched_h3_slope" not in prediction
    reliability = uncertainty_metrics(records, "ensemble", 7)
    assert all(abs(row["disagreement_error_spearman"] - 1) < 1e-12 for row in reliability["by_horizon"])
    risk = selective_risk_metrics(records, "ensemble", 7)
    assert risk["by_horizon"][0]["n_retained"] == 4
    assert risk["by_horizon"][0]["risk_80"] == 2.5
    assert risk["by_horizon"][0]["risk_reduction"] > 0


def test_relative_improvement_is_paired_by_training_seed():
    def run(long_mse):
        return {"prediction": {"mse@1": long_mse, "mse@2": long_mse,
                               "mse@3": long_mse, "long_mse": long_mse}}
    runs = {"baseline": {7: run(10.0), 17: run(20.0)},
            "rrt": {7: run(8.0), 17: run(10.0)}}
    report = _prediction_comparison(runs)["baseline_vs_rrt"]
    assert report["seeds"]["7"]["RI_long"] == 20.0
    assert report["seeds"]["17"]["RI_long"] == 50.0
    assert report["RI_long"]["mean"] == 35.0
    assert 21.2 < report["RI_long"]["std"] < 21.3


def test_stage1_pipeline_writes_revised_reports(tmp_path, dataset_factory):
    torch.set_num_threads(1)
    data_dir = tmp_path / "data"
    dataset_factory(data_dir, patients=12, latent_dim=4, action_dim=2,
                    min_timepoints=5, max_timepoints=5, seed=5)
    config = TrainingConfig(batch_size=16, epochs=1, hidden_dim=8, action_embed_dim=4,
                            time_embed_dim=4, ensemble_size=2,
                            early_stopping_patience=0, device="cpu", training_seeds=[17],
                            robustness_split_seeds=[23])
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(config)))
    output = tmp_path / "stage1"
    checkpoints = train_stage1(data_dir, config_path, output)
    assert len(checkpoints) == 4
    recursive = evaluate_stage1(output, "recursive", device="cpu")
    uncertainty = evaluate_stage1(output, "uncertainty", device="cpu")
    assert len(recursive) == 4
    assert len(uncertainty) == 2
    assert len(train_split_robustness(data_dir, config_path, output)) == 2
    assert len(train_horizon_ablation(data_dir, config_path, output)) == 3
    with pytest.raises(ValueError, match="Insufficient H4/H5 windows"):
        train_horizon_ablation(data_dir, config_path, output, include_stress=True,
                               min_stress_windows=1)
    summary = summarize_stage1(output)
    assert "stage1_pass" not in summary
    assert summary["main_split_seed"] == 17
    run = output / "seed_17" / "rrt_ensemble"
    training = json.loads((run / "training.json").read_text())
    assert sum(training["training_horizon_counts"].values()) == training["num_training_windows"]
    assert all(training["training_horizon_counts"][str(k)] > 0 for k in (1, 2, 3))
    assert training["max_horizon"] == 3
    assert "horizon_sampling_counts" not in training
    assert "horizon_embed_dim" not in training
    rrt = json.loads((output / "seed_17" / "rrt" / "training.json").read_text())
    assert rrt["horizon_strategy"] == "max_available"
    assert rrt["training_horizon_counts"] == training["training_horizon_counts"]
    ablation = json.loads((output / "reports" / "horizon_ablation.json").read_text())
    assert set(ablation) == {"K1", "K2", "K3"}
    assert {ablation[key]["max_horizon"] for key in ablation} == {1, 2, 3}
    assert all(ablation[key]["long_mse"] is not None for key in ablation)
    ablation_counts = []
    for key in ("k1", "k2", "k3"):
        metrics = json.loads((output / "horizon_ablation" / key / "recursive_metrics.json").read_text())
        ablation_counts.append([row["n_predictions"] for row in metrics["by_horizon"][:3]])
    assert ablation_counts[0] == ablation_counts[1] == ablation_counts[2]
    reports = output / "reports"
    for name in ("dataset_stats", "training_summary", "prediction_metrics", "prediction_comparison",
                 "uncertainty_metrics", "selective_risk", "split_robustness", "horizon_ablation", "stage1_summary"):
        assert (reports / f"{name}.json").exists()
    robustness = json.loads((reports / "split_robustness.json").read_text())
    assert {row["split_seed"] for row in robustness} == {17, 23}
    comparisons = json.loads((reports / "prediction_comparison.json").read_text())
    assert set(comparisons) == {"baseline_vs_rrt", "ensemble_vs_rrt_ensemble"}
    assert "RI_long" in comparisons["baseline_vs_rrt"]
    assert {row["rrt_long_mse"] is not None for row in robustness} == {True}
    assert not list(output.rglob("*.csv"))
    assert "bootstrap" not in (reports / "stage1.log").read_text().lower()
