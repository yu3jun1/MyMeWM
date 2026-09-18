import json
from dataclasses import asdict

import torch

from clarity_hauwm.ablation import evaluate_stage1, train_split_robustness, train_stage1
from clarity_hauwm.evaluation import recursive_metrics, selective_risk_metrics, uncertainty_metrics
from clarity_hauwm.reporting import _prediction_comparison, summarize_stage1
from clarity_hauwm.training import TrainingConfig


def test_recursive_and_selective_risk_metrics():
    records = []
    for horizon in (1, 2, 3):
        for index in range(5):
            records.append({"patient_id": f"p{index}", "start": 0, "horizon": horizon,
                            "mse": float(index + horizon), "cosine_distance": 0.1,
                            "uncertainty": float(index)})
    prediction = recursive_metrics(records, "baseline", 7)
    assert prediction["long_horizon_mse"] == 4.5
    assert "matched_h3_slope" not in prediction
    reliability = uncertainty_metrics(records, "ensemble", 7)
    assert all(abs(row["uncertainty_error_spearman"] - 1) < 1e-12 for row in reliability["by_horizon"])
    risk = selective_risk_metrics(records, "ensemble", 7)
    assert risk["by_horizon"][0]["n_retained"] == 4
    assert risk["by_horizon"][0]["risk_80"] == 2.5
    assert risk["by_horizon"][0]["risk_reduction"] > 0


def test_relative_improvement_is_paired_by_training_seed():
    def run(long_mse):
        return {"prediction": {"mse@1": long_mse, "mse@2": long_mse,
                               "mse@3": long_mse, "long_mse": long_mse}}
    runs = {"baseline": {7: run(10.0), 17: run(20.0)},
            "rhrt": {7: run(8.0), 17: run(10.0)}}
    report = _prediction_comparison(runs)["baseline_vs_rhrt"]
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
                            time_embed_dim=4, horizon_embed_dim=4, ensemble_size=2,
                            early_stopping_patience=0, device="cpu", training_seeds=[17],
                            robustness_split_seeds=[23])
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(config)))
    output = tmp_path / "stage1"
    checkpoints = train_stage1(data_dir, config_path, output)
    assert len(checkpoints) == 5
    recursive = evaluate_stage1(output, "recursive", device="cpu")
    uncertainty = evaluate_stage1(output, "uncertainty", device="cpu")
    assert len(recursive) == 5
    assert len(uncertainty) == 2
    assert len(train_split_robustness(data_dir, config_path, output)) == 3
    summary = summarize_stage1(output)
    assert "stage1_pass" not in summary
    assert summary["main_split_seed"] == 17
    run = output / "seed_17" / "rhrt_ensemble"
    training = json.loads((run / "training.json").read_text())
    assert sum(training["horizon_sampling_counts"].values()) == training["num_training_windows"] * 2
    assert all(training["horizon_sampling_counts"][str(k)] > 0 for k in (1, 2, 3))
    recursive_max = json.loads((output / "seed_17" / "recursive_max" / "training.json").read_text())
    assert recursive_max["horizon_strategy"] == "max_available"
    reports = output / "reports"
    for name in ("dataset_stats", "training_summary", "prediction_metrics", "prediction_comparison",
                 "uncertainty_metrics", "selective_risk", "split_robustness", "stage1_summary"):
        assert (reports / f"{name}.json").exists()
    robustness = json.loads((reports / "split_robustness.json").read_text())
    assert {row["split_seed"] for row in robustness} == {17, 23}
    comparisons = json.loads((reports / "prediction_comparison.json").read_text())
    assert "recursive_max_vs_rhrt" in comparisons
    assert "RI_long" in comparisons["baseline_vs_rhrt"]
    assert not list(output.rglob("*.csv"))
    assert "bootstrap" not in (reports / "stage1.log").read_text().lower()
