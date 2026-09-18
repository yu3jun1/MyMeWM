import json

import numpy as np

from clarity_hauwm.clarity_adapter import build_clarity_trajectories
from clarity_hauwm.data import (
    LatentNormalizer,
    TrainingHorizonDataset,
    load_dataset,
    split_patient_ids,
)


def test_round_trip_split_and_fixed_horizons(tmp_path, dataset_factory):
    data_dir = tmp_path / "synthetic"
    dataset_factory(data_dir, patients=12, seed=3)
    trajectories, metadata = load_dataset(data_dir)
    assert metadata["num_patients"] == 12
    split = split_patient_ids([item.patient_id for item in trajectories], 17, 0.7, 0.15)
    assert not (set(split["train"]) & set(split["test"]))
    normalizer = LatentNormalizer.fit(trajectories)
    for cap in (1, 2, 3, 5):
        dataset = TrainingHorizonDataset(trajectories, normalizer, cap, "max_available")
        assert len(dataset) == sum(len(item.latents) - 1 for item in trajectories)
        assert all(dataset[index]["horizon"] == min(cap, len(trajectories[ti].latents) - 1 - start)
                   for index, (ti, start) in enumerate(dataset.starts))
        assert sum(dataset.training_horizon_counts().values()) == len(dataset)
        assert dataset.available_horizon_counts(5)["1"] == len(dataset)
    one_step = TrainingHorizonDataset(trajectories, normalizer, 3, "one_step")
    assert one_step.training_horizon_counts() == {"1": len(one_step), "2": 0, "3": 0}
    with np.testing.assert_raises(ValueError):
        TrainingHorizonDataset(trajectories, normalizer, 3, "random_available")


def test_clarity_source_anchor_excludes_destination_action(tmp_path):
    payload = {
        "patients": {
            "PatientID_0001": {
                "timeline": [
                    {"tp_id": "T1", "mri_day": 0, "actions": {"chemotherapy": [{"agent": "Temodar"}]}},
                    {"tp_id": "T2", "mri_day": 90, "actions": {"radiation": [{"type": "external beam"}]}},
                    {"tp_id": "T3", "mri_day": 180, "actions": {}},
                ]
            }
        }
    }
    timeline = tmp_path / "timeline.json"
    timeline.write_text(json.dumps(payload), encoding="utf-8")
    latent_dir = tmp_path / "latents"
    latent_dir.mkdir()
    for index in range(1, 4):
        np.save(latent_dir / f"PatientID_0001_Timepoint_{index}.npy", np.ones(4) * index)
    output = tmp_path / "output"
    metadata = build_clarity_trajectories(timeline, latent_dir, output, action_anchor="source")
    trajectories, _ = load_dataset(output)
    vocabulary = metadata["action_vocab"]
    temozolomide = vocabulary.index("agent:temozolomide")
    radiation = vocabulary.index("category:radiation")
    assert trajectories[0].actions[0, temozolomide] == 1
    assert trajectories[0].actions[0, radiation] == 0
    assert trajectories[0].actions[1, radiation] == 1

