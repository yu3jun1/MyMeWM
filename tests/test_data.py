import json

import numpy as np

from clarity_hauwm.clarity_adapter import build_clarity_trajectories
from clarity_hauwm.data import (
    LatentNormalizer,
    TrainingHorizonDataset,
    load_dataset,
    split_patient_ids,
)


def test_round_trip_split_and_horizon_sampling(tmp_path, dataset_factory):
    data_dir = tmp_path / "synthetic"
    dataset_factory(data_dir, patients=12, seed=3)
    trajectories, metadata = load_dataset(data_dir)
    assert metadata["num_patients"] == 12
    split = split_patient_ids([item.patient_id for item in trajectories], 17, 0.7, 0.15)
    assert not (set(split["train"]) & set(split["test"]))
    normalizer = LatentNormalizer.fit(trajectories)
    dataset = TrainingHorizonDataset(trajectories, normalizer, 4, True, seed=9)
    first_epoch = [dataset[index]["horizon"] for index in range(len(dataset))]
    dataset.set_epoch(1)
    second_epoch = [dataset[index]["horizon"] for index in range(len(dataset))]
    assert all(1 <= horizon <= 4 for horizon in first_epoch + second_epoch)
    assert first_epoch != second_epoch


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

