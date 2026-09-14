from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import Trajectory, save_dataset


def generate_synthetic_dataset(
    output_dir: str | Path,
    patients: int = 80,
    latent_dim: int = 8,
    action_dim: int = 4,
    min_timepoints: int = 3,
    max_timepoints: int = 6,
    seed: int = 7,
) -> dict:
    """Generate an action-conditioned stochastic system for pipeline smoke tests."""
    if patients < 3:
        raise ValueError("patients must be >= 3")
    if min_timepoints < 2 or max_timepoints < min_timepoints:
        raise ValueError("Invalid timepoint range")
    rng = np.random.default_rng(seed)
    transition = rng.normal(0.0, 0.18, size=(latent_dim, latent_dim)).astype(np.float32)
    spectral_norm = np.linalg.norm(transition, ord=2)
    transition *= 0.65 / max(spectral_norm, 1e-6)
    action_effect = rng.normal(0.0, 0.35, size=(action_dim, latent_dim)).astype(np.float32)
    trajectories = []
    for patient_index in range(patients):
        length = int(rng.integers(min_timepoints, max_timepoints + 1))
        latents = np.zeros((length, latent_dim), dtype=np.float32)
        actions = np.zeros((length - 1, action_dim), dtype=np.float32)
        delta_days = rng.integers(30, 181, size=length - 1).astype(np.float32)
        latents[0] = rng.normal(0.0, 0.8, size=latent_dim)
        patient_drift = rng.normal(0.0, 0.04, size=latent_dim).astype(np.float32)
        for step in range(length - 1):
            logits = np.concatenate([latents[step, : min(action_dim, latent_dim)], np.zeros(action_dim)])
            probabilities = np.exp(logits[:action_dim] - np.max(logits[:action_dim]))
            probabilities /= probabilities.sum()
            primary_action = int(rng.choice(action_dim, p=probabilities))
            actions[step, primary_action] = 1.0
            if rng.random() < 0.18:
                actions[step, int(rng.integers(action_dim))] = 1.0
            scaled_time = delta_days[step] / 90.0
            deterministic = np.tanh(latents[step] @ transition + actions[step] @ action_effect)
            noise_scale = 0.025 * np.sqrt(scaled_time) * (1.0 + 0.8 * np.linalg.norm(latents[step]))
            latents[step + 1] = (
                0.82 * latents[step]
                + scaled_time * 0.32 * deterministic
                + patient_drift
                + rng.normal(0.0, noise_scale, size=latent_dim)
            )
        trajectories.append(
            Trajectory(
                patient_id=f"SYN_{patient_index:04d}",
                latents=latents,
                actions=actions,
                delta_days=delta_days,
                timepoints=np.asarray([f"T{index + 1}" for index in range(length)]),
            )
        )
    return save_dataset(
        output_dir,
        trajectories,
        [f"synthetic_action_{index}" for index in range(action_dim)],
        extra_metadata={"kind": "synthetic_smoke_test", "seed": seed},
    )

