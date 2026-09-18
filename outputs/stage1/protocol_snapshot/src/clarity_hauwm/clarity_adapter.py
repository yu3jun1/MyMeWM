from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from .data import Trajectory, save_dataset


AGENT_ALIASES = {
    "avastin": "bevacizumab",
    "temodar": "temozolomide",
    "optune ttf": "tumor treating fields",
    "gamma tiles": "gamma tile",
}


def _normalize_text(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
    return AGENT_ALIASES.get(text, text)


def action_tokens(actions: object) -> set[str]:
    if not isinstance(actions, dict):
        return set()
    tokens: set[str] = set()
    for raw_category, entries in actions.items():
        if not isinstance(entries, list) or not entries:
            continue
        category = _normalize_text(raw_category)
        if category:
            tokens.add(f"category:{category}")
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            agent = _normalize_text(entry.get("agent", ""))
            action_type = _normalize_text(entry.get("type", ""))
            if agent:
                tokens.add(f"agent:{agent}")
            if action_type:
                tokens.add(f"type:{action_type}")
    return tokens


def _timepoint_number(value: object) -> int:
    match = re.search(r"(\d+)", str(value))
    if not match:
        raise ValueError(f"Cannot parse timepoint number from {value!r}")
    return int(match.group(1))


def latent_filename(patient_id: str, timepoint: object) -> str:
    return f"{patient_id}_Timepoint_{_timepoint_number(timepoint)}.npy"


def load_latent_vector(path: Path, pooling: str = "mean") -> np.ndarray:
    value = np.load(path, allow_pickle=False).astype(np.float32)
    if value.ndim == 1:
        return value
    if value.ndim != 2:
        raise ValueError(f"Expected [D] or [tokens,D] in {path}, got {value.shape}")
    if pooling == "mean":
        return value.mean(axis=0)
    if pooling == "flatten":
        return value.reshape(-1)
    raise ValueError(f"Unsupported pooling: {pooling}")


def _interval_tokens(timeline: list[dict], left: int, right: int, action_anchor: str) -> set[str]:
    if action_anchor == "source":
        indices: Iterable[int] = range(left, right)
    elif action_anchor == "destination":
        indices = range(left + 1, right + 1)
    else:
        raise ValueError("action_anchor must be source or destination")
    tokens: set[str] = set()
    for index in indices:
        tokens.update(action_tokens(timeline[index].get("actions", {})))
    return tokens


def build_clarity_trajectories(
    timeline_path: str | Path,
    latent_dir: str | Path,
    output_dir: str | Path,
    action_anchor: str = "source",
    pooling: str = "mean",
    min_token_count: int = 1,
) -> dict:
    timeline_path = Path(timeline_path)
    latent_dir = Path(latent_dir)
    with timeline_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    patients = payload.get("patients")
    if not isinstance(patients, dict):
        raise ValueError("Expected CLARITY timeline JSON with a patients object")

    token_counts: Counter[str] = Counter()
    for patient in patients.values():
        for timepoint in patient.get("timeline", []):
            token_counts.update(action_tokens(timepoint.get("actions", {})))
    vocabulary = sorted(token for token, count in token_counts.items() if count >= min_token_count)
    if not vocabulary:
        raise ValueError("No treatment tokens found")
    token_index = {token: index for index, token in enumerate(vocabulary)}

    trajectories = []
    skipped_missing_latent = 0
    for patient_id, patient in patients.items():
        timeline = sorted(
            patient.get("timeline", []),
            key=lambda timepoint: (float(timepoint.get("mri_day", 0)), _timepoint_number(timepoint["tp_id"])),
        )
        available = []
        for timeline_index, timepoint in enumerate(timeline):
            path = latent_dir / latent_filename(str(patient_id), timepoint["tp_id"])
            if not path.exists():
                skipped_missing_latent += 1
                continue
            available.append((timeline_index, timepoint, path))
        if len(available) < 2:
            continue
        latents = np.stack([load_latent_vector(record[2], pooling) for record in available])
        actions = np.zeros((len(available) - 1, len(vocabulary)), dtype=np.float32)
        delta_days = np.zeros(len(available) - 1, dtype=np.float32)
        valid = True
        for interval_index, (current, following) in enumerate(zip(available[:-1], available[1:])):
            left_index, current_timepoint, _ = current
            right_index, following_timepoint, _ = following
            delta = float(following_timepoint.get("mri_day", 0)) - float(current_timepoint.get("mri_day", 0))
            if delta <= 0:
                valid = False
                break
            delta_days[interval_index] = delta
            for token in _interval_tokens(timeline, left_index, right_index, action_anchor):
                if token in token_index:
                    actions[interval_index, token_index[token]] = 1.0
        if not valid:
            continue
        trajectories.append(
            Trajectory(
                patient_id=str(patient_id),
                latents=latents,
                actions=actions,
                delta_days=delta_days,
                timepoints=np.asarray([str(record[1]["tp_id"]) for record in available]),
            )
        )
    if not trajectories:
        raise ValueError("No patient has at least two valid latent timepoints")
    return save_dataset(
        output_dir,
        trajectories,
        vocabulary,
        extra_metadata={
            "kind": "clarity",
            "timeline": str(timeline_path.resolve()),
            "latent_dir": str(latent_dir.resolve()),
            "latent_extraction": json.loads((latent_dir / "extraction_metadata.json").read_text(encoding="utf-8")) if (latent_dir / "extraction_metadata.json").is_file() else None,
            "action_anchor": action_anchor,
            "pooling": pooling,
            "min_token_count": min_token_count,
            "missing_latent_timepoints": skipped_missing_latent,
        },
    )

