"""Physionet EEG Motor Movement/Imagery Dataset loader for stage-4 experiments.

This module keeps the data contract used by mi_eeg.training.runner:
    subject_data[subject_id] = {"all": (X, y)}
where X has shape (n_trials, n_channels, n_timepoints) and y is 0-indexed.

Main protocol choices:
  - Use subject-level CV outside this loader; this loader never mixes subjects.
  - Use MI runs only: 4/8/12 and 6/10/14.
  - Map T1/T2 by run type to 4 classes:
      0 left fist, 1 right fist, 2 both fists, 3 both feet.
  - Apply optional bandpass within each EDF run before epoching.
  - Apply optional EA after all trials for a subject have been collected.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from mi_eeg.preprocessing.alignment import compute_ea_transform, apply_ea_transform
from mi_eeg.preprocessing.filtering import bandpass_filter

BCI2A_MATCHED_CHANNELS = [
    "FZ", "FC3", "FC1", "FCZ", "FC2", "FC4",
    "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
    "CP3", "CP1", "CPZ", "CP2", "CP4",
    "P1", "PZ", "P2", "POZ",
]

PHYSIONET_EXCLUDE = {88, 89, 92, 100}
PHYSIONET_MI_RUNS = (4, 6, 8, 10, 12, 14)
LEFT_RIGHT_RUNS = {4, 8, 12}
FISTS_FEET_RUNS = {6, 10, 14}


def _norm_ch(name: str) -> str:
    return name.strip().replace(".", "").replace(" ", "").upper()


def _find_edf(data_dir: str | Path, subject: int, run: int) -> Path:
    root = Path(data_dir)
    subj = f"S{subject:03d}"
    name = f"{subj}R{run:02d}.edf"
    candidates = [root / subj / name, root / name]
    for path in candidates:
        if path.exists():
            return path
    matches = list(root.rglob(name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Cannot find Physionet EDF for subject {subject}, run {run}: {name}")


def _select_channels(raw, channel_mode: str) -> Tuple[List[str], List[int], str]:
    names_norm = [_norm_ch(ch) for ch in raw.ch_names]
    lookup = {ch: i for i, ch in enumerate(names_norm)}

    if channel_mode == "matched":
        missing = [ch for ch in BCI2A_MATCHED_CHANNELS if ch not in lookup]
        if missing:
            raise ValueError(f"Physionet matched channels missing: {missing}")
        selected = BCI2A_MATCHED_CHANNELS
        return selected, [lookup[ch] for ch in selected], "matched22"

    if channel_mode == "all":
        # Physionet EDF files include 64 EEG channels in the standard montage.
        selected = names_norm
        return selected, list(range(len(names_norm))), "all64"

    raise ValueError(f"Unsupported Physionet channel_mode: {channel_mode}")


def _event_label(run: int, desc: str) -> Optional[int]:
    desc = desc.strip().upper()
    if desc == "T0":
        return None
    if run in LEFT_RIGHT_RUNS:
        if desc == "T1":
            return 0  # left fist imagery
        if desc == "T2":
            return 1  # right fist imagery
    if run in FISTS_FEET_RUNS:
        if desc == "T1":
            return 2  # both fists imagery
        if desc == "T2":
            return 3  # both feet imagery
    return None


def load_physionet_subject(
    subject: int,
    cfg: dict,
    data_dir: str | Path,
    channel_mode: str = "matched",
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load one Physionet subject as epoched MI trials."""
    try:
        import mne
        mne.set_log_level("WARNING")
    except ImportError as exc:
        raise ImportError("Physionet loader requires mne. Install with: pip install mne") from exc

    sfreq_expected = float(cfg.get("sfreq", 160.0))
    t_start = float(cfg.get("t_start", 0.0))
    t_end = float(cfg.get("t_end", 4.0))
    bandpass_freq = cfg.get("bandpass_freq") if cfg.get("bandpass", False) else None
    filter_order = int(cfg.get("filter_order", 4))

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    selected_channels: Optional[List[str]] = None
    resolved_mode: Optional[str] = None

    for run in PHYSIONET_MI_RUNS:
        edf_path = _find_edf(data_dir, subject, run)
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        picks_names, picks, mode_name = _select_channels(raw, channel_mode)
        selected_channels = picks_names
        resolved_mode = mode_name

        sfreq = float(raw.info["sfreq"])
        if abs(sfreq - sfreq_expected) > 1e-6:
            raise ValueError(f"Subject {subject} run {run} sfreq={sfreq}, expected {sfreq_expected}")

        data = raw.get_data(picks=picks).astype(np.float32)  # (C, T), volts
        if bandpass_freq is not None:
            data = bandpass_filter(data, bandpass_freq[0], bandpass_freq[1], sfreq, filter_order)

        start_offset = int(round(t_start * sfreq))
        n_tp = int(round((t_end - t_start) * sfreq))

        for ann in raw.annotations:
            label = _event_label(run, ann["description"])
            if label is None:
                continue
            start = int(round(float(ann["onset"]) * sfreq)) + start_offset
            end = start + n_tp
            if start < 0 or end > data.shape[1]:
                continue
            X_list.append(data[:, start:end])
            y_list.append(label)

    if not X_list:
        raise RuntimeError(f"No valid MI trials extracted for Physionet subject {subject}")

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)
    meta = {
        "subject": subject,
        "n_trials": int(len(y)),
        "class_counts": {str(i): int(np.sum(y == i)) for i in range(4)},
        "channels": selected_channels,
        "channel_mode_resolved": resolved_mode,
    }
    return X, y, meta


def load_physionet_dataset(
    cfg: dict,
    data_dir: str | Path,
    out_dir: str | Path,
) -> Tuple[Dict[int, dict], dict]:
    """Load all selected Physionet subjects and write dataset_qc.json/channel_check.json."""
    subjects = cfg.get("subjects") or [s for s in range(1, 110) if s not in PHYSIONET_EXCLUDE]
    channel_mode = cfg.get("channel_mode", "matched")
    min_class_trials = int(cfg.get("qc_min_class_trials", 10))

    dataset: Dict[int, dict] = {}
    qc = {
        "dataset": "physionet",
        "candidate_subjects": subjects,
        "excluded_predefined": sorted(PHYSIONET_EXCLUDE),
        "qc_min_class_trials": min_class_trials,
        "included_subjects": [],
        "excluded_by_qc": {},
        "subjects": {},
    }
    channel_check = {}

    for subject in subjects:
        try:
            X, y, meta = load_physionet_subject(subject, cfg, data_dir, channel_mode)
            counts = np.array([np.sum(y == i) for i in range(4)], dtype=int)
            meta["class_counts"] = {str(i): int(counts[i]) for i in range(4)}
            qc["subjects"][str(subject)] = meta
            if np.any(counts == 0) or counts.min() < min_class_trials:
                qc["excluded_by_qc"][str(subject)] = {
                    "reason": "insufficient_class_trials",
                    "class_counts": meta["class_counts"],
                }
                continue
            if cfg.get("alignment") == "ea":
                R = compute_ea_transform(X)
                X = apply_ea_transform(X, R)
            dataset[subject] = {"all": (X, y)}
            qc["included_subjects"].append(subject)
            if not channel_check:
                channel_check = {
                    "channel_mode_requested": channel_mode,
                    "channel_mode_resolved": meta.get("channel_mode_resolved"),
                    "channels": meta.get("channels"),
                    "n_channels": len(meta.get("channels") or []),
                }
        except Exception as exc:
            qc["excluded_by_qc"][str(subject)] = {"reason": "load_error", "error": str(exc)}

    if not dataset:
        raise RuntimeError("No Physionet subjects survived loading/QC")

    out = Path(out_dir) / "stage4" / "physionet"
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_qc.json").write_text(json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "channel_check.json").write_text(json.dumps(channel_check, indent=2, ensure_ascii=False), encoding="utf-8")
    return dataset, qc
