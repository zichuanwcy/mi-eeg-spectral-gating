"""OpenBMI MI loader for stage-4 experiments.

The loader is intentionally conservative:
  - Read channel names from the MAT file `chan` field when available.
  - Prefer BCI IV-2a matched 22 channels; fall back to OpenBMI MI20 if needed.
  - Downsample 1000 Hz data to 250 Hz with scipy.signal.resample_poly.
  - Keep subject-level data isolated; folds are generated in mi_eeg.training.runner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io
from scipy import signal

from mi_eeg.preprocessing.alignment import compute_ea_transform, apply_ea_transform
from mi_eeg.preprocessing.filtering import bandpass_filter

BCI2A_MATCHED_CHANNELS = [
    "FZ", "FC3", "FC1", "FCZ", "FC2", "FC4",
    "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
    "CP3", "CP1", "CPZ", "CP2", "CP4",
    "P1", "PZ", "P2", "POZ",
]

OPENBMI_MI20_CHANNELS = [
    "FC5", "FC3", "FC1", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "CZ", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
]


def _norm_ch(name: object) -> str:
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="ignore")
    if isinstance(name, np.ndarray):
        name = name.squeeze()
        if name.size == 1:
            name = name.item()
    return str(name).strip().replace(".", "").replace(" ", "").upper()


def _has_field(obj: object, name: str) -> bool:
    if isinstance(obj, dict):
        return name in obj
    return hasattr(obj, name)


def _get_field(obj: object, names: Tuple[str, ...]):
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    raise KeyError(f"None of fields found in MAT object: {names}")


def _extract_chan_names(obj: object, n_channels: int) -> List[str]:
    if not _has_field(obj, "chan"):
        return [f"CH{i+1}" for i in range(n_channels)]
    chan = np.asarray(_get_field(obj, ("chan",))).squeeze()
    names = [_norm_ch(c) for c in chan.flatten()]
    if len(names) != n_channels:
        # Some MATLAB structs store chan as nested cells; keep a robust fallback.
        names = []
        for item in chan.flatten():
            names.append(_norm_ch(item))
    if len(names) != n_channels:
        return [f"CH{i+1}" for i in range(n_channels)]
    return names


def _find_mat(data_dir: str | Path, subject: int, session: Optional[int] = None) -> Path:
    root = Path(data_dir)
    subj = f"subj{subject:02d}"
    patterns = []
    if session is not None:
        patterns.extend([
            f"sess{session:02d}_{subj}_EEG_MI.mat",
            f"sess{session}_{subj}_EEG_MI.mat",
            f"*sess*{session}*subj*{subject:02d}*MI*.mat",
        ])
    patterns.extend([
        f"*{subj}*EEG_MI*.mat",
        f"*subj{subject:02d}*MI*.mat",
    ])
    for pattern in patterns:
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Cannot find OpenBMI MI mat for subject {subject}, session={session}")



def _as_trials_channels_time(x: np.ndarray, n_channels_hint: Optional[int] = None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected 3D EEG array, got shape {x.shape}")

    # Common OpenBMI shapes include (time, channels, trials) or (trials, channels, time).
    dims = x.shape
    ch_axis = None
    if n_channels_hint is not None:
        for axis, size in enumerate(dims):
            if size == n_channels_hint:
                ch_axis = axis
                break
    if ch_axis is None:
        # Channels are usually the middle-sized dimension around 62.
        candidates = [i for i, s in enumerate(dims) if 16 <= s <= 128]
        ch_axis = candidates[0] if candidates else 1

    remaining = [i for i in range(3) if i != ch_axis]
    # Time axis is normally the larger of the remaining dimensions.
    time_axis = remaining[0] if dims[remaining[0]] > dims[remaining[1]] else remaining[1]
    trial_axis = remaining[1] if time_axis == remaining[0] else remaining[0]
    return np.transpose(x, (trial_axis, ch_axis, time_axis)).astype(np.float32)


def _extract_labels(obj: object) -> np.ndarray:
    y = _get_field(obj, ("y_dec", "y_class", "y", "label", "labels"))
    y = np.asarray(y).squeeze().astype(np.int64)
    # Convert to 0/1. OpenBMI commonly uses 1/2 for left/right.
    unique = sorted(np.unique(y).tolist())
    mapping = {old: i for i, old in enumerate(unique)}
    y = np.asarray([mapping[int(v)] for v in y], dtype=np.int64)
    if len(np.unique(y)) != 2:
        raise ValueError(f"OpenBMI MI expected 2 classes, got labels {unique}")
    return y


def _downsample_to_250(X: np.ndarray, orig_sfreq: float) -> Tuple[np.ndarray, float]:
    if abs(orig_sfreq - 250.0) < 1e-6:
        return X.astype(np.float32), 250.0
    if abs(orig_sfreq - 1000.0) < 1e-6:
        X_ds = signal.resample_poly(X, up=1, down=4, axis=2)
        return X_ds.astype(np.float32), 250.0
    # General fallback.
    target_len = int(round(X.shape[2] * 250.0 / orig_sfreq))
    X_ds = signal.resample(X, target_len, axis=2)
    return X_ds.astype(np.float32), 250.0


def _select_channels(names: List[str], channel_mode: str) -> Tuple[List[str], List[int], str]:
    lookup = {ch: i for i, ch in enumerate(names)}
    if channel_mode == "all":
        return names, list(range(len(names))), f"all{len(names)}"

    if channel_mode == "matched":
        missing = [ch for ch in BCI2A_MATCHED_CHANNELS if ch not in lookup]
        if not missing:
            return BCI2A_MATCHED_CHANNELS, [lookup[ch] for ch in BCI2A_MATCHED_CHANNELS], "matched22"
        missing_mi20 = [ch for ch in OPENBMI_MI20_CHANNELS if ch not in lookup]
        if missing_mi20:
            raise ValueError(
                f"OpenBMI neither matched22 nor MI20 available; "
                f"matched missing={missing}, mi20 missing={missing_mi20}"
            )
        return OPENBMI_MI20_CHANNELS, [lookup[ch] for ch in OPENBMI_MI20_CHANNELS], "mi20_fallback"

    if channel_mode == "mi20":
        missing = [ch for ch in OPENBMI_MI20_CHANNELS if ch not in lookup]
        if missing:
            raise ValueError(f"OpenBMI MI20 channels missing: {missing}")
        return OPENBMI_MI20_CHANNELS, [lookup[ch] for ch in OPENBMI_MI20_CHANNELS], "mi20"

    raise ValueError(f"Unsupported OpenBMI channel_mode: {channel_mode}")


def _extract_event_starts(obj: object) -> np.ndarray:
    starts = _get_field(obj, ("t", "trial", "trials", "event", "events", "pos"))
    starts = np.asarray(starts).squeeze().astype(np.int64)
    if starts.ndim != 1:
        starts = starts.reshape(-1)
    # MATLAB sample indices are commonly 1-based.
    if len(starts) and starts.min() >= 1:
        starts = starts - 1
    return starts


def _continuous_to_trials(
    raw_x: np.ndarray,
    starts: np.ndarray,
    fs: float,
    t_start: float,
    t_end: float,
    path: Path,
) -> np.ndarray:
    raw_x = np.asarray(raw_x, dtype=np.float32)
    if raw_x.ndim != 2:
        raise ValueError(f"Expected 2D continuous EEG array, got shape {raw_x.shape}")

    # Normalize to samples x channels.
    if raw_x.shape[0] < raw_x.shape[1]:
        raw_x = raw_x.T

    start_offset = int(round(t_start * fs))
    stop_offset = int(round(t_end * fs))
    n_times = stop_offset - start_offset
    if n_times <= 0:
        raise ValueError(f"Invalid OpenBMI time window: t_start={t_start}, t_end={t_end}")

    trials = []
    dropped = 0
    for start in starts:
        a = int(start) + start_offset
        b = int(start) + stop_offset
        if a < 0 or b > raw_x.shape[0]:
            dropped += 1
            continue
        trials.append(raw_x[a:b].T)

    if not trials:
        raise ValueError(f"No valid OpenBMI trials could be cut from {path}")
    if dropped:
        print(f"[OpenBMI] dropped {dropped} boundary trials in {path.name}")
    return np.stack(trials).astype(np.float32)


def _load_one_block(
    obj: object,
    path: Path,
    t_start: float,
    t_end: float,
) -> Tuple[np.ndarray, np.ndarray, List[str], float]:
    x = _get_field(obj, ("x", "X", "data", "EEG"))
    raw_x = np.asarray(x)
    fs = float(np.asarray(_get_field(obj, ("fs", "srate", "sfreq"))).squeeze()) if any(
        _has_field(obj, name) for name in ("fs", "srate", "sfreq")
    ) else 1000.0
    y = _extract_labels(obj)

    if raw_x.ndim == 3:
        # Guess channel count before rearranging.
        n_ch_hint = None
        for size in raw_x.shape:
            if 16 <= size <= 128:
                n_ch_hint = int(size)
                break
        X = _as_trials_channels_time(raw_x, n_ch_hint)
    elif raw_x.ndim == 2:
        starts = _extract_event_starts(obj)
        if len(starts) != len(y):
            raise ValueError(f"Event/label mismatch in {path}: t={len(starts)}, y={len(y)}")
        X = _continuous_to_trials(raw_x, starts, fs, t_start, t_end, path)
        if len(X) != len(y):
            # Only possible when boundary trials are dropped.
            y = y[:len(X)]
    else:
        raise ValueError(f"Expected 2D or 3D EEG array, got shape {raw_x.shape}")

    names = _extract_chan_names(obj, X.shape[1])
    if len(y) != len(X):
        raise ValueError(f"Label/trial mismatch in {path}: X={len(X)}, y={len(y)}")
    return X, y, names, fs


def _load_one_mat(
    path: Path,
    t_start: float = 0.0,
    t_end: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray, List[str], float]:
    mat = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    block_names = ("EEG_MI_train", "EEG_MI_test", "EEG_MI", "MI")
    blocks = [mat[name] for name in block_names if name in mat]
    if not blocks:
        blocks = [mat]

    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    names_ref: Optional[List[str]] = None
    fs_ref: Optional[float] = None

    for block in blocks:
        X, y, names, fs = _load_one_block(block, path, t_start, t_end)
        X_parts.append(X)
        y_parts.append(y)
        if names_ref is None:
            names_ref = names
        elif names != names_ref:
            raise ValueError(f"Channel names differ across blocks in {path}")
        if fs_ref is None:
            fs_ref = fs
        elif abs(fs - fs_ref) > 1e-6:
            raise ValueError(f"Sampling rate differs across blocks in {path}: {fs_ref} vs {fs}")

    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0), names_ref or [], fs_ref or 1000.0


def load_openbmi_subject(
    subject: int,
    cfg: dict,
    data_dir: str | Path,
    channel_mode: str = "matched",
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Load both available OpenBMI MI sessions for one subject."""
    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    selected_channels: Optional[List[str]] = None
    resolved_mode: Optional[str] = None
    fs_out = 250.0

    # Try session-specific files first, then fall back to any subject MI mat.
    candidate_paths = []
    for sess in (1, 2):
        try:
            candidate_paths.append(_find_mat(data_dir, subject, sess))
        except FileNotFoundError:
            pass
    if not candidate_paths:
        candidate_paths = [_find_mat(data_dir, subject, None)]
    # Avoid duplicate path when fallback pattern catches the same file.
    candidate_paths = sorted(set(candidate_paths))

    bandpass_freq = cfg.get("bandpass_freq") if cfg.get("bandpass", False) else None
    filter_order = int(cfg.get("filter_order", 4))

    for path in candidate_paths:
        X, y, names, fs = _load_one_mat(path, float(cfg.get("t_start", 0.0)), float(cfg.get("t_end", 4.0)))
        chosen, picks, mode = _select_channels(names, channel_mode)
        X = X[:, picks, :]
        X, fs_out = _downsample_to_250(X, fs)
        if bandpass_freq is not None:
            X = np.stack([
                bandpass_filter(X[i], bandpass_freq[0], bandpass_freq[1], fs_out, filter_order)
                for i in range(len(X))
            ]).astype(np.float32)
        selected_channels = chosen
        resolved_mode = mode
        X_parts.append(X)
        y_parts.append(y)

    X_all = np.concatenate(X_parts, axis=0).astype(np.float32)
    y_all = np.concatenate(y_parts, axis=0).astype(np.int64)

    meta = {
        "subject": subject,
        "files": [str(p) for p in candidate_paths],
        "n_trials": int(len(y_all)),
        "class_counts": {str(i): int(np.sum(y_all == i)) for i in range(2)},
        "channels": selected_channels,
        "channel_mode_resolved": resolved_mode,
        "sfreq": fs_out,
    }
    return X_all, y_all, meta


def load_openbmi_dataset(
    cfg: dict,
    data_dir: str | Path,
    out_dir: str | Path,
) -> Tuple[Dict[int, dict], dict]:
    subjects = cfg.get("subjects") or list(range(1, 55))
    channel_mode = cfg.get("channel_mode", "matched")

    dataset: Dict[int, dict] = {}
    qc = {
        "dataset": "openbmi",
        "candidate_subjects": subjects,
        "included_subjects": [],
        "excluded_by_qc": {},
        "subjects": {},
    }
    channel_check = {}

    for subject in subjects:
        try:
            X, y, meta = load_openbmi_subject(subject, cfg, data_dir, channel_mode)
            counts = np.array([np.sum(y == i) for i in range(2)], dtype=int)
            if np.any(counts == 0) or np.isnan(X).any():
                qc["excluded_by_qc"][str(subject)] = {
                    "reason": "invalid_labels_or_nan",
                    "class_counts": {str(i): int(counts[i]) for i in range(2)},
                    "has_nan": bool(np.isnan(X).any()),
                }
                continue
            if cfg.get("alignment") == "ea":
                R = compute_ea_transform(X)
                X = apply_ea_transform(X, R)
            dataset[subject] = {"all": (X, y)}
            qc["subjects"][str(subject)] = meta
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
        raise RuntimeError("No OpenBMI subjects survived loading/QC")

    out = Path(out_dir) / "stage4" / "openbmi"
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_qc.json").write_text(json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "channel_check.json").write_text(json.dumps(channel_check, indent=2, ensure_ascii=False), encoding="utf-8")
    return dataset, qc
