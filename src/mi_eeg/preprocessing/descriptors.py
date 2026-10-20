"""Subject-level unlabeled descriptors for adaptive EEG experiments.

All functions in this module operate on one subject at a time and never use
labels for feature extraction.  Labels may be passed only for QC bookkeeping by
callers, not for descriptor values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import signal
except Exception:  # pragma: no cover - handled at runtime by callers
    signal = None


Band = Tuple[str, float, float]

DEFAULT_BANDS: Sequence[Band] = (
    ("delta_0p5_4", 0.5, 4.0),
    ("theta_4_8", 4.0, 8.0),
    ("alpha_8_13", 8.0, 13.0),
    ("beta_13_30", 13.0, 30.0),
    ("gamma_30_40", 30.0, 40.0),
    ("high_40_100", 40.0, 100.0),
)


@dataclass(frozen=True)
class DescriptorConfig:
    sfreq: float
    bands: Sequence[Band] = DEFAULT_BANDS
    eps: float = 1e-8


def ensure_trials_channels_time(X: np.ndarray) -> np.ndarray:
    """Return X as float64 array shaped (trials, channels, time)."""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"Expected X with shape (trials, channels, time), got {arr.shape}")
    return arr


def bandpower_welch(X: np.ndarray, sfreq: float, bands: Sequence[Band] = DEFAULT_BANDS) -> Dict[str, float]:
    """Compute subject-level relative band powers using Welch PSD."""
    if signal is None:
        raise ImportError("scipy is required for bandpower_welch")
    X = ensure_trials_channels_time(X)
    nperseg = min(int(round(sfreq * 2.0)), X.shape[-1])
    freqs, psd = signal.welch(X, fs=sfreq, axis=-1, nperseg=nperseg)
    # Average over trials and channels, retaining frequency.
    psd_mean = np.nanmean(psd, axis=(0, 1))
    valid_total = (freqs >= min(b[1] for b in bands)) & (freqs <= min(max(b[2] for b in bands), freqs.max()))
    total_power = _trapz(psd_mean[valid_total], freqs[valid_total])
    out: Dict[str, float] = {}
    for name, lo, hi in bands:
        if lo >= freqs.max():
            out[f"bp_abs_{name}"] = float("nan")
            out[f"bp_rel_{name}"] = float("nan")
            continue
        mask = (freqs >= lo) & (freqs < min(hi, freqs.max() + 1e-12))
        power = _trapz(psd_mean[mask], freqs[mask]) if np.any(mask) else 0.0
        out[f"bp_abs_{name}"] = float(power)
        out[f"bp_rel_{name}"] = float(power / (total_power + 1e-12))
    out.update(peak_features(freqs, psd_mean, "alpha", 8.0, 13.0))
    out.update(peak_features(freqs, psd_mean, "beta", 13.0, 30.0))
    out["ratio_delta_alpha"] = safe_ratio(out.get("bp_abs_delta_0p5_4"), out.get("bp_abs_alpha_8_13"))
    out["ratio_delta_beta"] = safe_ratio(out.get("bp_abs_delta_0p5_4"), out.get("bp_abs_beta_13_30"))
    out["ratio_alpha_beta"] = safe_ratio(out.get("bp_abs_alpha_8_13"), out.get("bp_abs_beta_13_30"))
    return out


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    return float(np.trapz(y, x))


def peak_features(freqs: np.ndarray, psd: np.ndarray, prefix: str, lo: float, hi: float) -> Dict[str, float]:
    mask = (freqs >= lo) & (freqs <= hi)
    if not np.any(mask):
        return {f"{prefix}_peak_freq": float("nan"), f"{prefix}_peak_power": float("nan"), f"{prefix}_peak_sharpness": float("nan")}
    f = freqs[mask]
    p = psd[mask]
    idx = int(np.nanargmax(p))
    peak = float(p[idx])
    baseline = float(np.nanmedian(p))
    return {
        f"{prefix}_peak_freq": float(f[idx]),
        f"{prefix}_peak_power": peak,
        f"{prefix}_peak_sharpness": safe_ratio(peak, baseline),
    }


def safe_ratio(a: Optional[float], b: Optional[float], eps: float = 1e-12) -> float:
    if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return float(a / (b + eps))


def trial_covariances(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Return regularized per-trial covariance matrices."""
    X = ensure_trials_channels_time(X)
    covs = []
    for trial in X:
        z = trial - trial.mean(axis=1, keepdims=True)
        cov = (z @ z.T) / max(z.shape[1] - 1, 1)
        cov = cov + np.eye(cov.shape[0]) * eps
        covs.append(cov)
    return np.asarray(covs, dtype=np.float64)


def covariance_descriptors(X: np.ndarray) -> Dict[str, float]:
    covs = trial_covariances(X)
    mean_cov = np.nanmean(covs, axis=0)
    eig = np.linalg.eigvalsh(mean_cov)
    eig = np.maximum(eig, 1e-12)
    logeig = np.log(eig)
    traces = np.trace(covs, axis1=1, axis2=2)
    frob = np.linalg.norm(covs - mean_cov[None, :, :], axis=(1, 2))
    return {
        "cov_cond": float(eig.max() / eig.min()),
        "cov_logeig_mean": float(np.mean(logeig)),
        "cov_logeig_std": float(np.std(logeig)),
        "cov_logeig_range": float(np.max(logeig) - np.min(logeig)),
        "cov_trace_mean": float(np.mean(traces)),
        "cov_trace_std": float(np.std(traces)),
        "trial_cov_fro_dispersion": float(np.mean(frob)),
        "trial_cov_fro_dispersion_std": float(np.std(frob)),
    }


def correlation_descriptors(X: np.ndarray) -> Dict[str, float]:
    """Return low-dimensional, unlabeled channel-correlation descriptors.

    A full correlation upper triangle has 190--231 entries for the channel
    montages used here, which is not appropriate for 54/105-subject gate
    fitting. These summaries retain subject-level spatial synchrony while
    keeping the descriptor dimension small and stable.
    """
    X = ensure_trials_channels_time(X)
    trial_corrs = []
    for trial in X:
        corr = np.corrcoef(trial)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        corr = np.clip((corr + corr.T) * 0.5, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)
        trial_corrs.append(corr)
    corrs = np.asarray(trial_corrs, dtype=np.float64)
    mean_corr = np.nanmean(corrs, axis=0)
    tri = np.triu_indices(mean_corr.shape[0], k=1)
    upper = mean_corr[tri]
    abs_upper = np.abs(upper)
    channel_strength = (np.sum(np.abs(mean_corr), axis=1) - 1.0) / max(mean_corr.shape[0] - 1, 1)
    trial_mean_abs = np.mean(np.abs(corrs[:, tri[0], tri[1]]), axis=1)
    return {
        "corr_mean": float(np.mean(upper)),
        "corr_std": float(np.std(upper)),
        "corr_abs_mean": float(np.mean(abs_upper)),
        "corr_abs_std": float(np.std(abs_upper)),
        "corr_abs_q25": float(np.quantile(abs_upper, 0.25)),
        "corr_abs_q75": float(np.quantile(abs_upper, 0.75)),
        "corr_positive_fraction": float(np.mean(upper > 0.0)),
        "corr_strong_abs_fraction": float(np.mean(abs_upper >= 0.5)),
        "corr_channel_strength_mean": float(np.mean(channel_strength)),
        "corr_channel_strength_std": float(np.std(channel_strength)),
        "trial_corr_abs_mean": float(np.mean(trial_mean_abs)),
        "trial_corr_abs_std": float(np.std(trial_mean_abs)),
    }


def covariance_extended_descriptors(X: np.ndarray) -> Dict[str, float]:
    """Return complementary covariance stability summaries for gate ablation."""
    covs = trial_covariances(X)
    mean_cov = np.nanmean(covs, axis=0)
    traces = np.trace(covs, axis1=1, axis2=2)
    deviations = np.linalg.norm(covs - mean_cov[None, :, :], axis=(1, 2))
    diag = np.diag(mean_cov)
    offdiag = mean_cov[np.triu_indices(mean_cov.shape[0], k=1)]
    return {
        "cov_diag_cv": float(np.std(diag) / (np.mean(diag) + 1e-12)),
        "cov_offdiag_abs_mean": float(np.mean(np.abs(offdiag))),
        "cov_offdiag_std": float(np.std(offdiag)),
        "cov_trace_cv": float(np.std(traces) / (np.mean(traces) + 1e-12)),
        "cov_trial_fro_cv": float(np.std(deviations) / (np.mean(deviations) + 1e-12)),
        "cov_fro_per_trace": float(np.linalg.norm(mean_cov, ord="fro") / (float(np.trace(mean_cov)) + 1e-12)),
    }


def riemannian_distance_spd(A: np.ndarray, B: np.ndarray) -> float:
    """Affine-invariant distance between SPD matrices."""
    eig_a, vec_a = np.linalg.eigh(A)
    eig_a = np.maximum(eig_a, 1e-12)
    inv_sqrt_a = (vec_a * (1.0 / np.sqrt(eig_a))) @ vec_a.T
    C = inv_sqrt_a @ B @ inv_sqrt_a
    eig_c = np.linalg.eigvalsh(C)
    eig_c = np.maximum(eig_c, 1e-12)
    return float(np.linalg.norm(np.log(eig_c)))


def subject_covariance(X: np.ndarray) -> np.ndarray:
    covs = trial_covariances(X)
    return np.nanmean(covs, axis=0)


def source_similarity_descriptors(
    subject_cov: np.ndarray,
    source_covs: Mapping[str, np.ndarray],
    ks: Sequence[int] = (3, 5, 10, 20),
) -> Dict[str, float]:
    """Distances from one subject to a source-subject covariance pool."""
    if not source_covs:
        return {}
    distances = [(sid, riemannian_distance_spd(subject_cov, cov)) for sid, cov in source_covs.items()]
    vals = np.asarray([d for _, d in distances], dtype=float)
    distances.sort(key=lambda x: x[1])
    out = {
        "src_dist_mean": float(np.mean(vals)),
        "src_dist_std": float(np.std(vals)),
        "src_dist_min": float(np.min(vals)),
        "src_dist_max": float(np.max(vals)),
    }
    for k in ks:
        kk = min(k, len(distances))
        if kk <= 0:
            continue
        nearest = [d for _, d in distances[:kk]]
        out[f"src_top{kk}_dist_mean"] = float(np.mean(nearest))
        out[f"src_top{kk}_dist_std"] = float(np.std(nearest))
    return out


def compute_subject_descriptors(
    X: np.ndarray,
    sfreq: float,
    dataset: str,
    subject: str,
    channel_mode: str = "unknown",
    source_covs: Optional[Mapping[str, np.ndarray]] = None,
) -> Dict[str, float | str | int]:
    X = ensure_trials_channels_time(X)
    row: Dict[str, float | str | int] = {
        "dataset": dataset,
        "subject": subject,
        "n_trials": int(X.shape[0]),
        "n_channels": int(X.shape[1]),
        "n_times": int(X.shape[2]),
        "sfreq": float(sfreq),
        "channel_mode": channel_mode,
    }
    row.update(bandpower_welch(X, sfreq))
    row.update(covariance_descriptors(X))
    row.update(covariance_extended_descriptors(X))
    row.update(correlation_descriptors(X))
    cov = subject_covariance(X)
    if source_covs:
        row.update(source_similarity_descriptors(cov, source_covs))
    return row
