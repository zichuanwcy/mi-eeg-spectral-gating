"""Analyze stage-4 multidataset experiment results.

Expected inputs:
  results/stage4/physionet/*.json
  results/stage4/openbmi/*.json

Outputs:
  results/stage4/stage4_summary.csv
  results/stage4/stage4_subject_level.csv
  results/stage4/stage4_stats.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError as exc:
    raise ImportError("analyze_stage4_results.py requires pandas") from exc

try:
    from scipy import stats
except ImportError as exc:
    raise ImportError("analyze_stage4_results.py requires scipy") from exc

ROOT = Path("results") / "stage4"

EXPERIMENTS = [
    ("physionet", "Physionet_Traditional"),
    ("physionet", "Physionet_F0"),
    ("physionet", "Physionet_SE"),
    ("physionet", "Physionet_F0_allch"),
    ("physionet", "Physionet_SE_allch"),
    ("openbmi", "OpenBMI_Traditional"),
    ("openbmi", "OpenBMI_F0"),
    ("openbmi", "OpenBMI_SE"),
    ("openbmi", "OpenBMI_F0_allch"),
    ("openbmi", "OpenBMI_SE_allch"),
]

COMPARISONS = [
    ("Physionet_widthband_gain", "physionet", "Physionet_Traditional", "Physionet_F0"),
    ("Physionet_se_gain", "physionet", "Physionet_F0", "Physionet_SE"),
    ("Physionet_channel_density_f0", "physionet", "Physionet_F0", "Physionet_F0_allch"),
    ("Physionet_channel_density_se", "physionet", "Physionet_SE", "Physionet_SE_allch"),
    ("OpenBMI_widthband_gain", "openbmi", "OpenBMI_Traditional", "OpenBMI_F0"),
    ("OpenBMI_se_gain", "openbmi", "OpenBMI_F0", "OpenBMI_SE"),
    ("OpenBMI_channel_density_f0", "openbmi", "OpenBMI_F0", "OpenBMI_F0_allch"),
    ("OpenBMI_channel_density_se", "openbmi", "OpenBMI_SE", "OpenBMI_SE_allch"),
]


def load_result(dataset: str, exp_id: str) -> dict | None:
    path = ROOT / dataset / f"{exp_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def per_subject_series(result: dict) -> pd.Series:
    vals = result.get("per_subject_mean", {})
    if isinstance(vals, dict):
        return pd.Series({str(k): float(v) for k, v in vals.items()}, name=result["exp_id"])
    # Backward compatibility with BCI-style list outputs.
    return pd.Series({f"S{i+1}": float(v) for i, v in enumerate(vals)}, name=result["exp_id"])


def bootstrap_ci(diff: np.ndarray, n_boot: int = 10000, seed: int = 20260703) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float)
    if len(diff) == 0:
        return float("nan"), float("nan")
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(diff, size=len(diff), replace=True)
        means[i] = sample.mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def paired_stats(a: pd.Series, b: pd.Series) -> dict:
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    if joined.empty:
        return {"n": 0, "mean_delta": None, "std_delta": None}
    diff = joined.iloc[:, 1].to_numpy(dtype=float) - joined.iloc[:, 0].to_numpy(dtype=float)
    out = {
        "n": int(len(diff)),
        "mean_ref": float(joined.iloc[:, 0].mean()),
        "mean_new": float(joined.iloc[:, 1].mean()),
        "mean_delta": float(diff.mean()),
        "std_delta": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
        "positive_count": int(np.sum(diff > 0)),
        "negative_count": int(np.sum(diff < 0)),
    }
    if len(diff) >= 2:
        t_res = stats.ttest_rel(joined.iloc[:, 1], joined.iloc[:, 0])
        out["paired_t_stat"] = float(t_res.statistic)
        out["paired_t_p"] = float(t_res.pvalue)
        try:
            w_res = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
            out["wilcoxon_stat"] = float(w_res.statistic)
            out["wilcoxon_p"] = float(w_res.pvalue)
        except ValueError:
            out["wilcoxon_stat"] = None
            out["wilcoxon_p"] = None
    ci_lo, ci_hi = bootstrap_ci(diff)
    out["bootstrap95_delta"] = [ci_lo, ci_hi]
    return out


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    loaded: Dict[str, dict] = {}
    summary_rows: List[dict] = []
    subject_frames: List[pd.DataFrame] = []

    for dataset, exp_id in EXPERIMENTS:
        result = load_result(dataset, exp_id)
        if result is None:
            continue
        key = f"{dataset}:{exp_id}"
        loaded[key] = result
        summary_rows.append({
            "dataset": dataset,
            "exp_id": exp_id,
            "protocol": result.get("protocol"),
            "channel_mode": result.get("channel_mode"),
            "n_subjects": result.get("n_subjects", len(result.get("per_subject_mean", {}))),
            "mean": result.get("loso_mean"),
            "std": result.get("loso_std"),
            "kappa": result.get("mean_kappa"),
        })
        s = per_subject_series(result).rename("accuracy")
        df = s.reset_index().rename(columns={"index": "subject"})
        df["dataset"] = dataset
        df["exp_id"] = exp_id
        df["channel_mode"] = result.get("channel_mode")
        subject_frames.append(df[["dataset", "exp_id", "channel_mode", "subject", "accuracy"]])

    summary = pd.DataFrame(summary_rows)
    subject_level = pd.concat(subject_frames, ignore_index=True) if subject_frames else pd.DataFrame()
    summary.to_csv(ROOT / "stage4_summary.csv", index=False, encoding="utf-8-sig")
    subject_level.to_csv(ROOT / "stage4_subject_level.csv", index=False, encoding="utf-8-sig")

    stats_payload = {}
    series_cache = {key: per_subject_series(value) for key, value in loaded.items()}
    for name, dataset, ref, new in COMPARISONS:
        ref_key = f"{dataset}:{ref}"
        new_key = f"{dataset}:{new}"
        if ref_key not in series_cache or new_key not in series_cache:
            stats_payload[name] = {"status": "missing_result", "ref": ref, "new": new}
            continue
        stats_payload[name] = paired_stats(series_cache[ref_key], series_cache[new_key])
        stats_payload[name]["ref"] = ref
        stats_payload[name]["new"] = new
        stats_payload[name]["dataset"] = dataset

    # Combined SE gain across datasets, using main channel setting only.
    combined_pairs = []
    for dataset, ref, new in [("physionet", "Physionet_F0", "Physionet_SE"),
                              ("openbmi", "OpenBMI_F0", "OpenBMI_SE")]:
        ref_key = f"{dataset}:{ref}"
        new_key = f"{dataset}:{new}"
        if ref_key in series_cache and new_key in series_cache:
            joined = pd.concat([series_cache[ref_key], series_cache[new_key]], axis=1, join="inner").dropna()
            joined.index = [f"{dataset}:{idx}" for idx in joined.index]
            combined_pairs.append(joined)
    if combined_pairs:
        combined = pd.concat(combined_pairs, axis=0)
        stats_payload["Combined_se_gain"] = paired_stats(combined.iloc[:, 0], combined.iloc[:, 1])
        stats_payload["Combined_se_gain"]["datasets"] = ["physionet", "openbmi"]

    (ROOT / "stage4_stats.json").write_text(
        json.dumps(stats_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote {ROOT / 'stage4_summary.csv'}")
    print(f"Wrote {ROOT / 'stage4_subject_level.csv'}")
    print(f"Wrote {ROOT / 'stage4_stats.json'}")


if __name__ == "__main__":
    main()
