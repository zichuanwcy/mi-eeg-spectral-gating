"""Fit and evaluate subject-descriptor gate candidates.

Use evaluate.py for the retained repeated evaluation. Advanced
candidate options are available with python -m mi_eeg.gating.experiments --help.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    from scipy import stats
    from sklearn.base import clone
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError as exc:
    raise ImportError("run_stage5_direction1_experiments.py requires scipy and scikit-learn") from exc

DATASETS = ["physionet", "openbmi"]
DATASET_DEFAULT = {"physionet": "f0", "openbmi": "traditional"}

PHYSIONET_ALPHA_FEATURES = [
    "desc_alpha_peak_sharpness", "desc_ratio_alpha_beta", "desc_bp_rel_alpha_8_13",
    "desc_ratio_delta_alpha", "desc_alpha_peak_freq",
]
OPENBMI_HIGHFREQ_SOURCE_FEATURES = [
    "desc_bp_abs_gamma_30_40", "desc_bp_abs_high_40_100", "desc_beta_peak_power",
    "desc_beta_peak_freq", "desc_beta_peak_sharpness", "desc_src_top5_dist_std",
    "desc_src_top10_dist_std", "desc_src_top20_dist_std", "desc_cov_logeig_mean",
]
COMMON_COMPACT_FEATURES = [
    "desc_alpha_peak_sharpness", "desc_alpha_peak_freq", "desc_beta_peak_power",
    "desc_beta_peak_freq", "desc_ratio_delta_alpha", "desc_ratio_alpha_beta",
    "desc_cov_logeig_mean", "desc_trial_cov_fro_dispersion",
]
GROUP_PATTERNS = {
    "low_frequency": ["delta", "theta", "0_4", "4_8", "low"],
    "alpha_beta": ["alpha", "beta"],
    "high_frequency": ["gamma", "high", "30_40", "40_100"],
    "source_distance": ["src_", "source", "dist"],
    "covariance": ["cov", "logeig", "fro", "riemann"],
    "correlation": ["corr_", "correlation"],
}


def read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def as_float(value, default=float("nan")) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def finite(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values if np.isfinite(float(v))]


def safe_mean(values: Iterable[float]) -> float:
    vals = finite(values)
    return mean(vals) if vals else float("nan")


def unique(paths: Sequence[Path]) -> List[Path]:
    seen, out = set(), []
    for p in paths:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def find_dataset_file(results_dir: Path, dataset: str, gate_dir: Path | None) -> Path:
    roots = unique([
        gate_dir or results_dir,
        results_dir,
        results_dir / "stage5", results_dir / "stage5" / "gate",
        results_dir / "stage5" / "selected_conservative_gate",
        Path("results"), Path("results") / "stage5", Path("results") / "stage5" / "gate",
        Path("results") / "stage5" / "selected_conservative_gate",
    ])
    names = [f"{dataset}_gate_dataset.csv", f"{dataset}_dataset.csv", f"{dataset}_subject_gate_dataset.csv"]
    candidates: List[Path] = []
    for root in roots:
        for name in names:
            candidates.append(root / name)
        if root.exists():
            for name in names:
                candidates.extend(root.rglob(name))
    for path in unique(candidates):
        if path.exists() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(f"No gate dataset CSV found for {dataset}; searched under {', '.join(str(r) for r in roots)}")


def normalize_rows(rows: Sequence[dict]) -> List[dict]:
    out = []
    for row in rows:
        item = dict(row)
        trad = as_float(item.get("traditional"))
        f0 = as_float(item.get("f0"))
        if not np.isfinite(as_float(item.get("delta_f0_minus_traditional"))):
            item["delta_f0_minus_traditional"] = f0 - trad
        item["traditional"] = trad
        item["f0"] = f0
        item["oracle_accuracy"] = max(trad, f0)
        item["best_strategy"] = "f0" if f0 >= trad else "traditional"
        out.append(item)
    return out


def load_data(results_dir: Path, datasets: Sequence[str], gate_dir: Path | None) -> Dict[str, List[dict]]:
    data = {}
    for dataset in datasets:
        path = find_dataset_file(results_dir, dataset, gate_dir)
        rows = read_csv(path)
        if not rows:
            raise ValueError(f"Empty dataset file: {path}")
        missing = {"subject", "traditional", "f0"} - set(rows[0])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        data[dataset] = normalize_rows(rows)
        print(f"Loaded {dataset}: {len(rows)} subjects from {path}")
    return data


def desc_cols(rows: Sequence[dict]) -> List[str]:
    excluded = {"desc_source_file", "desc_channel_mode"}
    cols = [c for c in rows[0] if c.startswith("desc_") and c not in excluded]
    valid = []
    for col in cols:
        vals = finite(as_float(r.get(col)) for r in rows)
        if len(vals) >= 3 and len(set(round(v, 12) for v in vals)) > 1:
            valid.append(col)
    return valid


def present(rows: Sequence[dict], cols: Sequence[str]) -> List[str]:
    keys = set(rows[0])
    return [c for c in cols if c in keys]


def feature_sets(dataset: str, rows: Sequence[dict]) -> Dict[str, List[str]]:
    all_cols = desc_cols(rows)
    sets: Dict[str, List[str]] = {
        "all_features": all_cols,
        "common_compact": present(rows, COMMON_COMPACT_FEATURES),
    }
    if dataset == "physionet":
        sets["physionet_alpha"] = present(rows, PHYSIONET_ALPHA_FEATURES)
    if dataset == "openbmi":
        sets["openbmi_highfreq_source"] = present(rows, OPENBMI_HIGHFREQ_SOURCE_FEATURES)
    correlation_cols = [c for c in all_cols if c.startswith("desc_corr_") or c.startswith("desc_trial_corr_")]
    covariance_extended_cols = [
        c for c in all_cols
        if c.startswith("desc_cov_") or c.startswith("desc_trial_cov_")
    ]
    selected_cols = sets.get(selected_set(dataset), [])
    sets["correlation_only"] = correlation_cols
    sets["covariance_extended"] = covariance_extended_cols
    sets["selected_plus_correlation"] = list(dict.fromkeys([*selected_cols, *correlation_cols]))
    sets["selected_plus_covariance"] = list(dict.fromkeys([*selected_cols, *covariance_extended_cols]))
    lower = {c: c.lower() for c in all_cols}
    for name, pats in GROUP_PATTERNS.items():
        sets[name] = [c for c in all_cols if any(p in lower[c] for p in pats)]
    for name in list(GROUP_PATTERNS):
        removed = set(sets.get(name, []))
        if removed and len(removed) < len(all_cols):
            sets[f"all_minus_{name}"] = [c for c in all_cols if c not in removed]
    return {k: v for k, v in sets.items() if v}


def selected_set(dataset: str) -> str:
    return "openbmi_highfreq_source" if dataset == "openbmi" else "common_compact"


def resolve_sets(dataset: str, rows: Sequence[dict], names: Sequence[str]) -> List[str]:
    sets = feature_sets(dataset, rows)
    if not names or "all" in names:
        return list(sets)
    out = []
    for name in names:
        if name == "selected":
            name = selected_set(dataset)
        if name not in sets:
            raise ValueError(f"Unknown feature set {name!r} for {dataset}. Available: {sorted(sets)}")
        out.append(name)
    return list(dict.fromkeys(out))


def make_x(rows: Sequence[dict], features: Sequence[str]) -> np.ndarray:
    return np.asarray([[as_float(r.get(f)) for f in features] for r in rows], dtype=float)


def make_y(rows: Sequence[dict]) -> np.ndarray:
    return np.asarray([1 if as_float(r.get("delta_f0_minus_traditional")) > 0 else 0 for r in rows], dtype=int)


def models(seed: int) -> Dict[str, Pipeline]:
    return {
        "logreg": Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))]),
        "rf": Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", RandomForestClassifier(n_estimators=500, min_samples_leaf=3, class_weight="balanced", random_state=seed))]),
        "gb": Pipeline([("impute", SimpleImputer(strategy="median")), ("clf", GradientBoostingClassifier(random_state=seed))]),
    }


def splits_for(y: np.ndarray, n_splits: int, seed: int):
    counts = np.bincount(y.astype(int), minlength=2)
    if len(set(y.tolist())) < 2 or min(counts[counts > 0]) < 2:
        return []
    folds = min(n_splits, int(min(counts[counts > 0])))
    return list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def acc(row: dict, strategy: str) -> float:
    return as_float(row.get("f0")) if strategy == "f0" else as_float(row.get("traditional"))


def best_train_strategy(rows: Sequence[dict]) -> str:
    return "f0" if safe_mean(as_float(r.get("f0")) for r in rows) >= safe_mean(as_float(r.get("traditional")) for r in rows) else "traditional"


def opposite(strategy: str) -> str:
    return "traditional" if strategy == "f0" else "f0"


def oracle_recovery(value: float, best_single: float, oracle: float) -> float:
    denom = oracle - best_single
    return float("nan") if abs(denom) < 1e-12 else (value - best_single) / denom

# Experiment A: descriptor association.

def corr_ci(x: np.ndarray, y: np.ndarray, method: str, n_boot: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        xb, yb = x[idx], y[idx]
        if len(set(np.round(xb, 12))) < 2 or len(set(np.round(yb, 12))) < 2:
            continue
        r = stats.pearsonr(xb, yb)[0] if method == "pearson" else stats.spearmanr(xb, yb)[0]
        if np.isfinite(r):
            vals.append(float(r))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def perm_p(x: np.ndarray, y: np.ndarray, obs: float, method: str, n_perm: int, seed: int) -> float:
    if not np.isfinite(obs):
        return float("nan")
    rng = np.random.default_rng(seed)
    hit = valid = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        r = stats.pearsonr(x, yp)[0] if method == "pearson" else stats.spearmanr(x, yp)[0]
        if np.isfinite(r):
            valid += 1
            hit += int(abs(r) >= abs(obs))
    return (hit + 1) / (valid + 1) if valid else float("nan")


def run_association(data: Dict[str, List[dict]], out_dir: Path, n_perm: int, n_boot: int, seed: int) -> None:
    out = out_dir / "association"
    corr_rows, group_rows = [], []
    for dataset, rows in data.items():
        delta = np.asarray([as_float(r.get("delta_f0_minus_traditional")) for r in rows], dtype=float)
        best = np.asarray([1 if d > 0 else 0 for d in delta], dtype=int)
        for col in desc_cols(rows):
            x = np.asarray([as_float(r.get(col)) for r in rows], dtype=float)
            mask = np.isfinite(x) & np.isfinite(delta)
            if mask.sum() < 5 or len(set(np.round(x[mask], 12))) < 2:
                continue
            xm, ym = x[mask], delta[mask]
            pr, pp = stats.pearsonr(xm, ym)
            sr, sp = stats.spearmanr(xm, ym)
            pl, ph = corr_ci(xm, ym, "pearson", n_boot, seed)
            sl, sh = corr_ci(xm, ym, "spearman", n_boot, seed + 13)
            corr_rows.append({
                "dataset": dataset, "feature": col, "n": int(mask.sum()),
                "pearson_r": float(pr), "pearson_p": float(pp), "pearson_perm_p": perm_p(xm, ym, float(pr), "pearson", n_perm, seed),
                "pearson_ci_low": pl, "pearson_ci_high": ph,
                "spearman_r": float(sr), "spearman_p": float(sp), "spearman_perm_p": perm_p(xm, ym, float(sr), "spearman", n_perm, seed + 17),
                "spearman_ci_low": sl, "spearman_ci_high": sh, "abs_spearman_r": abs(float(sr)),
            })
            f0_vals = x[(best == 1) & np.isfinite(x)]
            tr_vals = x[(best == 0) & np.isfinite(x)]
            if len(f0_vals) >= 2 and len(tr_vals) >= 2:
                mw = stats.mannwhitneyu(f0_vals, tr_vals, alternative="two-sided")
                tt = stats.ttest_ind(f0_vals, tr_vals, equal_var=False, nan_policy="omit")
                group_rows.append({
                    "dataset": dataset, "feature": col, "n_f0_best": len(f0_vals), "n_traditional_best": len(tr_vals),
                    "f0_best_mean": float(np.mean(f0_vals)), "traditional_best_mean": float(np.mean(tr_vals)),
                    "mean_diff_f0_minus_traditional": float(np.mean(f0_vals) - np.mean(tr_vals)),
                    "mannwhitney_p": float(mw.pvalue), "welch_t_p": float(tt.pvalue),
                })
    corr_rows.sort(key=lambda r: (r["dataset"], -r["abs_spearman_r"]))
    group_rows.sort(key=lambda r: (r["dataset"], -abs(as_float(r["mean_diff_f0_minus_traditional"]))))
    write_csv(out / "descriptor_delta_correlations.csv", corr_rows)
    write_csv(out / "descriptor_delta_group_tests.csv", group_rows)
    lines = ["# Descriptor-Delta Association", "", "| Dataset | Feature | Pearson r | Spearman r | Spearman perm p | Spearman CI |", "|---|---|---:|---:|---:|---|"]
    for r in corr_rows[:50]:
        lines.append(f"| {r['dataset']} | {r['feature']} | {as_float(r['pearson_r']):+.4f} | {as_float(r['spearman_r']):+.4f} | {as_float(r['spearman_perm_p']):.4g} | [{as_float(r['spearman_ci_low']):+.3f}, {as_float(r['spearman_ci_high']):+.3f}] |")
    lines += ["", "## Largest Group Differences", "", "| Dataset | Feature | F0-best mean | Traditional-best mean | Diff | MW p |", "|---|---|---:|---:|---:|---:|"]
    for r in group_rows[:50]:
        lines.append(f"| {r['dataset']} | {r['feature']} | {as_float(r['f0_best_mean']):.4f} | {as_float(r['traditional_best_mean']):.4f} | {as_float(r['mean_diff_f0_minus_traditional']):+.4f} | {as_float(r['mannwhitney_p']):.4g} |")
    write_text(out / "descriptor_delta_association_report.md", "\n".join(lines) + "\n")


# Experiments B/C: baselines and conservative thresholds.

def summarize_subjects(subjects: Sequence[dict]) -> dict:
    selected = [as_float(r.get("selected_accuracy")) for r in subjects]
    oracle = [as_float(r.get("oracle_accuracy")) for r in subjects]
    regret = [as_float(r.get("regret")) for r in subjects]
    return {
        "mean_accuracy": safe_mean(selected), "oracle_mean_accuracy": safe_mean(oracle),
        "mean_regret": safe_mean(regret), "max_regret": max(regret) if regret else float("nan"),
        "harmed_count": sum(v > 1e-12 for v in regret),
        "default_count": sum(r.get("decision") == "default" for r in subjects),
        "f0_count": sum(r.get("selected_strategy") == "f0" for r in subjects),
        "traditional_count": sum(r.get("selected_strategy") == "traditional" for r in subjects),
    }


def apply_threshold(rows: Sequence[dict], probs: Sequence[float], default_strategy: str, low: float, high: float) -> List[dict]:
    out = []
    for row, p in zip(rows, probs):
        if float(p) >= high:
            strategy, decision = "f0", "high_conf_f0"
        elif float(p) <= low:
            strategy, decision = "traditional", "high_conf_traditional"
        else:
            strategy, decision = default_strategy, "default"
        selected = acc(row, strategy)
        oracle = max(as_float(row.get("traditional")), as_float(row.get("f0")))
        best_strategy = row.get("best_strategy")
        item = dict(row)
        item.update({
            "prob_f0_benefit": float(p), "selected_strategy": strategy, "decision": decision,
            "selected_accuracy": selected, "oracle_accuracy": oracle, "regret": oracle - selected,
            "correct_strategy": int(strategy == best_strategy),
            "actual_best_strategy": best_strategy,
            "is_harmed": int((oracle - selected) > 1e-12),
            "is_helped": int(selected > max(acc(row, default_strategy), acc(row, opposite(default_strategy))) + 1e-12),
            "used_default": int(decision == "default"),
            "traditional_accuracy": as_float(row.get("traditional")),
            "f0_accuracy": as_float(row.get("f0")),
        })
        out.append(item)
    return out


def choose_threshold(rows: Sequence[dict], probs: Sequence[float], default_strategy: str, lows: Sequence[float], highs: Sequence[float]) -> Tuple[float, float, dict]:
    best = None
    for low in lows:
        for high in highs:
            if low > high:
                continue
            subjects = apply_threshold(rows, probs, default_strategy, low, high)
            s = summarize_subjects(subjects)
            score = (s["mean_accuracy"], -s["max_regret"], -s["harmed_count"], s["default_count"])
            if best is None or score > best[0]:
                best = (score, float(low), float(high), s)
    assert best is not None
    return best[1], best[2], best[3]


def threshold_grid(dataset: str, rows: Sequence[dict], probs: Sequence[float], default_strategy: str, lows, highs, fold: int, feature_set: str, model: str, split: str) -> List[dict]:
    """Evaluate every threshold pair on the provided split.

    This must be called with held-out test rows when writing formal threshold
    sweep outputs. Train rows are only for selecting the deployed threshold.
    """
    out = []
    for low in lows:
        for high in highs:
            if low > high:
                continue
            subjects = apply_threshold(rows, probs, default_strategy, low, high)
            s = summarize_subjects(subjects)
            out.append({
                "dataset": dataset, "feature_set": feature_set, "model": model, "fold": fold, "split": split,
                "selection_scope": "heldout_grid_diagnostic",
                "default_strategy": default_strategy, "low_threshold": low, "high_threshold": high,
                "n_subjects": len(subjects),
                "coverage": 1.0 - s["default_count"] / max(len(subjects), 1),
                "mean_accuracy": s["mean_accuracy"], "mean_regret": s["mean_regret"], "max_regret": s["max_regret"],
                "harmed_count": s["harmed_count"], "default_count": s["default_count"],
                "f0_count": s["f0_count"], "traditional_count": s["traditional_count"],
            })
    return out


def fixed_baselines(dataset: str, rows: Sequence[dict]) -> List[dict]:
    trad = [as_float(r.get("traditional")) for r in rows]
    f0 = [as_float(r.get("f0")) for r in rows]
    oracle = [max(t, f) for t, f in zip(trad, f0)]
    global_strategy = "f0" if safe_mean(f0) >= safe_mean(trad) else "traditional"
    majority_strategy = Counter(r.get("best_strategy") for r in rows).most_common(1)[0][0]
    rng = random.Random(20260710)
    random_selected = [f if rng.random() >= 0.5 else t for t, f in zip(trad, f0)]
    cases = [
        ("always_traditional", trad, "traditional"), ("always_f0", f0, "f0"),
        ("global_best_single", f0 if global_strategy == "f0" else trad, global_strategy),
        ("majority_oracle_label", f0 if majority_strategy == "f0" else trad, majority_strategy),
        ("random_gate", random_selected, "random"), ("subject_oracle", oracle, "oracle"),
    ]
    best_single = max(safe_mean(trad), safe_mean(f0))
    oracle_mean = safe_mean(oracle)
    out = []
    for method, selected, strategy in cases:
        regret = [o - s for o, s in zip(oracle, selected)]
        out.append({
            "dataset": dataset, "method": method, "feature_set": "", "model": "", "strategy": strategy,
            "n_subjects": len(rows), "n_features": 0, "mean_accuracy": safe_mean(selected),
            "gain_vs_global_best_single": safe_mean(selected) - best_single,
            "oracle_recovery_ratio": oracle_recovery(safe_mean(selected), best_single, oracle_mean),
            "mean_regret": safe_mean(regret), "max_regret": max(regret),
            "harmed_count": sum(v > 1e-12 for v in regret),
        })
    return out

def gate_cv(dataset: str, rows: Sequence[dict], features: Sequence[str], feature_set: str, model_name: str, seed: int, n_splits: int, lows, highs):
    X, y = make_x(rows, features), make_y(rows)
    splits = splits_for(y, n_splits, seed)
    if not splits:
        return [], [], []
    model = models(seed)[model_name]
    fold_rows, subject_rows, grid_rows = [], [], []
    for fold, (tr_idx, te_idx) in enumerate(splits, start=1):
        train_rows = [rows[int(i)] for i in tr_idx]
        test_rows = [rows[int(i)] for i in te_idx]
        default_strategy = best_train_strategy(train_rows)
        clf = clone(model)
        clf.fit(X[tr_idx], y[tr_idx])
        train_prob = clf.predict_proba(X[tr_idx])[:, 1]
        test_prob = clf.predict_proba(X[te_idx])[:, 1]
        low, high, train_summary = choose_threshold(train_rows, train_prob, default_strategy, lows, highs)
        subjects = apply_threshold(test_rows, test_prob, default_strategy, low, high)
        test_summary = summarize_subjects(subjects)
        fold_rows.append({
            "dataset": dataset, "feature_set": feature_set, "model": model_name, "fold": fold,
            "default_strategy": default_strategy, "low_threshold": low, "high_threshold": high,
            "train_mean_accuracy": train_summary["mean_accuracy"], "train_mean_regret": train_summary["mean_regret"],
            "train_max_regret": train_summary["max_regret"], "test_mean_accuracy": test_summary["mean_accuracy"],
            "test_mean_regret": test_summary["mean_regret"], "test_max_regret": test_summary["max_regret"],
            "test_harmed_count": test_summary["harmed_count"], "test_default_count": test_summary["default_count"],
        })
        grid_rows.extend(threshold_grid(dataset, test_rows, test_prob, default_strategy, lows, highs, fold, feature_set, model_name, "test"))
        for s in subjects:
            item = dict(s)
            item.update({
                "dataset": dataset, "feature_set": feature_set, "model": model_name, "fold": fold,
                "default_strategy": default_strategy, "low_threshold": low, "high_threshold": high,
                "true_f0_benefit": int(as_float(s.get("delta_f0_minus_traditional")) > 0),
            })
            subject_rows.append(item)
    return fold_rows, subject_rows, grid_rows


def gate_summary(dataset: str, source_rows: Sequence[dict], subjects: Sequence[dict], feature_set: str, model: str, n_features: int, method: str) -> dict:
    trad_mean = safe_mean(as_float(r.get("traditional")) for r in source_rows)
    f0_mean = safe_mean(as_float(r.get("f0")) for r in source_rows)
    best_single = max(trad_mean, f0_mean)
    oracle_mean = safe_mean(max(as_float(r.get("traditional")), as_float(r.get("f0"))) for r in source_rows)
    selected_mean = safe_mean(as_float(r.get("selected_accuracy")) for r in subjects)
    regrets = [as_float(r.get("regret")) for r in subjects]
    return {
        "dataset": dataset, "method": method, "feature_set": feature_set, "model": model,
        "n_subjects": len(subjects), "n_features": n_features, "mean_accuracy": selected_mean,
        "gain_vs_global_best_single": selected_mean - best_single,
        "oracle_recovery_ratio": oracle_recovery(selected_mean, best_single, oracle_mean),
        "mean_regret": safe_mean(regrets), "max_regret": max(regrets) if regrets else float("nan"),
        "harmed_count": sum(v > 1e-12 for v in regrets),
        "default_count": sum(r.get("decision") == "default" for r in subjects),
        "f0_count": sum(r.get("selected_strategy") == "f0" for r in subjects),
        "traditional_count": sum(r.get("selected_strategy") == "traditional" for r in subjects),
    }


def run_baselines_thresholds(data, out_dir: Path, seed: int, n_splits: int, feature_names, model_names, lows, highs) -> None:
    baseline_rows, fold_rows, subject_rows, grid_rows, deployed_threshold_rows = [], [], [], [], []
    for dataset, rows in data.items():
        baseline_rows.extend(fixed_baselines(dataset, rows))
        sets = feature_sets(dataset, rows)
        for fs_name in resolve_sets(dataset, rows, feature_names):
            for model_name in model_names:
                folds, subjects, grid = gate_cv(dataset, rows, sets[fs_name], fs_name, model_name, seed, n_splits, lows, highs)
                fold_rows.extend(folds)
                subject_rows.extend(subjects)
                grid_rows.extend(grid)
                if subjects:
                    baseline_rows.append(gate_summary(dataset, rows, subjects, fs_name, model_name, len(sets[fs_name]), "conservative_gate"))
                    deployed = gate_summary(dataset, rows, subjects, fs_name, model_name, len(sets[fs_name]), "threshold_reject")
                    deployed["threshold_selection"] = "nested_train_fold"
                    deployed["n_folds"] = len(folds)
                    deployed_threshold_rows.append(deployed)
    write_csv(out_dir / "baselines" / "gate_baseline_comparison.csv", baseline_rows)
    write_csv(out_dir / "baselines" / "gate_subject_level_results.csv", subject_rows)
    write_csv(out_dir / "baselines" / "gate_fold_results.csv", fold_rows)
    threshold_summary_rows = aggregate_threshold_sweep(grid_rows)
    write_csv(out_dir / "thresholds" / "gate_threshold_sweep.csv", grid_rows)
    write_csv(out_dir / "thresholds" / "gate_threshold_sweep_summary.csv", threshold_summary_rows)
    write_csv(out_dir / "thresholds" / "gate_threshold_deployed_summary.csv", deployed_threshold_rows)
    write_csv(out_dir / "thresholds" / "gate_threshold_subjects.csv", subject_rows)
    write_table_report(out_dir / "baselines" / "gate_validation_report.md", "Gate Baseline Validation", baseline_rows)
    write_threshold_report(out_dir / "thresholds" / "gate_threshold_report.md", threshold_summary_rows)


def make_switch_labels(rows: Sequence[dict], default_strategy: str, gain_min: float) -> np.ndarray:
    alt = opposite(default_strategy)
    return np.asarray([1 if acc(r, alt) - acc(r, default_strategy) >= gain_min else 0 for r in rows], dtype=int)


def select_switch_threshold(rows: Sequence[dict], probs: Sequence[float], default_strategy: str, thresholds: Sequence[float]) -> float:
    best = None
    for thr in thresholds:
        subjects = []
        for row, p in zip(rows, probs):
            strategy = opposite(default_strategy) if float(p) >= thr else default_strategy
            selected = acc(row, strategy)
            oracle = max(as_float(row.get("traditional")), as_float(row.get("f0")))
            item = dict(row)
            item.update({"selected_accuracy": selected, "oracle_accuracy": oracle, "regret": oracle - selected})
            subjects.append(item)
        s = summarize_subjects(subjects)
        score = (s["mean_accuracy"], -s["max_regret"], -s["harmed_count"])
        if best is None or score > best[0]:
            best = (score, float(thr))
    return best[1] if best else 1.0


def risk_cv(dataset: str, rows: Sequence[dict], features: Sequence[str], feature_set: str, model_name: str, seed: int, n_splits: int, gain_min: float, switch_thresholds):
    X, y_split = make_x(rows, features), make_y(rows)
    splits = splits_for(y_split, n_splits, seed)
    if not splits:
        return []
    out = []
    for fold, (tr_idx, te_idx) in enumerate(splits, start=1):
        train_rows = [rows[int(i)] for i in tr_idx]
        test_rows = [rows[int(i)] for i in te_idx]
        default_strategy = best_train_strategy(train_rows)
        y_switch = make_switch_labels(train_rows, default_strategy, gain_min)
        if len(set(y_switch.tolist())) < 2:
            test_probs = np.zeros(len(te_idx), dtype=float)
            thr = 1.0
        else:
            clf = clone(models(seed)[model_name])
            clf.fit(X[tr_idx], y_switch)
            train_probs = clf.predict_proba(X[tr_idx])[:, 1]
            thr = select_switch_threshold(train_rows, train_probs, default_strategy, switch_thresholds)
            test_probs = clf.predict_proba(X[te_idx])[:, 1]
        for row, p in zip(test_rows, test_probs):
            switch = float(p) >= thr
            strategy = opposite(default_strategy) if switch else default_strategy
            selected = acc(row, strategy)
            oracle = max(as_float(row.get("traditional")), as_float(row.get("f0")))
            item = dict(row)
            item.update({
                "dataset": dataset, "feature_set": feature_set, "model": model_name, "fold": fold,
                "gain_threshold": gain_min, "default_strategy": default_strategy,
                "prob_switch_worth_it": float(p), "switch_threshold": thr,
                "selected_strategy": strategy, "decision": "switch" if switch else "default",
                "selected_accuracy": selected, "oracle_accuracy": oracle, "regret": oracle - selected,
                "correct_strategy": int(strategy == row.get("best_strategy")),
            })
            out.append(item)
    return out


def run_risk(data, out_dir: Path, seed: int, n_splits: int, feature_names, model_names, gain_thresholds, switch_thresholds) -> None:
    summary, subjects = [], []
    for dataset, rows in data.items():
        sets = feature_sets(dataset, rows)
        for fs_name in resolve_sets(dataset, rows, feature_names):
            for gain_min in gain_thresholds:
                for model_name in model_names:
                    pred = risk_cv(dataset, rows, sets[fs_name], fs_name, model_name, seed, n_splits, gain_min, switch_thresholds)
                    subjects.extend(pred)
                    if pred:
                        row = gate_summary(dataset, rows, pred, fs_name, model_name, len(sets[fs_name]), "risk_aware_gate")
                        row["gain_threshold"] = gain_min
                        summary.append(row)
    write_csv(out_dir / "risk" / "risk_aware_gate_results.csv", summary)
    write_csv(out_dir / "risk" / "risk_aware_gate_subjects.csv", subjects)
    write_table_report(out_dir / "risk" / "risk_aware_gate_report.md", "Risk-Aware Gate", summary)



# Confirmatory risk guardrails. Candidates and confidence are fixed by protocol.
GUARDRAIL_CANDIDATES = {
    "openbmi": {"name": "openbmi_correlation_gb", "feature_set": "selected_plus_correlation", "model": "gb", "mode": "conservative"},
    "physionet": {"name": "physionet_common_compact_logreg_risk01", "feature_set": "common_compact", "model": "logreg", "mode": "risk", "gain_threshold": 0.01},
}

def guardrail_cv(dataset, rows, features, candidate, seed, n_splits, confidence):
    """Evaluate a fixed strict-confidence fallback policy in outer CV."""
    X, y = make_x(rows, features), make_y(rows)
    splits = splits_for(y, n_splits, seed)
    if not splits:
        return []
    out = []
    for fold, (tr_idx, te_idx) in enumerate(splits, start=1):
        train_rows = [rows[int(i)] for i in tr_idx]
        test_rows = [rows[int(i)] for i in te_idx]
        default = best_train_strategy(train_rows)
        if candidate["mode"] == "conservative":
            clf = clone(models(seed)[candidate["model"]])
            clf.fit(X[tr_idx], y[tr_idx])
            probs = clf.predict_proba(X[te_idx])[:, 1]
            for row, prob in zip(test_rows, probs):
                if prob >= confidence:
                    strategy, decision = "f0", "guardrail_switch_f0"
                elif prob <= 1.0 - confidence:
                    strategy, decision = "traditional", "guardrail_switch_traditional"
                else:
                    strategy, decision = default, "guardrail_abstain_default"
                selected = acc(row, strategy)
                oracle = max(acc(row, "traditional"), acc(row, "f0"))
                item = dict(row)
                item.update({"dataset": dataset, "candidate": candidate["name"], "feature_set": candidate["feature_set"],
                             "model": candidate["model"], "method": "guardrail_conservative", "fold": fold,
                             "default_strategy": default, "guardrail_confidence": confidence, "prob_f0_benefit": float(prob),
                             "selected_strategy": strategy, "decision": decision, "selected_accuracy": selected,
                             "oracle_accuracy": oracle, "regret": oracle - selected,
                             "traditional_accuracy": acc(row, "traditional"), "f0_accuracy": acc(row, "f0"),
                             "used_guardrail_abstention": int(decision == "guardrail_abstain_default")})
                out.append(item)
        else:
            gain = float(candidate["gain_threshold"])
            labels = make_switch_labels(train_rows, default, gain)
            if len(set(labels.tolist())) < 2:
                probs = np.zeros(len(te_idx), dtype=float)
            else:
                clf = clone(models(seed)[candidate["model"]])
                clf.fit(X[tr_idx], labels)
                probs = clf.predict_proba(X[te_idx])[:, 1]
            for row, prob in zip(test_rows, probs):
                switch = float(prob) >= confidence
                strategy = opposite(default) if switch else default
                selected = acc(row, strategy)
                oracle = max(acc(row, "traditional"), acc(row, "f0"))
                item = dict(row)
                item.update({"dataset": dataset, "candidate": candidate["name"], "feature_set": candidate["feature_set"],
                             "model": candidate["model"], "method": "guardrail_risk_aware", "fold": fold,
                             "gain_threshold": gain, "default_strategy": default, "guardrail_confidence": confidence,
                             "prob_switch_worth_it": float(prob), "selected_strategy": strategy,
                             "decision": "guardrail_switch" if switch else "guardrail_abstain_default",
                             "selected_accuracy": selected, "oracle_accuracy": oracle, "regret": oracle - selected,
                             "traditional_accuracy": acc(row, "traditional"), "f0_accuracy": acc(row, "f0"),
                             "used_guardrail_abstention": int(not switch)})
                out.append(item)
    return out

def run_guardrail(data, out_dir, seed, n_splits, confidence):
    summary, subjects = [], []
    for dataset, candidate in GUARDRAIL_CANDIDATES.items():
        if dataset not in data:
            continue
        sets = feature_sets(dataset, data[dataset])
        if candidate["feature_set"] not in sets:
            raise ValueError(f"{dataset}: missing guardrail feature set {candidate['feature_set']}")
        pred = guardrail_cv(dataset, data[dataset], sets[candidate["feature_set"]], candidate, seed, n_splits, confidence)
        subjects.extend(pred)
        if pred:
            row = gate_summary(dataset, data[dataset], pred, candidate["feature_set"], candidate["model"], len(sets[candidate["feature_set"]]), pred[0]["method"])
            row.update({"candidate": candidate["name"], "guardrail_confidence": confidence, "abstention_count": sum(int(r["used_guardrail_abstention"]) for r in pred)})
            summary.append(row)
    write_csv(out_dir / "guardrail" / "guardrail_results.csv", summary)
    write_csv(out_dir / "guardrail" / "guardrail_subjects.csv", subjects)
    write_table_report(out_dir / "guardrail" / "guardrail_report.md", "Fixed Risk Guardrail", summary)

def run_feature_ablation(data, out_dir: Path, seed: int, n_splits: int, model_names, lows, highs) -> None:
    summary, subjects = [], []
    keep = {"all_features", "common_compact", "openbmi_highfreq_source", "physionet_alpha", "low_frequency", "alpha_beta", "high_frequency", "source_distance", "covariance", "correlation", "correlation_only", "selected_plus_correlation", "covariance_extended", "selected_plus_covariance"}
    for dataset, rows in data.items():
        sets = feature_sets(dataset, rows)
        names = [n for n in sets if n in keep or n.startswith("all_minus_")]
        for fs_name in names:
            for model_name in model_names:
                _, pred, _ = gate_cv(dataset, rows, sets[fs_name], fs_name, model_name, seed, n_splits, lows, highs)
                subjects.extend(pred)
                if pred:
                    summary.append(gate_summary(dataset, rows, pred, fs_name, model_name, len(sets[fs_name]), "feature_ablation"))
    write_csv(out_dir / "feature_ablation" / "gate_feature_set_ablation.csv", summary)
    write_csv(out_dir / "feature_ablation" / "gate_feature_set_subjects.csv", subjects)
    write_table_report(out_dir / "feature_ablation" / "gate_feature_set_ablation_report.md", "Feature-Set Ablation", summary)

# Experiment F: failure cases.

def subject_key(row: dict) -> Tuple[str, str, str, str]:
    return (row.get("dataset", ""), row.get("feature_set", ""), row.get("model", ""), str(row.get("gain_threshold", "")))


def best_subject_group(rows: Sequence[dict]) -> Tuple[str, str, str, str]:
    grouped: Dict[Tuple[str, str, str, str], List[dict]] = defaultdict(list)
    for row in rows:
        grouped[subject_key(row)].append(row)
    best_key, best_score = None, None
    for key, vals in grouped.items():
        score = (safe_mean(as_float(v.get("selected_accuracy")) for v in vals), -max(as_float(v.get("regret")) for v in vals))
        if best_score is None or score > best_score:
            best_key, best_score = key, score
    assert best_key is not None
    return best_key


def percentiles(rows: Sequence[dict], cols: Sequence[str]) -> Dict[str, Dict[str, float]]:
    out = {}
    for col in cols:
        vals = sorted([(str(r.get("subject")), as_float(r.get(col))) for r in rows if np.isfinite(as_float(r.get(col)))], key=lambda kv: kv[1])
        out[col] = {s: i / max(len(vals) - 1, 1) for i, (s, _) in enumerate(vals)}
    return out


def classify_error(pred: dict, source: dict) -> str:
    selected = pred.get("selected_strategy") or pred.get("chosen_strategy")
    actual = source.get("best_strategy")
    regret = as_float(pred.get("regret"))
    prob = as_float(pred.get("prob_f0_benefit"), as_float(pred.get("prob_switch_worth_it")))
    if selected == actual:
        return "correct"
    if regret >= 0.10 and np.isfinite(prob) and (prob >= 0.80 or prob <= 0.20):
        return "high_conf_high_regret_wrong"
    if regret >= 0.10:
        return "high_regret_wrong"
    if np.isfinite(prob) and 0.40 <= prob <= 0.60:
        return "low_margin_wrong"
    return "wrong"


def nearest_conflicts(dataset: str, subject: str, rows: Sequence[dict], cols: Sequence[str], k: int = 5) -> List[dict]:
    subjects = [str(r.get("subject")) for r in rows]
    if subject not in subjects or not cols:
        return []
    X = make_x(rows, cols)
    X = SimpleImputer(strategy="median").fit_transform(X)
    sd = np.std(X, axis=0)
    sd[sd == 0] = 1.0
    X = (X - np.mean(X, axis=0)) / sd
    idx = subjects.index(subject)
    dists = np.linalg.norm(X - X[idx], axis=1)
    order = np.argsort(dists)
    out = []
    for j in order:
        if int(j) == idx:
            continue
        out.append({
            "dataset": dataset, "subject": subject, "neighbor_subject": subjects[int(j)],
            "distance": float(dists[int(j)]),
            "subject_best_strategy": rows[idx].get("best_strategy"),
            "neighbor_best_strategy": rows[int(j)].get("best_strategy"),
            "strategy_conflict": int(rows[idx].get("best_strategy") != rows[int(j)].get("best_strategy")),
            "subject_delta": rows[idx].get("delta_f0_minus_traditional"),
            "neighbor_delta": rows[int(j)].get("delta_f0_minus_traditional"),
        })
        if len(out) >= k:
            break
    return out


def run_failure(data, out_dir: Path) -> None:
    candidates = []
    for rel in [
        "baselines/gate_subject_level_results.csv",
        "thresholds/gate_threshold_subjects.csv",
        "risk/risk_aware_gate_subjects.csv",
        "feature_ablation/gate_feature_set_subjects.csv",
    ]:
        path = out_dir / rel
        if path.exists() and path.stat().st_size > 0:
            candidates.extend(read_csv(path))
    if not candidates:
        raise FileNotFoundError("No subject-level outputs found; run baselines/thresholds/risk/feature_ablation first.")
    failures, profiles, near = [], [], []
    for dataset, rows in data.items():
        ds = [r for r in candidates if r.get("dataset") == dataset]
        if not ds:
            continue
        key = best_subject_group(ds)
        selected = [r for r in ds if subject_key(r) == key]
        cols = desc_cols(rows)
        row_by_subject = {str(r.get("subject")): r for r in rows}
        pct = percentiles(rows, cols)
        for pred in sorted(selected, key=lambda r: -as_float(r.get("regret"))):
            subject = str(pred.get("subject"))
            source = row_by_subject.get(subject, {})
            item = dict(pred)
            item.update({
                "actual_best_strategy": source.get("best_strategy"),
                "delta_f0_minus_traditional": source.get("delta_f0_minus_traditional"),
                "error_type": classify_error(pred, source),
            })
            failures.append(item)
            for col in cols:
                profiles.append({
                    "dataset": dataset, "subject": subject, "feature": col,
                    "value": source.get(col, ""), "percentile": pct.get(col, {}).get(subject, float("nan")),
                    "regret": pred.get("regret"), "error_type": item["error_type"],
                })
            near.extend(nearest_conflicts(dataset, subject, rows, cols))
    fdir = out_dir / "failure"
    write_csv(fdir / "gate_failure_cases.csv", failures)
    write_csv(fdir / "gate_high_regret_subject_profiles.csv", profiles)
    write_csv(fdir / "gate_nearest_subject_conflicts.csv", near)
    lines = ["# Failure Case / High-Regret Report", "", "| Dataset | Subject | Feature Set | Model | Selected | Best | Regret | Prob | Type |", "|---|---|---|---|---|---|---:|---:|---|"]
    for r in sorted(failures, key=lambda x: -as_float(x.get("regret")))[:60]:
        prob = as_float(r.get("prob_f0_benefit"), as_float(r.get("prob_switch_worth_it")))
        lines.append(f"| {r.get('dataset')} | {r.get('subject')} | {r.get('feature_set')} | {r.get('model')} | {r.get('selected_strategy')} | {r.get('actual_best_strategy')} | {as_float(r.get('regret')):.4f} | {prob:.4f} | {r.get('error_type')} |")
    conflict_rate = safe_mean(as_float(r.get("strategy_conflict")) for r in near)
    lines += ["", f"Mean nearest-neighbor strategy conflict rate: {conflict_rate:.4f}"]
    write_text(fdir / "gate_failure_case_report.md", "\n".join(lines) + "\n")


def aggregate_threshold_sweep(rows: Sequence[dict]) -> List[dict]:
    """Aggregate held-out threshold-grid rows across outer folds.

    `gate_threshold_sweep.csv` stores one row per outer fold and threshold pair.
    This summary is the formal report table: it averages held-out fold metrics
    weighted by fold test-subject counts and never uses train-side grid scores.
    """
    grouped = defaultdict(list)
    for row in rows:
        if row.get("split", "test") != "test":
            continue
        key = (row.get("dataset"), row.get("feature_set"), row.get("model"), as_float(row.get("low_threshold")), as_float(row.get("high_threshold")))
        grouped[key].append(row)
    out = []
    for (dataset, feature_set, model, low, high), vals in grouped.items():
        weights = [max(int(as_float(v.get("n_subjects"), 0)), 1) for v in vals]
        total = sum(weights)
        def wmean(col):
            return sum(as_float(v.get(col)) * w for v, w in zip(vals, weights)) / total if total else float("nan")
        out.append({
            "dataset": dataset, "feature_set": feature_set, "model": model,
            "low_threshold": low, "high_threshold": high, "n_folds": len(vals), "n_subjects": total,
            "coverage": wmean("coverage"), "mean_accuracy": wmean("mean_accuracy"),
            "mean_regret": wmean("mean_regret"), "max_regret": max(as_float(v.get("max_regret")) for v in vals),
            "harmed_count": sum(int(as_float(v.get("harmed_count"), 0)) for v in vals),
            "default_count": sum(int(as_float(v.get("default_count"), 0)) for v in vals),
            "f0_count": sum(int(as_float(v.get("f0_count"), 0)) for v in vals),
            "traditional_count": sum(int(as_float(v.get("traditional_count"), 0)) for v in vals),
        })
    return sorted(out, key=lambda r: (r.get("dataset", ""), r.get("feature_set", ""), r.get("model", ""), -as_float(r.get("mean_accuracy")), as_float(r.get("max_regret"))))


# Reports and entry point.

def write_table_report(path: Path, title: str, rows: Sequence[dict]) -> None:
    lines = [f"# {title}", "", "| Dataset | Method | Feature Set | Model | Mean Acc | Gain vs Best | Recovery | Mean Regret | Max Regret | Harmed |", "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda x: (x.get("dataset", ""), -as_float(x.get("mean_accuracy")))):
        lines.append(f"| {r.get('dataset')} | {r.get('method', '')} | {r.get('feature_set', '')} | {r.get('model', '')} | {as_float(r.get('mean_accuracy')):.4f} | {as_float(r.get('gain_vs_global_best_single')):+.4f} | {as_float(r.get('oracle_recovery_ratio')):.4f} | {as_float(r.get('mean_regret')):.4f} | {as_float(r.get('max_regret')):.4f} | {r.get('harmed_count', '')} |")
    write_text(path, "\n".join(lines) + "\n")


def write_threshold_report(path: Path, rows: Sequence[dict]) -> None:
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r.get("dataset"), r.get("feature_set"), r.get("model"))].append(r)
    best = [max(v, key=lambda r: (as_float(r.get("mean_accuracy")), -as_float(r.get("max_regret")), -as_float(r.get("harmed_count")))) for v in grouped.values()]
    lines = [
        "# Conservative Threshold / Reject Option",
        "",
        "This is a held-out threshold-grid diagnostic. It evaluates many threshold pairs on test folds and must not be used for deployable model selection.",
        "Use gate_threshold_deployed_summary.csv for formal nested deployable threshold results; its thresholds were selected using train subjects within each fold.",
        "",
        "| Dataset | Feature Set | Model | Low | High | Coverage | Mean Acc | Mean Regret | Max Regret | Harmed | Default |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(best, key=lambda x: (x.get("dataset", ""), -as_float(x.get("mean_accuracy")))):
        lines.append(f"| {r.get('dataset')} | {r.get('feature_set')} | {r.get('model')} | {as_float(r.get('low_threshold')):.2f} | {as_float(r.get('high_threshold')):.2f} | {as_float(r.get('coverage')):.3f} | {as_float(r.get('mean_accuracy')):.4f} | {as_float(r.get('mean_regret')):.4f} | {as_float(r.get('max_regret')):.4f} | {r.get('harmed_count')} | {r.get('default_count')} |")
    write_text(path, "\n".join(lines) + "\n")


def write_summary(out_dir: Path) -> None:
    parts = [
        "association/descriptor_delta_association_report.md",
        "baselines/gate_validation_report.md",
        "thresholds/gate_threshold_report.md",
        "risk/risk_aware_gate_report.md",
        "feature_ablation/gate_feature_set_ablation_report.md",
        "failure/gate_failure_case_report.md",
    ]
    lines = ["# Stage-5 Direction-1 Summary", ""]
    for rel in parts:
        path = out_dir / rel
        lines += [f"## {rel}", ""]
        if path.exists():
            lines.extend(path.read_text(encoding="utf-8").splitlines()[:80])
        else:
            lines.append("Not generated.")
        lines.append("")
    write_text(out_dir / "direction1_summary_report.md", "\n".join(lines) + "\n")


def parse_float_list(values: Sequence[str]) -> List[float]:
    return [float(v) for v in values]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--gate-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("results") / "stage5" / "direction1")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--experiments", nargs="+", default=["all"], choices=["all", "association", "baselines", "thresholds", "risk", "guardrail", "feature_ablation", "failure", "summary"])
    parser.add_argument("--feature-sets", nargs="+", default=["selected"], help="Use 'selected', 'all', or explicit feature-set names.")
    parser.add_argument("--models", nargs="+", default=["logreg", "gb"])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--low-thresholds", nargs="+", default=["0.20", "0.25", "0.30", "0.35", "0.40", "0.45", "0.50"])
    parser.add_argument("--high-thresholds", nargs="+", default=["0.50", "0.55", "0.60", "0.65", "0.70", "0.75", "0.80"])
    parser.add_argument("--switch-thresholds", nargs="+", default=["0.50", "0.55", "0.60", "0.65", "0.70", "0.75", "0.80"])
    parser.add_argument("--gain-thresholds", nargs="+", default=["0.00", "0.01", "0.02"])
    parser.add_argument("--guardrail-confidence", type=float, default=0.80, help="Fixed confidence needed to switch; other cases use fold default.")
    args = parser.parse_args()

    experiments = set(args.experiments)
    if "all" in experiments:
        experiments = {"association", "baselines", "thresholds", "risk", "guardrail", "feature_ablation", "failure", "summary"}
    if "thresholds" in experiments:
        experiments.add("baselines")

    data = load_data(args.results_dir, args.datasets, args.gate_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    lows = parse_float_list(args.low_thresholds)
    highs = parse_float_list(args.high_thresholds)
    switch_thresholds = parse_float_list(args.switch_thresholds)
    gain_thresholds = parse_float_list(args.gain_thresholds)

    if "association" in experiments:
        print("Running experiment A: descriptor association")
        run_association(data, args.out_dir, args.n_permutations, args.n_bootstrap, args.seed)
    if "baselines" in experiments:
        print("Running experiments B/C: baselines and conservative thresholds")
        run_baselines_thresholds(data, args.out_dir, args.seed, args.n_splits, args.feature_sets, args.models, lows, highs)
    if "risk" in experiments:
        print("Running experiment D: risk-aware gate")
        run_risk(data, args.out_dir, args.seed, args.n_splits, args.feature_sets, args.models, gain_thresholds, switch_thresholds)
    if "guardrail" in experiments:
        print("Running experiment G: fixed risk guardrail")
        run_guardrail(data, args.out_dir, args.seed, args.n_splits, args.guardrail_confidence)
    if "feature_ablation" in experiments:
        print("Running experiment E: feature-set ablation")
        run_feature_ablation(data, args.out_dir, args.seed, args.n_splits, args.models, lows, highs)
    if "failure" in experiments:
        print("Running experiment F: failure-case diagnostics")
        run_failure(data, args.out_dir)
    if "summary" in experiments:
        write_summary(args.out_dir)
    print(f"Wrote Direction-1 outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
