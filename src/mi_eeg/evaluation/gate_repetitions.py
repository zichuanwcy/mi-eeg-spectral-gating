"""Summarize predefined multi-seed repeated descriptor-gate validations.


The primary candidates are fixed before aggregation:
- OpenBMI: selected_plus_correlation + GB conservative gate.
- OpenBMI legacy reference: openbmi_highfreq_source + GB risk-aware gate (gain=0).
- PhysioNet: common_compact + logistic-regression risk-aware gate (gain=0.01).

For every candidate, this script compares each subject-level decision against
that seed's fixed global-best strategy, reports seed stability, bootstrap CIs,
and high-regret occurrence frequency. It never selects a per-seed winner.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable, Sequence

import numpy as np

try:
    from scipy import stats
except ImportError as exc:
    raise ImportError("This script requires scipy.") from exc


CANDIDATES = [
    {
        "name": "openbmi_correlation_gb",
        "dataset": "openbmi", "source": "baselines/gate_subject_level_results.csv",
        "feature_set": "selected_plus_correlation", "model": "gb", "method": "conservative_gate",
        "gain_threshold": None,
    },
    {
        "name": "openbmi_legacy_highfreq_source_gb",
        "dataset": "openbmi", "source": "risk/risk_aware_gate_subjects.csv",
        "feature_set": "openbmi_highfreq_source", "model": "gb", "method": "risk_aware_gate",
        "gain_threshold": "0.0",
    },
    {
        "name": "physionet_common_compact_logreg_risk01",
        "dataset": "physionet", "source": "risk/risk_aware_gate_subjects.csv",
        "feature_set": "common_compact", "model": "logreg", "method": "risk_aware_gate",
        "gain_threshold": "0.01",
    },
]


def read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value, default=float("nan")) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def text(value) -> str:
    return "" if value is None else str(value).strip()


def subject_id(value) -> str:
    raw = text(value)
    digits = "".join(ch for ch in raw if ch.isdigit())
    return f"S{int(digits):03d}" if digits else raw


def value(row: dict, *names: str) -> float:
    for name in names:
        x = number(row.get(name))
        if math.isfinite(x):
            return x
    return float("nan")


def normalized_method(row: dict, source: str) -> str:
    method = text(row.get("method")) or text(row.get("gate_method"))
    if method:
        return method
    return "risk_aware_gate" if source.startswith("risk/") else "conservative_gate"


def match_candidate(row: dict, candidate: dict) -> bool:
    if text(row.get("dataset")).lower() != candidate["dataset"]:
        return False
    if text(row.get("feature_set")) != candidate["feature_set"] or text(row.get("model")) != candidate["model"]:
        return False
    if normalized_method(row, candidate["source"]) != candidate["method"]:
        return False
    expected_gain = candidate["gain_threshold"]
    if expected_gain is not None:
        actual = number(row.get("gain_threshold"))
        if not math.isfinite(actual) or abs(actual - float(expected_gain)) > 1e-12:
            return False
    return True


def global_best(rows: Sequence[dict]) -> str:
    traditional = mean(value(row, "traditional_accuracy", "traditional") for row in rows)
    f0 = mean(value(row, "f0_accuracy", "f0") for row in rows)
    return "f0" if f0 > traditional else "traditional"


def bootstrap_ci(values: Sequence[float], rng: np.random.Generator, n: int) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    samples = np.asarray([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n)])
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def discover_seed_dirs(root: Path) -> list[Path]:
    return sorted([p for p in root.glob("seed_*") if p.is_dir()], key=lambda p: p.name)


def candidate_rows(seed_dir: Path, candidate: dict) -> list[dict]:
    rows = read_csv(seed_dir / candidate["source"])
    matched = [row for row in rows if match_candidate(row, candidate)]
    if not matched:
        raise ValueError(f"{seed_dir.name}: no rows for {candidate['name']}")
    subjects = [subject_id(row.get("subject")) for row in matched]
    if len(subjects) != len(set(subjects)):
        raise ValueError(f"{seed_dir.name}: duplicate subject rows for {candidate['name']}")
    return matched


def analyze_candidate(seed_dirs: Sequence[Path], candidate: dict, rng: np.random.Generator, n_bootstrap: int) -> tuple[list[dict], list[dict], dict]:
    seed_metrics, subject_rows = [], []
    by_subject = defaultdict(list)
    for seed_dir in seed_dirs:
        rows = candidate_rows(seed_dir, candidate)
        baseline = global_best(rows)
        selected = [value(row, "selected_accuracy") for row in rows]
        baseline_values = [value(row, f"{baseline}_accuracy", baseline) for row in rows]
        regrets = [value(row, "regret") for row in rows]
        deltas = [a - b for a, b in zip(selected, baseline_values)]
        seed_name = seed_dir.name
        seed = seed_name[len("seed_"):] if seed_name.startswith("seed_") else seed_name
        seed_metrics.append({
            "candidate": candidate["name"], "dataset": candidate["dataset"], "seed": seed,
            "n_subjects": len(rows), "global_best_strategy": baseline,
            "mean_accuracy": mean(selected), "mean_baseline_accuracy": mean(baseline_values),
            "gain_vs_global_best": mean(deltas), "mean_regret": mean(regrets),
            "max_regret": max(regrets), "harmed_count": sum(r > 1e-12 for r in regrets),
            "improved_vs_global_best": sum(d > 1e-12 for d in deltas),
            "worsened_vs_global_best": sum(d < -1e-12 for d in deltas),
        })
        for row, acc, base, regret, delta in zip(rows, selected, baseline_values, regrets, deltas):
            subject = subject_id(row.get("subject"))
            by_subject[subject].append((acc, base, regret, delta))
    gains = [number(row["gain_vs_global_best"]) for row in seed_metrics]
    ci_low, ci_high = bootstrap_ci(gains, rng, n_bootstrap)
    for subject, entries in by_subject.items():
        deltas = [item[3] for item in entries]
        regrets = [item[2] for item in entries]
        subject_rows.append({
            "candidate": candidate["name"], "dataset": candidate["dataset"], "subject": subject,
            "n_seeds": len(entries), "mean_accuracy": mean(item[0] for item in entries),
            "mean_baseline_accuracy": mean(item[1] for item in entries), "mean_gain": mean(deltas),
            "gain_positive_frequency": sum(delta > 1e-12 for delta in deltas) / len(deltas),
            "mean_regret": mean(regrets), "max_regret": max(regrets),
            "high_regret_frequency": sum(regret >= 0.10 for regret in regrets) / len(regrets),
        })
    subject_mean_deltas = [number(row["mean_gain"]) for row in subject_rows]
    try:
        wilcoxon_p = float(stats.wilcoxon(subject_mean_deltas, alternative="greater", zero_method="wilcox").pvalue)
    except ValueError:
        wilcoxon_p = float("nan")
    summary = {
        "candidate": candidate["name"], "dataset": candidate["dataset"], "n_seeds": len(seed_dirs),
        "seed_gain_mean": mean(gains), "seed_gain_std": stdev(gains) if len(gains) > 1 else 0.0,
        "seed_gain_ci95_low": ci_low, "seed_gain_ci95_high": ci_high,
        "subject_mean_gain": mean(subject_mean_deltas), "wilcoxon_greater_p": wilcoxon_p,
        "mean_accuracy": mean(number(row["mean_accuracy"]) for row in seed_metrics),
        "mean_regret": mean(number(row["mean_regret"]) for row in seed_metrics),
        "max_regret_across_seeds": max(number(row["max_regret"]) for row in seed_metrics),
        "mean_harmed_count": mean(number(row["harmed_count"]) for row in seed_metrics),
        "stable_high_regret_subjects": sum(number(row["high_regret_frequency"]) >= 0.6 for row in subject_rows),
    }
    return seed_metrics, subject_rows, summary


def fmt(value, digits: int = 4) -> str:
    x = number(value)
    return f"{x:.{digits}f}" if math.isfinite(x) else ""


def write_report(path: Path, summaries: Sequence[dict], seed_rows: Sequence[dict]) -> None:
    lines = [
        "# Stage-3 Multi-Seed Validation", "",
        "Candidates were fixed before aggregation; no per-seed winner selection is used.", "",
        "## Candidate Summary", "",
        "| Candidate | Dataset | Seeds | Mean Acc | Mean Gain | Gain SD | Bootstrap 95% CI | Wilcoxon p (greater) | Mean Regret | Max Regret | Mean Harmed | Stable High-Regret Subjects |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['candidate']} | {row['dataset']} | {row['n_seeds']} | {fmt(row['mean_accuracy'])} | "
            f"{fmt(row['seed_gain_mean'])} | {fmt(row['seed_gain_std'])} | "
            f"[{fmt(row['seed_gain_ci95_low'])}, {fmt(row['seed_gain_ci95_high'])}] | "
            f"{fmt(row['wilcoxon_greater_p'])} | {fmt(row['mean_regret'])} | "
            f"{fmt(row['max_regret_across_seeds'])} | {fmt(row['mean_harmed_count'], 2)} | {row['stable_high_regret_subjects']} |"
        )
    lines += ["", "## Per-Seed Results", ""]
    lines += ["| Candidate | Seed | Mean Acc | Baseline Acc | Gain | Mean Regret | Max Regret | Harmed |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in sorted(seed_rows, key=lambda r: (r['candidate'], int(r['seed']))):
        lines.append(f"| {row['candidate']} | {row['seed']} | {fmt(row['mean_accuracy'])} | {fmt(row['mean_baseline_accuracy'])} | {fmt(row['gain_vs_global_best'])} | {fmt(row['mean_regret'])} | {fmt(row['max_regret'])} | {row['harmed_count']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat-root", type=Path, default=Path("results/stage5/multiseed_correlation"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/stage5/multiseed_correlation/summary"))
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260720)
    args = parser.parse_args()

    seed_dirs = discover_seed_dirs(args.repeat_root)
    if len(seed_dirs) < 2:
        raise SystemExit(f"Need at least two completed seed directories under {args.repeat_root}; found {len(seed_dirs)}.")
    rng = np.random.default_rng(args.bootstrap_seed)
    all_seed_rows, all_subject_rows, summaries = [], [], []
    for candidate in CANDIDATES:
        seed_rows, subject_rows, summary = analyze_candidate(seed_dirs, candidate, rng, args.n_bootstrap)
        all_seed_rows.extend(seed_rows)
        all_subject_rows.extend(subject_rows)
        summaries.append(summary)
    write_csv(args.out_dir / "multiseed_candidate_summary.csv", summaries)
    write_csv(args.out_dir / "multiseed_seed_results.csv", all_seed_rows)
    write_csv(args.out_dir / "multiseed_subject_stability.csv", all_subject_rows)
    write_report(args.out_dir / "multiseed_validation_report.md", summaries, all_seed_rows)
    print(f"Wrote multi-seed summary for {len(seed_dirs)} seeds to {args.out_dir}")


if __name__ == "__main__":
    main()
