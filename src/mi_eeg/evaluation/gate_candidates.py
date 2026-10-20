"""Collect candidate comparison tables from existing gate outputs.

Run python -m mi_eeg.evaluation.gate_candidates --help for input paths.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence


def read_csv(path: Path) -> List[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({k for row in rows for k in row.keys()})
    priority = [
        "dataset", "family", "method", "feature_set", "model", "low_threshold", "high_threshold",
        "n_subjects", "mean_accuracy", "gain_vs_global_best_single", "mean_regret", "max_regret",
        "harmed_count", "default_count", "coverage", "source_file",
    ]
    fields = [f for f in priority if f in fields] + [f for f in fields if f not in priority]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
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


def fmt(value, digits: int = 4) -> str:
    x = as_float(value)
    return f"{x:.{digits}f}" if math.isfinite(x) else ""


def fmt_signed(value, digits: int = 4) -> str:
    x = as_float(value)
    return f"{x:+.{digits}f}" if math.isfinite(x) else ""


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def candidate_paths(direction1_dir: Path, stage5_dir: Path, rel: str) -> List[Path]:
    return [
        direction1_dir / rel,
        stage5_dir / "direction1" / rel,
        stage5_dir / rel,
        Path("results") / rel,
        Path("results") / Path(rel).name,
    ]


def add_rows(rows: List[dict], source_rows: Sequence[dict], family: str, source_file: Path) -> None:
    for row in source_rows:
        item = dict(row)
        item["family"] = family
        item["source_file"] = str(source_file)
        rows.append(item)


def threshold_best_rows(rows: Sequence[dict]) -> List[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("dataset"), row.get("feature_set"), row.get("model"))].append(row)
    best = []
    for vals in grouped.values():
        best.append(max(vals, key=lambda r: (
            as_float(r.get("mean_accuracy")),
            -as_float(r.get("max_regret")),
            -as_float(r.get("harmed_count")),
            -as_float(r.get("default_count")),
        )))
    return best


def is_deployable(row: dict) -> bool:
    """Return False for oracle upper-bound rows.

    Oracle rows remain in the full comparison table for context, but they must
    never be selected as a best method or included among deployable top rows.
    """
    method = str(row.get("method", "")).strip().lower()
    strategy = str(row.get("strategy", "")).strip().lower()
    scope = str(row.get("selection_scope", "")).strip().lower()
    return method not in {"subject_oracle", "oracle"} and strategy != "oracle" and scope != "heldout_grid_diagnostic"


def dataset_best_rows(rows: Sequence[dict]) -> List[dict]:
    grouped = defaultdict(list)
    for row in rows:
        if is_deployable(row):
            grouped[row.get("dataset", "")].append(row)
    best = []
    for vals in grouped.values():
        best.append(max(vals, key=lambda r: (
            as_float(r.get("mean_accuracy")),
            -as_float(r.get("max_regret")),
            -as_float(r.get("harmed_count")),
        )))
    return sorted(best, key=lambda r: r.get("dataset", ""))


def write_report(path: Path, rows: Sequence[dict], best_rows: Sequence[dict]) -> None:
    lines = [
        "# Direction-1 Formal Result Summary",
        "",
        "This table merges fixed baselines, ordinary conservative gates, risk-aware gates, feature ablations, and nested deployable threshold/reject summaries.",
        "Subject oracle and held-out threshold-grid maxima are contextual rows only and are excluded from all deployable rankings below.",
        "",
        "## Best Method Per Dataset",
        "",
        "| Dataset | Family | Method | Feature Set | Model | Low | High | Mean Acc | Gain vs Global Best | Mean Regret | Max Regret | Harmed | Default | Coverage |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in best_rows:
        lines.append(
            f"| {r.get('dataset', '')} | {r.get('family', '')} | {r.get('method', '')} | "
            f"{r.get('feature_set', '')} | {r.get('model', '')} | {fmt(r.get('low_threshold'), 2)} | "
            f"{fmt(r.get('high_threshold'), 2)} | {fmt(r.get('mean_accuracy'))} | "
            f"{fmt_signed(r.get('gain_vs_global_best_single'))} | {fmt(r.get('mean_regret'))} | "
            f"{fmt(r.get('max_regret'))} | {r.get('harmed_count', '')} | "
            f"{r.get('default_count', '')} | {fmt(r.get('coverage'), 3)} |"
        )
    lines += [
        "",
        "## Top Rows",
        "",
        "| Dataset | Family | Method | Feature Set | Model | Mean Acc | Mean Regret | Max Regret | Harmed | Default |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    top_rows = sorted((r for r in rows if is_deployable(r)), key=lambda r: (r.get("dataset", ""), -as_float(r.get("mean_accuracy"))))[:80]
    for r in top_rows:
        lines.append(
            f"| {r.get('dataset', '')} | {r.get('family', '')} | {r.get('method', '')} | "
            f"{r.get('feature_set', '')} | {r.get('model', '')} | {fmt(r.get('mean_accuracy'))} | "
            f"{fmt(r.get('mean_regret'))} | {fmt(r.get('max_regret'))} | {r.get('harmed_count', '')} | "
            f"{r.get('default_count', '')} |"
        )
    write_text(path, "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction1-dir", type=Path, default=Path("results/stage5/direction1"))
    parser.add_argument("--stage5-dir", type=Path, default=Path("results/stage5"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/stage5/direction1/final"))
    args = parser.parse_args()

    collected: List[dict] = []
    sources = [
        ("fixed_and_conservative", "baselines/gate_baseline_comparison.csv"),
        ("risk_aware", "risk/risk_aware_gate_results.csv"),
        ("feature_ablation", "feature_ablation/gate_feature_set_ablation.csv"),
    ]
    for family, rel in sources:
        path = first_existing(candidate_paths(args.direction1_dir, args.stage5_dir, rel))
        if path:
            add_rows(collected, read_csv(path), family, path)

    threshold_path = first_existing(candidate_paths(args.direction1_dir, args.stage5_dir, "thresholds/gate_threshold_deployed_summary.csv"))
    if threshold_path:
        threshold_rows = read_csv(threshold_path)
        for row in threshold_rows:
            row.setdefault("method", "threshold_reject")
            row.setdefault("threshold_selection", "nested_train_fold")
        add_rows(collected, threshold_rows, "threshold_reject", threshold_path)

    if not collected:
        raise SystemExit("No Direction-1 result files found. Provide the completed Direction-1 CSV outputs first.")

    best = dataset_best_rows(collected)
    write_csv(args.out_dir / "direction1_formal_comparison.csv", collected)
    write_csv(args.out_dir / "direction1_best_by_dataset.csv", best)
    write_report(args.out_dir / "direction1_formal_comparison.md", collected, best)
    write_text(
        args.out_dir / "direction1_key_findings.md",
        "# Direction-1 Key Findings\n\n"
        + "\n".join(
            f"- {r.get('dataset')}: best current row is {r.get('family')} / {r.get('method')} "
            f"with {r.get('feature_set')} + {r.get('model')}, mean accuracy {fmt(r.get('mean_accuracy'))}, "
            f"mean regret {fmt(r.get('mean_regret'))}, max regret {fmt(r.get('max_regret'))}."
            for r in best
        )
        + "\n",
    )
    print(f"Wrote formal Direction-1 summary to {args.out_dir}")


if __name__ == "__main__":
    main()
