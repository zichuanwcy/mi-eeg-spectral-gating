"""subject-level oracle and delta analysis.

This script consumes experiment logs under ``results/`` and produces normalized
subject-level tables for the adaptive-subject line of work. It is independent
of raw EEG data, so it can be run before any new GPU experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SUBJECT_RE = re.compile(r"\bS(\d{1,3}):\s+([0-9]+(?:\.[0-9]+)?)")
FOLD_RE = re.compile(r"fold acc=([0-9]+(?:\.[0-9]+)?), kappa=([0-9]+(?:\.[0-9]+)?)")


@dataclass(frozen=True)
class ExperimentSpec:
    dataset: str
    exp_id: str
    strategy: str
    pattern: str
    complete_required: bool = True


EXPERIMENTS: Sequence[ExperimentSpec] = (
    ExperimentSpec("physionet", "Physionet_Traditional", "traditional", "Physionet_Traditional_*.log"),
    ExperimentSpec("physionet", "Physionet_F0", "f0", "Physionet_F0_*.log"),
    ExperimentSpec("physionet", "Physionet_SE", "se", "Physionet_SE_*.log"),
    ExperimentSpec("openbmi", "OpenBMI_Traditional", "traditional", "OpenBMI_Traditional_*.log"),
    ExperimentSpec("openbmi", "OpenBMI_F0", "f0", "OpenBMI_F0_*.log"),
    ExperimentSpec("openbmi", "OpenBMI_SE", "se", "OpenBMI_SE_*.log", complete_required=False),
)


def latest_file(root: Path, pattern: str) -> Optional[Path]:
    matches = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _numbers_after(line: str, marker: str) -> List[float]:
    tail = line.split(marker, 1)[1].replace("?", " ")
    vals: List[float] = []
    for token in tail.split():
        try:
            vals.append(float(token))
        except ValueError:
            continue
    return vals


def parse_log(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    subjects = {f"S{m.group(1).zfill(3)}": float(m.group(2)) for m in SUBJECT_RE.finditer(text)}
    folds = [(float(acc), float(kappa)) for acc, kappa in FOLD_RE.findall(text)]
    completed = "??" in text and bool(subjects)

    reported_mean = None
    reported_std = None
    reported_kappa = None
    for line in text.splitlines():
        if "??:" in line:
            nums = _numbers_after(line, "??:")
            if len(nums) >= 2:
                reported_mean, reported_std = nums[0], nums[1]
        if "Kappa:" in line:
            nums = _numbers_after(line, "Kappa:")
            if nums:
                reported_kappa = nums[0]

    return {
        "path": str(path),
        "completed": completed,
        "subjects": subjects,
        "folds": folds,
        "reported_mean": reported_mean,
        "reported_std": reported_std,
        "reported_kappa": reported_kappa,
    }


def load_experiments(results_dir: Path) -> Dict[str, dict]:
    loaded: Dict[str, dict] = {}
    for spec in EXPERIMENTS:
        path = latest_file(results_dir, spec.pattern)
        if path is None:
            continue
        parsed = parse_log(path)
        parsed.update({
            "dataset": spec.dataset,
            "exp_id": spec.exp_id,
            "strategy": spec.strategy,
            "complete_required": spec.complete_required,
        })
        loaded[f"{spec.dataset}:{spec.strategy}"] = parsed
    return loaded


def summarize_experiment(result: dict) -> dict:
    vals = list(result["subjects"].values())
    folds = result["folds"]
    return {
        "dataset": result["dataset"],
        "exp_id": result["exp_id"],
        "strategy": result["strategy"],
        "completed": result["completed"],
        "n_subjects": len(vals),
        "subject_mean": mean(vals) if vals else None,
        "subject_std": stdev(vals) if len(vals) > 1 else None,
        "fold_mean": mean(acc for acc, _ in folds) if folds else None,
        "fold_kappa_mean": mean(k for _, k in folds) if folds else None,
        "reported_mean": result["reported_mean"],
        "reported_std": result["reported_std"],
        "reported_kappa": result["reported_kappa"],
        "source_path": result["path"],
    }


def build_subject_rows(loaded: Dict[str, dict]) -> List[dict]:
    rows: List[dict] = []
    for result in loaded.values():
        for subject, accuracy in sorted(result["subjects"].items()):
            rows.append({
                "dataset": result["dataset"],
                "subject": subject,
                "exp_id": result["exp_id"],
                "strategy": result["strategy"],
                "accuracy": accuracy,
                "completed": result["completed"],
                "source_path": result["path"],
            })
    return rows


def pivot_subjects(rows: Iterable[dict]) -> Dict[Tuple[str, str], dict]:
    pivot: Dict[Tuple[str, str], dict] = {}
    for row in rows:
        key = (row["dataset"], row["subject"])
        pivot.setdefault(key, {"dataset": row["dataset"], "subject": row["subject"]})
        pivot[key][row["strategy"]] = row["accuracy"]
    return pivot


def build_delta_rows(pivot: Dict[Tuple[str, str], dict]) -> List[dict]:
    out: List[dict] = []
    for row in sorted(pivot.values(), key=lambda r: (r["dataset"], r["subject"])):
        item = dict(row)
        if "traditional" in row and "f0" in row:
            item["delta_f0_minus_traditional"] = row["f0"] - row["traditional"]
            item["best_traditional_f0"] = "f0" if row["f0"] >= row["traditional"] else "traditional"
            item["oracle_traditional_f0"] = max(row["traditional"], row["f0"])
        if "f0" in row and "se" in row:
            item["delta_se_minus_f0"] = row["se"] - row["f0"]
            item["best_f0_se"] = "se" if row["se"] >= row["f0"] else "f0"
            item["oracle_f0_se"] = max(row["f0"], row["se"])
        candidates = {k: row[k] for k in ("traditional", "f0", "se") if k in row}
        if candidates:
            best_strategy, best_acc = max(candidates.items(), key=lambda kv: kv[1])
            item["oracle_available_strategy"] = best_strategy
            item["oracle_available_accuracy"] = best_acc
        out.append(item)
    return out


def oracle_summary(delta_rows: Sequence[dict]) -> List[dict]:
    rows: List[dict] = []
    for dataset in sorted({r["dataset"] for r in delta_rows}):
        subset = [r for r in delta_rows if r["dataset"] == dataset]
        row = {"dataset": dataset, "n_subjects": len(subset)}
        for strategy in ("traditional", "f0", "se"):
            vals = [r[strategy] for r in subset if strategy in r]
            if vals:
                row[f"{strategy}_mean"] = mean(vals)
        vals = [r["oracle_traditional_f0"] for r in subset if "oracle_traditional_f0" in r]
        if vals:
            row["oracle_traditional_f0_mean"] = mean(vals)
            best_single = max(row.get("traditional_mean", float("-inf")), row.get("f0_mean", float("-inf")))
            row["oracle_gain_vs_best_single"] = row["oracle_traditional_f0_mean"] - best_single
            row["f0_wins"] = sum(r.get("best_traditional_f0") == "f0" for r in subset)
            row["traditional_wins"] = sum(r.get("best_traditional_f0") == "traditional" for r in subset)
        vals_all = [r["oracle_available_accuracy"] for r in subset if "oracle_available_accuracy" in r]
        if vals_all:
            row["oracle_available_mean"] = mean(vals_all)
        rows.append(row)
    return rows


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


def fmt(value: object) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def write_report(path: Path, summary_rows: Sequence[dict], oracle_rows: Sequence[dict]) -> None:
    lines = ["# Stage-5 Oracle Report", ""]
    lines.append("## Experiment Summary")
    lines.append("")
    lines.append("| Dataset | Strategy | Subjects | Subject Mean | Fold Mean | Kappa | Completed |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for row in summary_rows:
        kappa = row["reported_kappa"] if row["reported_kappa"] is not None else row["fold_kappa_mean"]
        lines.append(
            f"| {row['dataset']} | {row['strategy']} | {row['n_subjects']} | "
            f"{fmt(row['subject_mean'])} | {fmt(row['fold_mean'])} | {fmt(kappa)} | {row['completed']} |"
        )
    lines.append("")
    lines.append("## Oracle Summary")
    lines.append("")
    lines.append("| Dataset | Traditional | F0 | SE | Oracle(T,F0) | Gain vs Best Single | F0 Wins | Traditional Wins |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in oracle_rows:
        lines.append(
            f"| {row['dataset']} | {fmt(row.get('traditional_mean'))} | "
            f"{fmt(row.get('f0_mean'))} | {fmt(row.get('se_mean'))} | "
            f"{fmt(row.get('oracle_traditional_f0_mean'))} | "
            f"{fmt(row.get('oracle_gain_vs_best_single'))} | "
            f"{row.get('f0_wins', '')} | {row.get('traditional_wins', '')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results", type=Path)
    parser.add_argument("--out-dir", default=Path("results") / "stage5", type=Path)
    args = parser.parse_args()

    loaded = load_experiments(args.results_dir)
    if not loaded:
        raise SystemExit(f"No logs found under {args.results_dir}")

    summary_rows = [summarize_experiment(v) for v in loaded.values()]
    subject_rows = build_subject_rows(loaded)
    delta_rows = build_delta_rows(pivot_subjects(subject_rows))
    oracle_rows = oracle_summary(delta_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "experiment_summary.csv", summary_rows)
    write_csv(args.out_dir / "subject_level_summary.csv", subject_rows)
    write_csv(args.out_dir / "subject_delta_all.csv", delta_rows)
    for dataset in sorted({r["dataset"] for r in delta_rows}):
        write_csv(args.out_dir / f"subject_delta_{dataset}.csv", [r for r in delta_rows if r["dataset"] == dataset])
    write_csv(args.out_dir / "oracle_summary.csv", oracle_rows)
    write_report(args.out_dir / "oracle_report.md", summary_rows, oracle_rows)
    (args.out_dir / "oracle_summary.json").write_text(
        json.dumps({"experiments": summary_rows, "oracle": oracle_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote stage-5 oracle files to {args.out_dir}")
    print("Next: run extract_subject_descriptors.py once raw EEG data is available locally or on AutoDL.")


if __name__ == "__main__":
    main()
