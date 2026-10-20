"""Auxiliary subject diagnostics for candidate gate outputs.

These diagnostics are not the authoritative locked-candidate summaries.
Run python -m mi_eeg.evaluation.gate_subjects --help for input paths.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence

EXPECTED_SUBJECTS = {"openbmi": 54, "physionet": 105}


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
        "dataset", "subject", "family", "method", "feature_set", "model", "fold",
        "selected_strategy", "actual_best_strategy", "decision", "selected_accuracy",
        "traditional_accuracy", "f0_accuracy", "oracle_accuracy", "regret",
        "is_harmed", "is_helped", "used_default", "prob_f0_benefit",
        "gain_threshold", "switch_threshold", "low_threshold", "high_threshold",
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


def subject_id(value) -> str:
    text = str(value).strip()
    if text.upper().startswith("S"):
        digits = "".join(ch for ch in text if ch.isdigit())
        return f"S{int(digits):03d}" if digits else text
    try:
        return f"S{int(float(text)):03d}"
    except ValueError:
        return text


def clean_text(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def normalize_row(row: dict, family: str, source_file: Path) -> dict:
    item = dict(row)
    item["family"] = family
    item["source_file"] = str(source_file)
    item["dataset"] = clean_text(item.get("dataset", "")).lower()
    item["subject"] = subject_id(item.get("subject", ""))
    item["method"] = clean_text(item.get("method", item.get("gate_method", family))) or family
    if "traditional_accuracy" not in item:
        item["traditional_accuracy"] = item.get("traditional", "")
    if "f0_accuracy" not in item:
        item["f0_accuracy"] = item.get("f0", "")
    traditional = as_float(item.get("traditional_accuracy"))
    f0 = as_float(item.get("f0_accuracy"))
    selected = as_float(item.get("selected_accuracy"))
    oracle = as_float(item.get("oracle_accuracy"), max(traditional, f0))
    if not math.isfinite(oracle) and math.isfinite(traditional) and math.isfinite(f0):
        oracle = max(traditional, f0)
    item["oracle_accuracy"] = oracle
    if clean_text(item.get("actual_best_strategy", "")) == "":
        if math.isfinite(f0) and math.isfinite(traditional):
            item["actual_best_strategy"] = "f0" if f0 > traditional else "traditional"
        else:
            item["actual_best_strategy"] = clean_text(item.get("best_strategy", ""))
    if clean_text(item.get("regret", "")) == "":
        item["regret"] = oracle - selected if math.isfinite(oracle) and math.isfinite(selected) else ""
    regret = as_float(item.get("regret"))
    item["is_harmed"] = int(math.isfinite(regret) and regret > 1e-12)
    baseline = max(traditional, f0) if math.isfinite(traditional) and math.isfinite(f0) else float("nan")
    item["is_helped"] = int(math.isfinite(selected) and math.isfinite(baseline) and selected > baseline + 1e-12)
    item["used_default"] = int(str(item.get("decision", "")).lower() == "default")
    return item


def analysis_group_key(row: dict) -> tuple:
    """Group rows for one deployable subject-level configuration.

    Do not include low/high thresholds here. In nested CV those thresholds can be
    fold-specific, so including them splits one complete experiment into partial
    subject subsets. Risk-aware parameters are kept because they define distinct
    deployed gate variants.
    """
    return (
        row.get("dataset", ""),
        row.get("family", ""),
        row.get("method", ""),
        row.get("feature_set", ""),
        row.get("model", ""),
        clean_text(row.get("gain_threshold", "")),
        clean_text(row.get("switch_threshold", "")),
    )


def is_deployable_formal_row(row: dict) -> bool:
    method = clean_text(row.get("method", "")).lower()
    strategy = clean_text(row.get("strategy", "")).lower()
    scope = clean_text(row.get("selection_scope", "")).lower()
    return method not in {"subject_oracle", "oracle"} and strategy != "oracle" and scope != "heldout_grid_diagnostic"


def formal_row_key(row: dict) -> tuple:
    """Map a formal-summary row to the matching raw subject-row group key."""
    family = clean_text(row.get("family", ""))
    # Formal fixed/conservative rows are sourced from a combined summary file,
    # while subject-level rows use the concrete conservative-gate family name.
    if family == "fixed_and_conservative":
        family = "conservative_gate"
    method = clean_text(row.get("method", ""))
    # Raw risk rows are normalized to their concrete family name because they
    # do not store the aggregate-summary method label.
    if family == "risk_aware" and method == "risk_aware_gate":
        method = "risk_aware"
    return (
        clean_text(row.get("dataset", "")).lower(),
        family,
        method,
        clean_text(row.get("feature_set", "")),
        clean_text(row.get("model", "")),
        clean_text(row.get("gain_threshold", "")),
        clean_text(row.get("switch_threshold", "")),
    )



def formal_key_matches_subject_row(formal_key: tuple, subject_row: dict) -> bool:
    """Match formal configuration to fold-level subject rows.

    Empty formal gain/switch fields are wildcards because the deployed value
    may vary by outer fold and is intentionally absent from its aggregate row.
    """
    actual = analysis_group_key(subject_row)
    if actual[:5] != formal_key[:5]:
        return False
    return all(expected == "" or expected == observed for expected, observed in zip(formal_key[5:], actual[5:]))


def formal_comparison_candidates(direction1_dir: Path, stage5_dir: Path, explicit: Path | None) -> List[Path]:
    paths = []
    if explicit is not None:
        paths.append(explicit)
    paths.extend([
        direction1_dir / "final" / "direction1_formal_comparison.csv",
        stage5_dir / "direction1" / "final" / "direction1_formal_comparison.csv",
        stage5_dir / "final" / "direction1_formal_comparison.csv",
        Path("results") / "stage5" / "direction1" / "final" / "direction1_formal_comparison.csv",
    ])
    return paths


def formal_best_keys(direction1_dir: Path, stage5_dir: Path, explicit: Path | None) -> tuple[dict, Path | None]:
    """Return formal-best deployable group keys, or an empty mapping if absent."""
    path = first_existing(formal_comparison_candidates(direction1_dir, stage5_dir, explicit))
    if path is None:
        return {}, None
    grouped = defaultdict(list)
    for row in read_csv(path):
        if is_deployable_formal_row(row):
            grouped[clean_text(row.get("dataset", "")).lower()].append(row)
    selected = {}
    for dataset, rows in grouped.items():
        best = max(rows, key=lambda r: (
            as_float(r.get("mean_accuracy")),
            -as_float(r.get("max_regret")),
            -as_float(r.get("harmed_count")),
        ))
        selected[dataset] = formal_row_key(best)
    return selected, path


def unique_subject_count(rows: Sequence[dict]) -> int:
    return len({r.get("subject", "") for r in rows if r.get("subject", "") != ""})


def best_group(rows: Sequence[dict]) -> tuple:
    grouped = defaultdict(list)
    for row in rows:
        grouped[analysis_group_key(row)].append(row)
    best_key, best_score = None, None
    for key, vals in grouped.items():
        acc_vals = [as_float(v.get("selected_accuracy")) for v in vals if math.isfinite(as_float(v.get("selected_accuracy")))]
        reg_vals = [as_float(v.get("regret")) for v in vals if math.isfinite(as_float(v.get("regret")))]
        acc = sum(acc_vals) / len(acc_vals) if acc_vals else float("nan")
        max_regret = max(reg_vals) if reg_vals else float("nan")
        harmed = sum(int(as_float(v.get("is_harmed"), 0)) for v in vals)
        n_subjects = unique_subject_count(vals)
        score = (n_subjects, acc, -max_regret, -harmed)
        if best_score is None or score > best_score:
            best_key, best_score = key, score
    return best_key


def summarize_subjects(rows: Sequence[dict]) -> List[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("dataset", ""), row.get("subject", ""))].append(row)
    out = []
    for (dataset, subject), vals in grouped.items():
        regrets = [as_float(v.get("regret")) for v in vals if math.isfinite(as_float(v.get("regret")))]
        accs = [as_float(v.get("selected_accuracy")) for v in vals if math.isfinite(as_float(v.get("selected_accuracy")))]
        out.append({
            "dataset": dataset,
            "subject": subject,
            "n_rows": len(vals),
            "mean_selected_accuracy": sum(accs) / len(accs) if accs else float("nan"),
            "max_regret": max(regrets) if regrets else float("nan"),
            "mean_regret": sum(regrets) / len(regrets) if regrets else float("nan"),
            "harmed_count": sum(int(as_float(v.get("is_harmed"), 0)) for v in vals),
            "default_count": sum(int(as_float(v.get("used_default"), 0)) for v in vals),
            "selected_strategies": ";".join(sorted(set(str(v.get("selected_strategy", "")) for v in vals if v.get("selected_strategy", "") != ""))),
            "actual_best_strategy": vals[0].get("actual_best_strategy", ""),
        })
    return sorted(out, key=lambda r: (r["dataset"], -as_float(r["max_regret"]), r["subject"]))


def validate_coverage(matrix: Sequence[dict], allow_partial: bool) -> None:
    by_dataset = defaultdict(set)
    for row in matrix:
        dataset = clean_text(row.get("dataset", "")).lower()
        subject = clean_text(row.get("subject", ""))
        if dataset and subject:
            by_dataset[dataset].add(subject)
    errors = []
    for dataset, expected in EXPECTED_SUBJECTS.items():
        if dataset in by_dataset and len(by_dataset[dataset]) != expected:
            errors.append(f"{dataset}: expected {expected} subjects, got {len(by_dataset[dataset])}")
    if errors and not allow_partial:
        raise SystemExit(
            "Subject-level coverage check failed. "
            + "; ".join(errors)
            + ". This usually means the subject rows are partial or were split by an incompatible grouping. "
            + "Use --allow-partial only for smoke/subset runs."
        )


def write_report(path: Path, matrix: Sequence[dict], high_regret: Sequence[dict], defaults: Sequence[dict], selected_keys: Sequence[tuple], selection_source: str) -> None:
    by_dataset = defaultdict(list)
    for row in matrix:
        by_dataset[row.get("dataset", "")].append(row)
    lines = [
        "# Direction-1 Subject-Level Diagnostics",
        "",
        "This report uses the formal-best deployable configuration when a formal comparison CSV is available.",
        f"Selection source: `{selection_source}`.",
        "Low/high thresholds are not used as grouping keys because nested-CV thresholds can differ by fold.",
        "",
        "## Selected Groups",
        "",
        "| Dataset | Family | Method | Feature Set | Model | Gain Thr | Switch Thr |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for key in selected_keys:
        lines.append(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]} | {key[4]} | {key[5]} | {key[6]} |")
    lines += [
        "",
        "## Dataset Summary",
        "",
        "| Dataset | N Subjects | Mean Selected Acc | Mean Regret | Max Regret | Harmed Rows | Default Rows |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, vals in sorted(by_dataset.items()):
        accs = [as_float(v.get("mean_selected_accuracy")) for v in vals if math.isfinite(as_float(v.get("mean_selected_accuracy")))]
        regrets = [as_float(v.get("mean_regret")) for v in vals if math.isfinite(as_float(v.get("mean_regret")))]
        lines.append(
            f"| {dataset} | {len(vals)} | {fmt(sum(accs) / len(accs) if accs else float('nan'))} | "
            f"{fmt(sum(regrets) / len(regrets) if regrets else float('nan'))} | "
            f"{fmt(max(as_float(v.get('max_regret')) for v in vals))} | "
            f"{sum(int(as_float(v.get('harmed_count'), 0)) for v in vals)} | "
            f"{sum(int(as_float(v.get('default_count'), 0)) for v in vals)} |"
        )
    lines += [
        "",
        "## Highest-Regret Subjects",
        "",
        "| Dataset | Subject | Max Regret | Mean Regret | Harmed Rows | Default Rows | Selected Strategies | Best Strategy |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for r in high_regret[:60]:
        lines.append(
            f"| {r.get('dataset')} | {r.get('subject')} | {fmt(r.get('max_regret'))} | "
            f"{fmt(r.get('mean_regret'))} | {r.get('harmed_count')} | {r.get('default_count')} | "
            f"{r.get('selected_strategies')} | {r.get('actual_best_strategy')} |"
        )
    lines += [
        "",
        "## Defaulted Subjects",
        "",
        "| Dataset | Subject | Default Rows | Max Regret | Mean Regret |",
        "|---|---|---:|---:|---:|",
    ]
    for r in defaults[:60]:
        lines.append(f"| {r.get('dataset')} | {r.get('subject')} | {r.get('default_count')} | {fmt(r.get('max_regret'))} | {fmt(r.get('mean_regret'))} |")
    write_text(path, "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction1-dir", type=Path, default=Path("results/stage5/direction1"))
    parser.add_argument("--stage5-dir", type=Path, default=Path("results/stage5"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/stage5/direction1/subject_analysis"))
    parser.add_argument("--formal-comparison", type=Path, default=None, help="Optional formal comparison CSV. When available, its best deployable row determines the audited group.")
    parser.add_argument("--allow-partial", action="store_true", help="Allow subset/smoke runs without 54/105-subject coverage.")
    args = parser.parse_args()

    sources = [
        ("conservative_gate", "baselines/gate_subject_level_results.csv"),
        ("threshold_reject", "thresholds/gate_threshold_subjects.csv"),
        ("risk_aware", "risk/risk_aware_gate_subjects.csv"),
        ("feature_ablation", "feature_ablation/gate_feature_set_subjects.csv"),
    ]
    all_rows: List[dict] = []
    for family, rel in sources:
        path = first_existing(candidate_paths(args.direction1_dir, args.stage5_dir, rel))
        if not path:
            continue
        all_rows.extend(normalize_row(row, family, path) for row in read_csv(path))

    if not all_rows:
        raise SystemExit("No subject-level Direction-1 files found. Provide gate_subject_level/risk/threshold subject CSV files first.")

    formal_keys, formal_path = formal_best_keys(args.direction1_dir, args.stage5_dir, args.formal_comparison)
    selected = []
    selected_keys = []
    for dataset in sorted(set(r.get("dataset", "") for r in all_rows)):
        ds = [r for r in all_rows if r.get("dataset", "") == dataset]
        key = formal_keys.get(dataset)
        if key is not None:
            matches = [r for r in ds if formal_key_matches_subject_row(key, r)]
            if not matches:
                raise SystemExit(f"Formal best configuration for {dataset} did not match any subject-level rows: {key}.")
        else:
            key = best_group(ds)
            matches = [r for r in ds if analysis_group_key(r) == key]
        selected_keys.append(key)
        selected.extend(matches)
    selection_source = str(formal_path) if formal_path is not None else "raw subject rows fallback (no formal comparison CSV found)"

    matrix = summarize_subjects(selected)
    validate_coverage(matrix, args.allow_partial)
    high_regret = sorted(matrix, key=lambda r: (r.get("dataset", ""), -as_float(r.get("max_regret"))))
    helped_harmed = sorted(selected, key=lambda r: (r.get("dataset", ""), -as_float(r.get("regret"))))
    defaults = sorted(
        [r for r in matrix if int(as_float(r.get("default_count"), 0)) > 0],
        key=lambda r: (r.get("dataset", ""), -int(as_float(r.get("default_count"), 0)), -as_float(r.get("max_regret"))),
    )

    write_csv(args.out_dir / "subject_decision_rows.csv", selected)
    write_csv(args.out_dir / "subject_decision_matrix.csv", matrix)
    write_csv(args.out_dir / "high_regret_subjects.csv", high_regret)
    write_csv(args.out_dir / "helped_harmed_subjects.csv", helped_harmed)
    write_csv(args.out_dir / "threshold_default_subjects.csv", defaults)
    write_report(args.out_dir / "subject_level_report.md", matrix, high_regret, defaults, selected_keys, selection_source)
    print(f"Wrote Direction-1 subject-level diagnostics to {args.out_dir}")


if __name__ == "__main__":
    main()
