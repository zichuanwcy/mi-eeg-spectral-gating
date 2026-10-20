"""Join fixed-pipeline outcomes to subject descriptors, without fitting a gate."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Sequence

TARGET_COLUMN = "delta_f0_minus_traditional"
SUBJECT_COUNTS = {"openbmi": 54, "physionet": 105}


def merge_delta_descriptors(delta_rows: Sequence[dict], descriptor_rows: Sequence[dict], dataset: str) -> List[dict]:
    desc = {(r["dataset"], r["subject"]): r for r in descriptor_rows if r.get("dataset") == dataset}
    merged = []
    for row in delta_rows:
        if row.get("dataset") != dataset or row.get(TARGET_COLUMN, "") == "":
            continue
        key = (row["dataset"], row["subject"])
        if key not in desc:
            continue
        item = dict(row)
        item.update({f"desc_{k}": v for k, v in desc[key].items() if k not in ("dataset", "subject")})
        merged.append(item)
    return merged


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def prepare_inputs(delta_csv: Path, descriptor_csv: Sequence[Path], datasets: Sequence[str], output_dir: Path) -> dict[str, int]:
    """Validate unique subject keys and manuscript subsets before writing CSVs."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Use an empty output directory to avoid overwriting gate inputs.")
    delta = read_rows(delta_csv)
    descriptors = [row for path in descriptor_csv for row in read_rows(path)]
    for rows, label in [(delta, "outcomes"), (descriptors, "descriptors")]:
        if not rows or any(not row.get("dataset") or not row.get("subject") for row in rows):
            raise ValueError(f"Missing dataset/subject fields in {label}.")
        keys = [(r["dataset"], r["subject"]) for r in rows if r["dataset"] in datasets]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate dataset/subject keys in {label}.")
    merged = {dataset: merge_delta_descriptors(delta, descriptors, dataset) for dataset in datasets}
    for dataset, rows in merged.items():
        if len(rows) != SUBJECT_COUNTS[dataset]:
            raise ValueError(f"{dataset}: expected {SUBJECT_COUNTS[dataset]} subjects; found {len(rows)}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset, rows in merged.items():
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (output_dir / f"{dataset}_gate_dataset.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return {dataset: len(rows) for dataset, rows in merged.items()}
