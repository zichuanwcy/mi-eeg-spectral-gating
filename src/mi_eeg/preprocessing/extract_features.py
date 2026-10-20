"""Memory-safe subject descriptor extraction.



This version loads one subject at a time, writes progress after every subject,
and stores only small covariance matrices for optional source-similarity
features. It is designed for CPU/no-card mode and low RAM pressure.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


from mi_eeg.preprocessing.descriptors import (
    compute_subject_descriptors,
    source_similarity_descriptors,
    subject_covariance,
)


PHYSIONET_EXCLUDE = {88, 89, 92, 100}
DEFAULT_SUBJECTS = {
    "physionet": [s for s in range(1, 110) if s not in PHYSIONET_EXCLUDE],
    "openbmi": list(range(1, 55)),
}
DEFAULT_SFREQ = {"physionet": 160.0, "openbmi": 250.0}


def descriptor_cfg(dataset: str, channel_mode: str) -> dict:
    if dataset == "physionet":
        return {
            "exp_id": "Physionet_Descriptor",
            "dataset": "physionet",
            "channel_mode": channel_mode,
            "n_classes": 4,
            "sfreq": 160.0,
            "t_start": 0.0,
            "t_end": 4.0,
            "bandpass": False,
            "bandpass_freq": (4, 40),
            "filter_order": 4,
            "alignment": "none",
            "qc_min_class_trials": 10,
        }
    if dataset == "openbmi":
        return {
            "exp_id": "OpenBMI_Descriptor",
            "dataset": "openbmi",
            "channel_mode": channel_mode,
            "n_classes": 2,
            "sfreq": 250.0,
            "t_start": 0.0,
            "t_end": 4.0,
            "bandpass": False,
            "bandpass_freq": (4, 40),
            "filter_order": 4,
            "alignment": "none",
        }
    raise ValueError(f"Unsupported dataset: {dataset}")


def load_one_subject(dataset: str, subject: int, cfg: dict, data_dir: Path, channel_mode: str):
    if dataset == "physionet":
        from mi_eeg.data.physionet import load_physionet_subject
        return load_physionet_subject(subject, cfg, data_dir, channel_mode)
    if dataset == "openbmi":
        from mi_eeg.data.openbmi import load_openbmi_subject
        return load_openbmi_subject(subject, cfg, data_dir, channel_mode)
    raise ValueError(dataset)


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


def dump_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_subjects(dataset: str, raw: str | None) -> List[int]:
    if not raw:
        return list(DEFAULT_SUBJECTS[dataset])
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    if dataset == "physionet":
        out = [s for s in out if s not in PHYSIONET_EXCLUDE]
    return sorted(set(out))


def process_dataset(
    dataset: str,
    data_dir: Path,
    out_dir: Path,
    channel_mode: str,
    subjects: List[int],
    add_source_similarity: bool,
) -> List[dict]:
    cfg = descriptor_cfg(dataset, channel_mode)
    rows: List[dict] = []
    covs: Dict[str, np.ndarray] = {}
    errors: Dict[str, str] = {}
    partial_path = out_dir / f"{dataset}_descriptors_partial.csv"
    status_path = out_dir / f"{dataset}_descriptor_status.json"

    print(f"[{dataset}] start: {len(subjects)} subjects, channel_mode={channel_mode}", flush=True)
    for idx, subject in enumerate(subjects, start=1):
        sid = f"S{subject:03d}"
        try:
            print(f"[{dataset}] loading {sid} ({idx}/{len(subjects)})", flush=True)
            X, _y, meta = load_one_subject(dataset, subject, cfg, data_dir, channel_mode)
            resolved_mode = meta.get("channel_mode_resolved", channel_mode) if isinstance(meta, dict) else channel_mode
            row = compute_subject_descriptors(
                X,
                sfreq=DEFAULT_SFREQ[dataset],
                dataset=dataset,
                subject=sid,
                channel_mode=resolved_mode,
                source_covs=None,
            )
            row["loader_n_trials"] = int(meta.get("n_trials", len(X))) if isinstance(meta, dict) else int(len(X))
            row["class_counts"] = json.dumps(meta.get("class_counts", {}), ensure_ascii=False) if isinstance(meta, dict) else "{}"
            rows.append(row)
            if add_source_similarity:
                covs[sid] = subject_covariance(X)
            # Release large array before next subject.
            del X
            write_csv(partial_path, rows)
            dump_status(status_path, {"dataset": dataset, "completed": len(rows), "errors": errors, "last_subject": sid})
            print(f"[{dataset}] done {sid}; descriptors={len(rows)}", flush=True)
        except Exception as exc:
            errors[sid] = str(exc)
            dump_status(status_path, {"dataset": dataset, "completed": len(rows), "errors": errors, "last_error_subject": sid})
            print(f"[{dataset}] ERROR {sid}: {exc}", flush=True)

    if add_source_similarity and covs:
        print(f"[{dataset}] computing source-similarity descriptors", flush=True)
        for row in rows:
            sid = str(row["subject"])
            source_covs = {k: v for k, v in covs.items() if k != sid}
            row.update(source_similarity_descriptors(covs[sid], source_covs))

    final_path = out_dir / f"{dataset}_descriptors.csv"
    write_csv(final_path, rows)
    dump_status(status_path, {"dataset": dataset, "completed": len(rows), "errors": errors, "final_path": str(final_path)})
    print(f"[{dataset}] wrote {final_path} ({len(rows)} subjects, errors={len(errors)})", flush=True)
    return rows


def write_readme(path: Path, args, completed: Dict[str, int]) -> None:
    lines = [
        "# Stage-5 Subject Descriptors", "",
        "Generated by `extract_subject_descriptors_stream.py`.", "",
        "This script loads one subject at a time and is safe for CPU/no-card mode.", "",
        "Descriptor schema: v2 (spectral, covariance, low-dimensional channel-correlation, and source-similarity features).", "",
        "Correlation descriptors are summary statistics rather than flattened channel-pair matrices, to avoid subject-level overfitting.", "",
        f"Data dir: `{args.data_dir}`", "",
        "## Completed", "",
    ]
    for dataset, n in completed.items():
        lines.append(f"- {dataset}: {n} subjects")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["physionet", "openbmi"], choices=["physionet", "openbmi"])
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "raw")
    parser.add_argument("--out-dir", type=Path, default=Path("results") / "stage5" / "descriptors")
    parser.add_argument("--channel-mode", default="matched", choices=["matched", "all", "mi20"])
    parser.add_argument("--subjects", default=None, help="Optional subject list, e.g. 1-10 or 1,2,3. Applies to all selected datasets.")
    parser.add_argument("--no-source-similarity", action="store_true", help="Skip Riemannian source-distance features for faster first pass.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    completed: Dict[str, int] = {}
    all_rows: List[dict] = []
    for dataset in args.datasets:
        subjects = parse_subjects(dataset, args.subjects)
        rows = process_dataset(
            dataset=dataset,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            channel_mode=args.channel_mode,
            subjects=subjects,
            add_source_similarity=not args.no_source_similarity,
        )
        completed[dataset] = len(rows)
        all_rows.extend(rows)

    if all_rows:
        write_csv(args.out_dir / "all_descriptors.csv", all_rows)
    write_readme(args.out_dir / "descriptor_readme.md", args, completed)
    dump_status(args.out_dir / "descriptor_status.json", {"completed": completed, "data_dir": str(args.data_dir), "channel_mode": args.channel_mode})
    print(f"All done. Wrote descriptor files to {args.out_dir}", flush=True)
    print("Next: run analyze_subject_oracle.py, then train_subject_gate.py.", flush=True)


if __name__ == "__main__":
    main()
