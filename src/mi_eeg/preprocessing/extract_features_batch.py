"""Batch extraction of unlabeled, wideband pre-EA subject descriptors.

Use the streaming entry extract_features.py for lower memory use.
This backend also accepts the historical input modes through its --help.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


from mi_eeg.preprocessing.descriptors import compute_subject_descriptors, subject_covariance


DATASET_DEFAULT_SFREQ = {
    "physionet": 160.0,
    "openbmi": 250.0,
    "bci": 250.0,
}


def descriptor_cfg(dataset: str, channel_mode: str) -> dict:
    """Return a loader cfg for unlabeled descriptor extraction."""
    if dataset == "physionet":
        return {
            "exp_id": "Physionet_Descriptor",
            "dataset": "physionet",
            "channel_mode": channel_mode,
            "n_classes": 4,
            "n_channels": 22 if channel_mode == "matched" else 64,
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
            "n_channels": 22,
            "sfreq": 250.0,
            "t_start": 0.0,
            "t_end": 4.0,
            "bandpass": False,
            "bandpass_freq": (4, 40),
            "filter_order": 4,
            "alignment": "none",
        }
    raise ValueError(f"Loader mode currently supports physionet/openbmi, got {dataset}")


def load_dataset_with_stage4_loader(dataset: str, data_dir: Path, out_dir: Path, channel_mode: str) -> tuple[dict, dict]:
    cfg = descriptor_cfg(dataset, channel_mode)
    if dataset == "physionet":
        from mi_eeg.data.physionet import load_physionet_dataset
        return load_physionet_dataset(cfg, data_dir, out_dir)
    if dataset == "openbmi":
        from mi_eeg.data.openbmi import load_openbmi_dataset
        return load_openbmi_dataset(cfg, data_dir, out_dir)
    raise ValueError(dataset)


def rows_from_loader(dataset: str, data_dir: Path, out_dir: Path, channel_mode: str) -> List[dict]:
    stage_data, qc = load_dataset_with_stage4_loader(dataset, data_dir, out_dir, channel_mode)
    payloads = []
    for subject_id in sorted(stage_data):
        X, _y = stage_data[subject_id]["all"]
        payloads.append({
            "subject": f"S{int(subject_id):03d}",
            "X": X,
            "sfreq": DATASET_DEFAULT_SFREQ[dataset],
            "channel_mode": channel_mode,
        })
    return descriptor_rows_from_payloads(dataset, payloads)


def iter_npz_subjects(root: Path, dataset: str):
    folder = root / dataset
    if not folder.exists():
        return
    for path in sorted(folder.glob("S*.npz")):
        subject = path.stem
        payload = np.load(path, allow_pickle=True)
        if "X" not in payload:
            raise ValueError(f"{path} does not contain key 'X'")
        X = payload["X"]
        sfreq = float(payload["sfreq"]) if "sfreq" in payload else DATASET_DEFAULT_SFREQ.get(dataset, 250.0)
        channel_mode = str(payload["channel_mode"]) if "channel_mode" in payload else "unknown"
        yield {"subject": subject, "X": X, "sfreq": sfreq, "channel_mode": channel_mode, "source_file": str(path)}


def descriptor_rows_from_payloads(dataset: str, payloads: List[dict]) -> List[dict]:
    if not payloads:
        return []
    source_covs_all = {item["subject"]: subject_covariance(item["X"]) for item in payloads}
    rows = []
    for item in payloads:
        source_covs = {k: v for k, v in source_covs_all.items() if k != item["subject"]}
        row = compute_subject_descriptors(
            item["X"],
            sfreq=float(item["sfreq"]),
            dataset=dataset,
            subject=item["subject"],
            channel_mode=str(item.get("channel_mode", "unknown")),
            source_covs=source_covs,
        )
        if "source_file" in item:
            row["source_file"] = item["source_file"]
        rows.append(row)
    return rows


def extract_from_npz(npz_root: Path, dataset: str) -> List[dict]:
    payloads = list(iter_npz_subjects(npz_root, dataset) or [])
    return descriptor_rows_from_payloads(dataset, payloads)


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(path: Path, datasets: Iterable[str], completed: Dict[str, int], mode: str, data_dir: Path, npz_root: Path) -> None:
    lines = [
        "# Subject Descriptor Extraction", "",
        "This folder stores unlabeled subject descriptors for stage-5 gate and source-selection experiments.", "",
        f"Input mode: `{mode}`", "",
        "## Completed", "",
    ]
    for dataset in datasets:
        lines.append(f"- {dataset}: {completed.get(dataset, 0)} subjects")
    lines.extend([
        "", "## Loader Mode", "",
        f"Data dir: `{data_dir}`", "",
        "Loader mode uses wideband, no-EA trials to describe the target subject.",
        "Labels are loaded by the dataset loader only for QC; descriptor values do not use labels.",
        "", "## NPZ Mode", "",
        f"NPZ root: `{npz_root}`", "",
        "Each subject file should be named like `S001.npz` and contain `X=(trials, channels, time)`.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-mode", choices=["loader", "npz"], default="loader")
    parser.add_argument("--data-dir", type=Path, default=Path("data") / "raw")
    parser.add_argument("--npz-root", type=Path, default=Path("data") / "stage5_npz")
    parser.add_argument("--out-dir", type=Path, default=Path("results") / "stage5" / "descriptors")
    parser.add_argument("--datasets", nargs="+", default=["physionet", "openbmi"])
    parser.add_argument("--channel-mode", default="matched", choices=["matched", "all", "mi20"])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    completed: Dict[str, int] = {}
    all_rows: List[dict] = []
    missing = []

    for dataset in args.datasets:
        if args.input_mode == "loader":
            rows = rows_from_loader(dataset, args.data_dir, Path("results"), args.channel_mode)
        else:
            rows = extract_from_npz(args.npz_root, dataset)
        completed[dataset] = len(rows)
        if rows:
            write_csv(args.out_dir / f"{dataset}_descriptors.csv", rows)
            all_rows.extend(rows)
        else:
            missing.append(dataset)

    if all_rows:
        write_csv(args.out_dir / "all_descriptors.csv", all_rows)
    write_readme(args.out_dir / "descriptor_readme.md", args.datasets, completed, args.input_mode, args.data_dir, args.npz_root)
    (args.out_dir / "descriptor_status.json").write_text(
        json.dumps({"completed": completed, "missing": missing, "input_mode": args.input_mode, "data_dir": str(args.data_dir), "npz_root": str(args.npz_root)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote descriptor files to {args.out_dir}")
    if missing:
        print("Missing descriptor rows for: " + ", ".join(missing))
    else:
        print("Next: run python src/train_subject_gate.py to test whether descriptors predict F0-vs-Traditional benefit.")


if __name__ == "__main__":
    main()
