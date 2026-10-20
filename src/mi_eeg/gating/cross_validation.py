"""Repeat gate candidate evaluation with the five recorded random seeds.

Each seed evaluates baselines, nested thresholds and risk-aware gates
from existing outcomes and descriptors, without decoder training.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_SEEDS = [20260710, 20260711, 20260712, 20260713, 20260714]


def run(cmd: Sequence[str]) -> None:
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(list(cmd), check=True)


def parse_seeds(values: Iterable[str]) -> list[int]:
    result: list[int] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                result.append(int(item))
    return list(dict.fromkeys(result))


def completed(seed_dir: Path) -> bool:
    required = [
        seed_dir / "final" / "direction1_best_by_dataset.csv",
        seed_dir / "subject_analysis" / "subject_decision_matrix.csv",
        seed_dir / "subject_analysis" / "subject_decision_rows.csv",
    ]
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--gate-dir", type=Path, default=Path("results/stage5/gate_correlation"))
    parser.add_argument("--out-root", type=Path, default=Path("results/stage5/multiseed_correlation"))
    parser.add_argument("--datasets", nargs="+", default=["openbmi", "physionet"])
    parser.add_argument("--seeds", nargs="+", default=[str(seed) for seed in DEFAULT_SEEDS])
    parser.add_argument("--feature-sets", nargs="+", default=["selected", "selected_plus_correlation"])
    parser.add_argument("--models", nargs="+", default=["logreg", "gb"])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--force", action="store_true", help="Rerun a seed even when its final artifacts already exist.")
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    if not seeds:
        raise SystemExit("At least one seed is required.")
    args.out_root.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    for seed in seeds:
        seed_dir = args.out_root / f"seed_{seed}"
        if completed(seed_dir) and not args.force:
            print(f"Skipping seed={seed}: completed artifacts already exist under {seed_dir}", flush=True)
            continue
        seed_dir.mkdir(parents=True, exist_ok=True)
        run([
            py, "-u", "-m", "mi_eeg.gating.experiments",
            "--results-dir", str(args.results_dir),
            "--gate-dir", str(args.gate_dir),
            "--out-dir", str(seed_dir),
            "--datasets", *args.datasets,
            "--experiments", "baselines", "thresholds", "risk",
            "--feature-sets", *args.feature_sets,
            "--models", *args.models,
            "--n-splits", str(args.n_splits),
            "--seed", str(seed),
        ])
        run([
            py, "-m", "mi_eeg.evaluation.gate_candidates",
            "--direction1-dir", str(seed_dir),
            "--stage5-dir", str(args.results_dir / "stage5"),
            "--out-dir", str(seed_dir / "final"),
        ])
        run([
            py, "-m", "mi_eeg.evaluation.gate_subjects",
            "--direction1-dir", str(seed_dir),
            "--stage5-dir", str(args.results_dir / "stage5"),
            "--formal-comparison", str(seed_dir / "final" / "direction1_formal_comparison.csv"),
            "--out-dir", str(seed_dir / "subject_analysis"),
        ])
    print(f"Completed multi-seed repetitions under {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
