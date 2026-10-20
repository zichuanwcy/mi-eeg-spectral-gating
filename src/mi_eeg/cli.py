"""Argument parsing and task dispatch; scientific backends are imported on use."""
from __future__ import annotations

import argparse
import importlib
import shlex
import sys
from pathlib import Path

from mi_eeg.config import get_configs, list_configs, partial_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mi-eeg", description="Subject-level spectral preprocessing selection for MI EEG.")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list-configs", help="List the 25 retained decoder configurations.")
    listing.add_argument("--config-dir", type=Path, help="Directory containing bci2a.json, openbmi.json and physionet.json.")
    train = commands.add_parser("train", help="Train explicitly selected neural decoders.")
    train.add_argument("--experiments", nargs="+", required=True)
    train.add_argument("--config-dir", type=Path, help="Read decoder settings from this directory instead of the bundled configs/.")
    train.add_argument("--data-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    extract = commands.add_parser("extract-features", help="Extract wideband, pre-EA subject descriptors.")
    extract.add_argument("--datasets", nargs="+", default=["physionet", "openbmi"], choices=["openbmi", "physionet"])
    extract.add_argument("--data-dir", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    evaluate = commands.add_parser("evaluate", help="Run repeated subject-level gate evaluation.")
    evaluate.add_argument("--analysis", choices=["gates"], default="gates")
    evaluate.add_argument("--results-dir", type=Path, required=True, help="Parent of stage4/ and stage5/ decoder/outcome inputs.")
    evaluate.add_argument("--gate-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    summarize = commands.add_parser("summarize", help="Summarize existing per-repetition gate outputs.")
    summarize.add_argument("--analysis", choices=["gates"], default="gates")
    summarize.add_argument("--input-dir", type=Path, required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)
    prepare = commands.add_parser("prepare-inputs", help="Join outcome and descriptor CSVs; no model fitting.")
    prepare.add_argument("--delta-csv", type=Path, required=True)
    prepare.add_argument("--descriptor-csv", type=Path, nargs="+", required=True)
    prepare.add_argument("--datasets", nargs="+", default=["openbmi", "physionet"], choices=["openbmi", "physionet"])
    prepare.add_argument("--output-dir", type=Path, required=True)
    plot = commands.add_parser("plot", help="Render historical quantitative panels from matching frozen CSVs.")
    plot.add_argument("--source-dir", type=Path, required=True)
    plot.add_argument("--output-dir", type=Path, required=True)
    plot.add_argument("--qa-dir", type=Path, required=True)
    for command in [train, extract, evaluate, summarize, prepare, plot]:
        command.add_argument("--dry-run", action="store_true", help="Print settings/paths without importing or running a scientific backend.")
    return parser


def backend_command(args: argparse.Namespace) -> tuple[str, list[str]]:
    """Translate public path arguments into the original backend's options."""
    if args.command == "extract-features":
        return "mi_eeg.preprocessing.extract_features", ["--datasets", *args.datasets, "--data-dir", str(args.data_dir), "--out-dir", str(args.output_dir), "--channel-mode", "matched"]
    if args.command == "evaluate":
        module = "mi_eeg.gating.cross_validation"
        return module, ["--results-dir", str(args.results_dir), "--gate-dir", str(args.gate_dir), "--out-root", str(args.output_dir)]
    if args.command == "summarize":
        module = "mi_eeg.evaluation.gate_repetitions"
        return module, ["--repeat-root", str(args.input_dir), "--out-dir", str(args.output_dir)]
    raise ValueError(f"No argparse backend for {args.command}")


def invoke(module_name: str, arguments: list[str]) -> None:
    module = importlib.import_module(module_name)
    # Restore argv even if a backend raises, allowing reuse from Python.
    previous = sys.argv
    try:
        sys.argv = [module_name, *arguments]
        module.main()
    finally:
        sys.argv = previous


def print_dry_run(args: argparse.Namespace) -> None:
    print(f"Dry run: {args.command}")
    if args.command == "train":
        for cfg in get_configs(args.experiments, args.config_dir):
            print(f"  {cfg['exp_id']}: dataset={cfg.get('dataset', 'bci2a')}, protocol={cfg['protocol']}, additional_bandpass={cfg['bandpass']}, alignment={cfg['alignment']}, seeds={cfg['seeds']}")
        print(f"  data: {args.data_dir}\n  output: {args.output_dir}")
    elif args.command in {"extract-features", "evaluate", "summarize"}:
        module, arguments = backend_command(args)
        print("  " + shlex.join([sys.executable, "-m", module, *arguments]))
        if args.command == "evaluate":
            seeds = list(range(20260710, 20260715))
            print(f"  stratified gate folds: 5; repetition seeds: {seeds}")
    else:
        for key, value in vars(args).items():
            if key not in {"command", "dry_run"}:
                print(f"  {key}: {value}")


def require_new_output(path: Path) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Use a new or empty output directory: {path}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {"list-configs", "train"}:
        try:
            if args.command == "list-configs":
                list_configs(args.config_dir)
                return
            get_configs(args.experiments, args.config_dir)
            partial = partial_results(args.config_dir)
            for eid in args.experiments:
                if eid in partial and not partial[eid].get("allow_training_via_cli", False) and not args.dry_run:
                    parser.error(f"{eid} is a six-matched-fold historical result (seeds 42/43); this entry does not extend it to five repetitions.")
        except (ValueError, KeyError, FileNotFoundError) as exc:
            parser.error(str(exc))
    if args.dry_run:
        print_dry_run(args)
        return
    try:
        require_new_output(args.output_dir)
        if args.command == "train":
            runner = importlib.import_module("mi_eeg.training.runner")
            # Call the original computation directly so a failure propagates to the CLI.
            for cfg in get_configs(args.experiments, args.config_dir):
                runner.run_one(cfg, str(args.data_dir), str(args.output_dir))
        elif args.command in {"extract-features", "evaluate", "summarize"}:
            module, arguments = backend_command(args)
            invoke(module, arguments)
        elif args.command == "prepare-inputs":
            from mi_eeg.data.gate_inputs import prepare_inputs
            counts = prepare_inputs(args.delta_csv, args.descriptor_csv, args.datasets, args.output_dir)
            print("Joined subjects:", counts)
        elif args.command == "plot":
            require_new_output(args.qa_dir)
            module = importlib.import_module("mi_eeg.evaluation.plotting")
            module.SOURCE, module.OUT, module.QA = args.source_dir, args.output_dir, args.qa_dir
            module.main()
    except (ValueError, FileNotFoundError, ModuleNotFoundError) as exc:
        parser.exit(1, f"{exc}\n")
