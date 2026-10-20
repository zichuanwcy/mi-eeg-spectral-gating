"""Check real configuration loading and CLI boundaries without EEG fitting."""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mi_eeg import cli
from mi_eeg.config import default_config_dir, get_config, get_experiment_ids, load_configs
from mi_eeg.data.gate_inputs import merge_delta_descriptors, prepare_inputs


class InterfaceTests(unittest.TestCase):
    def copy_configs(self, tmp):
        return Path(shutil.copytree(ROOT / "configs", Path(tmp) / "configs"))

    def test_scope_matches_real_configuration_files(self):
        configs = load_configs()
        self.assertEqual(len(configs), 25)
        self.assertEqual(sum(cfg.get("dataset", "bci2a") == "bci2a" for cfg in configs.values()), 19)
        self.assertFalse(any("MixUp" in eid or "P3" in eid for eid in configs))
        self.assertEqual(default_config_dir(), ROOT / "configs")
        for dataset in ["OpenBMI", "Physionet"]:
            self.assertTrue(get_config(dataset + "_Traditional")["bandpass"])
            self.assertFalse(get_config(dataset + "_F0")["bandpass"])

    def test_json_edits_take_effect_without_python_parameter_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = self.copy_configs(tmp)
            path = folder / "bci2a.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            eid = document["experiments"][0]["exp_id"]
            original = get_config(eid, folder)["max_epochs"]
            document["experiments"][0]["max_epochs"] = original + 1
            path.write_text(json.dumps(document), encoding="utf-8")
            self.assertEqual(get_config(eid, folder)["max_epochs"], original + 1)

    def test_original_array_types_and_independent_copies(self):
        cfg = get_config("MS_2branch_ea")
        self.assertIsInstance(cfg["bandpass_freq"], tuple)
        self.assertEqual(cfg["branches"], [(125, 6), (250, 2)])
        cfg["seeds"].append(99)
        self.assertNotIn(99, get_config("MS_2branch_ea")["seeds"])

    def test_duplicate_configuration_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = self.copy_configs(tmp)
            path = folder / "bci2a.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            document["experiments"].append(document["experiments"][0])
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_configs(folder)

    def test_invalid_json_identifies_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = self.copy_configs(tmp)
            (folder / "openbmi.json").write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "openbmi.json"):
                load_configs(folder)

    def test_train_dispatch_uses_explicit_json_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = self.copy_configs(tmp)
            path = folder / "bci2a.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            eid = document["experiments"][0]["exp_id"]
            document["experiments"][0]["max_epochs"] = 321
            path.write_text(json.dumps(document), encoding="utf-8")
            runner = SimpleNamespace(run_one=Mock())
            output = Path(tmp) / "absent"
            with patch.object(cli.importlib, "import_module", return_value=runner):
                cli.main(["train", "--experiments", eid, "--config-dir", str(folder), "--data-dir", "input", "--output-dir", str(output)])
            runner.run_one.assert_called_once()
            self.assertEqual(runner.run_one.call_args.args[0]["max_epochs"], 321)
            self.assertFalse(output.exists())

    def test_all_dry_runs_leave_no_outputs_or_scientific_imports(self):
        cases = [
            ["train", "--experiments", "OpenBMI_Traditional", "OpenBMI_F0", "OpenBMI_SE", "--data-dir", "missing"],
            ["extract-features", "--data-dir", "missing"],
            ["evaluate", "--results-dir", "missing", "--gate-dir", "missing"],
            ["summarize", "--input-dir", "missing"],
            ["prepare-inputs", "--delta-csv", "missing.csv", "--descriptor-csv", "missing.csv"],
            ["plot", "--source-dir", "missing", "--qa-dir", "missing"],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "absent"
            env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
            for arguments in cases:
                code = "from mi_eeg.cli import main; import sys; main(" + repr(arguments + ["--output-dir", str(output), "--dry-run"]) + "); assert not any(m in sys.modules for m in ['torch','numpy','scipy','sklearn','pandas','matplotlib'])"
                result = subprocess.run([sys.executable, "-B", "-X", "utf8", "-c", code], env=env, cwd=tmp, capture_output=True, text=True, encoding="utf-8")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("Dry run:", result.stdout)
                self.assertFalse(output.exists())

    def test_root_entries_work_outside_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ["train.py", "extract_features.py", "evaluate.py"]:
                result = subprocess.run([sys.executable, "-B", "-X", "utf8", str(ROOT / name), "--help"], cwd=tmp, capture_output=True, text=True, encoding="utf-8")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--dry-run", result.stdout)

    def test_partial_openbmi_se_cannot_start_backend(self):
        with patch.object(cli.importlib, "import_module") as imports, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                cli.main(["train", "--experiments", "OpenBMI_SE", "--data-dir", "missing", "--output-dir", "unused"])
            self.assertEqual(caught.exception.code, 2)
            imports.assert_not_called()

    def test_unreported_experiments_are_not_public_cli_options(self):
        with patch.object(cli.importlib, "import_module") as imports, contextlib.redirect_stderr(io.StringIO()):
            for arguments in [["train", "--experiments", "MixUp", "--data-dir", "missing", "--output-dir", "unused"], ["evaluate", "--analysis", "confidence", "--results-dir", "missing", "--gate-dir", "missing", "--output-dir", "unused"]]:
                with self.assertRaises(SystemExit) as caught:
                    cli.main(arguments)
                self.assertEqual(caught.exception.code, 2)
            imports.assert_not_called()

    def test_gate_evaluation_forwards_paths_and_original_seeds(self):
        args = cli.build_parser().parse_args(["evaluate", "--results-dir", "results dir", "--gate-dir", "input dir", "--output-dir", "output dir"])
        self.assertEqual(cli.backend_command(args), ("mi_eeg.gating.cross_validation", ["--results-dir", "results dir", "--gate-dir", "input dir", "--out-root", "output dir"]))
        from mi_eeg.gating import cross_validation
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(cross_validation, "run") as run, patch.object(sys, "argv", ["test", "--out-root", tmp]), contextlib.redirect_stdout(io.StringIO()):
                cross_validation.main()
            self.assertEqual(run.call_count, 15)
            commands = [call.args[0] for call in run.call_args_list if "mi_eeg.gating.experiments" in call.args[0]]
            self.assertEqual(len(commands), 5)
            for index, command in enumerate(commands):
                self.assertEqual(command[:4], [sys.executable, "-u", "-m", "mi_eeg.gating.experiments"])
                self.assertEqual(str(command[command.index("--seed") + 1]), str(20260710 + index))
                self.assertEqual(str(command[command.index("--n-splits") + 1]), "5")

    def test_failed_backend_restores_argv_and_propagates(self):
        class Backend:
            @staticmethod
            def main():
                raise RuntimeError("failure")
        before = sys.argv
        with patch.object(cli.importlib, "import_module", return_value=Backend):
            with self.assertRaisesRegex(RuntimeError, "failure"):
                cli.invoke("example.backend", ["--x", "value"])
        self.assertIs(sys.argv, before)

    def test_merge_uses_dataset_subject_and_retains_target(self):
        delta = [{"dataset": "openbmi", "subject": "S001", "delta_f0_minus_traditional": "0.02"}, {"dataset": "physionet", "subject": "S001", "delta_f0_minus_traditional": "0.03"}]
        desc = [{"dataset": "openbmi", "subject": "S001", "alpha": "1.5"}]
        self.assertEqual(merge_delta_descriptors(delta, desc, "openbmi"), [{**delta[0], "desc_alpha": "1.5"}])
        self.assertEqual(merge_delta_descriptors(delta, desc, "physionet"), [])

    def test_duplicate_input_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            delta = root / "delta.csv"
            desc = root / "desc.csv"
            delta.write_text("dataset,subject,delta_f0_minus_traditional\nopenbmi,S001,0.02\nopenbmi,S001,0.03\n", encoding="utf-8")
            desc.write_text("dataset,subject,alpha\nopenbmi,S001,1.5\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                prepare_inputs(delta, [desc], ["openbmi"], root / "out")
            self.assertFalse((root / "out").exists())


if __name__ == "__main__":
    unittest.main()
