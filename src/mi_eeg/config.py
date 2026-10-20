"""Read the experiment JSON files in configs/; no numerical settings live here."""
from __future__ import annotations

import copy
import json
from importlib import metadata
from pathlib import Path
from typing import Iterable

CONFIG_FILES = ("bci2a.json", "openbmi.json", "physionet.json")
INSTALLED_CONFIG_PATH = "share/mi-eeg-spectral-gating/configs/"
TUPLE_FIELDS = ("bandpass_freq", "low_band", "high_band")


def default_config_dir() -> Path:
    """Locate the source checkout or the wheel-installed configuration files."""
    checkout = Path(__file__).resolve().parents[2] / "configs"
    if all((checkout / name).is_file() for name in CONFIG_FILES):
        return checkout
    try:
        distribution = metadata.distribution("mi-eeg-spectral-gating")
        for entry in distribution.files or ():
            if entry.as_posix().endswith(INSTALLED_CONFIG_PATH + CONFIG_FILES[0]):
                folder = Path(distribution.locate_file(entry)).parent
                if all((folder / name).is_file() for name in CONFIG_FILES):
                    return folder
    except metadata.PackageNotFoundError:
        pass
    raise FileNotFoundError("Experiment JSON files were not found. Use a complete checkout or provide --config-dir.")


def read_documents(config_dir: Path | None = None) -> list[dict]:
    folder = Path(config_dir) if config_dir is not None else default_config_dir()
    documents = []
    for filename in CONFIG_FILES:
        path = folder / filename
        try:
            with path.open(encoding="utf-8") as handle:
                document = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc.msg}") from exc
        if not isinstance(document, dict) or document.get("dataset") != Path(filename).stem or not isinstance(document.get("experiments"), list):
            raise ValueError(f"{path}: expected dataset and experiments fields.")
        documents.append(document)
    return documents


def load_configs(config_dir: Path | None = None) -> dict[str, dict]:
    """Read and validate files on each call so edits take effect immediately."""
    configs = {}
    for document in read_documents(config_dir):
        for raw in document["experiments"]:
            cfg = copy.deepcopy(raw)
            required = {"exp_id", "protocol", "bandpass", "alignment", "seeds", "max_epochs", "patience_es"}
            if not isinstance(cfg, dict) or not required <= cfg.keys():
                raise ValueError("Each experiment needs an ID, protocol and complete preprocessing/training settings.")
            eid = cfg["exp_id"]
            if not isinstance(eid, str) or not eid or eid in configs:
                raise ValueError(f"Duplicate or invalid experiment ID: {eid!r}")
            if cfg.get("dataset", "bci2a") != document["dataset"]:
                raise ValueError(f"{eid}: experiment dataset does not match its JSON file.")
            if type(cfg["bandpass"]) is not bool or not isinstance(cfg["seeds"], list) or not cfg["seeds"] or any(type(seed) is not int for seed in cfg["seeds"]):
                raise ValueError(f"{eid}: bandpass must be Boolean and seeds must be a nonempty integer list.")
            if cfg["bandpass"] and "bandpass_freq" not in cfg:
                raise ValueError(f"{eid}: bandpass_freq is required when the additional filter is enabled.")
            for field in TUPLE_FIELDS:
                if field in cfg:
                    value = cfg[field]
                    if not isinstance(value, list) or len(value) != 2 or any(type(v) not in (int, float) for v in value) or value[0] >= value[1]:
                        raise ValueError(f"{eid}: {field} must contain increasing numeric bounds.")
                    cfg[field] = tuple(value)
            if "branches" in cfg:
                if not isinstance(cfg["branches"], list) or any(not isinstance(branch, list) or len(branch) != 2 for branch in cfg["branches"]):
                    raise ValueError(f"{eid}: branches must contain kernel/filter pairs.")
                cfg["branches"] = [tuple(branch) for branch in cfg["branches"]]
            configs[eid] = cfg
    return configs


def get_experiment_ids(config_dir: Path | None = None) -> tuple[str, ...]:
    return tuple(load_configs(config_dir))


def get_config(exp_id: str, config_dir: Path | None = None) -> dict:
    return get_configs([exp_id], config_dir)[0]


def get_configs(exp_ids: Iterable[str], config_dir: Path | None = None) -> list[dict]:
    available = load_configs(config_dir)
    selected = []
    for eid in exp_ids:
        if eid not in available:
            raise KeyError(f"Unknown experiment {eid!r}. Available IDs: {', '.join(available)}")
        selected.append(copy.deepcopy(available[eid]))
    return selected


def partial_results(config_dir: Path | None = None) -> dict[str, dict]:
    result = {}
    for document in read_documents(config_dir):
        for item in document.get("partial_results", []):
            result[item["exp_id"]] = copy.deepcopy(item)
    return result


def list_configs(config_dir: Path | None = None) -> None:
    print(f"{'ID':<24} {'Dataset':<12} {'Protocol':<16} Additional band-pass")
    for cfg in load_configs(config_dir).values():
        band = cfg["bandpass_freq"] if cfg["bandpass"] else "omitted"
        print(f"{cfg['exp_id']:<24} {cfg.get('dataset', 'bci2a'):<12} {cfg['protocol']:<16} {band}")
