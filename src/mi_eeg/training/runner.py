"""Neural-decoder training with the configurations in mi_eeg.config.

Use train.py or `mi-eeg train` to choose experiments and paths.
"""

import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path


from mi_eeg.config import get_configs, list_configs
from mi_eeg.data.bci import load_bci2a_raw, cut_trials
from mi_eeg.preprocessing.filtering import bandpass_filter
from mi_eeg.preprocessing.normalization import exponential_moving_standardize, apply_zscore_per_subject
from mi_eeg.preprocessing.alignment import (
    apply_ea_per_subject, apply_ea_cross_session,
    compute_ea_transform, apply_ea_transform,
)
from mi_eeg.models.eegnet import EEGNet
from mi_eeg.models.eegnet_se import EEGNetSE
from mi_eeg.models.multiscale import MultiScaleEEGNet
from mi_eeg.models.dual_branch import DualBranchEEGNet
from mi_eeg.models.dual_branch_attention import DualBranchAttnEEGNet
from mi_eeg.training.trainer import Trainer
from mi_eeg.evaluation.metrics import compute_metrics
from mi_eeg.training.logging import setup_logger
from mi_eeg.data.physionet import load_physionet_dataset
from mi_eeg.data.openbmi import load_openbmi_dataset

# ============================================================
# Defaults overridden by the command-line entry point
# ============================================================

EXP_IDS  = []    # 要运行的实验 ID 列表，串行执行
                            # Set explicitly through train.py.

DATA_DIR = "data/raw"       # 数据目录
OUT_DIR  = "results"        # 结果保存目录

# ============================================================
# 以下为正式代码，不含可调参数
# ============================================================

SUBJECTS    = list(range(1, 10))
N_CLASSES   = 4
N_CHANNELS  = 22
SFREQ       = 250.0
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
F1          = 8
D           = 2
KERN_LENGTH = 125
DROPOUT     = 0.5


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def load_subject(subj: int, cfg: dict, data_dir: str) -> dict:
    """
    加载单个被试的 T 和 E 两个 Session。

    预处理顺序：
        1. 带通滤波（per-run，在 loader 内部完成）
        2. EMS（连续信号，若启用）
        3. trial 切割

    EA 在所有被试加载完毕后统一执行（见 run_loso）。
    """
    bandpass_freq = cfg['bandpass_freq'] if cfg['bandpass'] else None
    use_ems       = cfg['alignment'] in ('ems', 'ea_ems')

    result = {}
    for session in ['T', 'E']:
        fp = os.path.join(data_dir, f"A0{subj}{session}.mat")
        if not os.path.exists(fp):
            raise FileNotFoundError(
                f"找不到数据文件: {fp}\n"
                f"请先运行: python download_data.py"
            )

        raw, trials, labels, artifacts, _ = load_bci2a_raw(
            fp, N_CHANNELS, bandpass_freq, SFREQ, cfg['filter_order']
        )

        if use_ems:
            raw = exponential_moving_standardize(raw)

        X, y = cut_trials(
            raw, trials, labels, artifacts,
            cfg['t_start'], cfg['t_end'], SFREQ,
            cue_onset_offset=0.0, filter_artifacts=True,
        )
        result[session] = (X, y)

    return result


def run_loso(cfg: dict, data_dir: str, logger) -> dict:
    """Strict LOSO 交叉验证。"""

    logger.info("加载所有被试数据...")
    all_data = {}
    for subj in SUBJECTS:
        all_data[subj] = load_subject(subj, cfg, data_dir)
        nT = len(all_data[subj]['T'][1])
        nE = len(all_data[subj]['E'][1])
        logger.info(f"  S{subj}: T={nT} trials, E={nE} trials")

    use_ea = cfg['alignment'] in ('ea', 'ea_ems')
    if use_ea:
        logger.info("EA: 逐被试独立对齐（T+E合并计算参考矩阵）...")
        all_data = apply_ea_per_subject(all_data)

    use_zscore = cfg['alignment'] == 'zscore'
    if use_zscore:
        logger.info("Z-score: 逐被试独立标准化（T+E合并计算均值/标准差）...")
        all_data = apply_zscore_per_subject(all_data)

    logger.info("开始 LOSO 交叉验证...")
    per_subject_accs     = []
    per_subject_kappas   = []
    per_subject_per_seed = {}
    saved_models         = {}

    for test_subj in SUBJECTS:
        train_subjs = [s for s in SUBJECTS if s != test_subj]

        X_test = np.concatenate([all_data[test_subj]['T'][0],
                                  all_data[test_subj]['E'][0]])
        y_test = np.concatenate([all_data[test_subj]['T'][1],
                                  all_data[test_subj]['E'][1]])
        X_train = np.concatenate([
            np.concatenate([all_data[s]['T'][0], all_data[s]['E'][0]])
            for s in train_subjs
        ])
        y_train = np.concatenate([
            np.concatenate([all_data[s]['T'][1], all_data[s]['E'][1]])
            for s in train_subjs
        ])

        # MixUp：构建 subject_id 数组（与 X_train/y_train 完全对应）
        mixup_alpha = cfg.get('mixup_alpha', 0.0)
        subject_ids_train = None
        if mixup_alpha > 0:
            from mi_eeg.training.augmentation import build_subject_id_array
            subject_ids_train = build_subject_id_array(all_data, train_subjs)

        logger.info(f"\n--- test=S{test_subj} | train={len(y_train)} | "
                    f"test={len(y_test)} ---")

        seed_accs, seed_kappas, seed_probas = [], [], []
        n_timepoints = X_train.shape[2]

        for seed in cfg['seeds']:
            set_seed(seed)
            model = _build_model(cfg, n_timepoints)
            trainer = Trainer(
                model, DEVICE,
                lr=1e-3, weight_decay=1e-4,
                max_epochs=cfg['max_epochs'],
                patience_es=cfg['patience_es'],
                batch_size=64,
                use_max_norm=True,
                verbose=False,
                es_strategy=cfg['es_strategy'],
                mixup_alpha=mixup_alpha,
                n_classes=N_CLASSES,
            )

            if cfg['es_strategy'] == 'test_set':
                trainer.fit(X_train, y_train, X_test=X_test, y_test=y_test,
                            subject_ids_train=subject_ids_train)
            else:
                trainer.fit(X_train, y_train,
                            subject_ids_train=subject_ids_train)

            proba   = trainer.predict_proba(X_test)
            metrics = compute_metrics(y_test, proba.argmax(axis=1), N_CLASSES)
            seed_accs.append(metrics['accuracy'])
            seed_kappas.append(metrics['kappa'])
            seed_probas.append(proba)

            logger.info(f"  Seed {seed}: acc={metrics['accuracy']:.4f} "
                        f"kappa={metrics['kappa']:.4f}")

        saved_models[test_subj] = trainer

        mean_acc = float(np.mean(seed_accs))
        per_subject_accs.append(mean_acc)
        per_subject_kappas.append(float(np.mean(seed_kappas)))
        per_subject_per_seed[f"S{test_subj}"] = seed_accs
        logger.info(f"  S{test_subj}: {mean_acc:.4f} ± {np.std(seed_accs):.4f}")

    return {
        "per_subject_mean":     per_subject_accs,
        "per_subject_kappa":    per_subject_kappas,
        "per_subject_per_seed": per_subject_per_seed,
        "loso_mean":            float(np.mean(per_subject_accs)),
        "loso_std":             float(np.std(per_subject_accs)),
        "mean_kappa":           float(np.mean(per_subject_kappas)),
        "_saved_models":        saved_models,
    }


def run_cross_session(cfg: dict, data_dir: str, logger) -> dict:
    """Cross-Session 验证（被试内，仅作参考上限）。"""

    logger.info("加载所有被试数据...")
    all_data = {}
    for subj in SUBJECTS:
        all_data[subj] = load_subject(subj, cfg, data_dir)

    logger.info("开始 Cross-Session 评估...")
    per_subject_accs     = []
    per_subject_kappas   = []
    per_subject_per_seed = {}

    for subj in SUBJECTS:
        X_T, y_T = all_data[subj]['T']
        X_E, y_E = all_data[subj]['E']

        use_ea = cfg['alignment'] in ('ea', 'ea_ems')
        if use_ea:
            X_T, X_E = apply_ea_cross_session(X_T, X_E)

        logger.info(f"\n--- Subject {subj} | train={len(y_T)} | test={len(y_E)} ---")

        seed_accs, seed_kappas = [], []
        n_timepoints = X_T.shape[2]

        for seed in cfg['seeds']:
            set_seed(seed)
            model = EEGNet(N_CLASSES, N_CHANNELS, n_timepoints,
                           F1, D, F1 * D, KERN_LENGTH, DROPOUT)
            trainer = Trainer(
                model, DEVICE,
                lr=1e-3, weight_decay=1e-4,
                max_epochs=cfg['max_epochs'],
                patience_es=cfg['patience_es'],
                batch_size=64,
                use_max_norm=True,
                verbose=False,
                es_strategy=cfg['es_strategy'],
            )

            if cfg['es_strategy'] == 'test_set':
                trainer.fit(X_T, y_T, X_test=X_E, y_test=y_E)
            else:
                trainer.fit(X_T, y_T)

            proba   = trainer.predict_proba(X_E)
            metrics = compute_metrics(y_E, proba.argmax(axis=1), N_CLASSES)
            seed_accs.append(metrics['accuracy'])
            seed_kappas.append(metrics['kappa'])
            logger.info(f"  Seed {seed}: acc={metrics['accuracy']:.4f}")

        mean_acc = float(np.mean(seed_accs))
        per_subject_accs.append(mean_acc)
        per_subject_kappas.append(float(np.mean(seed_kappas)))
        per_subject_per_seed[f"S{subj}"] = seed_accs
        logger.info(f"  S{subj}: {mean_acc:.4f} ± {np.std(seed_accs):.4f}")

    return {
        "per_subject_mean":     per_subject_accs,
        "per_subject_kappa":    per_subject_kappas,
        "per_subject_per_seed": per_subject_per_seed,
        "loso_mean":            float(np.mean(per_subject_accs)),
        "loso_std":             float(np.std(per_subject_accs)),
        "mean_kappa":           float(np.mean(per_subject_kappas)),
    }


def _prepare_dual_data(
    all_data: dict,
    cfg: dict,
) -> dict:
    """
    为阶段三双路/C类实验准备预处理后的数据。

    处理逻辑：
      - C1：对每个被试用高频滤波后的数据计算 EA 变换矩阵，
            将该矩阵应用于宽带（未滤波）数据。
            返回：{ subj: {'T': (X_wide_ea, y), 'E': (X_wide_ea, y)} }

      - C3：低频路跳过EA，高频路正常EA，两路时域叠加。
            返回：{ subj: {'T': (X_low + X_high_ea, y), ...} }

      - dual_branch / dual_branch_attn：
            低频路不做EA，高频路做EA，分别保存两路数据。
            返回：{ subj: {'T': ((X_low, X_high_ea), y), ...} }

    Parameters
    ----------
    all_data : dict
        原始宽带数据，结构：{ subj_id: {'T': (X, y), 'E': (X, y)} }
    cfg : dict
        实验配置，须含 model_type / low_band / high_band / filter_order

    Returns
    -------
    processed : dict，结构同 all_data（双路模型的 X 为 tuple）
    """
    model_type   = cfg['model_type']
    low_band     = cfg['low_band']
    high_band    = cfg['high_band']
    filter_order = cfg.get('filter_order', 4)

    processed = {}

    for subj_id, sessions in all_data.items():
        subj_result = {}

        for sess in ['T', 'E']:
            X, y = sessions[sess]  # X: (N, C, T)

            # 分别做低频/高频滤波（逐 trial）
            X_low = np.stack([
                bandpass_filter(X[i], low_band[0],  low_band[1],  SFREQ, filter_order)
                for i in range(len(X))
            ])
            X_high = np.stack([
                bandpass_filter(X[i], high_band[0], high_band[1], SFREQ, filter_order)
                for i in range(len(X))
            ])

            subj_result[sess] = (X, X_low, X_high, y)

        processed[subj_id] = subj_result

    # 对高频路做 EA（逐被试，T+E 合并计算参考矩阵）
    for subj_id in processed:
        X_high_T = processed[subj_id]['T'][2]
        X_high_E = processed[subj_id]['E'][2]
        X_high_all = np.concatenate([X_high_T, X_high_E], axis=0)
        R = compute_ea_transform(X_high_all)

        processed[subj_id]['_R_high'] = R   # 保存变换矩阵

    # 按 model_type 组装最终格式
    result = {}
    for subj_id in processed:
        R       = processed[subj_id]['_R_high']
        result[subj_id] = {}

        for sess in ['T', 'E']:
            X_wide, X_low, X_high, y = processed[subj_id][sess]
            X_high_ea = apply_ea_transform(X_high, R)

            if model_type == 'c1':
                # 高频协方差变换矩阵 → 应用于宽带信号
                X_out = apply_ea_transform(X_wide, R)
                result[subj_id][sess] = (X_out, y)

            elif model_type == 'c3':
                # 低频不做EA，高频EA，时域叠加
                X_out = X_low + X_high_ea
                result[subj_id][sess] = (X_out, y)

            else:
                # dual_branch / dual_branch_attn：返回 (X_low, X_high_ea) tuple
                result[subj_id][sess] = ((X_low, X_high_ea), y)

    return result


def _build_model(cfg: dict, n_timepoints: int, n_channels: int = None, n_classes: int = None):
    """根据 model_type 实例化对应模型。"""
    model_type  = cfg.get('model_type', 'eegnet')
    kern_length = cfg.get('kern_length', KERN_LENGTH)
    n_channels  = int(n_channels if n_channels is not None else cfg.get('n_channels', N_CHANNELS))
    n_classes   = int(n_classes if n_classes is not None else cfg.get('n_classes', N_CLASSES))

    if model_type == 'eegnet':
        return EEGNet(n_classes, n_channels, n_timepoints,
                      F1, D, F1 * D, kern_length, DROPOUT)
    elif model_type in ('c1', 'c3'):
        return EEGNet(n_classes, n_channels, n_timepoints,
                      F1, D, F1 * D, kern_length, DROPOUT)
    elif model_type == 'eegnet_se':
        return EEGNetSE(n_classes, n_channels, n_timepoints,
                        F1, D, F1 * D, kern_length, DROPOUT)
    elif model_type == 'dual_branch':
        return DualBranchEEGNet(n_classes, n_channels, n_timepoints,
                                F1, D, kern_length, DROPOUT)
    elif model_type == 'dual_branch_attn':
        return DualBranchAttnEEGNet(n_classes, n_channels, n_timepoints,
                                    F1, D, kern_length, DROPOUT)
    elif model_type == 'multiscale':
        branches = cfg.get('branches')
        if not branches:
            raise ValueError(
                "model_type='multiscale' 须在配置中提供 'branches' 字段，"
                "格式为 [(kern_length, n_filters), ...]"
            )
        return MultiScaleEEGNet(n_classes, n_channels, n_timepoints,
                                branches, D, DROPOUT)
    else:
        raise ValueError(f"未知 model_type: '{model_type}'")

def _load_pretrained_branch(branch: nn.Module, pretrained_exp_id: str,
                            test_subj: int, out_dir: str) -> None:
    """
    将已训练好的单路 EEGNet 权重加载到 DualBranchEEGNet 的某一路上。

    跳过 classifier 层（维度不匹配：单路4分类 vs 双路32维输入），
    其余 backbone 层（temporal_conv/depthwise_conv/separable等）严格加载。
    """
    model_dir = Path(out_dir) / pretrained_exp_id / "models"
    pth_files = sorted(model_dir.glob(f"*S{test_subj}_seed*.pth"))
    if not pth_files:
        raise FileNotFoundError(
            f"找不到预训练模型: {model_dir}/*S{test_subj}_seed*.pth\n"
            f"请确认 {pretrained_exp_id} 已运行完成"
        )

    state = torch.load(str(pth_files[-1]), map_location='cpu')
    state = {k: v for k, v in state.items() if not k.startswith('classifier')}
    branch.load_state_dict(state, strict=False)


def run_loso_phase3(cfg: dict, data_dir: str, logger, out_dir: str) -> dict:
    """
    阶段三 Strict LOSO 交叉验证。

    支持 model_type: c1 | c3 | dual_branch | dual_branch_attn

    与原 run_loso 的主要差异：
      1. 预处理：调用 _prepare_dual_data 做频段分离+差异化对齐
      2. 模型：调用 _build_model 实例化对应架构
      3. 双路模型的 forward 接受 (X_low, X_high) 两个输入
    """
    model_type = cfg['model_type']

    logger.info("加载所有被试数据（宽带，无滤波）...")
    all_data_raw = {}
    for subj in SUBJECTS:
        result = {}
        for session in ['T', 'E']:
            fp = os.path.join(data_dir, f"A0{subj}{session}.mat")
            if not os.path.exists(fp):
                raise FileNotFoundError(f"找不到数据文件: {fp}")
            raw, trials, labels, artifacts, _ = load_bci2a_raw(
                fp, N_CHANNELS, None, SFREQ, cfg['filter_order']
            )
            X, y = cut_trials(raw, trials, labels, artifacts,
                               cfg['t_start'], cfg['t_end'], SFREQ, 0.0, True)
            result[session] = (X, y)
            logger.info(f"  S{subj} {session}: {len(y)} trials")
        all_data_raw[subj] = result

    logger.info(f"预处理（model_type={model_type}）...")
    all_data = _prepare_dual_data(all_data_raw, cfg)

    logger.info("开始 LOSO 交叉验证...")
    per_subject_accs     = []
    per_subject_kappas   = []
    per_subject_per_seed = {}
    saved_models         = {}

    is_dual = model_type in ('dual_branch', 'dual_branch_attn')

    for test_subj in SUBJECTS:
        train_subjs = [s for s in SUBJECTS if s != test_subj]

        # 组装测试集
        if is_dual:
            X_test_low  = np.concatenate([all_data[test_subj]['T'][0][0],
                                           all_data[test_subj]['E'][0][0]])
            X_test_high = np.concatenate([all_data[test_subj]['T'][0][1],
                                           all_data[test_subj]['E'][0][1]])
            y_test = np.concatenate([all_data[test_subj]['T'][1],
                                      all_data[test_subj]['E'][1]])
        else:
            X_test = np.concatenate([all_data[test_subj]['T'][0],
                                      all_data[test_subj]['E'][0]])
            y_test = np.concatenate([all_data[test_subj]['T'][1],
                                      all_data[test_subj]['E'][1]])

        # 组装训练集
        if is_dual:
            X_train_low  = np.concatenate([
                np.concatenate([all_data[s]['T'][0][0], all_data[s]['E'][0][0]])
                for s in train_subjs
            ])
            X_train_high = np.concatenate([
                np.concatenate([all_data[s]['T'][0][1], all_data[s]['E'][0][1]])
                for s in train_subjs
            ])
            y_train = np.concatenate([
                np.concatenate([all_data[s]['T'][1], all_data[s]['E'][1]])
                for s in train_subjs
            ])
            n_timepoints = X_train_low.shape[2]
        else:
            X_train = np.concatenate([
                np.concatenate([all_data[s]['T'][0], all_data[s]['E'][0]])
                for s in train_subjs
            ])
            y_train = np.concatenate([
                np.concatenate([all_data[s]['T'][1], all_data[s]['E'][1]])
                for s in train_subjs
            ])
            n_timepoints = X_train.shape[2]

        logger.info(f"\n--- test=S{test_subj} | train={len(y_train)} | "
                    f"test={len(y_test)} ---")

        seed_accs, seed_kappas, seed_probas = [], [], []

        for seed in cfg['seeds']:
            set_seed(seed)
            model = _build_model(cfg, n_timepoints)

            if cfg.get('freeze_backbone', False):
                _load_pretrained_branch(
                    model.low_branch, cfg['pretrained_low'], test_subj, out_dir
                )
                _load_pretrained_branch(
                    model.high_branch, cfg['pretrained_high'], test_subj, out_dir
                )
                for branch in (model.low_branch, model.high_branch):
                    for p in branch.parameters():
                        p.requires_grad_(False)
                logger.info(f"  Seed {seed}: 已加载预训练权重 "
                            f"({cfg['pretrained_low']}/{cfg['pretrained_high']})，"
                            f"backbone已冻结，仅训练分类头")

            trainer = Trainer(
                model, DEVICE,
                lr=1e-3, weight_decay=1e-4,
                max_epochs=cfg['max_epochs'],
                patience_es=cfg['patience_es'],
                batch_size=64,
                use_max_norm=True,
                verbose=False,
                es_strategy=cfg['es_strategy'],
            )

            if is_dual:
                # 双路模型：Trainer 接受拼接后的 (X_low, X_high) 作为扩展输入
                if cfg['es_strategy'] == 'test_set':
                    trainer.fit_dual(
                        X_train_low, X_train_high, y_train,
                        X_test_low,  X_test_high,  y_test,
                    )
                else:
                    trainer.fit_dual(X_train_low, X_train_high, y_train)

                proba = trainer.predict_proba_dual(X_test_low, X_test_high)
            else:
                if cfg['es_strategy'] == 'test_set':
                    trainer.fit(X_train, y_train, X_test=X_test, y_test=y_test)
                else:
                    trainer.fit(X_train, y_train)
                proba = trainer.predict_proba(X_test)

            metrics = compute_metrics(y_test, proba.argmax(axis=1), N_CLASSES)
            seed_accs.append(metrics['accuracy'])
            seed_kappas.append(metrics['kappa'])
            seed_probas.append(proba)
            logger.info(f"  Seed {seed}: acc={metrics['accuracy']:.4f} "
                        f"kappa={metrics['kappa']:.4f}")

        saved_models[test_subj] = trainer

        mean_acc = float(np.mean(seed_accs))
        per_subject_accs.append(mean_acc)
        per_subject_kappas.append(float(np.mean(seed_kappas)))
        per_subject_per_seed[f"S{test_subj}"] = seed_accs
        logger.info(f"  S{test_subj}: {mean_acc:.4f} ± {np.std(seed_accs):.4f}")

    return {
        "per_subject_mean":     per_subject_accs,
        "per_subject_kappa":    per_subject_kappas,
        "per_subject_per_seed": per_subject_per_seed,
        "loso_mean":            float(np.mean(per_subject_accs)),
        "loso_std":             float(np.std(per_subject_accs)),
        "mean_kappa":           float(np.mean(per_subject_kappas)),
        "_saved_models":        saved_models,
    }


def _stage4_dataset_dir(cfg: dict, out_dir: str) -> Path:
    dataset = cfg.get('dataset', 'bci2a')
    if dataset == 'physionet':
        return Path(out_dir) / 'stage4' / 'physionet'
    if dataset == 'openbmi':
        return Path(out_dir) / 'stage4' / 'openbmi'
    return Path(out_dir)


def _load_stage4_dataset(cfg: dict, data_dir: str, out_dir: str, logger):
    dataset = cfg.get('dataset')
    logger.info(f"加载第四阶段数据集: {dataset}, channel_mode={cfg.get('channel_mode')}")
    if dataset == 'physionet':
        return load_physionet_dataset(cfg, data_dir, out_dir)
    if dataset == 'openbmi':
        return load_openbmi_dataset(cfg, data_dir, out_dir)
    raise ValueError(f"subject_kfold 协议暂只支持 physionet/openbmi，实际 dataset={dataset}")


def _make_subject_folds(subjects, n_folds: int, seed: int):
    subjects = list(subjects)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(subjects).tolist()
    folds = [list(map(int, fold)) for fold in np.array_split(perm, n_folds)]
    return folds


def _concat_subjects(stage4_data: dict, subjects: list):
    X = np.concatenate([stage4_data[s]['all'][0] for s in subjects], axis=0)
    y = np.concatenate([stage4_data[s]['all'][1] for s in subjects], axis=0)
    return X, y


def run_subject_kfold(cfg: dict, data_dir: str, out_dir: str, logger) -> dict:
    """
    第四阶段主协议：3-fold subject-level CV × 5 repeats。

    每个repeat用一个seed生成subject split并训练3个fold。
    每个被试跨repeat聚合为一个均值，作为后续配对统计单位。
    """
    stage4_data, qc = _load_stage4_dataset(cfg, data_dir, out_dir, logger)
    subjects = sorted(stage4_data.keys())
    n_folds = int(cfg.get('n_folds', 3))
    n_repeats = int(cfg.get('n_repeats', len(cfg['seeds'])))
    seeds = list(cfg['seeds'])[:n_repeats]
    n_classes = int(cfg.get('n_classes', N_CLASSES))
    dataset_dir = _stage4_dataset_dir(cfg, out_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"subject_kfold: n_subjects={len(subjects)}, n_folds={n_folds}, repeats={n_repeats}")

    per_subject_scores = {int(s): [] for s in subjects}
    per_subject_kappas = {int(s): [] for s in subjects}
    fold_records = []

    for repeat_idx, seed in enumerate(seeds, 1):
        folds = _make_subject_folds(subjects, n_folds, seed)
        split_path = dataset_dir / f"fold_splits_seed{seed}.json"
        split_payload = {
            "seed": int(seed),
            "n_folds": n_folds,
            "folds": {str(i + 1): folds[i] for i in range(n_folds)},
        }
        with open(split_path, 'w', encoding='utf-8') as f:
            json.dump(split_payload, f, indent=2, ensure_ascii=False)

        for fold_idx, test_subjects in enumerate(folds, 1):
            train_subjects = [s for s in subjects if s not in test_subjects]
            X_train, y_train = _concat_subjects(stage4_data, train_subjects)
            X_test, y_test = _concat_subjects(stage4_data, test_subjects)

            n_channels = int(X_train.shape[1])
            n_timepoints = int(X_train.shape[2])
            set_seed(int(seed))
            model = _build_model(cfg, n_timepoints, n_channels=n_channels, n_classes=n_classes)
            trainer = Trainer(
                model, DEVICE,
                lr=1e-3, weight_decay=1e-4,
                max_epochs=cfg['max_epochs'],
                patience_es=cfg['patience_es'],
                batch_size=64,
                use_max_norm=True,
                verbose=False,
                es_strategy=cfg['es_strategy'],
                n_classes=n_classes,
            )

            logger.info(
                f"repeat={repeat_idx}/{n_repeats}, fold={fold_idx}/{n_folds}, "
                f"seed={seed}, train_subjects={len(train_subjects)}, "
                f"test_subjects={len(test_subjects)}, train_trials={len(y_train)}, test_trials={len(y_test)}"
            )

            if cfg['es_strategy'] == 'test_set':
                trainer.fit(X_train, y_train, X_test=X_test, y_test=y_test)
            else:
                trainer.fit(X_train, y_train)

            proba = trainer.predict_proba(X_test)
            y_pred = proba.argmax(axis=1)
            fold_metrics = compute_metrics(y_test, y_pred, n_classes)

            # Subject-level metrics within this fold.
            offset = 0
            subject_metrics = {}
            for subj in test_subjects:
                X_subj, y_subj = stage4_data[subj]['all']
                n = len(y_subj)
                p_subj = y_pred[offset:offset + n]
                metrics_subj = compute_metrics(y_subj, p_subj, n_classes)
                per_subject_scores[int(subj)].append(float(metrics_subj['accuracy']))
                per_subject_kappas[int(subj)].append(float(metrics_subj['kappa']))
                subject_metrics[str(subj)] = {
                    "accuracy": float(metrics_subj['accuracy']),
                    "kappa": float(metrics_subj['kappa']),
                    "n_trials": int(n),
                }
                offset += n

            fold_records.append({
                "repeat": repeat_idx,
                "seed": int(seed),
                "fold": fold_idx,
                "train_subjects": list(map(int, train_subjects)),
                "test_subjects": list(map(int, test_subjects)),
                "fold_accuracy": float(fold_metrics['accuracy']),
                "fold_kappa": float(fold_metrics['kappa']),
                "subject_metrics": subject_metrics,
            })
            logger.info(f"  fold acc={fold_metrics['accuracy']:.4f}, kappa={fold_metrics['kappa']:.4f}")

    per_subject_mean = {
        f"S{int(s):03d}": float(np.mean(vals))
        for s, vals in per_subject_scores.items()
        if vals
    }
    per_subject_kappa = {
        f"S{int(s):03d}": float(np.mean(vals))
        for s, vals in per_subject_kappas.items()
        if vals
    }

    acc_values = np.array(list(per_subject_mean.values()), dtype=float)
    kappa_values = np.array(list(per_subject_kappa.values()), dtype=float)

    return {
        "protocol": "subject_kfold",
        "dataset": cfg.get('dataset'),
        "channel_mode": cfg.get('channel_mode'),
        "n_folds": n_folds,
        "n_repeats": n_repeats,
        "n_subjects": int(len(per_subject_mean)),
        "included_subjects": [int(s) for s in subjects],
        "fold_records": fold_records,
        "per_subject_mean": per_subject_mean,
        "per_subject_kappa": per_subject_kappa,
        "per_subject_per_repeat": {f"S{int(s):03d}": vals for s, vals in per_subject_scores.items()},
        "loso_mean": float(acc_values.mean()),
        "loso_std": float(acc_values.std()),
        "mean_kappa": float(kappa_values.mean()) if len(kappa_values) else float('nan'),
        "qc_summary": qc,
    }

def save_model_weights(results: dict, out_dir: str, exp_id: str) -> None:
    """将各 fold 最后一个 seed 的模型权重保存到磁盘，供 analyze_filters.py 使用。"""
    saved_models = results.get('_saved_models', {})
    if not saved_models:
        return
    model_dir = Path(out_dir) / exp_id / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for subj, trainer in saved_models.items():
        pth_path = model_dir / f"{exp_id}_S{subj}_seed_last.pth"
        torch.save(trainer.model.state_dict(), str(pth_path))


def run_one(cfg: dict, data_dir: str, out_dir: str) -> None:
    """执行单个实验配置，保存结果到 JSON。"""

    exp_id    = cfg['exp_id']
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path  = _stage4_dataset_dir(cfg, out_dir) if cfg.get('protocol') == 'subject_kfold' else Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log_dir   = str(out_path / "logs")

    logger = setup_logger(f"{exp_id}_{timestamp}", log_dir=log_dir)

    logger.info("=" * 60)
    logger.info(f"实验 {exp_id}: {cfg['description']}")
    logger.info(f"  协议:    {cfg['protocol'].upper()}")
    logger.info(f"  带通:    {'✓ ' + str(cfg['bandpass_freq']) if cfg['bandpass'] else '✗ 宽带'}")
    logger.info(f"  对齐:    {cfg['alignment']}")
    logger.info(f"  模型:    {cfg.get('model_type', 'eegnet')}")
    logger.info(f"  时间窗:  [{cfg['t_start']}, {cfg['t_end']}]s")
    logger.info(f"  早停:    {cfg['es_strategy']}")
    logger.info(f"  种子:    {cfg['seeds']}")
    logger.info(f"  设备:    {DEVICE}")
    logger.info("=" * 60)

    t0 = time.time()

    model_type = cfg.get('model_type', 'eegnet')

    if cfg['protocol'] == 'loso':
        if model_type in ('eegnet', 'multiscale', 'eegnet_se'):
            raw_results = run_loso(cfg, data_dir, logger)
        else:
            raw_results = run_loso_phase3(cfg, data_dir, logger, out_dir)
    elif cfg['protocol'] == 'subject_kfold':
        raw_results = run_subject_kfold(cfg, data_dir, out_dir, logger)
    else:
        raw_results = run_cross_session(cfg, data_dir, logger)

    elapsed = time.time() - t0

    if cfg['protocol'] == 'loso':
        save_model_weights(raw_results, out_dir, exp_id)

    save_results = {k: v for k, v in raw_results.items() if not k.startswith('_')}
    save_results.update({
        "exp_id":       exp_id,
        "config":       cfg,
        "device":       DEVICE,
        "elapsed_sec":  round(elapsed, 1),
        "completed_at": datetime.now().isoformat(),
    })

    out_file = out_path / f"{exp_id}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"实验 {exp_id} 完成")
    subj_means = raw_results['per_subject_mean']
    if isinstance(subj_means, dict):
        for subj, acc in subj_means.items():
            logger.info(f"  {subj}: {acc:.4f}")
    else:
        for i, subj in enumerate(SUBJECTS):
            logger.info(f"  S{subj}: {subj_means[i]:.4f}")
    logger.info(f"均值: {raw_results['loso_mean']:.4f} ± {raw_results['loso_std']:.4f}")
    logger.info(f"Kappa: {raw_results['mean_kappa']:.4f}")
    logger.info(f"耗时: {elapsed:.0f}s")
    logger.info(f"结果保存至: {out_file}")


def main():
    cfgs  = get_configs(EXP_IDS)
    total = len(cfgs)

    print(f"将串行运行 {total} 个实验：{EXP_IDS}")
    print(f"数据目录: {DATA_DIR}")
    print(f"结果目录: {OUT_DIR}\n")

    for i, cfg in enumerate(cfgs, 1):
        print(f"[{i}/{total}] 开始实验 {cfg['exp_id']}: {cfg['description']}")
        try:
            run_one(cfg, DATA_DIR, OUT_DIR)
            print(f"[{i}/{total}] ✅ {cfg['exp_id']} 完成\n")
        except Exception as e:
            print(f"[{i}/{total}] ❌ {cfg['exp_id']} 失败: {e}\n")
            import traceback
            traceback.print_exc()

    print(f"全部实验完成。结果保存在 {OUT_DIR}/")


if __name__ == "__main__":
    main()
