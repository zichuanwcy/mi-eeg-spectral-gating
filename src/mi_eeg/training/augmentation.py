"""Cross-subject MixUp after Euclidean alignment.

Pairs must come from different subjects. The trainer uses soft mixed labels
and the corresponding cross-entropy calculation in _mixup_criterion.
"""

import numpy as np
import torch


def apply_mixup_to_batch(
    x_batch:       torch.Tensor,
    y_batch:       torch.Tensor,
    subject_ids:   torch.Tensor,
    alpha:         float,
    n_classes:     int,
) -> tuple:
    """
    对一个 batch 执行强制跨被试 MixUp。

    每个样本 i 从 batch 中找一个来自不同被试的样本 j，
    按 lambda ~ Beta(alpha, alpha) 做线性混合。
    若 batch 内某个被试只有自己一个样本（无法找到不同被试的配对），
    则该样本不参与 MixUp（保持原样），这是边界情况下的保守处理。

    Parameters
    ----------
    x_batch     : (B, 1, C, T)  — 当前 batch 的 EEG 信号
    y_batch     : (B,)          — 硬标签（整数）
    subject_ids : (B,)          — 每个样本的被试编号（整数）
    alpha       : float         — Beta 分布参数，控制混合强度
    n_classes   : int           — 分类数，用于 one-hot 编码软标签

    Returns
    -------
    x_mixed   : (B, 1, C, T)   — 混合后的信号
    y_soft    : (B, n_classes)  — 软标签（float32）
    """
    B = x_batch.size(0)
    device = x_batch.device

    # 采样 lambda，shape (B,)
    lam = torch.from_numpy(
        np.random.beta(alpha, alpha, size=B).astype(np.float32)
    ).to(device)

    # 构建 one-hot 软标签基础
    y_onehot = torch.zeros(B, n_classes, device=device, dtype=torch.float32)
    y_onehot.scatter_(1, y_batch.view(-1, 1), 1.0)

    x_mixed = x_batch.clone()
    y_soft  = y_onehot.clone()

    subj_np = subject_ids.cpu().numpy()

    for i in range(B):
        # 找出 batch 内所有来自不同被试的候选索引
        candidates = np.where(subj_np != subj_np[i])[0]
        if len(candidates) == 0:
            # 边界情况：batch 内只有一个被试的样本，不做 MixUp
            continue

        j = int(np.random.choice(candidates))
        l = lam[i].item()

        x_mixed[i] = l * x_batch[i] + (1 - l) * x_batch[j]
        y_soft[i]  = l * y_onehot[i] + (1 - l) * y_onehot[j]

    return x_mixed, y_soft


def build_subject_id_array(
    all_data: dict,
    train_subjs: list,
) -> np.ndarray:
    """
    为 LOSO 训练集构建 subject_id 数组，与 X_train/y_train concat 顺序完全对应。

    Parameters
    ----------
    all_data     : dict {subj_id: {'T': (X, y), 'E': (X, y)}}
    train_subjs  : list — 训练被试列表（按 SUBJECTS 顺序）

    Returns
    -------
    subject_ids : (N_train,) int32
    """
    arrays = []
    for s in train_subjs:
        n_T = len(all_data[s]['T'][1])
        n_E = len(all_data[s]['E'][1])
        arrays.append(np.full(n_T + n_E, s, dtype=np.int32))
    return np.concatenate(arrays)
