"""评估指标计算模块

提供：
  - 分类准确率
  - Cohen's Kappa 系数
  - 混淆矩阵
  - 汇总接口 compute_metrics
"""

import numpy as np
from typing import Dict


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = 4,
) -> Dict:
    """
    计算分类评估指标。

    Parameters
    ----------
    y_true : (N,)  真实标签，0-indexed
    y_pred : (N,)  预测标签，0-indexed
    n_classes : int  类别数

    Returns
    -------
    dict with keys:
        accuracy     : float  — 分类准确率
        kappa        : float  — Cohen's Kappa 系数
        confusion    : ndarray (n_classes, n_classes) — 混淆矩阵
        n_samples    : int    — 样本数
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    assert len(y_true) == len(y_pred), "y_true 和 y_pred 长度不一致"

    accuracy  = float((y_true == y_pred).mean())
    kappa_val = cohen_kappa(y_true, y_pred, n_classes)
    conf_mat  = confusion_matrix(y_true, y_pred, n_classes)

    return {
        "accuracy":  accuracy,
        "kappa":     kappa_val,
        "confusion": conf_mat,
        "n_samples": len(y_true),
    }


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算分类准确率。"""
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def cohen_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = 4,
) -> float:
    """
    计算 Cohen's Kappa 系数。

    Kappa = (P_o - P_e) / (1 - P_e)

    其中：
      P_o = 观察到的一致性（即准确率）
      P_e = 随机一致性期望值

    对于平衡 4 类任务，随机猜测 Kappa = 0，完全正确 Kappa = 1。
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    n      = len(y_true)

    if n == 0:
        return 0.0

    conf = confusion_matrix(y_true, y_pred, n_classes)

    # P_o：观察一致性
    P_o = conf.diagonal().sum() / n

    # P_e：随机一致性
    true_counts = conf.sum(axis=1)  # 每类真实数量
    pred_counts = conf.sum(axis=0)  # 每类预测数量
    P_e = (true_counts * pred_counts).sum() / (n ** 2)

    if P_e >= 1.0:
        return 1.0  # 避免除零

    kappa = (P_o - P_e) / (1.0 - P_e)
    return float(kappa)


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int = 4,
) -> np.ndarray:
    """
    计算混淆矩阵。

    Returns
    -------
    conf : ndarray, shape (n_classes, n_classes)
        conf[i, j] = 真实类别为 i，预测为 j 的样本数
    """
    y_true = np.asarray(y_true).flatten().astype(int)
    y_pred = np.asarray(y_pred).flatten().astype(int)

    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            conf[t, p] += 1

    return conf
