"""PyTorch Dataset 封装模块

将 numpy 格式的 (X, y) 包装为 PyTorch Dataset，
并自动添加 EEGNet 所需的 channel 维度（N, 1, C, T）。
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class EEGDataset(Dataset):
    """
    EEG trial Dataset。

    Parameters
    ----------
    X : np.ndarray, shape (N_trials, N_channels, N_timepoints)
        EEG 特征数组，float32
    y : np.ndarray, shape (N_trials,)
        类别标签，int64，0-indexed

    Notes
    -----
    __getitem__ 返回 (x, y) 其中 x shape 为 (1, N_channels, N_timepoints)，
    即为 EEGNet 所需的 4D 输入格式中的单样本（batch 维度由 DataLoader 添加）。
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        # 确保数据类型正确
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        # 添加 channel 维度：(N, C, T) → (N, 1, C, T)
        if X.ndim == 3:
            X = X[:, np.newaxis, :, :]  # (N, 1, C, T)
        elif X.ndim == 4:
            pass  # 已有 channel 维度，不处理
        else:
            raise ValueError(f"期望 X.ndim == 3 或 4，实际为 {X.ndim}")

        self.X = torch.from_numpy(X)  # (N, 1, C, T), float32
        self.y = torch.from_numpy(y)  # (N,), int64

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class EEGDatasetWithSubject(Dataset):
    """
    EEG trial Dataset，额外携带被试编号，供跨被试 MixUp 使用。

    __getitem__ 返回 (x, y, subject_id)，
    其中 x shape 为 (1, N_channels, N_timepoints)。
    """

    def __init__(
        self,
        X:           np.ndarray,
        y:           np.ndarray,
        subject_ids: np.ndarray,
    ):
        X           = np.asarray(X,           dtype=np.float32)
        y           = np.asarray(y,           dtype=np.int64)
        subject_ids = np.asarray(subject_ids, dtype=np.int32)

        if X.ndim == 3:
            X = X[:, np.newaxis, :, :]
        elif X.ndim == 4:
            pass
        else:
            raise ValueError(f"期望 X.ndim == 3 或 4，实际为 {X.ndim}")

        self.X           = torch.from_numpy(X)
        self.y           = torch.from_numpy(y)
        self.subject_ids = torch.from_numpy(subject_ids)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.subject_ids[idx]
