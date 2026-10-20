"""双路并行 EEGNet 模型（实验3-B）

低频路（0.5–13Hz）和高频路（13–40Hz）各自独立的 EEGNet 特征提取器，
分类器前将两路 16 维特征拼接为 32 维，接全连接分类头。

预处理约定（在 mi_eeg.training.runner 中完成，模型本身不感知）：
  - 低频路：带通后不做 EA
  - 高频路：带通后做 EA
  - 两路数据分别以 X_low / X_high 的形式传入

接口设计与单路 EEGNet 保持一致：
  - forward(x_low, x_high) → logits
  - _forward_features(x_low, x_high) → 32 维特征向量
  - apply_max_norm_() → 对两路分别施加 MaxNorm 约束
"""

import torch
import torch.nn as nn
from mi_eeg.models.eegnet import EEGNet, _apply_max_norm_to_layer, _apply_max_norm_to_linear


class DualBranchEEGNet(nn.Module):
    """
    双路并行 EEGNet。

    Parameters
    ----------
    n_classes    : int   — 分类数，默认 4
    n_channels   : int   — EEG 通道数，默认 22
    n_timepoints : int   — 单 trial 时间点数
    F1           : int   — 时间卷积滤波器数，默认 8
    D            : int   — Depthwise 乘子，默认 2
    kern_length  : int   — 时间卷积核长度，默认 125
    dropout      : float — Dropout 比率，默认 0.5
    """

    def __init__(
        self,
        n_classes:    int   = 4,
        n_channels:   int   = 22,
        n_timepoints: int   = 1125,
        F1:           int   = 8,
        D:            int   = 2,
        kern_length:  int   = 125,
        dropout:      float = 0.5,
    ):
        super().__init__()

        F2 = F1 * D

        # 低频路特征提取器（不含分类头）
        self.low_branch  = EEGNet(n_classes, n_channels, n_timepoints,
                                  F1, D, F2, kern_length, dropout)
        # 高频路特征提取器（不含分类头）
        self.high_branch = EEGNet(n_classes, n_channels, n_timepoints,
                                  F1, D, F2, kern_length, dropout)

        # 计算单路特征维度（由 EEGNet._compute_flatten_size 确定）
        single_feat_dim = self.low_branch.classifier.in_features

        # 替换两路各自的分类头为 Identity（特征提取只用 _forward_features）
        # 联合分类头：32维 → n_classes
        self.classifier = nn.Linear(single_feat_dim * 2, n_classes)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def _forward_features(
        self,
        x_low:  torch.Tensor,
        x_high: torch.Tensor,
    ) -> torch.Tensor:
        """
        提取双路拼接特征。

        Parameters
        ----------
        x_low  : (B, 1, C, T) — 低频输入
        x_high : (B, 1, C, T) — 高频输入

        Returns
        -------
        features : (B, 32)
        """
        feat_low  = self.low_branch._forward_features(x_low)    # (B, 16)
        feat_high = self.high_branch._forward_features(x_high)  # (B, 16)
        return torch.cat([feat_low, feat_high], dim=1)           # (B, 32)

    def forward(
        self,
        x_low:  torch.Tensor,
        x_high: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_low  : (B, 1, C, T)
        x_high : (B, 1, C, T)

        Returns
        -------
        logits : (B, n_classes)
        """
        features = self._forward_features(x_low, x_high)
        return self.classifier(features)

    def apply_max_norm_(
        self,
        depthwise_max_norm:  float = 1.0,
        classifier_max_norm: float = 0.25,
    ) -> None:
        """对两路分别施加 MaxNorm 约束，并约束联合分类头。"""
        for branch in [self.low_branch, self.high_branch]:
            _apply_max_norm_to_layer(branch.depthwise_conv, depthwise_max_norm)
        _apply_max_norm_to_linear(self.classifier, classifier_max_norm)
