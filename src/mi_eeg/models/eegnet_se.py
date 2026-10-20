"""EEGNet with a squeeze-and-excitation control block.

The block follows depthwise spatial convolution and batch normalization,
before ELU. The training interface matches EEGNet.
"""

import torch
import torch.nn as nn

from mi_eeg.models.eegnet import _apply_max_norm_to_layer, _apply_max_norm_to_linear


class SEBlock(nn.Module):
    """
    轻量级 Squeeze-and-Excitation 通道注意力模块。

    流程：
        输入 (B, C, 1, T) → Global Average Pooling → (B, C, 1, 1)
        → FC(C → C//r) → ReLU → FC(C//r → C) → Sigmoid → Scale

    Parameters
    ----------
    n_channels : int   — 输入通道数（即 F2 = F1 * D）
    reduction  : int   — 压缩比，默认 4
    """

    def __init__(self, n_channels: int, reduction: int = 4):
        super().__init__()
        mid = max(n_channels // reduction, 1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, n_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, 1, T)

        Returns
        -------
        (B, C, 1, T)  — 各通道乘以对应的注意力权重
        """
        scale = self.fc(self.gap(x))          # (B, C)
        return x * scale.view(scale.shape[0], -1, 1, 1)


class EEGNetSE(nn.Module):
    """
    EEGNet + SE Block。

    与原 EEGNet 的唯一结构差异：在 Depthwise Conv → BN2 之后、ELU1 之前
    插入 SEBlock，让模型对 Depthwise 输出的 F2 个空间通道做软加权。

    Parameters
    ----------
    n_classes    : int   — 分类数，默认 4
    n_channels   : int   — EEG 通道数，默认 22
    n_timepoints : int   — 单 trial 时间点数
    F1           : int   — 时间卷积滤波器数，默认 8
    D            : int   — Depthwise 乘子，默认 2，F2 = F1 * D
    kern_length  : int   — 时间卷积核长度，默认 125
    dropout      : float — Dropout 比率，默认 0.5
    se_reduction : int   — SE Block 压缩比，默认 4
    """

    def __init__(
        self,
        n_classes:    int   = 4,
        n_channels:   int   = 22,
        n_timepoints: int   = 1125,
        F1:           int   = 8,
        D:            int   = 2,
        F2:           int   = None,
        kern_length:  int   = 125,
        dropout:      float = 0.5,
        se_reduction: int   = 4,
    ):
        super().__init__()

        if F2 is None:
            F2 = F1 * D

        self.F1          = F1
        self.D           = D
        self.F2          = F2
        self.kern_length = kern_length

        # ── Block 1: Temporal Convolution（与原 EEGNet 完全相同）───
        self.temporal_conv = nn.Conv2d(
            in_channels=1, out_channels=F1,
            kernel_size=(1, kern_length), padding=(0, kern_length // 2),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1, eps=1e-5, momentum=0.1)

        # ── Block 1: Depthwise Spatial Convolution ──────────────────
        self.depthwise_conv = nn.Conv2d(
            in_channels=F1, out_channels=F1 * D,
            kernel_size=(n_channels, 1), groups=F1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(F1 * D, eps=1e-5, momentum=0.1)

        # ── SE Block（插在 BN2 之后、ELU1 之前）─────────────────────
        self.se = SEBlock(F1 * D, reduction=se_reduction)

        self.elu1     = nn.ELU()
        self.pool1    = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(p=dropout)

        # ── Block 2: Separable Convolution（与原 EEGNet 完全相同）───
        self.sep_depthwise = nn.Conv2d(
            in_channels=F1 * D, out_channels=F1 * D,
            kernel_size=(1, 16), padding=(0, 8),
            groups=F1 * D, bias=False,
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels=F1 * D, out_channels=F2,
            kernel_size=(1, 1), bias=False,
        )
        self.bn3      = nn.BatchNorm2d(F2, eps=1e-5, momentum=0.1)
        self.elu2     = nn.ELU()
        self.pool2    = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(p=dropout)

        # ── Classifier ────────────────────────────────────────────────
        flatten_size = self._compute_flatten_size(n_channels, n_timepoints)
        self.classifier = nn.Linear(flatten_size, n_classes)

        self._initialize_weights()

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征（不含分类头）。x: (B, 1, C, T)"""
        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.se(x)           # SE Block 插入点
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.sep_depthwise(x)
        x = self.pointwise_conv(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        return x.flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._forward_features(x))

    def apply_max_norm_(
        self,
        depthwise_max_norm:  float = 1.0,
        classifier_max_norm: float = 0.25,
    ) -> None:
        _apply_max_norm_to_layer(self.depthwise_conv, depthwise_max_norm)
        _apply_max_norm_to_linear(self.classifier, classifier_max_norm)

    def _compute_flatten_size(self, n_channels: int, n_timepoints: int) -> int:
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            out   = self._forward_features(dummy)
        return out.shape[1]

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
