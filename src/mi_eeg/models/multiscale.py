"""EEGNet with parallel temporal convolutions for the retained architecture control.

Branch outputs are cropped to a common time length before concatenation,
then passed to the shared depthwise and separable convolutions.
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from mi_eeg.models.eegnet import _apply_max_norm_to_layer, _apply_max_norm_to_linear


class MultiScaleEEGNet(nn.Module):
    """
    多尺度时域卷积 EEGNet。

    Parameters
    ----------
    n_classes    : int   — 分类数，默认 4
    n_channels   : int   — EEG 通道数，默认 22
    n_timepoints : int   — 单 trial 时间点数
    branches     : List[Tuple[int, int]]
        每个分支的 (kern_length, n_filters)，例如
        [(25, 1), (125, 5), (250, 2)] 表示三分支，
        滤波器总数 F1 = 所有分支 n_filters 之和。
        分支数量不限于3，例如双分支可传 [(125, 6), (250, 2)]。
    D            : int   — Depthwise 乘子，默认 2（F2 = F1_total * D）
    dropout      : float — Dropout 比率，默认 0.5
    """

    def __init__(
        self,
        n_classes:    int = 4,
        n_channels:   int = 22,
        n_timepoints: int = 1125,
        branches:     List[Tuple[int, int]] = None,
        D:            int = 2,
        dropout:      float = 0.5,
    ):
        super().__init__()

        if not branches:
            raise ValueError("branches不能为空，须传入[(kern_length, n_filters), ...]")

        self.branch_specs = list(branches)
        F1_total = sum(n_filt for _, n_filt in self.branch_specs)
        F2 = F1_total * D

        self.n_channels = n_channels
        self.F1_total   = F1_total
        self.D          = D
        self.F2         = F2

        # ── Block 1a：多尺度并行时域卷积分支 ──────────────────────
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels=1, out_channels=n_filt,
                    kernel_size=(1, kl), padding=(0, kl // 2),
                    bias=False,
                ),
                nn.BatchNorm2d(n_filt, eps=1e-5, momentum=0.1),
            )
            for kl, n_filt in self.branch_specs
        ])

        # ── Block 1b：Depthwise Spatial Convolution（与原EEGNet相同）──
        self.depthwise_conv = nn.Conv2d(
            in_channels=F1_total, out_channels=F1_total * D,
            kernel_size=(n_channels, 1), groups=F1_total, bias=False,
        )
        self.bn2      = nn.BatchNorm2d(F1_total * D, eps=1e-5, momentum=0.1)
        self.elu1     = nn.ELU()
        self.pool1    = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(p=dropout)

        # ── Block 2：Separable Convolution（与原EEGNet相同）──────────
        self.sep_depthwise = nn.Conv2d(
            in_channels=F1_total * D, out_channels=F1_total * D,
            kernel_size=(1, 16), padding=(0, 8),
            groups=F1_total * D, bias=False,
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels=F1_total * D, out_channels=F2,
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

    # ──────────────────────────────────────────────────────────────────
    # 前向传播
    # ──────────────────────────────────────────────────────────────────

    def _forward_branches(self, x: torch.Tensor) -> torch.Tensor:
        """
        并行跑各分支并concat，裁剪到最短分支长度后拼接。

        x : (B, 1, C, T) → (B, F1_total, C, T')
        """
        branch_outs = [branch(x) for branch in self.branches]
        min_t = min(out.shape[-1] for out in branch_outs)
        branch_outs = [out[..., :min_t] for out in branch_outs]
        return torch.cat(branch_outs, dim=1)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征（不含分类头）。x: (B, 1, C, T)"""
        x = self._forward_branches(x)   # (B, F1_total, C, T')

        x = self.depthwise_conv(x)      # (B, F1_total*D, 1, T')
        x = self.bn2(x)
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
        """
        Parameters
        ----------
        x : (B, 1, C, T)

        Returns
        -------
        logits : (B, n_classes)
        """
        features = self._forward_features(x)
        return self.classifier(features)

    # ──────────────────────────────────────────────────────────────────
    # MaxNorm 约束（必须在每个 optimizer step 后调用，与原EEGNet一致）
    # ──────────────────────────────────────────────────────────────────

    def apply_max_norm_(
        self,
        depthwise_max_norm:  float = 1.0,
        classifier_max_norm: float = 0.25,
    ) -> None:
        _apply_max_norm_to_layer(self.depthwise_conv, depthwise_max_norm)
        _apply_max_norm_to_linear(self.classifier, classifier_max_norm)

    # ──────────────────────────────────────────────────────────────────
    # 辅助方法
    # ──────────────────────────────────────────────────────────────────

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
