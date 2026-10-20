"""EEGNet 模型实现

参考：Lawhern et al., "EEGNet: A Compact Convolutional Neural Network
      for EEG-based Brain-Computer Interfaces", J. Neural Eng., 2018.

关键实现细节（影响复现性的高风险项）：
  - kernLength = 125（采样率 // 2 = 250 // 2），不可使用 64 或 32
  - Depthwise Conv MaxNorm = 1（每个 optimizer step 后手动调用 apply_max_norm_()）
  - Pointwise Conv MaxNorm = 0.25（同上）
  - 不在 EEGNet 外部额外添加带通滤波（EEGNet 时间卷积自学习频率）
  - 归一化使用 exponential moving standardize，而非 z-score
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    EEGNet 紧凑型 EEG 分类网络。

    Parameters
    ----------
    n_classes : int
        分类数，默认 4（BCI IV-2a 四类运动想象）
    n_channels : int
        EEG 通道数，默认 22
    n_timepoints : int
        单 trial 时间点数，默认 1125（4.5s @ 250Hz）
    F1 : int
        时间卷积滤波器数，默认 8
    D : int
        Depthwise 乘子，默认 2
    F2 : int
        Separable Conv 输出滤波器数，默认 F1 * D = 16
    kern_length : int
        时间卷积核长度，默认 125（= 采样率 // 2）
    dropout : float
        Dropout 比率，默认 0.5

    Architecture
    ------------
    Input: (B, 1, C, T)
      ↓ Block 1: Temporal Conv (1×kernLength) + BN
      ↓          Depthwise Spatial Conv (C×1) + BN + ELU + AvgPool(1×4) + Dropout
      ↓ Block 2: Separable Conv (Depthwise 1×16 + Pointwise 1×1) + BN + ELU
      ↓          AvgPool(1×8) + Dropout
      ↓ Flatten → Dense(n_classes)
    """

    def __init__(
        self,
        n_classes: int = 4,
        n_channels: int = 22,
        n_timepoints: int = 1125,
        F1: int = 8,
        D: int = 2,
        F2: int = None,
        kern_length: int = 125,
        dropout: float = 0.5,
    ):
        super().__init__()

        if F2 is None:
            F2 = F1 * D

        self.n_classes    = n_classes
        self.n_channels   = n_channels
        self.n_timepoints = n_timepoints
        self.F1           = F1
        self.D            = D
        self.F2           = F2
        self.kern_length  = kern_length

        # ── Block 1: Temporal Convolution ──────────────────────────────
        # Conv2d: (1, F1, kernel=(1, kernLength), padding='same')
        # 使用 padding=(0, kern_length//2) 实现 same padding（奇数核长）
        self.temporal_conv = nn.Conv2d(
            in_channels  = 1,
            out_channels = F1,
            kernel_size  = (1, kern_length),
            padding      = (0, kern_length // 2),
            bias         = False,
        )
        self.bn1 = nn.BatchNorm2d(F1, eps=1e-5, momentum=0.1)

        # ── Block 1: Depthwise Spatial Convolution ─────────────────────
        # groups=F1 实现 depthwise（每个输入滤波器有独立的空间滤波器）
        # MaxNorm = 1 须在每个 optimizer step 后调用 apply_max_norm_()
        self.depthwise_conv = nn.Conv2d(
            in_channels  = F1,
            out_channels = F1 * D,
            kernel_size  = (n_channels, 1),
            groups       = F1,
            bias         = False,
        )
        self.bn2      = nn.BatchNorm2d(F1 * D, eps=1e-5, momentum=0.1)
        self.elu1     = nn.ELU()
        self.pool1    = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout1 = nn.Dropout(p=dropout)

        # ── Block 2: Separable Convolution ─────────────────────────────
        # Separable = Depthwise (1×16) + Pointwise (1×1)
        # Depthwise 部分：groups = F1*D，无 MaxNorm 约束
        self.sep_depthwise = nn.Conv2d(
            in_channels  = F1 * D,
            out_channels = F1 * D,
            kernel_size  = (1, 16),
            padding      = (0, 8),   # 与 Keras 'same' 近似，动态 flatten 会处理尺寸差异
            groups       = F1 * D,
            bias         = False,
        )
        # Pointwise 部分：MaxNorm = 0.25
        self.pointwise_conv = nn.Conv2d(
            in_channels  = F1 * D,
            out_channels = F2,
            kernel_size  = (1, 1),
            bias         = False,
        )
        self.bn3      = nn.BatchNorm2d(F2, eps=1e-5, momentum=0.1)
        self.elu2     = nn.ELU()
        self.pool2    = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout2 = nn.Dropout(p=dropout)

        # ── Classifier ─────────────────────────────────────────────────
        # 通过一次虚拟前向传播动态计算 flatten 尺寸
        flatten_size = self._compute_flatten_size(n_channels, n_timepoints)
        self.classifier = nn.Linear(flatten_size, n_classes)

        # 权重初始化
        self._initialize_weights()

    # ──────────────────────────────────────────────────────────────────
    # 前向传播
    # ──────────────────────────────────────────────────────────────────

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征（不含分类头）。x: (B, 1, C, T)"""
        # Block 1
        x = self.temporal_conv(x)     # (B, F1, C, T)
        x = self.bn1(x)
        x = self.depthwise_conv(x)    # (B, F1*D, 1, T)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.pool1(x)             # (B, F1*D, 1, T//4)
        x = self.dropout1(x)

        # Block 2
        x = self.sep_depthwise(x)     # (B, F1*D, 1, T//4 ± 1)
        x = self.pointwise_conv(x)    # (B, F2,   1, T//4 ± 1)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pool2(x)             # (B, F2, 1, ~T//32)
        x = self.dropout2(x)

        return x.flatten(start_dim=1)  # (B, F2 * ~T//32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, C, T)  — EEGDataset 已添加 channel 维度

        Returns
        -------
        logits : (B, n_classes)
        """
        features = self._forward_features(x)
        return self.classifier(features)

    # ──────────────────────────────────────────────────────────────────
    # MaxNorm 约束（必须在每个 optimizer step 后调用）
    # ──────────────────────────────────────────────────────────────────

    def apply_max_norm_(
        self,
        depthwise_max_norm: float = 1.0,
        classifier_max_norm: float = 0.25,
    ) -> None:
        """
        原地（in-place）应用 MaxNorm 权重约束。

        必须在每个 optimizer.step() 之后立即调用：
            optimizer.step()
            model.apply_max_norm_()

        约束对象（与原始 EEGNet 论文及参考实现一致）：
          - Depthwise Conv (Block 1): max_norm = 1.0
          - 分类器 fc Linear:        max_norm = 0.25
        """
        # Depthwise spatial conv: max_norm = 1
        _apply_max_norm_to_layer(self.depthwise_conv, depthwise_max_norm)

        # 分类器 fc: max_norm = 0.25（原论文约束对象，非 pointwise conv）
        _apply_max_norm_to_linear(self.classifier, classifier_max_norm)

    # ──────────────────────────────────────────────────────────────────
    # 辅助方法
    # ──────────────────────────────────────────────────────────────────

    def _compute_flatten_size(self, n_channels: int, n_timepoints: int) -> int:
        """通过虚拟前向传播动态计算 flatten 后的维度。"""
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            out   = self._forward_features(dummy)
        return out.shape[1]

    def _initialize_weights(self) -> None:
        """He 初始化卷积层，常数初始化 BN。"""
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


# ──────────────────────────────────────────────────────────────────────
# MaxNorm 工具函数
# ──────────────────────────────────────────────────────────────────────

def _apply_max_norm_to_layer(layer: nn.Conv2d, max_norm: float) -> None:
    """
    对单个 Conv2d 层应用 MaxNorm 约束（按输出通道行归一化）。
    """
    with torch.no_grad():
        w     = layer.weight                                     # (out_ch, ...)
        norm  = w.view(w.shape[0], -1).norm(p=2, dim=1)         # (out_ch,)
        scale = (max_norm / norm.clamp(min=max_norm))            # ≤ 1
        layer.weight.data = w * scale.view(-1, *([1] * (w.ndim - 1)))


def _apply_max_norm_to_linear(layer: nn.Linear, max_norm: float) -> None:
    """
    对单个 Linear 层应用 MaxNorm 约束（按输出行归一化）。
    """
    with torch.no_grad():
        w     = layer.weight                                     # (out_features, in_features)
        norm  = w.norm(p=2, dim=1, keepdim=True)                # (out_features, 1)
        scale = (max_norm / norm.clamp(min=max_norm))
        layer.weight.data = w * scale


def apply_max_norm(model: nn.Module) -> None:
    """
    对 EEGNet 模型应用 MaxNorm 约束的外部接口。

    如果模型有 apply_max_norm_() 方法，直接调用；
    否则跳过（兼容非 EEGNet 模型）。
    """
    if hasattr(model, 'apply_max_norm_'):
        model.apply_max_norm_()
