"""双路 + Cross-Attention EEGNet 模型（实验3-C）

在 DualBranchEEGNet 的基础上，在两路 Block1 输出后插入轻量级
Cross-Attention 模块：高频特征作为 Query，低频特征作为 Key/Value。
注意力输出与原高频特征做残差相加后，继续各自的 Block2 流程。

Cross-Attention 操作在 Block1 输出的特征图（高维）上进行，
而非最终 16 维 flatten 特征，保留更丰富的空间-时间结构。

架构示意：
  X_low  → Block1_low  → feat_low_b1  ──────────────────────┐
                                        Cross-Attn           │
  X_high → Block1_high → feat_high_b1 → attn_out + residual │
                                                ↓            │
  X_high → Block2_high → feat_high_16         [cat]
  feat_low_b1 → Block2_low → feat_low_16 ─────────────────→ [cat] → classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mi_eeg.models.eegnet import EEGNet, _apply_max_norm_to_layer, _apply_max_norm_to_linear


class CrossFreqAttention(nn.Module):
    """
    轻量级跨频段注意力模块。

    在 EEGNet Block1 输出的特征图上操作。
    Block1 输出 shape：(B, F1*D, 1, T//4)，将空间维（1）squeeze 后
    视为序列 (B, F1*D, T//4)，在时间维做注意力。

    高频特征作为 Query，低频特征作为 Key/Value：
      - 引导模型学习"在特定低频状态下应关注高频的哪些时间片段"

    使用单头注意力，保持轻量。

    Parameters
    ----------
    feat_dim : int — 特征通道数，即 F1*D，默认 16
    """

    def __init__(self, feat_dim: int = 16):
        super().__init__()
        self.feat_dim = feat_dim

        # Q/K/V 投影（1×1 卷积，不增加参数量）
        self.q_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.k_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.v_proj = nn.Linear(feat_dim, feat_dim, bias=False)
        self.out_proj = nn.Linear(feat_dim, feat_dim, bias=False)

        self.scale = feat_dim ** -0.5

        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(
        self,
        feat_high: torch.Tensor,
        feat_low:  torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        feat_high : (B, F1*D, 1, T//4) — 高频 Block1 输出
        feat_low  : (B, F1*D, 1, T//4) — 低频 Block1 输出

        Returns
        -------
        out : (B, F1*D, 1, T//4) — 注意力增强后的高频特征（含残差）
        """
        B, C, _, T = feat_high.shape

        # squeeze 空间维，转为序列：(B, T, C)
        h = feat_high.squeeze(2).permute(0, 2, 1)  # (B, T, C)
        l = feat_low.squeeze(2).permute(0, 2, 1)   # (B, T, C)

        Q = self.q_proj(h)  # (B, T, C)  高频为 Query
        K = self.k_proj(l)  # (B, T, C)  低频为 Key
        V = self.v_proj(l)  # (B, T, C)  低频为 Value

        # 注意力权重：(B, T, T)
        attn = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # 加权聚合：(B, T, C)
        out = torch.bmm(attn, V)
        out = self.out_proj(out)

        # 转回 (B, C, 1, T) 并做残差相加
        out = out.permute(0, 2, 1).unsqueeze(2)  # (B, C, 1, T)
        return feat_high + out


class DualBranchAttnEEGNet(nn.Module):
    """
    双路 + Cross-Attention EEGNet。

    在 Block1 输出后对高频特征注入低频上下文，
    之后两路各自继续 Block2，最终特征拼接送入分类头。

    接口与 DualBranchEEGNet 完全一致。
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

        # 两路共享参数结构，但权重独立
        self.low_branch  = EEGNet(n_classes, n_channels, n_timepoints,
                                  F1, D, F2, kern_length, dropout)
        self.high_branch = EEGNet(n_classes, n_channels, n_timepoints,
                                  F1, D, F2, kern_length, dropout)

        # Cross-Attention 模块
        self.cross_attn = CrossFreqAttention(feat_dim=F2)

        # 联合分类头
        single_feat_dim = self.low_branch.classifier.in_features
        self.classifier = nn.Linear(single_feat_dim * 2, n_classes)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def _forward_block1(
        self,
        branch: EEGNet,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        执行 EEGNet Block1，返回 AvgPool 和 Dropout 之前的特征图。

        Block1 流程：temporal_conv → bn1 → depthwise_conv → bn2 → elu1
                    → pool1 → dropout1
        返回 pool1 输出（dropout 之前），shape: (B, F1*D, 1, T//4)
        """
        x = branch.temporal_conv(x)
        x = branch.bn1(x)
        x = branch.depthwise_conv(x)
        x = branch.bn2(x)
        x = branch.elu1(x)
        x = branch.pool1(x)
        return x  # dropout 在注意力之后再施加

    def _forward_block2_and_flatten(
        self,
        branch: EEGNet,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        从 Block1 pool1 输出开始，执行 dropout1 + Block2 + flatten。

        Returns
        -------
        features : (B, 16)
        """
        x = branch.dropout1(x)
        x = branch.sep_depthwise(x)
        x = branch.pointwise_conv(x)
        x = branch.bn3(x)
        x = branch.elu2(x)
        x = branch.pool2(x)
        x = branch.dropout2(x)
        return x.flatten(start_dim=1)

    def _forward_features(
        self,
        x_low:  torch.Tensor,
        x_high: torch.Tensor,
    ) -> torch.Tensor:
        """
        提取双路拼接特征（含 Cross-Attention）。

        Returns
        -------
        features : (B, 32)
        """
        # Block1
        feat_low_b1  = self._forward_block1(self.low_branch,  x_low)
        feat_high_b1 = self._forward_block1(self.high_branch, x_high)

        # Cross-Attention：低频注入高频
        feat_high_b1 = self.cross_attn(feat_high_b1, feat_low_b1)

        # Block2 + flatten
        feat_low  = self._forward_block2_and_flatten(self.low_branch,  feat_low_b1)
        feat_high = self._forward_block2_and_flatten(self.high_branch, feat_high_b1)

        return torch.cat([feat_low, feat_high], dim=1)  # (B, 32)

    def forward(
        self,
        x_low:  torch.Tensor,
        x_high: torch.Tensor,
    ) -> torch.Tensor:
        features = self._forward_features(x_low, x_high)
        return self.classifier(features)

    def apply_max_norm_(
        self,
        depthwise_max_norm:  float = 1.0,
        classifier_max_norm: float = 0.25,
    ) -> None:
        """对两路分别施加 MaxNorm，并约束联合分类头。"""
        for branch in [self.low_branch, self.high_branch]:
            _apply_max_norm_to_layer(branch.depthwise_conv, depthwise_max_norm)
        _apply_max_norm_to_linear(self.classifier, classifier_max_norm)
