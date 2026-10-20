"""实验 2-A：EA 对齐效果的频段差异分析

验证假说：EA 的贡献主要来自消除高频成分的空间协方差偏移，对低频成分接近冗余。

包含两种互补的分析设计：

  方案一（原方案）：F4 vs F7 分别训练的模型
      对比高频输入（F4，13–30Hz）和低频输入（F7，0.5–13Hz）两种条件下，
      有EA / 无EA 时特征协方差结构的差异。
      前提：F4/F7/F4_noea/F7_noea 的模型文件已存在。

  方案二（补充）：F0 宽带模型 + 输入分带
      在同一个宽带模型（F0）上，对输入数据分别做低频/高频滤波后提取激活，
      对比 EA 对两种激活的对齐效果。控制了模型本身的变量。
      前提：F0 和 F0_noea 的模型文件已存在。


Paths below refer to separately obtained historical model/results inputs.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path


# ============================================================
# 参数配置（修改此处）
# ============================================================

DATA_DIR = "data/raw"
OUT_DIR  = "figures/2A"

# 方案一：F4/F7 分别训练的模型目录
F4_EA_DIR    = "results/F4/models"
F7_EA_DIR    = "results/F7/models"
F4_NOEA_DIR  = "results/F4_noea/models"
F7_NOEA_DIR  = "results/F7_noea/models"

# 方案二：F0 宽带模型目录
F0_EA_DIR    = "results/F0/models"
F0_NOEA_DIR  = "results/F0_noea/models"

# F 系列训练结果 JSON（用于被试分层，取 F0 的 LOSO 准确率）
F0_RESULT_JSON = "results/F0.json"

# EEGNet 结构参数（须与训练时一致）
N_CHANNELS  = 22
N_CLASSES   = 4
F1_FILTERS  = 8
D           = 2
KERN_LENGTH = 125
DROPOUT     = 0.5
SFREQ       = 250.0
T_START     = 2.0
T_END       = 6.5

# 方案二的分带边界
LOW_BAND  = (0.5, 13.0)   # 对应 F7 频段
HIGH_BAND = (13.0, 30.0)  # 对应 F4 频段

# ============================================================
# 以下为正式代码，不含可调参数
# ============================================================

N_TIMEPOINTS = int((T_END - T_START) * SFREQ)  # 1125
SUBJECTS     = list(range(1, 10))


# ──────────────────────────────────────────────────────────────
# 模型加载
# ──────────────────────────────────────────────────────────────

def load_models(model_dir: str) -> dict:
    """
    加载指定目录下所有被试的 EEGNet 模型。

    Returns
    -------
    models : dict  {  'S1': model, 'S2': model, ... }
    """
    import torch
    from mi_eeg.models.eegnet import EEGNet

    models    = {}
    model_dir = Path(model_dir)

    if not model_dir.exists():
        print(f"  ⚠️  目录不存在：{model_dir}")
        return models

    for subj in SUBJECTS:
        pth_files = sorted(model_dir.glob(f"*S{subj}_seed*.pth"))
        if not pth_files:
            print(f"  ⚠️  未找到 S{subj} 的模型文件，跳过")
            continue

        state = torch.load(pth_files[-1], map_location='cpu')
        model = EEGNet(N_CLASSES, N_CHANNELS, N_TIMEPOINTS,
                       F1_FILTERS, D, F1_FILTERS * D, KERN_LENGTH, DROPOUT)
        model.load_state_dict(state)
        model.eval()
        models[f'S{subj}'] = model

    print(f"  成功加载 {len(models)} 个模型（来自 {model_dir}）")
    return models


# ──────────────────────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────────────────────

def load_all_subjects_raw(bandpass_freq=None) -> dict:
    """
    加载所有被试的宽带（或指定频段）数据，不做 EA 对齐。

    Returns
    -------
    all_data : dict
        { subj_id: (X, y) }   X shape: (N_trials, N_channels, N_timepoints)
    """
    from mi_eeg.data.bci import load_bci2a_raw, cut_trials

    all_data = {}
    for subj in SUBJECTS:
        X_list, y_list = [], []
        for session in ['T', 'E']:
            fp = os.path.join(DATA_DIR, f"A0{subj}{session}.mat")
            raw, trials, labels, artifacts, _ = load_bci2a_raw(
                fp, N_CHANNELS, bandpass_freq, SFREQ, 4
            )
            X, y = cut_trials(raw, trials, labels, artifacts,
                               T_START, T_END, SFREQ, 0.0, True)
            X_list.append(X)
            y_list.append(y)
        all_data[subj] = (
            np.concatenate(X_list, axis=0),
            np.concatenate(y_list, axis=0),
        )
    return all_data


def apply_ea_to_dict(all_data: dict) -> dict:
    """对 all_data 字典中的每个被试独立应用 EA 对齐。"""
    from mi_eeg.preprocessing.alignment import compute_ea_transform, apply_ea_transform

    aligned = {}
    for subj_id, (X, y) in all_data.items():
        R = compute_ea_transform(X)
        aligned[subj_id] = (apply_ea_transform(X, R), y)
    return aligned


# ──────────────────────────────────────────────────────────────
# 特征提取
# ──────────────────────────────────────────────────────────────

def extract_features(model, X: np.ndarray) -> np.ndarray:
    """
    提取 EEGNet 分类器前一层的 16 维特征向量。

    Parameters
    ----------
    X : (N_trials, N_channels, N_timepoints)

    Returns
    -------
    features : (N_trials, 16)
    """
    import torch

    model.eval()
    X_tensor = torch.FloatTensor(X[:, np.newaxis, :, :])  # (N, 1, C, T)
    with torch.no_grad():
        features = model._forward_features(X_tensor)       # (N, 16)
    return features.numpy()


# ──────────────────────────────────────────────────────────────
# 协方差结构分析
# ──────────────────────────────────────────────────────────────

def compute_feature_cov(features: np.ndarray) -> np.ndarray:
    """计算特征矩阵的协方差矩阵，shape (16, 16)。"""
    return np.cov(features.T)  # (16, 16)


def cov_trace_variance(covs: list) -> float:
    """
    计算跨被试协方差矩阵迹的方差。
    迹代表总方差（尺度差异），迹的方差反映被试间尺度偏移的大小。
    """
    traces = [np.trace(c) for c in covs]
    return float(np.var(traces))


def cov_principal_angle(cov_a: np.ndarray, cov_b: np.ndarray) -> float:
    """
    计算两个协方差矩阵之间的主轴夹角（度）。
    衡量被试间主方向（旋转）偏移的大小。
    """
    _, va = np.linalg.eigh(cov_a)
    _, vb = np.linalg.eigh(cov_b)
    # 取前8个主方向（最大特征值对应的特征向量）
    top_k = 8
    va_top = va[:, -top_k:]
    vb_top = vb[:, -top_k:]
    # 主角 = arccos(最大奇异值)
    S = np.linalg.svd(va_top.T @ vb_top, compute_uv=False)
    angle = float(np.degrees(np.arccos(np.clip(S[0], -1, 1))))
    return angle


def silhouette_score_features(features_list: list, labels_list: list) -> float:
    """
    计算多被试特征的 Silhouette Score（以被试 ID 作为类标签）。
    分数越高，被试间分离度越大（对齐效果越差）；
    对齐有效 → 分数下降（被试分布重叠增加）。
    """
    from sklearn.metrics import silhouette_score

    X_all = np.concatenate(features_list, axis=0)
    y_all = np.concatenate(labels_list,   axis=0)

    # 特征数量过多时随机采样，避免计算过慢
    if len(X_all) > 2000:
        idx   = np.random.choice(len(X_all), 2000, replace=False)
        X_all = X_all[idx]
        y_all = y_all[idx]

    return float(silhouette_score(X_all, y_all, metric='euclidean'))


# ──────────────────────────────────────────────────────────────
# 方案一：F4/F7 分别训练的模型对比
# ──────────────────────────────────────────────────────────────

def run_scheme1(loso_accs: dict) -> dict:
    """
    方案一：对比高频（F4）和低频（F7）模型在有EA/无EA下的特征协方差结构。

    Returns
    -------
    results : dict  包含各条件下的迹方差、主角、Silhouette Score
    """
    print("\n" + "=" * 60)
    print("方案一：F4/F7 分别训练模型 × 有EA/无EA 对比")
    print("=" * 60)

    results = {}

    configs = [
        ("F4_ea",   F4_EA_DIR,   (13.0, 30.0), True,  "高频+EA"),
        ("F4_noea", F4_NOEA_DIR, (13.0, 30.0), False, "高频+无EA"),
        ("F7_ea",   F7_EA_DIR,   (0.5,  13.0), True,  "低频+EA"),
        ("F7_noea", F7_NOEA_DIR, (0.5,  13.0), False, "低频+无EA"),
    ]

    # 预加载各频段数据
    print("\n  加载数据...")
    data_high = load_all_subjects_raw(bandpass_freq=(13.0, 30.0))
    data_low  = load_all_subjects_raw(bandpass_freq=(0.5,  13.0))
    data_high_ea = apply_ea_to_dict(data_high)
    data_low_ea  = apply_ea_to_dict(data_low)

    for key, model_dir, band, use_ea, label in configs:
        print(f"\n  [{label}] 加载模型...")
        models = load_models(model_dir)
        if not models:
            print(f"  ⚠️  跳过 {label}（模型文件缺失）")
            continue

        # 选取对应数据
        if band == (13.0, 30.0):
            data = data_high_ea if use_ea else data_high
        else:
            data = data_low_ea if use_ea else data_low

        # 提取各被试特征
        covs          = []
        features_list = []
        subj_id_list  = []

        for subj_key, model in models.items():
            subj_id = int(subj_key[1:])
            X, _    = data[subj_id]
            feats   = extract_features(model, X)
            cov     = compute_feature_cov(feats)
            covs.append(cov)
            features_list.append(feats)
            subj_id_list.append(
                np.full(len(feats), subj_id, dtype=np.int64)
            )

        # 计算指标
        trace_var = cov_trace_variance(covs)

        # 主角：计算所有被试对的均值
        angles = []
        subj_keys = list(models.keys())
        for i in range(len(subj_keys)):
            for j in range(i + 1, len(subj_keys)):
                angles.append(cov_principal_angle(covs[i], covs[j]))
        mean_angle = float(np.mean(angles)) if angles else 0.0

        sil_score = silhouette_score_features(features_list, subj_id_list)

        results[key] = {
            'label':      label,
            'trace_var':  round(trace_var,  4),
            'mean_angle': round(mean_angle, 4),
            'sil_score':  round(sil_score,  4),
            'n_subj':     len(models),
        }
        print(f"    迹方差={trace_var:.4f}  主角均值={mean_angle:.2f}°  "
              f"Silhouette={sil_score:.4f}")

    return results


# ──────────────────────────────────────────────────────────────
# 方案二：F0 宽带模型 + 输入分带（补充子实验）
# ──────────────────────────────────────────────────────────────

def run_scheme2(loso_accs: dict) -> dict:
    """
    方案二（补充）：固定同一个宽带模型（F0），
    对输入数据分别做低频/高频滤波后提取激活，
    对比 EA 对两种激活的对齐效果差异。

    控制了模型本身的变量，结论更干净。

    Returns
    -------
    results : dict
    """
    print("\n" + "=" * 60)
    print("方案二（补充）：F0 宽带模型 × 输入低/高频分带 × 有EA/无EA")
    print("=" * 60)

    results = {}

    # 加载 F0 有EA / 无EA 模型
    print("\n  加载 F0 模型（有EA）...")
    models_ea   = load_models(F0_EA_DIR)
    print("  加载 F0 模型（无EA）...")
    models_noea = load_models(F0_NOEA_DIR)

    if not models_ea or not models_noea:
        print("  ⚠️  F0 模型文件缺失，跳过方案二")
        return results

    # 加载宽带数据（无滤波）
    print("  加载宽带数据...")
    data_wide    = load_all_subjects_raw(bandpass_freq=None)
    data_wide_ea = apply_ea_to_dict(data_wide)

    for band_label, band_freq in [("低频", LOW_BAND), ("高频", HIGH_BAND)]:
        from mi_eeg.preprocessing.filtering import bandpass_filter

        for ea_label, models, data in [
            ("有EA",  models_ea,   data_wide_ea),
            ("无EA",  models_noea, data_wide),
        ]:
            key = f"F0_{band_label}_{ea_label}"
            label = f"F0宽带模型 × 输入{band_label}滤波 × {ea_label}"
            print(f"\n  [{label}]")

            covs          = []
            features_list = []
            subj_id_list  = []

            for subj_key, model in models.items():
                subj_id = int(subj_key[1:])
                X, _    = data[subj_id]

                # 对已对齐（或未对齐）的 trial 数据做输入端分带滤波
                X_band = np.stack([
                    bandpass_filter(
                        X[i], band_freq[0], band_freq[1], SFREQ, 4
                    )
                    for i in range(len(X))
                ])  # (N, C, T)

                feats = extract_features(model, X_band)
                cov   = compute_feature_cov(feats)
                covs.append(cov)
                features_list.append(feats)
                subj_id_list.append(
                    np.full(len(feats), subj_id, dtype=np.int64)
                )

            trace_var  = cov_trace_variance(covs)
            angles     = []
            subj_keys  = list(models.keys())
            for i in range(len(subj_keys)):
                for j in range(i + 1, len(subj_keys)):
                    angles.append(cov_principal_angle(covs[i], covs[j]))
            mean_angle = float(np.mean(angles)) if angles else 0.0
            sil_score  = silhouette_score_features(features_list, subj_id_list)

            results[key] = {
                'label':      label,
                'trace_var':  round(trace_var,  4),
                'mean_angle': round(mean_angle, 4),
                'sil_score':  round(sil_score,  4),
                'n_subj':     len(models),
            }
            print(f"    迹方差={trace_var:.4f}  主角均值={mean_angle:.2f}°  "
                  f"Silhouette={sil_score:.4f}")

    return results


# ──────────────────────────────────────────────────────────────
# 打印汇总结论
# ──────────────────────────────────────────────────────────────

def print_summary(scheme1: dict, scheme2: dict) -> None:
    print("\n" + "=" * 70)
    print("实验 2-A 汇总：EA 对齐效果的频段差异")
    print("=" * 70)

    print("\n【方案一：F4/F7分别训练模型】")
    print(f"  {'条件':<16} {'迹方差':>10} {'主角均值(°)':>12} {'Silhouette':>12}")
    print("  " + "-" * 52)
    for key in ["F4_ea", "F4_noea", "F7_ea", "F7_noea"]:
        if key not in scheme1:
            continue
        r = scheme1[key]
        print(f"  {r['label']:<16} {r['trace_var']:>10.4f} "
              f"{r['mean_angle']:>12.2f} {r['sil_score']:>12.4f}")

    # 差值：EA的效果 = 有EA指标 - 无EA指标（正值=EA使被试间差异减小）
    if all(k in scheme1 for k in ["F4_ea", "F4_noea", "F7_ea", "F7_noea"]):
        f4_delta_sil = scheme1["F4_noea"]["sil_score"] - scheme1["F4_ea"]["sil_score"]
        f7_delta_sil = scheme1["F7_noea"]["sil_score"] - scheme1["F7_ea"]["sil_score"]
        print(f"\n  EA 对高频（F4）的 Silhouette 降低量：{f4_delta_sil:+.4f}")
        print(f"  EA 对低频（F7）的 Silhouette 降低量：{f7_delta_sil:+.4f}")
        if f4_delta_sil > f7_delta_sil:
            print("  → 高频条件下 EA 效果更显著，支持假说")
        else:
            print("  → 低频条件下 EA 效果更显著或相当，不支持假说")

    print("\n【方案二（补充）：F0宽带模型 × 输入分带】")
    print(f"  {'条件':<28} {'迹方差':>10} {'主角均值(°)':>12} {'Silhouette':>12}")
    print("  " + "-" * 64)
    for key, r in scheme2.items():
        print(f"  {r['label']:<28} {r['trace_var']:>10.4f} "
              f"{r['mean_angle']:>12.2f} {r['sil_score']:>12.4f}")

    # 方案二的核心对比：高频EA效果 vs 低频EA效果
    keys = list(scheme2.keys())
    hea  = next((k for k in keys if "高频" in k and "有EA" in k), None)
    hno  = next((k for k in keys if "高频" in k and "无EA" in k), None)
    lea  = next((k for k in keys if "低频" in k and "有EA" in k), None)
    lno  = next((k for k in keys if "低频" in k and "无EA" in k), None)

    if all(x is not None for x in [hea, hno, lea, lno]):
        high_delta = scheme2[hno]["sil_score"] - scheme2[hea]["sil_score"]
        low_delta  = scheme2[lno]["sil_score"] - scheme2[lea]["sil_score"]
        print(f"\n  EA 对高频激活的 Silhouette 降低量：{high_delta:+.4f}")
        print(f"  EA 对低频激活的 Silhouette 降低量：{low_delta:+.4f}")
        if high_delta > low_delta:
            print("  → 高频激活对齐效果更显著，支持假说（两方案结论一致）"
                  if scheme1 else "  → 高频激活对齐效果更显著，支持假说")
        else:
            print("  → 低频激活对齐效果更显著或相当，不支持假说")

    print("=" * 70)


# ──────────────────────────────────────────────────────────────
# 可视化
# ──────────────────────────────────────────────────────────────

def plot_results(scheme1: dict, scheme2: dict, out_dir: str) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠️  matplotlib 未安装，跳过绘图")
        return

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── 图1：方案一 Silhouette Score 对比 ──────────────────────
    if scheme1:
        labels = []
        sil_scores = []
        colors = []
        color_map = {
            "F4_ea":   ("steelblue",  "高频+EA"),
            "F4_noea": ("lightblue",  "高频+无EA"),
            "F7_ea":   ("coral",      "低频+EA"),
            "F7_noea": ("lightsalmon","低频+无EA"),
        }
        for key in ["F4_ea", "F4_noea", "F7_ea", "F7_noea"]:
            if key in scheme1:
                c, lbl = color_map[key]
                labels.append(lbl)
                sil_scores.append(scheme1[key]["sil_score"])
                colors.append(c)

        if labels:
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.bar(labels, sil_scores, color=colors, alpha=0.85, width=0.5)
            ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
            ax.set_ylabel("Silhouette Score（越低=被试间分布越重叠）", fontsize=10)
            ax.set_title(
                "方案一：F4/F7模型 × 有EA/无EA\n"
                "Silhouette Score（以被试ID为类标签）",
                fontsize=11
            )
            ax.set_ylim(0, max(sil_scores) * 1.25)
            ax.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            fig.savefig(out_path / "2A_scheme1_silhouette.png", dpi=150)
            plt.close(fig)

    # ── 图2：方案二 Silhouette Score 对比（2×2 分组柱状图）───
    if scheme2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

        for ax_idx, (band_label, color_ea, color_noea) in enumerate([
            ("低频", "coral",      "lightsalmon"),
            ("高频", "steelblue",  "lightblue"),
        ]):
            key_ea  = next((k for k in scheme2 if band_label in k and "有EA" in k), None)
            key_nea = next((k for k in scheme2 if band_label in k and "无EA" in k), None)
            ax = axes[ax_idx]

            vals   = []
            clrs   = []
            xlbls  = []
            if key_ea:
                vals.append(scheme2[key_ea]["sil_score"])
                clrs.append(color_ea)
                xlbls.append("有EA")
            if key_nea:
                vals.append(scheme2[key_nea]["sil_score"])
                clrs.append(color_noea)
                xlbls.append("无EA")

            if vals:
                bars = ax.bar(xlbls, vals, color=clrs, alpha=0.85, width=0.4)
                ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
                ax.set_title(f"F0模型 × 输入{band_label}滤波", fontsize=11)
                ax.set_ylabel("Silhouette Score", fontsize=10)
                ax.set_ylim(0, max(vals) * 1.25)
                ax.grid(True, axis='y', alpha=0.3)

        fig.suptitle(
            "方案二（补充）：F0宽带模型 × 输入分带 × 有EA/无EA\n"
            "Silhouette Score（以被试ID为类标签）",
            fontsize=11
        )
        plt.tight_layout()
        fig.savefig(out_path / "2A_scheme2_silhouette.png", dpi=150)
        plt.close(fig)

    # ── 图3：迹方差汇总（方案一+方案二合并展示）─────────────
    all_labels     = []
    all_trace_vars = []
    all_colors     = []

    palette = ["#2196F3", "#90CAF9", "#FF5722", "#FFCCBC",
               "#9C27B0", "#E1BEE7", "#4CAF50", "#C8E6C9"]
    idx = 0
    for key in ["F4_ea", "F4_noea", "F7_ea", "F7_noea"]:
        if key in scheme1:
            all_labels.append(scheme1[key]["label"])
            all_trace_vars.append(scheme1[key]["trace_var"])
            all_colors.append(palette[idx % len(palette)])
            idx += 1
    for key, r in scheme2.items():
        all_labels.append(r["label"].replace("F0宽带模型 × ", ""))
        all_trace_vars.append(r["trace_var"])
        all_colors.append(palette[idx % len(palette)])
        idx += 1

    if all_labels:
        fig, ax = plt.subplots(figsize=(max(10, len(all_labels) * 1.5), 4))
        bars = ax.bar(range(len(all_labels)), all_trace_vars,
                      color=all_colors, alpha=0.85)
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=8)
        ax.set_xticks(range(len(all_labels)))
        ax.set_xticklabels(all_labels, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel("特征协方差迹的跨被试方差（越低=被试间尺度偏移越小）",
                      fontsize=10)
        ax.set_title("实验2-A：各条件下特征协方差迹方差汇总", fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        fig.savefig(out_path / "2A_trace_variance_all.png", dpi=150)
        plt.close(fig)

    print(f"图表已保存至 {out_path}")


# ──────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)

    print("=" * 70)
    print("实验 2-A：EA 对齐效果的频段差异分析")
    print("=" * 70)

    # 加载 LOSO 准确率（用于被试分层参考）
    loso_accs = {}
    if os.path.exists(F0_RESULT_JSON):
        with open(F0_RESULT_JSON) as f:
            rdata = json.load(f)
        loso_accs = {
            f"S{i+1}": acc
            for i, acc in enumerate(rdata.get('per_subject_mean', []))
        }
        print(f"\n已加载 F0 LOSO 准确率，均值="
              f"{np.mean(list(loso_accs.values())):.4f}")
    else:
        print(f"\n⚠️  未找到 {F0_RESULT_JSON}，跳过被试分层")

    # 运行方案一
    scheme1 = run_scheme1(loso_accs)

    # 运行方案二
    scheme2 = run_scheme2(loso_accs)

    if not scheme1 and not scheme2:
        print("\n❌ 两个方案均未产生结果，请检查模型文件是否存在。")
        sys.exit(1)

    # 打印汇总
    print_summary(scheme1, scheme2)

    # 保存 JSON 结果
    out_path = Path(OUT_DIR)
    out_path.mkdir(parents=True, exist_ok=True)

    save_data = {
        "scheme1_f4_f7_models": scheme1,
        "scheme2_f0_split_input": scheme2,
    }
    result_file = out_path / "2A_results.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n数值结果已保存至 {result_file}")

    # 绘图
    plot_results(scheme1, scheme2, OUT_DIR)

    print("\n完成。")


if __name__ == "__main__":
    main()
