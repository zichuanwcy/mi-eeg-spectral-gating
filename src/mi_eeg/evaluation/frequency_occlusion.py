"""Evaluate frequency occlusion using existing F0 decoder weights.

Zero selected FFT components before applying the original unperturbed EA
transform, then evaluate the fixed decoder. No decoder is retrained.
Paths below expect the historical F0 subject weights and compatible raw data.
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path


from mi_eeg.data.bci import load_bci2a_raw, cut_trials
from mi_eeg.preprocessing.alignment import compute_ea_transform, apply_ea_transform
from mi_eeg.models.eegnet import EEGNet
from mi_eeg.evaluation.metrics import compute_metrics

# ============================================================
# 参数配置
# ============================================================

DATA_DIR    = "data/raw"
MODEL_DIR   = "results/F0/models"
OUT_PATH    = "results/frequency_occlusion.json"
SFREQ       = 250.0
T_START     = 2.0
T_END       = 6.5
N_CHANNELS  = 22
N_CLASSES   = 4
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'

# F0训练时的模型超参数（须与F0实际配置一致，否则state_dict加载会失败）
F1          = 8
D           = 2
KERN_LENGTH = 125
DROPOUT     = 0.5

CHECK_SUBJECTS = list(range(1, 10))

# 待测频段：覆盖0.5–100Hz全范围，复用F-series边界，补充40–100Hz
OCCLUSION_BANDS = [
    ("delta(0.5-4)",  0.5,  4.0),
    ("theta(4-8)",    4.0,  8.0),
    ("alpha(8-13)",   8.0,  13.0),
    ("beta(13-30)",   13.0, 30.0),
    ("gamma(30-40)",  30.0, 40.0),
    ("high(40-100)",  40.0, 100.0),
]


# ──────────────────────────────────────────────────────────────
# 数据加载（与F0配置一致：无带通、[2.0,6.5]s）
# ──────────────────────────────────────────────────────────────

def load_subject_wideband(subj: int):
    """加载指定被试的宽带数据（T+E合并），不做任何带通滤波。"""
    X_list, y_list = [], []
    for session in ['T', 'E']:
        fp = os.path.join(DATA_DIR, f"A0{subj}{session}.mat")
        raw, trials, labels, artifacts, _ = load_bci2a_raw(
            fp, N_CHANNELS, None, SFREQ
        )
        X, y = cut_trials(raw, trials, labels, artifacts,
                           T_START, T_END, SFREQ, 0.0, True)
        X_list.append(X)
        y_list.append(y)

    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


# ──────────────────────────────────────────────────────────────
# 频域遮蔽
# ──────────────────────────────────────────────────────────────

def occlude_band(X: np.ndarray, low: float, high: float, sfreq: float) -> np.ndarray:
    """
    对信号做频域遮蔽：[low, high)Hz范围内的频率分量置零后逆FFT还原。

    Parameters
    ----------
    X : np.ndarray, shape (N_trials, N_channels, N_timepoints)
    low, high : float — 待遮蔽频段边界（Hz）
    sfreq : float — 采样率

    Returns
    -------
    np.ndarray, 与输入shape相同，float32
    """
    n_tp  = X.shape[-1]
    freqs = np.fft.rfftfreq(n_tp, d=1.0 / sfreq)
    mask  = (freqs >= low) & (freqs < high)

    X_fft = np.fft.rfft(X, axis=-1)
    X_fft[..., mask] = 0.0
    X_occluded = np.fft.irfft(X_fft, n=n_tp, axis=-1)

    return X_occluded.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# 模型加载与推理
# ──────────────────────────────────────────────────────────────

def load_f0_model(subj: int, n_timepoints: int) -> EEGNet:
    """加载指定被试的F0已训练模型（最后一个种子的权重）。"""
    pth_path = Path(MODEL_DIR) / f"F0_S{subj}_seed_last.pth"
    if not pth_path.exists():
        raise FileNotFoundError(
            f"找不到F0模型文件: {pth_path}\n"
            f"请确认F0已运行完成（见run_experiment.py / configs.py中的F0配置）"
        )

    model = EEGNet(N_CLASSES, N_CHANNELS, n_timepoints,
                   F1, D, F1 * D, KERN_LENGTH, DROPOUT)
    state = torch.load(str(pth_path), map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def predict_accuracy(model: EEGNet, X: np.ndarray, y: np.ndarray) -> float:
    """用固定模型对给定输入做推理，返回准确率。"""
    with torch.no_grad():
        x_tensor = torch.from_numpy(X[:, np.newaxis, :, :]).to(DEVICE)
        logits   = model(x_tensor)
        y_pred   = logits.argmax(dim=1).cpu().numpy()
    return compute_metrics(y, y_pred, N_CLASSES)['accuracy']


# ──────────────────────────────────────────────────────────────
# 主流程：逐被试逐频段遮蔽测试
# ──────────────────────────────────────────────────────────────

def run_occlusion_for_subject(subj: int) -> dict:
    """对单个被试运行完整的频域遮蔽测试，返回基线及各频段的准确率/下降幅度。"""
    X_raw, y = load_subject_wideband(subj)
    n_timepoints = X_raw.shape[-1]

    # EA变换矩阵固定用未扰动数据计算（与F0原始训练/评估时一致）
    R = compute_ea_transform(X_raw)

    model = load_f0_model(subj, n_timepoints)

    # 基线：不遮蔽，仅做EA
    X_baseline_ea = apply_ea_transform(X_raw, R)
    acc_baseline  = predict_accuracy(model, X_baseline_ea, y)

    band_results = {}
    for band_name, low, high in OCCLUSION_BANDS:
        X_occ    = occlude_band(X_raw, low, high, SFREQ)
        X_occ_ea = apply_ea_transform(X_occ, R)
        acc_occ  = predict_accuracy(model, X_occ_ea, y)
        drop     = acc_baseline - acc_occ
        band_results[band_name] = {'acc': acc_occ, 'drop': drop}

    return {'acc_baseline': acc_baseline, 'bands': band_results}


def main():
    print("=" * 70)
    print("频域遮蔽测试：各频段对F0（宽带）模型的真实重要性")
    print("=" * 70)

    all_results = {}

    for subj in CHECK_SUBJECTS:
        print(f"\n--- S{subj} ---")
        result = run_occlusion_for_subject(subj)
        all_results[subj] = result

        print(f"  基线准确率（不遮蔽）: {result['acc_baseline']:.4f}")
        for band_name, _, _ in OCCLUSION_BANDS:
            r = result['bands'][band_name]
            print(f"  遮蔽{band_name:<16s}: acc={r['acc']:.4f}  "
                  f"下降={r['drop']:+.4f}")

    # 跨被试汇总
    print("\n" + "=" * 70)
    print("跨被试汇总（各频段平均下降幅度，降幅越大说明模型对该频段依赖越强）")
    print("=" * 70)

    mean_baseline = np.mean([all_results[s]['acc_baseline'] for s in CHECK_SUBJECTS])
    print(f"跨被试平均基线准确率: {mean_baseline:.4f}\n")

    summary = {}
    for band_name, _, _ in OCCLUSION_BANDS:
        drops = [all_results[s]['bands'][band_name]['drop'] for s in CHECK_SUBJECTS]
        mean_drop = float(np.mean(drops))
        std_drop  = float(np.std(drops))
        summary[band_name] = {'mean_drop': mean_drop, 'std_drop': std_drop,
                               'per_subj_drop': dict(zip(CHECK_SUBJECTS, drops))}
        print(f"  {band_name:<16s}: 平均下降={mean_drop:+.4f}  跨被试std={std_drop:.4f}")

    print("\n按平均下降幅度排序（重要性从高到低）：")
    ranked = sorted(summary.items(), key=lambda kv: kv[1]['mean_drop'], reverse=True)
    for rank, (band_name, stats) in enumerate(ranked, 1):
        print(f"  {rank}. {band_name:<16s} 平均下降={stats['mean_drop']:+.4f}")

    print("\n请结合各频段下降幅度及跨被试std（std过大可能提示个体异质性，")
    print("而非统一的重要性模式），决定方案C的滤波器分配比例（Q2）和输入")
    print("频段范围（Q3）。若40-100Hz降幅显著但仅集中在少数被试上，需结合")
    print("F9（40-100Hz单独）结果一并判断是否为肌电伪迹而非真实神经信息。")
    print("=" * 70)

    save_path = Path(OUT_PATH)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_payload = {
        "per_subject": {
            str(subj): {
                "acc_baseline": all_results[subj]['acc_baseline'],
                "bands": {
                    band_name: {
                        "acc":  all_results[subj]['bands'][band_name]['acc'],
                        "drop": all_results[subj]['bands'][band_name]['drop'],
                    }
                    for band_name, _, _ in OCCLUSION_BANDS
                },
            }
            for subj in CHECK_SUBJECTS
        },
        "summary": {
            band_name: {
                "mean_drop": summary[band_name]['mean_drop'],
                "std_drop":  summary[band_name]['std_drop'],
                "per_subj_drop": {
                    str(s): d for s, d in summary[band_name]['per_subj_drop'].items()
                },
            }
            for band_name, _, _ in OCCLUSION_BANDS
        },
        "mean_baseline": mean_baseline,
    }
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_payload, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至: {save_path}")

    return all_results, summary


if __name__ == "__main__":
    main()
