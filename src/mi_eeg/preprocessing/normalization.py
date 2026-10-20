"""归一化模块

实现 exponential moving standardize（指数移动标准化），
参考 EEG-TCNet 实现（https://github.com/iis-eth-zurich/eeg-tcnet）。
"""

import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


def exponential_moving_standardize(
    data: np.ndarray,
    factor_new: float = 0.001,
    init_block_size: int = 1000,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    指数移动标准化（Exponential Moving Standardize）。

    对信号进行在线式去均值 + 方差归一化，
    使用指数加权移动平均（EWM）估计局部均值和方差。

    相比简单 z-score，此方法对非平稳信号更鲁棒，
    且不会引入未来信息泄露（因果滤波器方向）。

    Parameters
    ----------
    data : np.ndarray
        输入信号，shape (N_channels, N_timepoints)
    factor_new : float
        EWM alpha 参数（衰减因子），越小记忆越长，默认 0.001
    init_block_size : int
        初始化块大小（预留参数，pandas EWM 无需显式使用）
    eps : float
        防止除零的小量，默认 1e-4

    Returns
    -------
    np.ndarray
        归一化后的信号，shape 与输入相同，dtype float32

    Notes
    -----
    - 建议在 trial 切割之前对连续信号应用，以利用更长的历史信息
    - factor_new=0.001 对应约 1000 个样本的有效记忆（250Hz 下约 4 秒）
    """
    if _HAS_PANDAS:
        return _ems_pandas(data, factor_new, eps)
    else:
        return _ems_numpy(data, factor_new, eps)


def _ems_pandas(data: np.ndarray, factor_new: float, eps: float) -> np.ndarray:
    """基于 pandas EWM 的实现（推荐，与参考实现一致）。"""
    import pandas as pd

    # (N_channels, N_timepoints) → (N_timepoints, N_channels)
    data_t = data.T

    df      = pd.DataFrame(data_t)
    meaned  = df.ewm(alpha=factor_new).mean()
    demeaned = df - meaned
    sqrd    = demeaned ** 2
    var     = sqrd.ewm(alpha=factor_new).mean()
    std     = np.maximum(np.sqrt(var.values), eps)
    normalized = demeaned.values / std

    return normalized.T.astype(np.float32)  # (N_channels, N_timepoints)


def _ems_numpy(data: np.ndarray, factor_new: float, eps: float) -> np.ndarray:
    """纯 numpy 实现（pandas 不可用时的备用方案）。"""
    # (N_channels, N_timepoints)
    n_ch, n_tp = data.shape
    result = np.zeros_like(data, dtype=np.float32)

    for ch in range(n_ch):
        x = data[ch].astype(np.float64)
        mean_est = np.zeros(n_tp)
        var_est  = np.zeros(n_tp)

        # 初始化
        mean_est[0] = x[0]
        var_est[0]  = 0.0

        for t in range(1, n_tp):
            mean_est[t] = (1 - factor_new) * mean_est[t-1] + factor_new * x[t]
            diff = x[t] - mean_est[t]
            var_est[t]  = (1 - factor_new) * var_est[t-1] + factor_new * diff**2

        std = np.maximum(np.sqrt(var_est), eps)
        result[ch] = ((x - mean_est) / std).astype(np.float32)

    return result


# ──────────────────────────────────────────────────────────────────────
# 被试级 z-score 标准化（F7_zscore 实验专用）
# ──────────────────────────────────────────────────────────────────────

def apply_zscore_per_subject(subject_data: dict) -> dict:
    """
    LOSO 协议下对所有被试逐一独立做被试级 z-score 标准化。

    每名被试用自身全部 trial（T+E 合并）按通道计算均值和标准差，
    再用这组固定的 (mean, std) 对该被试所有 trial 做标准化。

    粒度与 EA 一致（逐被试一套变换，不随 trial 变化，仅用被试自身数据，
    不引入跨被试信息泄露），用于拆分 EA 对低频端到端任务的贡献中，
    "尺度归一化"与"协方差白化/旋转"两种效应各自的占比（F7_zscore 实验）。

    与 exponential_moving_standardize 的区别：后者是逐通道、随时间滑动的
    在线估计，用于连续信号；本函数是基于已切割 trial 的、被试级别的
    一次性统计量，只做尺度归一化，不做空间白化/旋转。

    Parameters
    ----------
    subject_data : dict
        {
            subj_id: {
                'T': (X_T, y_T),   # Session T 的 (trials, labels)
                'E': (X_E, y_E),   # Session E 的 (trials, labels)
            },
            ...
        }

    Returns
    -------
    standardized_data : dict
        结构与输入相同，X 已替换为标准化后的版本，y 不变
    """
    standardized_data = {}
    eps = 1e-8

    for subj_id, sessions in subject_data.items():
        X_T, y_T = sessions['T']
        X_E, y_E = sessions['E']

        # 用 T+E 合并计算该被试逐通道的均值/标准差（沿 trial 和时间两个轴）
        X_all = np.concatenate([X_T, X_E], axis=0)          # (N, C, T)
        mean  = X_all.mean(axis=(0, 2), keepdims=True)       # (1, C, 1)
        std   = X_all.std(axis=(0, 2), keepdims=True)
        std   = np.maximum(std, eps)

        X_T_z = ((X_T - mean) / std).astype(np.float32)
        X_E_z = ((X_E - mean) / std).astype(np.float32)

        standardized_data[subj_id] = {
            'T': (X_T_z, y_T),
            'E': (X_E_z, y_E),
        }

    return standardized_data
