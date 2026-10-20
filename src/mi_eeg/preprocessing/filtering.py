"""带通滤波模块

提供 Butterworth 零相位带通滤波（filtfilt），
应用于连续 EEG 信号（在 trial 切割之前）以避免边缘效应。
"""

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(
    data: np.ndarray,
    low: float,
    high: float,
    sfreq: float = 250.0,
    order: int = 4,
) -> np.ndarray:
    """
    对 EEG 信号进行 Butterworth 零相位带通滤波。

    Parameters
    ----------
    data : np.ndarray
        输入 EEG 信号，shape 为 (N_channels, N_timepoints) 或 (N_timepoints,)
    low : float
        低截止频率（Hz）
    high : float
        高截止频率（Hz）
    sfreq : float
        采样率（Hz），默认 250
    order : int
        Butterworth 滤波器阶数，默认 4

    Returns
    -------
    np.ndarray
        滤波后的信号，与输入 shape 相同，dtype 保持 float32

    Notes
    -----
    - 使用 filtfilt（零相位）避免相位畸变
    - 建议在 trial 切割之前对连续信号滤波，以避免 epoch 边缘效应
    - 对于 EEGNet，也可选择跳过此步骤（EEGNet 的时间卷积层可自学习频率滤波）
    """
    nyq = 0.5 * sfreq
    low_norm  = low  / nyq
    high_norm = high / nyq

    # 确保截止频率在有效范围内
    low_norm  = np.clip(low_norm,  1e-6, 1.0 - 1e-6)
    high_norm = np.clip(high_norm, 1e-6, 1.0 - 1e-6)

    b, a = butter(order, [low_norm, high_norm], btype='bandpass')

    was_1d = (data.ndim == 1)
    if was_1d:
        data = data[np.newaxis, :]  # (1, T)

    filtered = np.array([filtfilt(b, a, ch) for ch in data], dtype=np.float32)

    if was_1d:
        filtered = filtered[0]

    return filtered


def notch_filter(
    data: np.ndarray,
    freq: float = 50.0,
    sfreq: float = 250.0,
    quality: float = 30.0,
) -> np.ndarray:
    """
    陷波滤波，用于去除工频干扰（50Hz 或 60Hz）。

    Parameters
    ----------
    data : (N_channels, N_timepoints)
    freq : 陷波频率（Hz）
    sfreq : 采样率（Hz）
    quality : 品质因数，越大陷波带越窄

    Returns
    -------
    np.ndarray, same shape as input
    """
    from scipy.signal import iirnotch
    b, a = iirnotch(freq / (0.5 * sfreq), quality)
    return np.array([filtfilt(b, a, ch) for ch in data], dtype=np.float32)
