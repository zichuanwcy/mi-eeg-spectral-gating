"""BCI Competition IV-2a 数据加载模块

文件格式：
  - 每个 .mat 文件包含 9 个 run（data[0]~data[8]）
  - Run 0:   EOG 校准 run，无 trial
  - Run 1-2: 无 trial（过渡 run）
  - Run 3-8: 每个 MI run 含 48 个 trial，共 288 个 trial
  - trial 索引相对于各 run 起点，1-indexed（MATLAB 惯例）
  - X 原始格式为 (N_timepoints, N_channels)，需转置

重要设计决策：
  带通滤波在各 run 内部单独进行（在 run 拼接之前），
  避免 run 边界处的滤波振铃（ringing artifact）污染 trial。
"""

import os
import numpy as np
import scipy.io
from pathlib import Path
from typing import Tuple, Optional


def download_bci2a(save_dir: str = "data/raw") -> None:
    """
    通过 MOABB 下载 BCI Competition IV-2a 数据集并转存为 .mat 格式。

    下载后文件命名为 A0{subj}T.mat / A0{subj}E.mat，与原始竞赛格式一致。

    Parameters
    ----------
    save_dir : str
        数据保存目录
    """
    try:
        from moabb.datasets import BNCI2014_001
        import mne
        mne.set_log_level('WARNING')
    except ImportError:
        raise ImportError(
            "请先安装依赖：pip install moabb mne\n"
            "或手动从 https://www.bbci.de/competition/iv/ 下载数据"
        )

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"正在通过 MOABB 下载 BCI IV-2a 数据到 {save_path.resolve()} ...")
    dataset = BNCI2014_001()

    for subj in range(1, 10):
        for session_label, session_key in [('T', 'session_T'), ('E', 'session_E')]:
            out_file = save_path / f"A0{subj}{session_label}.mat"
            if out_file.exists():
                print(f"  A0{subj}{session_label}.mat 已存在，跳过")
                continue

            print(f"  下载 Subject {subj} Session {session_label} ...")
            try:
                data = dataset.get_data(subjects=[subj])
                # MOABB 返回结构：data[subj][session][run]
                subj_data = data[subj]

                # 找到对应 session
                session_keys = list(subj_data.keys())
                # session_T 对应 0T，session_E 对应 1T
                target_idx = 0 if session_label == 'T' else 1
                if target_idx >= len(session_keys):
                    print(f"    ⚠️  Subject {subj} 没有 Session {session_label}，跳过")
                    continue

                raw_session = subj_data[session_keys[target_idx]]

                # 将 MNE Raw 对象转存为 .mat
                _save_moabb_as_mat(raw_session, out_file, subj, session_label)
                print(f"    ✅ 已保存 {out_file.name}")

            except Exception as e:
                print(f"    ❌ Subject {subj} Session {session_label} 下载失败: {e}")

    print("下载完成。")


def _save_moabb_as_mat(raw_session: dict, out_file: Path, subj: int, session: str) -> None:
    """将 MOABB 返回的 session 数据转存为与竞赛格式兼容的 .mat 文件。"""
    import mne

    run_list = []
    for run_key, raw in raw_session.items():
        if not isinstance(raw, mne.io.BaseRaw):
            continue

        raw.load_data()
        data_np = raw.get_data()  # (n_channels, n_timepoints)
        ch_names = raw.ch_names
        sfreq = raw.info['sfreq']

        # 提取 events
        events, event_id = mne.events_from_annotations(raw, verbose=False)

        # 构建与竞赛格式兼容的 run 结构
        trial_starts = []
        trial_labels = []
        artifacts_arr = []

        mi_event_ids = {v: k for k, v in event_id.items() if 'imagery' in k.lower()
                        or k in ['1', '2', '3', '4', 769, 770, 771, 772]}

        for ev in events:
            if ev[2] in [769, 770, 771, 772]:
                trial_starts.append(ev[0] + 1)  # 1-indexed
                trial_labels.append(ev[2] - 768)  # 1-4
                artifacts_arr.append(0)

        run_dict = {
            'X': data_np.T.astype(np.float32),  # (T, C) 与竞赛格式一致
            'trial': np.array(trial_starts, dtype=np.int64),
            'y': np.array(trial_labels, dtype=np.int64),
            'artifacts': np.array(artifacts_arr, dtype=np.int64),
            'fs': sfreq,
        }
        run_list.append(run_dict)

    scipy.io.savemat(str(out_file), {'data': np.array(run_list, dtype=object)})


def load_bci2a_raw(
    filepath: str,
    n_eeg_channels: int = 22,
    bandpass_freq: Optional[Tuple[float, float]] = None,
    sfreq: float = 250.0,
    filter_order: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    加载 BCI IV-2a .mat 文件。

    各 run 单独滤波后再拼接，避免 run 边界振铃污染 trial。

    Parameters
    ----------
    filepath : str
    n_eeg_channels : int
        保留的 EEG 通道数，默认 22（丢弃索引 22-24 的 EOG）
    bandpass_freq : (low, high) or None
        带通滤波范围（Hz）。None = 不滤波
    sfreq : float
        采样率，默认 250 Hz
    filter_order : int
        Butterworth 阶数

    Returns
    -------
    raw       : (N_channels, N_total_timepoints), float32
    trials    : (N_trials,), int64  — 全局 trial 起始索引（0-indexed）
    labels    : (N_trials,), int64  — 0-indexed 类别标签（0-3）
    artifacts : (N_trials,), int64  — 伪迹标记（1=有伪迹）
    fs        : float
    """
    filepath = str(filepath)
    mat = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)

    if 'data' not in mat:
        raise ValueError(
            f"'{filepath}' 中未找到 'data' 键。\n"
            f"可用键: {[k for k in mat.keys() if not k.startswith('_')]}"
        )

    data_array = mat['data']
    if not (isinstance(data_array, np.ndarray) and data_array.dtype == object):
        raise ValueError(
            f"data 字段格式不符合预期（期望 object array）。\n"
            f"实际: {type(data_array)}, shape={getattr(data_array, 'shape', 'N/A')}"
        )

    if bandpass_freq is not None:
        from mi_eeg.preprocessing.filtering import bandpass_filter

    raw_segments  = []
    all_trials    = []
    all_labels    = []
    all_artifacts = []
    fs            = sfreq
    time_offset   = 0

    for run_idx in range(len(data_array)):
        run = data_array[run_idx]
        if not hasattr(run, 'X'):
            continue

        X = np.asarray(run.X, dtype=np.float32)
        if X.ndim == 2 and X.shape[0] > X.shape[1]:
            X = X.T  # (C, T)

        if run_idx == 0:
            try:
                fs = float(np.squeeze(np.asarray(run.fs)))
            except Exception:
                fs = sfreq

        if X.shape[0] > n_eeg_channels:
            X = X[:n_eeg_channels, :]

        # run 内部单独做带通滤波（避免 run 边界振铃）
        if bandpass_freq is not None:
            X = bandpass_filter(X, bandpass_freq[0], bandpass_freq[1], fs, filter_order)

        raw_segments.append(X)

        trial_arr = np.asarray(run.trial).flatten()
        if trial_arr.size == 0:
            time_offset += X.shape[1]
            continue

        trial_arr = trial_arr.astype(np.int64)
        y_arr     = np.asarray(run.y).flatten().astype(np.int64)

        if hasattr(run, 'artifacts') and np.asarray(run.artifacts).size == trial_arr.size:
            art_arr = np.asarray(run.artifacts).flatten().astype(np.int64)
        else:
            art_arr = np.zeros(len(trial_arr), dtype=np.int64)

        global_trials = (trial_arr - 1) + time_offset

        all_trials.append(global_trials)
        all_labels.append(y_arr)
        all_artifacts.append(art_arr)

        time_offset += X.shape[1]

    if not raw_segments:
        raise RuntimeError(f"'{filepath}' 中未找到任何有效的 EEG 数据。")

    raw_cat = np.concatenate(raw_segments, axis=1)  # (C, T_total)

    if not all_trials:
        raise RuntimeError(f"'{filepath}' 中所有 run 的 trial 均为空。")

    trials_cat    = np.concatenate(all_trials).astype(np.int64)
    labels_cat    = np.concatenate(all_labels).astype(np.int64)
    artifacts_cat = np.concatenate(all_artifacts).astype(np.int64)

    labels_cat = labels_cat - 1  # 1-indexed → 0-indexed

    return raw_cat, trials_cat, labels_cat, artifacts_cat, fs


def cut_trials(
    raw: np.ndarray,
    trials: np.ndarray,
    labels: np.ndarray,
    artifacts: np.ndarray,
    t_start: float = 2.0,
    t_end: float = 6.5,
    sfreq: float = 250.0,
    cue_onset_offset: float = 0.0,
    filter_artifacts: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从连续 EEG 中切割 trial epoch。

    BCI IV-2a 时间轴（相对 trial 起点）：
        0s ——fixation—— 2s ——cue onset—— 3.5s ——MI—— 6s ——休息—— ~8s

    切割窗口 [t_start, t_end] 相对于 trial 起点（秒）。

    Returns
    -------
    X : (N_valid_trials, N_channels, N_timepoints), float32
    y : (N_valid_trials,), int64
    """
    start_offset = int((cue_onset_offset + t_start) * sfreq)
    end_offset   = int((cue_onset_offset + t_end)   * sfreq)
    n_tp         = end_offset - start_offset

    X_list, y_list = [], []

    for i in range(len(trials)):
        if filter_artifacts and artifacts[i] == 1:
            continue

        start = int(trials[i]) + start_offset
        end   = int(trials[i]) + end_offset

        if start < 0 or end > raw.shape[1]:
            continue

        epoch = raw[:, start:end]
        if epoch.shape[1] != n_tp:
            continue

        X_list.append(epoch)
        y_list.append(labels[i])

    if len(X_list) == 0:
        raise RuntimeError(
            "切割后没有有效 trial。\n"
            f"  trial 数: {len(trials)}, raw shape: {raw.shape}\n"
            f"  start_offset={start_offset}, end_offset={end_offset}\n"
            "请检查时间窗口参数是否正确。"
        )

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y


def load_bci2a(
    filepath: str,
    t_start: float = 2.0,
    t_end: float = 6.5,
    sfreq: float = 250.0,
    n_eeg_channels: int = 22,
    bandpass_freq: Optional[Tuple[float, float]] = None,
    filter_artifacts: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """便捷接口：加载 → 滤波 → 切割，不含归一化与对齐。"""
    raw, trials, labels, artifacts, _ = load_bci2a_raw(
        filepath, n_eeg_channels, bandpass_freq, sfreq
    )
    return cut_trials(raw, trials, labels, artifacts,
                      t_start, t_end, sfreq, 0.0, filter_artifacts)
