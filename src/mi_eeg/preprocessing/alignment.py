"""欧几里得对齐（Euclidean Alignment, EA）模块

EA 通过将每个被试所有 trial 的协方差矩阵对齐到单位矩阵，
消除被试间的个体幅度差异，提升跨被试泛化能力。

关键约定：
  - EA 参考矩阵须基于单个被试自身的全部 trial 计算（逐被试独立计算）
  - 不得将多个被试的 trial 混合后计算统一参考矩阵（会退化为全局归一化，失去个体对齐意义）
  - LOSO 协议下：训练集每名被试独立对齐，测试被试同样独立对齐（无标签泄露）
  - CrossSession 协议下：用 Session T 计算参考矩阵，对 T 和 E 统一应用
"""

import numpy as np
import scipy.linalg


# ──────────────────────────────────────────────────────────────────────
# 核心工具函数（单被试）
# ──────────────────────────────────────────────────────────────────────

def compute_ea_transform(X: np.ndarray) -> np.ndarray:
    """
    计算单个被试的 EA 变换矩阵。

    基于该被试自身所有 trial 的协方差矩阵均值，
    计算使均值协方差趋向单位矩阵的变换矩阵。

    步骤：
      1. 计算每个 trial 的协方差矩阵（np.cov，自动去均值，除以 T-1）
      2. 计算均值协方差的矩阵平方根逆：R_inv_sqrt = (mean_cov)^{-1/2}

    Parameters
    ----------
    X : np.ndarray, shape (N_trials, N_channels, N_timepoints)
        单个被试的全部 trial（T+E 合并或单 Session 均可）

    Returns
    -------
    R_inv_sqrt : np.ndarray, shape (N_channels, N_channels), float32
        EA 变换矩阵，传入 apply_ea_transform 使用
    """
    n_trials, n_ch, n_tp = X.shape

    covs = np.zeros((n_trials, n_ch, n_ch), dtype=np.float64)
    for i in range(n_trials):
        x = X[i].astype(np.float64)  # (C, T)
        covs[i] = np.cov(x)          # 自动去均值，除以 T-1

    R_mean    = covs.mean(axis=0)                        # (C, C)
    R_sqrt    = scipy.linalg.sqrtm(R_mean).real          # 取实部避免数值虚部
    R_inv_sqrt = np.linalg.inv(R_sqrt)

    return R_inv_sqrt.astype(np.float32)


def apply_ea_transform(X: np.ndarray, R_inv_sqrt: np.ndarray) -> np.ndarray:
    """
    对 EEG 数据应用 EA 变换。

    变换：x_aligned = R_inv_sqrt @ x

    Parameters
    ----------
    X : np.ndarray, shape (N_trials, N_channels, N_timepoints)
    R_inv_sqrt : np.ndarray, shape (N_channels, N_channels)
        由 compute_ea_transform 计算得到的变换矩阵

    Returns
    -------
    np.ndarray, same shape as X, float32
    """
    X_aligned = np.einsum('ij,njk->nik', R_inv_sqrt, X.astype(np.float64))
    return X_aligned.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# LOSO 协议：逐被试独立对齐
# ──────────────────────────────────────────────────────────────────────

def apply_ea_per_subject(
    subject_data: dict,
) -> dict:
    """
    LOSO 协议下对所有被试逐一独立做 EA 对齐。

    每名被试用自身的全部 trial（T+E 合并）计算参考矩阵，
    并对自身数据应用变换。被试之间互不影响。

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
    aligned_data : dict
        结构与输入相同，X 已替换为 EA 对齐后的版本，y 不变
    """
    aligned_data = {}

    for subj_id, sessions in subject_data.items():
        X_T, y_T = sessions['T']
        X_E, y_E = sessions['E']

        # 用 T+E 合并计算该被试的参考矩阵
        X_all = np.concatenate([X_T, X_E], axis=0)
        R_inv_sqrt = compute_ea_transform(X_all)

        # 对 T 和 E 分别应用同一变换
        X_T_aligned = apply_ea_transform(X_T, R_inv_sqrt)
        X_E_aligned = apply_ea_transform(X_E, R_inv_sqrt)

        aligned_data[subj_id] = {
            'T': (X_T_aligned, y_T),
            'E': (X_E_aligned, y_E),
        }

    return aligned_data


# ──────────────────────────────────────────────────────────────────────
# CrossSession 协议：单被试 Session T → Session E 对齐
# ──────────────────────────────────────────────────────────────────────

def apply_ea_cross_session(
    X_T: np.ndarray,
    X_E: np.ndarray,
) -> tuple:
    """
    CrossSession 协议下对单个被试做 EA 对齐。

    用 Session T（训练集）计算参考矩阵，
    对 T 和 E 统一应用相同变换。

    Parameters
    ----------
    X_T : (N_trials_T, N_channels, N_timepoints)  Session T
    X_E : (N_trials_E, N_channels, N_timepoints)  Session E

    Returns
    -------
    X_T_aligned : float32, same shape as X_T
    X_E_aligned : float32, same shape as X_E
    """
    # 用 T+E 合并计算参考矩阵（与 LOSO 保持一致，利用全部可用数据）
    X_all = np.concatenate([X_T, X_E], axis=0)
    R_inv_sqrt = compute_ea_transform(X_all)

    X_T_aligned = apply_ea_transform(X_T, R_inv_sqrt)
    X_E_aligned = apply_ea_transform(X_E, R_inv_sqrt)

    return X_T_aligned, X_E_aligned


# ──────────────────────────────────────────────────────────────────────
# 兼容旧接口（保留，避免破坏已有依赖）
# ──────────────────────────────────────────────────────────────────────

def euclidean_alignment(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple:
    """
    旧版便捷接口（保留用于兼容）。

    ⚠️  注意：此接口用 X_train 计算参考矩阵后应用于 X_test，
    适用于单被试 CrossSession 场景。
    在 LOSO 多被试场景下请使用 apply_ea_per_subject()，
    避免将多被试 trial 混合传入此函数。

    Parameters
    ----------
    X_train : (N_train, N_channels, N_timepoints)
    X_test  : (N_test,  N_channels, N_timepoints)

    Returns
    -------
    X_train_ea, X_test_ea : float32
    """
    R_inv_sqrt = compute_ea_transform(X_train)
    return apply_ea_transform(X_train, R_inv_sqrt), apply_ea_transform(X_test, R_inv_sqrt)
