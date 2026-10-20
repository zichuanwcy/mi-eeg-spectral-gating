"""训练器模块

封装 PyTorch 模型的完整训练流程，包括：
  - CosineAnnealingLR 学习率调度
  - 两种早停策略（通过 es_strategy 参数控制）：
      "fixed_epoch" : 跑满 max_epochs，不早停
      "test_set"    : 用测试集全量数据监控 acc 早停，Patience=150
  - MaxNorm 约束（每个 optimizer step 后自动调用）
  - 最优模型权重保存与恢复
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mi_eeg.data.datasets import EEGDataset
from mi_eeg.models.eegnet import apply_max_norm


class EarlyStopping:
    """监控准确率，patience 个 epoch 内无提升则触发停止。"""

    def __init__(self, patience: int = 150):
        self.patience    = patience
        self.counter     = 0
        self.best_acc    = -1.0
        self.should_stop = False

    def step(self, val_acc: float) -> bool:
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.counter  = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    def reset(self):
        self.counter     = 0
        self.best_acc    = -1.0
        self.should_stop = False


class Trainer:
    """
    EEG 分类模型训练器。

    Parameters
    ----------
    es_strategy : str
        早停策略，二选一：
        - "fixed_epoch" : 跑满 max_epochs，不早停，保存最后一个 epoch 权重
        - "test_set"    : 用测试集监控 acc 早停（Patience=150），保存最优权重
    """

    def __init__(
        self,
        model:        nn.Module,
        device:       str   = 'cpu',
        lr:           float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs:   int   = 600,
        patience_es:  int   = 150,
        batch_size:   int   = 64,
        use_max_norm: bool  = True,
        verbose:      bool  = False,
        es_strategy:  str   = "test_set",
        mixup_alpha:  float = 0.0,
        n_classes:    int   = 4,
    ):
        if es_strategy not in ("fixed_epoch", "test_set"):
            raise ValueError(
                f"es_strategy 须为 'fixed_epoch' 或 'test_set'，"
                f"当前传入: '{es_strategy}'"
            )

        self.model        = model.to(device)
        self.device       = torch.device(device)
        self.max_epochs   = max_epochs
        self.batch_size   = batch_size
        self.use_max_norm = use_max_norm
        self.verbose      = verbose
        self.es_strategy  = es_strategy
        self.mixup_alpha  = mixup_alpha
        self.n_classes    = n_classes

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs
        )
        self.early_stopping = EarlyStopping(patience=patience_es)

        self._best_state   = None
        self._best_val_acc = -1.0
        self.history       = {'train_loss': [], 'val_acc': []}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test:  np.ndarray = None,
        y_test:  np.ndarray = None,
        subject_ids_train: np.ndarray = None,
    ) -> 'Trainer':
        """
        训练模型。

        Parameters
        ----------
        X_train, y_train          : 训练集，必须传入
        X_test,  y_test           : 测试集，es_strategy="test_set" 时必须传入
        subject_ids_train         : 每条训练 trial 的被试编号数组，
                                    mixup_alpha > 0 时必须传入，用于强制跨被试采样
        """
        if self.es_strategy == "test_set":
            if X_test is None or y_test is None:
                raise ValueError("es_strategy='test_set' 时须传入 X_test 和 y_test")

        if self.mixup_alpha > 0 and subject_ids_train is None:
            raise ValueError("mixup_alpha > 0 时须传入 subject_ids_train")

        use_mixup = self.mixup_alpha > 0

        if use_mixup:
            from mi_eeg.data.datasets import EEGDatasetWithSubject
            train_dataset = EEGDatasetWithSubject(X_train, y_train,
                                                  subject_ids_train)
        else:
            train_dataset = EEGDataset(X_train, y_train)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=(torch.device(self.device).type == 'cuda'),
        )

        monitor_loader = None
        if self.es_strategy == "test_set":
            monitor_loader = DataLoader(
                EEGDataset(X_test, y_test),
                batch_size=self.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
            )

        # 重置状态
        self.early_stopping.reset()
        self._best_state   = None
        self._best_val_acc = -1.0
        self.history       = {'train_loss': [], 'val_acc': []}

        for epoch in range(self.max_epochs):
            if use_mixup:
                train_loss = self._train_one_epoch_mixup(train_loader)
            else:
                train_loss = self._train_one_epoch(train_loader)
            self.scheduler.step()
            self.history['train_loss'].append(train_loss)

            if self.es_strategy == "fixed_epoch":
                self.history['val_acc'].append(None)
                if self.verbose and (epoch + 1) % 100 == 0:
                    print(f"  Epoch {epoch+1:3d}/{self.max_epochs} | "
                          f"loss={train_loss:.4f}")

            else:  # test_set
                monitor_acc = self._evaluate_acc(monitor_loader)
                self.history['val_acc'].append(monitor_acc)

                if monitor_acc > self._best_val_acc:
                    self._best_val_acc = monitor_acc
                    self._best_state   = copy.deepcopy(self.model.state_dict())

                if self.verbose and (epoch + 1) % 100 == 0:
                    print(f"  Epoch {epoch+1:3d}/{self.max_epochs} | "
                          f"loss={train_loss:.4f} | test_acc={monitor_acc:.4f}")

                if self.early_stopping.step(monitor_acc):
                    if self.verbose:
                        print(f"  Early stop @ epoch {epoch+1}, "
                              f"best_acc={self._best_val_acc:.4f}")
                    break

        # 加载最优权重
        if self.es_strategy == "fixed_epoch":
            self._best_state = copy.deepcopy(self.model.state_dict())
        else:
            if self._best_state is not None:
                self.model.load_state_dict(self._best_state)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        loader = DataLoader(
            EEGDataset(X, np.zeros(len(X), dtype=np.int64)),
            batch_size=self.batch_size, shuffle=False,
            drop_last=False, num_workers=0,
        )
        self.model.eval()
        preds = []
        with torch.no_grad():
            for X_batch, _ in loader:
                preds.extend(
                    self.model(X_batch.to(self.device)).argmax(1).cpu().numpy()
                )
        return np.array(preds, dtype=np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch.nn.functional as F
        loader = DataLoader(
            EEGDataset(X, np.zeros(len(X), dtype=np.int64)),
            batch_size=self.batch_size, shuffle=False,
            drop_last=False, num_workers=0,
        )
        self.model.eval()
        probas = []
        with torch.no_grad():
            for X_batch, _ in loader:
                logits = self.model(X_batch.to(self.device))
                probas.append(F.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probas, axis=0).astype(np.float32)

    def get_features(self, X: np.ndarray) -> np.ndarray:
        """
        提取分类器前一层的特征向量（16维）。
        用于 t-SNE 可视化、协方差结构分析等。
        """
        loader = DataLoader(
            EEGDataset(X, np.zeros(len(X), dtype=np.int64)),
            batch_size=self.batch_size, shuffle=False,
            drop_last=False, num_workers=0,
        )
        self.model.eval()
        features = []
        with torch.no_grad():
            for X_batch, _ in loader:
                feat = self.model._forward_features(X_batch.to(self.device))
                features.append(feat.cpu().numpy())
        return np.concatenate(features, axis=0).astype(np.float32)

    def get_temporal_conv_weights(self) -> np.ndarray:
        """
        提取 EEGNet 首层 Temporal Convolution 的滤波器权重。
        用于实验 1-A 的频域响应分析。

        Returns
        -------
        weights : (F1, kern_length), float32
            F1 个滤波器，每个滤波器的时域权重
        """
        w = self.model.temporal_conv.weight  # (F1, 1, 1, kern_length)
        return w.squeeze().detach().cpu().numpy().astype(np.float32)

    def get_best_val_acc(self) -> float:
        return self._best_val_acc

    def fit_dual(
        self,
        X_low_train:  np.ndarray,
        X_high_train: np.ndarray,
        y_train:      np.ndarray,
        X_low_test:   np.ndarray = None,
        X_high_test:  np.ndarray = None,
        y_test:       np.ndarray = None,
    ) -> 'Trainer':
        """
        双路模型训练入口。

        模型 forward 签名须为 forward(x_low, x_high) → logits。
        除输入为两路外，训练流程与 fit() 完全一致。
        """
        if self.es_strategy == "test_set":
            if X_low_test is None or X_high_test is None or y_test is None:
                raise ValueError(
                    "es_strategy='test_set' 时须传入 X_low_test / X_high_test / y_test"
                )

        from mi_eeg.data.datasets import EEGDataset

        train_loader = DataLoader(
            _DualEEGDataset(X_low_train, X_high_train, y_train),
            batch_size=self.batch_size, shuffle=True,
            drop_last=True, num_workers=0,
            pin_memory=(torch.device(self.device).type == 'cuda'),
        )

        monitor_loader = None
        if self.es_strategy == "test_set":
            monitor_loader = DataLoader(
                _DualEEGDataset(X_low_test, X_high_test, y_test),
                batch_size=self.batch_size, shuffle=False,
                drop_last=False, num_workers=0,
            )

        self.early_stopping.reset()
        self._best_state   = None
        self._best_val_acc = -1.0
        self.history       = {'train_loss': [], 'val_acc': []}

        for epoch in range(self.max_epochs):
            train_loss = self._train_one_epoch_dual(train_loader)
            self.scheduler.step()
            self.history['train_loss'].append(train_loss)

            if self.es_strategy == "fixed_epoch":
                self.history['val_acc'].append(None)
            else:
                monitor_acc = self._evaluate_acc_dual(monitor_loader)
                self.history['val_acc'].append(monitor_acc)

                if monitor_acc > self._best_val_acc:
                    self._best_val_acc = monitor_acc
                    self._best_state   = copy.deepcopy(self.model.state_dict())

                if self.early_stopping.step(monitor_acc):
                    break

        if self.es_strategy == "fixed_epoch":
            self._best_state = copy.deepcopy(self.model.state_dict())
        else:
            if self._best_state is not None:
                self.model.load_state_dict(self._best_state)

        return self

    def predict_proba_dual(
        self,
        X_low:  np.ndarray,
        X_high: np.ndarray,
    ) -> np.ndarray:
        """双路模型推理，返回 softmax 概率，shape (N, n_classes)。"""
        import torch.nn.functional as F

        loader = DataLoader(
            _DualEEGDataset(X_low, X_high,
                            np.zeros(len(X_low), dtype=np.int64)),
            batch_size=self.batch_size, shuffle=False,
            drop_last=False, num_workers=0,
        )
        self.model.eval()
        probas = []
        with torch.no_grad():
            for X_l, X_h, _ in loader:
                logits = self.model(X_l.to(self.device), X_h.to(self.device))
                probas.append(F.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(probas, axis=0).astype(np.float32)

    # ── 私有方法 ──────────────────────────────────────────────

    def _train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_samples  = 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(X_batch), y_batch)
            loss.backward()
            self.optimizer.step()
            if self.use_max_norm:
                apply_max_norm(self.model)
            total_loss += loss.item() * len(y_batch)
            n_samples  += len(y_batch)
        return total_loss / max(n_samples, 1)

    def _train_one_epoch_mixup(self, loader: DataLoader) -> float:
        """跨被试 MixUp 训练单 epoch。DataLoader 须包含 subject_id 字段。"""
        from mi_eeg.training.augmentation import apply_mixup_to_batch
        self.model.train()
        total_loss = 0.0
        n_samples  = 0
        for X_batch, y_batch, subj_batch in loader:
            X_batch    = X_batch.to(self.device)
            y_batch    = y_batch.to(self.device)
            subj_batch = subj_batch.to(self.device)

            X_mixed, y_soft = apply_mixup_to_batch(
                X_batch, y_batch, subj_batch,
                self.mixup_alpha, self.n_classes,
            )

            self.optimizer.zero_grad()
            logits = self.model(X_mixed)
            loss   = self._mixup_criterion(logits, y_soft)
            loss.backward()
            self.optimizer.step()
            if self.use_max_norm:
                apply_max_norm(self.model)
            total_loss += loss.item() * len(y_batch)
            n_samples  += len(y_batch)
        return total_loss / max(n_samples, 1)

    @staticmethod
    def _mixup_criterion(
        logits:  torch.Tensor,
        y_soft:  torch.Tensor,
    ) -> torch.Tensor:
        """
        支持软标签的交叉熵损失。

        CrossEntropyLoss 的标准实现只接受硬标签，MixUp 使用软标签须手动计算：
            L = -sum(y_soft * log_softmax(logits), dim=1).mean()
        这与硬标签的 CrossEntropyLoss 在 y_soft 为 one-hot 时等价。
        """
        import torch.nn.functional as F
        log_p = F.log_softmax(logits, dim=1)      # (B, n_classes)
        return -(y_soft * log_p).sum(dim=1).mean()

    def _evaluate_acc(self, loader: DataLoader) -> float:
        self.model.eval()
        correct   = 0
        n_samples = 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                preds    = self.model(X_batch).argmax(dim=1)
                correct  += (preds == y_batch).sum().item()
                n_samples += len(y_batch)
        return correct / max(n_samples, 1)

    def _train_one_epoch_dual(self, loader: DataLoader) -> float:
        """双路模型单 epoch 训练。"""
        self.model.train()
        total_loss = 0.0
        n_samples  = 0
        for X_l, X_h, y_batch in loader:
            X_l     = X_l.to(self.device)
            X_h     = X_h.to(self.device)
            y_batch = y_batch.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(X_l, X_h), y_batch)
            loss.backward()
            self.optimizer.step()
            if self.use_max_norm:
                apply_max_norm(self.model)
            total_loss += loss.item() * len(y_batch)
            n_samples  += len(y_batch)
        return total_loss / max(n_samples, 1)

    def _evaluate_acc_dual(self, loader: DataLoader) -> float:
        """双路模型准确率评估。"""
        self.model.eval()
        correct   = 0
        n_samples = 0
        with torch.no_grad():
            for X_l, X_h, y_batch in loader:
                X_l     = X_l.to(self.device)
                X_h     = X_h.to(self.device)
                y_batch = y_batch.to(self.device)
                preds    = self.model(X_l, X_h).argmax(dim=1)
                correct  += (preds == y_batch).sum().item()
                n_samples += len(y_batch)
        return correct / max(n_samples, 1)


class _DualEEGDataset(torch.utils.data.Dataset):
    """
    双路 EEG Dataset，同时持有低频路和高频路数据。

    __getitem__ 返回 (x_low, x_high, y)，
    其中 x_low / x_high shape 为 (1, C, T)。
    """

    def __init__(
        self,
        X_low:  np.ndarray,
        X_high: np.ndarray,
        y:      np.ndarray,
    ):
        X_low  = np.asarray(X_low,  dtype=np.float32)
        X_high = np.asarray(X_high, dtype=np.float32)
        y      = np.asarray(y,      dtype=np.int64)

        # (N, C, T) → (N, 1, C, T)
        if X_low.ndim == 3:
            X_low  = X_low[:, np.newaxis, :, :]
            X_high = X_high[:, np.newaxis, :, :]

        self.X_low  = torch.from_numpy(X_low)
        self.X_high = torch.from_numpy(X_high)
        self.y      = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X_low[idx], self.X_high[idx], self.y[idx]
