"""
Dummy Detector — DeepfakeBench 最小可執行整合範例 (B-1 驗證)

用途：
    驗證「新增 custom detector」的最小工作步驟：
        1. 繼承 AbstractDetector
        2. 以 @DETECTOR.register_module() 裝飾
        3. 實作 7 個抽象方法
        4. 提供 1 個 YAML 設定檔
        5. 在 __init__.py 新增 import

    此 Detector 不使用任何 pretrained 權重，不依賴外部模型，
    適合在無 GPU / 無資料集的環境下快速確認整合流程是否正確。

使用方式：
    python training/train.py \
        --detector_path training/config/detector/dummy.yaml \
        --phase test
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

from .base_detector import AbstractDetector
from detectors import DETECTOR
from loss import LOSSFUNC
from metrics.base_metrics_class import calculate_metrics_for_train

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='dummy')
class DummyDetector(AbstractDetector):
    """
    最小可執行的 DeepfakeBench Detector。

    架構：
        輸入圖像 (B, 3, H, W)
        → 全域平均池化 → (B, 3)
        → 線性層 → (B, 2) logits

    此架構刻意設計得非常簡單，目的是驗證整合介面，
    而非追求偵測效能。
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        # 簡單線性分類頭：3 (avg RGB) → 2 (real/fake)
        self.fc = nn.Linear(3, 2)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        self.loss_func = self.build_loss(config)

        # 供 Trainer 計算 video-level 指標用的累積列表
        self.prob: list = []
        self.label: list = []
        self.video_names: list = []
        self.correct: int = 0
        self.total: int = 0

    # ── 1. build_backbone ─────────────────────────────────────────────
    def build_backbone(self, config: dict):
        # DummyDetector 不使用獨立 backbone；回傳 None 即可
        return None

    # ── 2. build_loss ─────────────────────────────────────────────────
    def build_loss(self, config: dict) -> nn.Module:
        loss_name = config.get('loss_func', 'cross_entropy')
        try:
            loss_cls = LOSSFUNC[loss_name]
            return loss_cls()
        except (KeyError, Exception):
            logger.warning("Loss function '%s' not found; fallback to CrossEntropyLoss.", loss_name)
            return nn.CrossEntropyLoss()

    # ── 3. features ───────────────────────────────────────────────────
    def features(self, data_dict: dict) -> torch.Tensor:
        """
        全域平均池化：(B, 3, H, W) → (B, 3)

        Parameters
        ----------
        data_dict : dict
            必須含有 'image' key，形狀 (B, 3, H, W)，值域 [-1, 1]
        """
        images = data_dict['image']          # (B, 3, H, W)
        # 全域平均池化：每幀三個 channel 的均值
        feat = images.mean(dim=[2, 3])       # (B, 3)
        return feat

    # ── 4. classifier ─────────────────────────────────────────────────
    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        """
        線性分類：(B, 3) → (B, 2) logits
        """
        return self.fc(features)

    # ── 5. forward ────────────────────────────────────────────────────
    def forward(self, data_dict: dict, inference: bool = False) -> dict:
        """
        完整前向傳播。

        Returns
        -------
        dict with keys:
            cls  : (B, 2) logits
            prob : (B,)   fake 機率（softmax 後 index=1 的值）
            feat : (B, 3) 中間特徵
        """
        feat = self.features(data_dict)
        pred = self.classifier(feat)
        prob = torch.softmax(pred, dim=1)[:, 1]   # fake class probability
        return {'cls': pred, 'prob': prob, 'feat': feat}

    # ── 6. get_losses ────────────────────────────────────────────────
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        """
        計算 cross-entropy 損失。

        Returns
        -------
        dict with keys:
            overall : scalar Tensor（trainer 用此做 backward）
            cls     : 同 overall（此 detector 無額外輔助損失）
        """
        label = data_dict['label']
        pred  = pred_dict['cls']
        loss  = self.loss_func(pred, label)
        return {'overall': loss, 'cls': loss}

    # ── 7. get_train_metrics ─────────────────────────────────────────
    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        """
        計算批次訓練指標（使用 DFB 內建 calculate_metrics_for_train）。

        Returns
        -------
        dict with keys: acc, auc, eer, ap
        """
        label = data_dict['label']
        pred  = pred_dict['cls']
        auc, eer, acc, ap = calculate_metrics_for_train(label.detach(), pred.detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
