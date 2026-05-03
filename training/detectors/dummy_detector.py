"""
dummy_detector.py — DeepfakeBench 最小 Custom Detector 範例
============================================================
作者: Nia（Framework & Integration）
日期: 2026-05-03
目的: 驗證 DeepfakeBench 的「新增 custom detector 最小工作步驟」
      並示範如何讓 DeepfakeBench detector 輸出符合 Week 11 data contract 的格式。

[B-1 Checklist 對應]
✓ 新增 custom detector 的最小工作步驟驗證
✓ 輸出格式與 BaseDetector schema 對齊
✓ 可作為 P2（真實 Xception detector）的改寫模板

[與 AbstractDetector 的關係]
DeepfakeBench 的 AbstractDetector（training/detectors/base_detector.py）
是 PyTorch nn.Module，定義了 training pipeline 所需的 7 個方法。
本檔案的 DummyDetector 繼承它，並額外實作 predict_with_timing() 方法，
將 forward() 的輸出轉換為 Week 11 統一的 DetectorOutput 格式。

[使用方式]
1. 將此檔案放到 training/detectors/dummy_detector.py
2. 在 training/detectors/__init__.py 加入：
       from .dummy_detector import DummyDetector
3. 建立 training/config/detector/dummy.yaml（見文末的 YAML 範例）
4. 執行：
       python training/train.py --detector_path training/config/detector/dummy.yaml
"""

import time
import random
import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────
# 導入 DeepfakeBench 的 abstract class
# 實際部署時取消下方的 try/except，直接 import
# ──────────────────────────────────────────
try:
    from detectors.base_detector import AbstractDetector
    from detectors import DETECTOR
    IN_DEEPFAKEBENCH = True
except ImportError:
    # 獨立執行時的 stub（用於本地測試）
    import abc
    class AbstractDetector(nn.Module, metaclass=abc.ABCMeta):
        def __init__(self): super().__init__()
        @abc.abstractmethod
        def features(self, data_dict): pass
        @abc.abstractmethod
        def forward(self, data_dict, inference=False): pass
        @abc.abstractmethod
        def classifier(self, features): pass
        @abc.abstractmethod
        def build_backbone(self, config): pass
        @abc.abstractmethod
        def build_loss(self, config): pass
        @abc.abstractmethod
        def get_losses(self, data_dict, pred_dict): pass
        @abc.abstractmethod
        def get_train_metrics(self, data_dict, pred_dict): pass

    class _DummyRegistry:
        def register_module(self, module_name=None):
            def decorator(cls): return cls
            return decorator
    DETECTOR = _DummyRegistry()
    IN_DEEPFAKEBENCH = False

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────
# §1. 最小 Backbone（3 層 Conv + FC）
#     真實使用時換成 Xception / EfficientNet-B4
# ──────────────────────────────────────────

class MinimalBackbone(nn.Module):
    """
    用於驗證流程的極小 backbone，不需要 pretrained weights。
    輸入: (B, 3, 224, 224) 的 RGB 圖像
    輸出: (B, 2) 的 logit
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # → (16, 112, 112)
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # → (32, 56, 56)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # → (64, 28, 28)
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),               # → (64, 1, 1)
        )
        self.fc = nn.Linear(64, num_classes)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def classifier(self, feat: torch.Tensor) -> torch.Tensor:
        return self.fc(feat.view(feat.size(0), -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ──────────────────────────────────────────
# §2. DummyDetector（DeepfakeBench 格式）
# ──────────────────────────────────────────

@DETECTOR.register_module(module_name="dummy")
class DummyDetector(AbstractDetector):
    """
    最小可運行的 DeepfakeBench custom detector。

    目的:
      1. 驗證新增 detector 的完整流程（4 個 method + 1 個 YAML）
      2. 作為 B-1 checklist 第一條的驗證依據
      3. 示範如何橋接 DeepfakeBench 的 pred_dict 與 Week 11 的 DetectorOutput

    流程驗證點:
      - __init__ 呼叫 build_backbone + build_loss ✓
      - features() 回傳 feature tensor ✓
      - classifier() 回傳 logit ✓
      - forward() 回傳 pred_dict（含 cls, prob, feat）✓
      - get_losses() 回傳 loss_dict ✓
      - get_train_metrics() 回傳 metric_dict ✓
      - predict_with_timing() 輸出 Week 11 統一格式 ✓
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        if config is None:
            # 獨立測試時的預設 config（不需要 YAML 檔）
            config = {
                "num_classes": 2,
                "pretrained": False,
            }
        self.config = config
        self.backbone = self.build_backbone(config)
        self.loss_func = self.build_loss(config)
        logger.info(
            f"[DummyDetector] 初始化完成 | backbone: MinimalBackbone | "
            f"params: {sum(p.numel() for p in self.parameters()):,}"
        )

    # ── DeepfakeBench 必要方法 ──────────────────

    def build_backbone(self, config: Dict) -> nn.Module:
        """建立 backbone。真實模型換成 Xception / EfficientNet-B4。"""
        num_classes = config.get("num_classes", 2)
        return MinimalBackbone(num_classes=num_classes)

    def build_loss(self, config: Dict) -> nn.Module:
        """建立 loss function。此處使用最簡單的 CrossEntropyLoss。"""
        return nn.CrossEntropyLoss()

    def features(self, data_dict: Dict) -> torch.Tensor:
        """
        從 data_dict['image'] 提取特徵。
        data_dict['image'] shape: (B, 3, H, W)
        """
        return self.backbone.features(data_dict["image"])

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        """將特徵分類為 logit。"""
        return self.backbone.classifier(features)

    def get_losses(self, data_dict: Dict, pred_dict: Dict) -> Dict:
        """計算 loss，回傳 loss_dict（含 'overall' key）。"""
        label = data_dict["label"]
        pred = pred_dict["cls"]
        loss = self.loss_func(pred, label)
        return {"overall": loss, "cls": loss}

    def get_train_metrics(self, data_dict: Dict, pred_dict: Dict) -> Dict:
        """計算 training metrics（batch level）。"""
        label = data_dict["label"]
        pred = pred_dict["cls"]
        prob = pred_dict["prob"]

        # 計算 batch accuracy
        predicted_class = torch.argmax(pred, dim=1)
        correct = (predicted_class == label).float().mean().item()

        return {"acc": correct}

    def forward(self, data_dict: Dict, inference: bool = False) -> Dict:
        """
        DeepfakeBench 標準 forward pass。
        回傳 pred_dict = {cls: logit, prob: softmax[:,1], feat: feature}
        """
        feat = self.features(data_dict)
        logit = self.classifier(feat)
        prob = torch.softmax(logit, dim=1)[:, 1]  # 取 fake class 的機率
        return {"cls": logit, "prob": prob, "feat": feat}

    # ── Week 11 橋接方法 ────────────────────────

    def predict_with_timing(
        self,
        sample_id: str,
        image_tensor: torch.Tensor,
        device: str = "cpu",
    ) -> Dict:
        """
        橋接方法：將 DeepfakeBench 的 forward() 輸出轉換為 Week 11 統一格式。
        輸出格式對應 DetectorOutput schema（§3.4 of Week 11 規格書）。

        Args:
            sample_id    : metadata.csv 中的唯一識別碼
            image_tensor : (1, 3, H, W) 的 RGB 圖像 tensor
            device       : 推論裝置

        Returns:
            dict — 對應 DetectorOutput 的欄位（可直接傳入 export_to_score_csv）
        """
        t_start = time.perf_counter()

        try:
            self.eval()
            image_tensor = image_tensor.to(device)
            data_dict = {"image": image_tensor, "label": torch.zeros(1, dtype=torch.long)}

            with torch.no_grad():
                pred_dict = self.forward(data_dict, inference=True)

            fake_score = float(pred_dict["prob"][0].cpu())
            elapsed_ms = (time.perf_counter() - t_start) * 1000

            return {
                "sample_id": sample_id,
                "detector_name": "DummyDetector",
                "modality": "visual",
                "fake_score": round(fake_score, 4),
                "score_type": "probability",
                "confidence": round(abs(fake_score - 0.5) * 2, 4),
                "inference_time_ms": round(elapsed_ms, 3),
                "window_start_sec": None,
                "window_end_sec": None,
                "status": "ok",
                "error_message": None,
            }

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                error_msg = "cuda_oom"
            else:
                error_msg = str(e)
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            return {
                "sample_id": sample_id,
                "detector_name": "DummyDetector",
                "modality": "visual",
                "fake_score": None,
                "score_type": "probability",
                "confidence": None,
                "inference_time_ms": round(elapsed_ms, 3),
                "window_start_sec": None,
                "window_end_sec": None,
                "status": "failed",
                "error_message": error_msg,
            }

        except Exception as e:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            return {
                "sample_id": sample_id,
                "detector_name": "DummyDetector",
                "modality": "visual",
                "fake_score": None,
                "score_type": "probability",
                "confidence": None,
                "inference_time_ms": round(elapsed_ms, 3),
                "window_start_sec": None,
                "window_end_sec": None,
                "status": "failed",
                "error_message": str(e),
            }


# ──────────────────────────────────────────
# §3. YAML 範例（放到 training/config/detector/dummy.yaml）
# ──────────────────────────────────────────

DUMMY_YAML_EXAMPLE = """
# training/config/detector/dummy.yaml
# 用於驗證 DeepfakeBench 新增 custom detector 流程

# 基本設定
pretrained: false
model_name: dummy
backbone_name: null

# 資料集（使用小樣本驗證）
train_dataset: [FaceForensics++]
test_dataset: [FaceForensics++]
compression: c23

# 訓練參數（dummy 用極小值加速驗證）
train_batchSize: 4
test_batchSize: 4
workers: 2
frame_num: {train: 4, test: 4}
resolution: 224
with_mask: false
with_landmark: false

# Loss
loss_func: cross_entropy

# Optimizer
optimizer:
  type: adam
  adam:
    lr: 0.0002
    beta1: 0.9
    beta2: 0.999
    eps: 0.00000001
    weight_decay: 0.0005
    amsgrad: false

# 訓練 epoch（驗證用只跑 1 epoch）
nEpochs: 1
metric_scoring: auc
cuda: true
cudnn: true
save_ckpt: false
save_feat: false
"""


# ──────────────────────────────────────────
# §4. 獨立 Self-Test（不需要 DeepfakeBench 環境）
# ──────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("DummyDetector — DeepfakeBench Custom Detector Verification")
    print("=" * 60)

    if not IN_DEEPFAKEBENCH:
        print("[NOTE] 在 DeepfakeBench 環境外執行，使用 stub AbstractDetector\n")

    # 建立 detector
    detector = DummyDetector()

    # 建立假的 batch（模擬 DeepfakeBench dataloader 的 data_dict）
    batch_size = 4
    fake_images = torch.randn(batch_size, 3, 224, 224)
    fake_labels = torch.randint(0, 2, (batch_size,))
    data_dict = {
        "image": fake_images,
        "label": fake_labels,
        "mask": None,
        "landmark": None,
    }

    print(f"[Test 1] forward() — batch_size={batch_size}")
    pred_dict = detector.forward(data_dict)
    assert "cls" in pred_dict, "pred_dict 缺少 'cls' key"
    assert "prob" in pred_dict, "pred_dict 缺少 'prob' key"
    assert pred_dict["prob"].shape == (batch_size,), f"prob shape 錯誤: {pred_dict['prob'].shape}"
    assert all(0 <= p <= 1 for p in pred_dict["prob"]), "prob 超出 [0,1] 範圍"
    print(f"  ✓ pred shape: cls={pred_dict['cls'].shape}, prob={pred_dict['prob'].tolist()}")

    print(f"\n[Test 2] get_losses()")
    losses = detector.get_losses(data_dict, pred_dict)
    assert "overall" in losses, "loss_dict 缺少 'overall' key"
    print(f"  ✓ overall loss = {losses['overall'].item():.4f}")

    print(f"\n[Test 3] get_train_metrics()")
    metrics = detector.get_train_metrics(data_dict, pred_dict)
    assert "acc" in metrics, "metric_dict 缺少 'acc' key"
    print(f"  ✓ batch acc = {metrics['acc']:.4f}")

    print(f"\n[Test 4] predict_with_timing() — Week 11 output format")
    single_image = torch.randn(1, 3, 224, 224)
    result = detector.predict_with_timing("ffpp_0001", single_image)

    required_keys = [
        "sample_id", "detector_name", "modality", "fake_score",
        "score_type", "inference_time_ms", "status"
    ]
    for k in required_keys:
        assert k in result, f"output 缺少 '{k}' key"

    assert result["status"] == "ok", f"status 不是 ok: {result['status']}"
    assert 0 <= result["fake_score"] <= 1, f"fake_score 超出範圍: {result['fake_score']}"
    print(f"  ✓ output: {result}")

    print(f"\n[Test 5] YAML 範例列印")
    print("  提示：複製以下內容到 training/config/detector/dummy.yaml")
    for line in DUMMY_YAML_EXAMPLE.strip().split("\n")[:10]:
        print(f"  {line}")
    print("  ...")

    print(f"\n{'=' * 60}")
    print("所有 Test 通過 ✓")
    print("下一步：")
    print("  1. 將此檔案複製到 training/detectors/dummy_detector.py")
    print("  2. 在 training/detectors/__init__.py 加入 import")
    print("  3. 建立 training/config/detector/dummy.yaml（見 DUMMY_YAML_EXAMPLE）")
    print("  4. 執行: python training/train.py --detector_path ./training/config/detector/dummy.yaml")
    print(f"{'=' * 60}")
