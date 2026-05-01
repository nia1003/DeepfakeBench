# DeepfakeBench 整合技術備忘錄

**版本**：v1.0 · **日期**：2026-04-30 · **作者**：Nia（Framework & Integration 組）

---

## 一、DeepfakeBench 目錄結構

```
DeepfakeBench/
├── training/
│   ├── detectors/                  ← 所有偵測器實作
│   │   ├── base_detector.py        ← AbstractDetector 抽象基類（不可修改）
│   │   ├── xception_detector.py    ← 完整參考範例
│   │   ├── dummy_detector.py       ← [新增] 最小整合驗證範例
│   │   ├── adb_visual_detector.py  ← [新增] ADB 視覺偵測器適配器
│   │   ├── adb_rppg_detector.py    ← [新增] ADB rPPG 偵測器適配器
│   │   ├── adb_sync_detector.py    ← [新增] ADB 音視頻同步偵測器適配器
│   │   └── __init__.py             ← [修改] 新增 import 觸發 Registry 登記
│   ├── networks/                   ← Backbone 定義（Xception、EfficientNet 等）
│   ├── config/
│   │   └── detector/               ← 每個偵測器對應一個 YAML
│   │       ├── dummy.yaml          ← [新增]
│   │       ├── adb_visual.yaml     ← [新增]
│   │       ├── adb_rppg.yaml       ← [新增]
│   │       └── adb_sync.yaml       ← [新增]
│   ├── loss/                       ← 損失函數（LOSSFUNC registry）
│   ├── metrics/
│   │   ├── registry.py             ← DETECTOR / BACKBONE / LOSSFUNC 登記系統
│   │   └── base_metrics_class.py   ← calculate_metrics_for_train()
│   └── trainer/trainer.py          ← 訓練主迴圈
├── docs/
│   └── deepfakebench_integration.md  ← 本文件
└── preprocessing/                  ← 資料預處理腳本
```

---

## 二、Registry（登記系統）運作原理

`training/metrics/registry.py` 定義了一個輕量型 Registry：

```python
class Registry(object):
    def __init__(self):
        self.data = {}   # {module_name: class}

    def register_module(self, module_name=None):
        def _register(cls):
            self.data[module_name or cls.__name__] = cls
            return cls
        return _register

    def __getitem__(self, key):
        return self.data[key]

DETECTOR = Registry()   # 偵測器登記
BACKBONE = Registry()   # Backbone 登記
LOSSFUNC = Registry()   # 損失函數登記
```

**觸發時機**：只要 `training/detectors/__init__.py` 執行對應的 `from .xxx import XxxDetector`，裝飾器就會自動把 class 存入 `DETECTOR.data`。Trainer 讀取 YAML 中的 `model_name` 欄位，透過 `DETECTOR[config['model_name']](config)` 實例化對應模型。

---

## 三、AbstractDetector — 7 個必須實作的抽象方法

繼承自 `AbstractDetector`（`training/detectors/base_detector.py`），必須實作以下方法：

| # | 方法名稱 | 輸入 | 輸出 | 說明 |
|---|----------|------|------|------|
| 1 | `build_backbone(config)` | dict | `nn.Module` or `None` | 建立 backbone；若無獨立 backbone 可回傳 None |
| 2 | `build_loss(config)` | dict | `nn.Module` | 建立損失函數，通常 `LOSSFUNC[config['loss_func']]()` |
| 3 | `features(data_dict)` | `{'image': Tensor(B,3,H,W)}` | `Tensor` | 從輸入提取特徵 |
| 4 | `classifier(features)` | `Tensor` | `Tensor(B, 2)` | 特徵 → 2-class logits |
| 5 | `forward(data_dict, inference)` | dict | `{'cls', 'prob', 'feat'}` | 完整前向傳播 |
| 6 | `get_losses(data_dict, pred_dict)` | dict, dict | `{'overall', 'cls'}` | 計算訓練損失（`overall` 為 backward 目標） |
| 7 | `get_train_metrics(data_dict, pred_dict)` | dict, dict | `{'acc', 'auc', 'eer', 'ap'}` | 計算批次訓練指標 |

**`forward` 回傳格式規範**：
```python
return {
    'cls':  pred,   # Tensor(B, 2)  — 2-class logits
    'prob': prob,   # Tensor(B,)    — fake 機率（softmax[:,1]）
    'feat': feat,   # Tensor(B, D)  — 中間特徵
}
```

**`data_dict` 格式**：
```python
{
    'image':    Tensor(B, 3, H, W),   # 正規化圖像（mean=0.5, std=0.5）
    'label':    Tensor(B,),            # 0=real, 1=fake
    'vid_name': List[str],             # 影片名稱（測試時使用）
}
```

---

## 四、新增自訂 Detector 的最小步驟（已驗證）

### Step 1：建立 `training/detectors/my_detector.py`

```python
from detectors import DETECTOR
from .base_detector import AbstractDetector
from loss import LOSSFUNC
from metrics.base_metrics_class import calculate_metrics_for_train

@DETECTOR.register_module(module_name='my_model')
class MyDetector(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config   = config
        self.fc       = torch.nn.Linear(3, 2)
        self.loss_func = self.build_loss(config)
        self.prob, self.label = [], []
        self.correct, self.total = 0, 0

    def build_backbone(self, config): return None
    def build_loss(self, config):     return LOSSFUNC[config['loss_func']]()

    def features(self, data_dict):
        return data_dict['image'].mean(dim=[2, 3])     # (B, 3)

    def classifier(self, features):
        return self.fc(features)                        # (B, 2)

    def forward(self, data_dict, inference=False):
        feat = self.features(data_dict)
        pred = self.classifier(feat)
        prob = torch.softmax(pred, dim=1)[:, 1]
        return {'cls': pred, 'prob': prob, 'feat': feat}

    def get_losses(self, data_dict, pred_dict):
        loss = self.loss_func(pred_dict['cls'], data_dict['label'])
        return {'overall': loss, 'cls': loss}

    def get_train_metrics(self, data_dict, pred_dict):
        auc, eer, acc, ap = calculate_metrics_for_train(
            data_dict['label'].detach(), pred_dict['cls'].detach())
        return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}
```

### Step 2：在 `training/detectors/__init__.py` 末尾追加

```python
from .my_detector import MyDetector
```

### Step 3：建立 `training/config/detector/my_model.yaml`

```yaml
model_name: my_model
backbone_name: null
pretrained: null
train_dataset: [FaceForensics++]
test_dataset: [FaceForensics++]
compression: c23
train_batchSize: 8
test_batchSize: 8
frame_num: {train: 4, test: 4}
resolution: 256
with_mask: false
with_landmark: false
use_data_augmentation: false
mean: [0.5, 0.5, 0.5]
std: [0.5, 0.5, 0.5]
optimizer:
  type: adam
  adam: {lr: 0.001, beta1: 0.9, beta2: 0.999, eps: 1.0e-8, weight_decay: 0.0}
nEpochs: 2
save_epoch: 1
rec_iter: 10
manualSeed: 42
save_ckpt: false
save_feat: false
loss_func: cross_entropy
metric_scoring: auc
cuda: true
cudnn: true
log_dir: ./logs
logdir: ./logs
```

### Step 4：執行測試

```bash
cd /home/user/DeepfakeBench
python training/train.py \
    --detector_path training/config/detector/my_model.yaml \
    --phase test
```

> ✅ **驗證結果**：以 `dummy_detector.py` 實際跑通上述流程，確認 4 method + 1 YAML 的最小步驟正確。

---

## 五、ADB 三模態適配器安裝說明（B-2）

ADB（Anti-Deepfake-Box）適配器已放置於：
- `training/detectors/adb_visual_detector.py`（視覺紋理 XceptionNet）
- `training/detectors/adb_rppg_detector.py`（生理訊號 POS rPPG）
- `training/detectors/adb_sync_detector.py`（音視頻同步 LatentSync SyncNet）

**前置條件**：將 anti-deepfake-box 加入 Python 路徑

```bash
export PYTHONPATH=/path/to/anti-deepfake-box:$PYTHONPATH
```

**對應 YAML 設定**：
```
training/config/detector/adb_visual.yaml
training/config/detector/adb_rppg.yaml
training/config/detector/adb_sync.yaml
```

**執行各模態評估**：
```bash
python training/train.py --detector_path training/config/detector/adb_visual.yaml --phase test
python training/train.py --detector_path training/config/detector/adb_rppg.yaml   --phase test
python training/train.py --detector_path training/config/detector/adb_sync.yaml   --phase test
```

### 適配器架構設計（B-2 介面說明）

每個適配器的 `forward()` 統一輸出：

| Key | 型別 | 說明 |
|-----|------|------|
| `cls` | `Tensor(B, 2)` | 2-class logits（real / fake） |
| `prob` | `Tensor(B,)` | fake 機率（0 ~ 1） |
| `feat` | `Tensor(B, D)` | 中間特徵（供 DFB analysis 使用） |

DFB `data_dict['image']` → 適配器內部轉換為 `FaceTrack` → ADB 原生 detector → fake_score → 回包成 DFB 格式。

---

## 六、Video vs Image Detector 差異（rPPG 時序輸入）

| 面向 | Image Detector（Xception 等）| Video Detector（I3D、FTCN 等）|
|------|------------------------------|-------------------------------|
| 輸入格式 | `(B, 3, H, W)` — 單幀 | `(B, T, 3, H, W)` 或 `(B, 3, T, H, W)` |
| YAML `frame_num` | 整數或 dict | 通常需 `video_mode: true` |
| DataLoader | 圖像級別 batch | clip 級別 batch（`clip_size` 設定） |
| 時序建模 | 幀間獨立，結果 aggregate | 顯式時序建模（3D Conv / Transformer） |
| rPPG 相關性 | ❌ 無時序關係 | ✅ 可借用 video dataloader 機制 |

**rPPG 整合建議**：
ADB rPPG 適配器目前在 `features()` 中對 DFB batch 做逐幀處理後 aggregate；若要充分利用 DFB 的 video dataloader，可參考 `i3d_detector.py` 的 `clip_size` / `video_mode` 設定，要求 DataLoader 輸出連續幀序列。

---

## 七、音訊擴充工程估算（Audio Extension Engineering Estimate）

**目標**：在 DFB preprocessing pipeline 中新增 FFmpeg 音訊提取步驟，不破壞原有 JSON schema。

**現況**：DFB 的 `dataset/` 資料夾僅處理視覺輸入，無音訊欄位。

### 工作量估算

| 項目 | 工作量 | 說明 |
|------|--------|------|
| 修改 `dataset/` DataLoader 以支援音訊路徑欄位 | 1 天 | 新增 `audio_path` 欄位到 `data_dict`，設為 Optional |
| FFmpeg 音訊提取整合（subprocess 封裝） | 0.5 天 | 參考 ADB `preprocessing/audio_extractor.py` |
| 確保無音訊影片 graceful degradation（回傳 None） | 0.5 天 | SyncDetector 已實作此邏輯，可移植 |
| 更新 `adb_sync_detector.py` 讀取 `data_dict['audio_path']` | 0.5 天 | 目前適配器使用 ADB 內建音訊提取 |
| 測試（FakeAVCeleb，含音訊的資料集） | 1 天 | 需 FakeAVCeleb 資料集就緒 |
| **合計** | **~3.5 天** | 中等風險 |

### 風險點

1. **DFB DataLoader 並行**：DFB 使用多 worker DataLoader，音訊提取若在 `__getitem__` 中呼叫 FFmpeg subprocess，可能造成大量平行進程（建議預先提取，存快取）
2. **音訊與視訊幀對齊**：DFB sample 幀是隨機的（`random_crop_frames`），音訊必須對應到相同時間段（需傳入 `start_frame_idx`）
3. **FakeAVCeleb 資料格式**：部分影片音訊為 cloned audio（複製自真實影片），與偽造視覺不同步，是偵測重點也是資料清理重點

---

## 八、DF40 / FakeAVCeleb 整合工作量評估

| 資料集 | 整合工作量 | 主要挑戰 |
|--------|-----------|----------|
| DF40 | 中（2-3 天） | 需確認 DFB 的 `all_dataset` list 是否包含，以及 face crop 的 JSON 格式是否相同 |
| FakeAVCeleb | 高（3-5 天） | 需音訊支援（見上節）；申請流程 1-3 天；四個子類別（RAFV/FAFV/FARV/RAVF）標籤對應 |

---

## 九、已知 YAML Schema 注意事項

| 欄位 | 常見錯誤 | 正確寫法 |
|------|---------|---------|
| `frame_num` | 直接寫整數（部分 detector 期望 dict） | `frame_num: {train: 32, test: 32}` |
| `backbone_name: null` | DFB 部分 code 直接用 `BACKBONE[config['backbone_name']]`，null 會 KeyError | 確認 `build_backbone()` 有處理 null case |
| `pretrained: null` | `torch.load(None)` 會報錯 | `build_backbone()` 中加 `if config.get('pretrained'):` 判斷 |
| `loss_func` | 必須是 LOSSFUNC 中已登記的名稱 | `cross_entropy`（見 `loss/__init__.py`） |
| `metric_scoring` | 拼錯 key 名稱 | 僅接受 `auc`、`acc`、`eer`、`ap` 之一 |
