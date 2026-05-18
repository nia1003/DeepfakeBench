# 怎麼把你的程式碼放到正確位置

這份文件給所有想把 detector 整合進 DeepfakeBench 的人看。  
不用讀整個 codebase，只需要知道「你的東西放哪裡」。

---

## 一句話原則

**你的所有新程式碼只能放在這四個地方，其他地方不要動。**

| 你要做的事 | 放哪裡 |
|-----------|--------|
| 新增 detector 邏輯 | `training/detectors/你的名字_detector.py` |
| Detector 的訓練設定 | `training/config/detector/你的名字.yaml` |
| Inference 腳本 | `evaluation/run_你的名字_inference.py` |
| 跑出來的 score CSV | `evaluation/scores_模態_detector名_v0.csv` |

---

## 1. Detector 檔案

### 放哪裡
```
training/detectors/rppg_detector.py      ← 子寰
training/detectors/syncnet_detector.py   ← 音訊（已有）
training/detectors/dummy_detector.py     ← 範例，直接抄結構
```

### 最小結構（照 dummy_detector.py 抄）

```python
# training/detectors/rppg_detector.py

try:
    from detectors.base_detector import AbstractDetector
    from metrics.registry import DETECTOR
except (ImportError, RuntimeError, Exception):
    # 獨立執行時的 stub，不用改
    import abc
    class AbstractDetector(nn.Module, metaclass=abc.ABCMeta): ...
    class _R:
        def register_module(self, module_name=None):
            def d(cls): return cls
            return d
    DETECTOR = _R()

@DETECTOR.register_module(module_name='rppg')   # ← 這個名字要跟 yaml 的 model_name 一致
class RPPGDetector(AbstractDetector):

    def __init__(self, config):
        super().__init__()
        # 你的初始化

    def build_backbone(self, config): ...
    def build_loss(self, config): ...
    def features(self, data_dict): ...
    def classifier(self, features): ...
    def get_losses(self, data_dict, pred_dict): ...
    def get_train_metrics(self, data_dict, pred_dict): ...
    def forward(self, data_dict, inference=False): ...

    # Week 11 推論介面（必須加）
    def predict_with_timing(self, sample_id, ...) -> dict:
        """回傳格式見下方 §輸出格式"""
```

### 加進 `__init__.py`（只加一行）

```python
# training/detectors/__init__.py 最底部加：
from .rppg_detector import RPPGDetector
```

---

## 2. Config 檔案

### 放哪裡
```
training/config/detector/rppg.yaml
```

### 最小範例（從 dummy.yaml 改）

```yaml
model_name: rppg          # 必須跟 @DETECTOR.register_module(module_name=...) 一致
pretrained: null
train_dataset: [FaceForensics++]
test_dataset: [FaceForensics++]
compression: c23
train_batchSize: 8
test_batchSize: 8
workers: 4
frame_num: {train: 30, test: 30}   # rPPG 需要連續幀，調高
resolution: 224
with_mask: false
with_landmark: false
with_audio: false         # rPPG 不需要音訊
loss_func: cross_entropy
optimizer:
  type: adam
  adam:
    lr: 0.0001
    beta1: 0.9
    beta2: 0.999
    eps: 1e-8
    weight_decay: 0.0005
    amsgrad: false
nEpochs: 30
metric_scoring: auc
manualSeed: 1024
cuda: true
cudnn: true
save_ckpt: true
save_feat: false
rgb_dir: /datasets
lmdb: false
dataset_json_folder: preprocessing/dataset_json_v3
use_data_augmentation: false
data_aug:
  flip_prob: 0.5
  rotate_prob: 0.3
  rotate_limit: [-10, 10]
  blur_prob: 0.3
  blur_limit: [3, 7]
  brightness_prob: 0.3
  brightness_limit: [-0.1, 0.1]
  contrast_limit: [-0.1, 0.1]
  quality_lower: 40
  quality_upper: 100
lr_scheduler: null
start_epoch: 0
mean: [0.5, 0.5, 0.5]
std: [0.5, 0.5, 0.5]
```

---

## 3. Inference 腳本

### 放哪裡
```
evaluation/run_rppg_inference.py
```

### 模仿 `run_xception_inference.py` 的結構

腳本必須：
1. 讀 `evaluation/mini_eval_metadata.csv`（或接受 `--videos` 參數）
2. 跑 inference
3. 輸出 CSV 到 `evaluation/scores_rppg_你的detector名_v0.csv`

### 輸出 CSV 格式（每個人的 CSV schema 必須一樣）

```
sample_id,dataset,label,detector_name,modality,fake_score,score_type,inference_time_ms,window_start_sec,window_end_sec,status,error_message
real_000,FF++,0,POS_rppg,rppg,0.3421,snr,245.3,0.0,4.0,ok,
fake_001,FF++,1,POS_rppg,rppg,0.7812,snr,251.1,0.0,4.0,ok,
```

**必填欄位說明：**

| 欄位 | 你填什麼 | 注意 |
|------|---------|------|
| `sample_id` | metadata 裡的 sample_id | 不能自己亂取 |
| `modality` | `"rppg"` | 固定，不能打錯 |
| `fake_score` | 0~1 浮點數，越高越像 fake | 必須統一方向 |
| `score_type` | rPPG 用 `"snr"`，音訊用 `"sync_error"` | 視覺用 `"probability"` |
| `status` | `"ok"` / `"failed"` / `"skipped"` | 只有這三種 |
| `window_start_sec` / `window_end_sec` | 你分析的時間區間（秒）| 單幀 detector 填 `"N/A"` |

### `fake_score` 方向規則

> **越高 = 越像 fake**

- rPPG：血流信號弱（低 SNR）→ 可疑 → **高分**。公式：`fake_score = 1.0 - normalize(snr)`
- 音訊同步：sync_error 大 → 可疑 → **高分**。直接用 cosine distance
- 視覺：softmax 的 fake class 機率 → 直接就是高分

---

## 4. predict_with_timing() 輸出格式

每個 detector 的 `predict_with_timing()` 必須回傳這個 dict（跟 `evaluation/base_detector.py` 的 `DetectorOutput` 一致）：

```python
{
    'sample_id': 'real_000',
    'detector_name': 'POS_rppg',         # 你的 detector 名字
    'modality': 'rppg',                  # 'visual' | 'rppg' | 'av_sync'
    'fake_score': 0.3421,                # 0~1, 越高越 fake
    'score_type': 'snr',                 # 'probability'|'logit'|'snr'|'sync_error'
    'confidence': None,
    'inference_time_ms': 245.3,
    'window_start_sec': 0.0,             # 分析的時間段起點（沒有填 None）
    'window_end_sec': 4.0,               # 分析的時間段終點（沒有填 None）
    'status': 'ok',                      # 'ok' | 'failed' | 'skipped'
    'error_message': None,
}
```

---

## 5. 現有已完成的範例

看這些學怎麼做：

| 模態 | Detector 檔案 | Inference 腳本 | Score CSV |
|------|-------------|--------------|----------|
| 視覺 | （xception_detector.py，原本就有）| `evaluation/run_xception_inference.py` | `scores_visual_xception_v0.csv` |
| 音訊 | `training/detectors/syncnet_detector.py` | `evaluation/run_syncnet_inference.py` | `scores_avsync_syncnet_v0.csv` |
| rPPG | **你要做的** | **你要做的** | **你要做的** |

---

## 6. 常見錯誤

**❌ 不要做這些：**

```
# 錯：把檔案放在 repo 根目錄
/rppg_model.py
/my_inference.py

# 錯：自己建新的資料夾
/my_stuff/rppg.py

# 錯：CSV 欄位名字打錯或順序錯
fake_probability,  ← 應該是 fake_score
rPPG,              ← 應該是 rppg（全小寫）

# 錯：score 方向反了
fake_score = snr_value  ← 高 SNR = 真人 = 應該是低分，方向反了

# 錯：不加 predict_with_timing()
# fusion solver 讀不到你的輸出
```

**✅ 要做這些：**

```
# 放檔案前先確認路徑
training/detectors/rppg_detector.py      ✓
training/config/detector/rppg.yaml       ✓
evaluation/run_rppg_inference.py         ✓
evaluation/scores_rppg_pos_v0.csv        ✓

# 跑完確認 CSV header 跟 Xception 的一樣
head -1 evaluation/scores_visual_xception_v0.csv
head -1 evaluation/scores_rppg_pos_v0.csv
# 兩行必須一樣

# 跑 smoke test 確認 detector 可以 import
python training/detectors/rppg_detector.py
```

---

## 7. Git 工作流程

```bash
# 從 main 開新 branch
git checkout main && git pull
git checkout -b feature/rppg-detector

# 只 commit 你的檔案，不要 commit 別人的
git add training/detectors/rppg_detector.py
git add training/config/detector/rppg.yaml
git add evaluation/run_rppg_inference.py
git add evaluation/scores_rppg_pos_v0.csv
git commit -m "feat(rPPG): add POS rPPG detector and inference script"

# 推上去開 PR
git push -u origin feature/rppg-detector
```

PR 開好之後貼連結給 Nia review。

---

## 8. 程式碼在實驗室機器但還沒進 repo？照這裡做

你已經在實驗室機器跑通了，但東西放在自己的資料夾而不是 repo 裡面。  
以下以 rPPG 為例，其他模態照著改就好。

### Step 1：確認你有哪些檔案

你的 rPPG 程式碼可能長這樣（路徑隨便舉例）：
```
~/my_work/rppg_model.py          ← 模型主體
~/my_work/run_rppg.py            ← 推論腳本
~/my_work/results/scores.csv     ← 跑出來的分數
```

### Step 2：把檔案複製到 repo 的正確位置

```bash
# 先確認你在 repo 根目錄
cd /path/to/DeepfakeBench

# detector → training/detectors/
cp ~/my_work/rppg_model.py training/detectors/rppg_detector.py

# inference 腳本 → evaluation/
cp ~/my_work/run_rppg.py evaluation/run_rppg_inference.py

# score CSV → evaluation/
cp ~/my_work/results/scores.csv evaluation/scores_rppg_pos_v0.csv
```

### Step 3：修掉 detector 檔案裡的外部路徑依賴

最常見的問題是 `sys.path` 指向你自己的資料夾：

```python
# ❌ 不能有這種東西
import sys
sys.path.insert(0, '/home/你的名字/my_project')
from my_utils import something

# ✅ 把用到的 utility 直接複製進來，或改用 repo 內的 import
```

如果你的程式碼引用了外部 package，加進 `requirements.txt` 就好（但確認 Nia 同意再加）。

### Step 4：在 detector 檔案最上面加 try/except 保護

讓 detector 在 DeepfakeBench 框架外也能獨立執行：

```python
import torch
import torch.nn as nn

try:
    from detectors.base_detector import AbstractDetector
    from metrics.registry import DETECTOR
except (ImportError, RuntimeError, Exception):
    import abc
    class AbstractDetector(nn.Module, metaclass=abc.ABCMeta):
        @abc.abstractmethod
        def build_backbone(self, config): ...
        @abc.abstractmethod
        def build_loss(self, config): ...
        @abc.abstractmethod
        def features(self, data_dict): ...
        @abc.abstractmethod
        def classifier(self, features): ...
        @abc.abstractmethod
        def get_losses(self, data_dict, pred_dict): ...
        @abc.abstractmethod
        def get_train_metrics(self, data_dict, pred_dict): ...
        @abc.abstractmethod
        def forward(self, data_dict, inference=False): ...
    class _R:
        def register_module(self, module_name=None):
            def d(cls): return cls
            return d
    DETECTOR = _R()
```

### Step 5：確認 class 名稱和 register_module 一致

```python
@DETECTOR.register_module(module_name='rppg')   # ← yaml 的 model_name 要跟這個一樣
class RPPGDetector(AbstractDetector):
    ...
```

### Step 6：加進 `__init__.py`

```bash
echo "from .rppg_detector import RPPGDetector" >> training/detectors/__init__.py
```

### Step 7：建 config 檔

```bash
# 從 syncnet.yaml 複製一份改
cp training/config/detector/syncnet.yaml training/config/detector/rppg.yaml
# 然後編輯：model_name 改成 rppg，with_audio 改成 false，其他參數自己調
```

### Step 8：修 inference 腳本的輸出格式

確認 `evaluation/run_rppg_inference.py` 輸出的 CSV 欄位順序跟現有的一樣：

```bash
head -1 evaluation/scores_visual_xception_v0.csv
head -1 evaluation/scores_rppg_pos_v0.csv
# 兩行必須完全一樣
```

如果你的 CSV 欄位不對，參考 `run_syncnet_inference.py` 的 `SCORE_CSV_FIELDS` 清單。

### Step 9：smoke test

```bash
# 確認 detector 可以獨立 import 不爆炸
/path/to/DeepfakeBench/venv/bin/python training/detectors/rppg_detector.py
```

### Step 10：開 PR

```bash
git checkout main && git pull
git checkout -b feature/rppg-detector

git add training/detectors/rppg_detector.py
git add training/detectors/__init__.py
git add training/config/detector/rppg.yaml
git add evaluation/run_rppg_inference.py
git add evaluation/scores_rppg_pos_v0.csv

git commit -m "feat(rPPG): add POS rPPG detector and inference script"
git push -u origin feature/rppg-detector
```

然後貼 PR 連結給 Nia。

---

## 快速 checklist

開 PR 之前確認：

- [ ] `training/detectors/我的detector.py` 存在，有 `predict_with_timing()` 方法
- [ ] `training/detectors/__init__.py` 最底部有 `from .我的detector import 我的Class`
- [ ] `training/config/detector/我的detector.yaml` 存在，`model_name` 跟 `register_module` 一致
- [ ] `evaluation/run_我的detector_inference.py` 存在，可以跑
- [ ] `evaluation/scores_模態_我的detector_v0.csv` header 跟 `scores_visual_xception_v0.csv` 完全一致
- [ ] `fake_score` 方向正確（越高越 fake）
- [ ] 跑 `python training/detectors/我的detector.py` 不爆炸
