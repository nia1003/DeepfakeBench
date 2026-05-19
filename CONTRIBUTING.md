# DeepfakeBench 整合指南

給所有要把程式碼放進 repo 的人看。  
不用讀整個 codebase，只需要知道「你的東西放哪裡」。

---

## 一、誰負責什麼、要放哪裡

| 組員 | 模態 | 核心實作（你要放進來的） | Adapter（已在 repo） | Inference 腳本 | Score CSV |
|------|------|------------------------|----------------------|----------------|-----------|
| 子寰 | rPPG | `training/detectors/rppg_detector.py` | `adb_rppg_detector.py` ✓ | `evaluation/run_rppg_inference.py` | `scores_rppg_pos_v0.csv` |
| 音訊同步 | AV-sync | `training/detectors/sync_detector.py` | `adb_sync_detector.py` ✓ | `evaluation/run_sync_inference.py` | `scores_avsync_sync_v0.csv` |
| 視覺 | visual | `training/detectors/visual_detector.py` | `adb_visual_detector.py` ✓ | `evaluation/run_visual_inference.py` | `scores_visual_adb_v0.csv` |
| Nia | Framework | SyncNet + abstract_dataset + preprocess | — | `run_syncnet_inference.py` ✓ | `scores_avsync_syncnet_v0.csv` ✓ |

**共用工具（也需要放進 repo）：**

| 工具 | 放哪裡 | 誰用到 |
|------|--------|--------|
| `face_extractor.py` | `training/preprocessing/face_extractor.py` | rPPG、sync、visual 三個都用 |
| `audio_extractor.py` | `training/preprocessing/audio_extractor.py` | sync 用 |

> adapter 檔案在 `framework` branch 已有，merge 進來就好，**不用自己再寫**。

---

## 二、資料集放哪裡

所有資料集統一放在：

```
datasets/
└── rgb/
    ├── FaceForensics++/          ← 已有（FF++ c23 視覺資料集）
    │   ├── original_sequences/
    │   ├── manipulated_sequences/
    │   ├── train.json
    │   ├── val.json
    │   └── test.json
    ├── FakeAVCeleb/              ← 音訊 AV-sync 用
    │   ├── RealVideo-RealAudio/
    │   ├── FakeVideo-RealAudio/
    │   ├── RealVideo-FakeAudio/
    │   └── FakeVideo-FakeAudio/
    ├── DFDC/                     ← DFDC（若有申請到）
    │   ├── videos/
    │   └── labels.csv
    └── LRS3/                     ← 唇語 / AV 同步測試用（若需要）
        └── ...
```

**規則：**
- 資料集只放 `datasets/rgb/` 底下，**不要放在 home 目錄或其他任何地方**
- 大型資料集不 commit 進 git（`.gitignore` 已設定 `datasets/rgb/*/` 路徑）
- 給組員看的是「路徑結構」，實際資料自己抓

---

## 三、Pretrained Weights 放哪裡

```
training/
└── checkpoints/                  ← 所有人的 weights 統一放這裡
    ├── physnet_ubfc.pth          ← 子寰的 PhysNet (rPPG)
    ├── latentsync_syncnet.pth    ← 音訊 SyncNet weights
    └── xception_ff_c23.pth       ← 視覺 XceptionNet weights
```

config 裡的路徑（例如 `adb_rppg.yaml`）寫的是相對 repo 根目錄的路徑，所以：
```yaml
rppg_pretrained: 'training/checkpoints/physnet_ubfc.pth'
```

weights 檔案同樣不 commit 進 git，給組員看放在哪、自己去實驗室機器的對應路徑抓。

---

## 四、子寰（rPPG）要做什麼

### 要放的檔案

```
training/detectors/rppg_detector.py        ← 你的 PhysNet + SNR 核心實作
training/preprocessing/face_extractor.py  ← FaceTrack 工具（跟 sync / visual 共用）
training/checkpoints/physnet_ubfc.pth     ← weights（gitignore，自己放）
evaluation/run_rppg_inference.py           ← inference 腳本（你要新建）
evaluation/scores_rppg_pos_v0.csv          ← 跑出來的分數（你要生成）
```

### adapter 怎麼找到你的程式碼

`adb_rppg_detector.py` 的第 37–41 行：

```python
_ADB_ROOT = Path(__file__).parent.parent   # = training/
sys.path.insert(0, str(_ADB_ROOT))
from detectors.rppg_detector import RPPGDetector as ADBRPPGDetector_impl, compute_ppg_snr, snr_to_fake_score
from preprocessing.face_extractor import FaceTrack
```

`_ADB_ROOT` = `training/`，所以：
- `training/detectors/rppg_detector.py` → 被找到 ✓
- `training/preprocessing/face_extractor.py` → 被找到 ✓

**你的 `rppg_detector.py` 必須 export 三個東西：**
```python
class RPPGDetector: ...
def compute_ppg_snr(signal: np.ndarray) -> float: ...
def snr_to_fake_score(snr: float) -> float: ...   # 越高越 fake
```

### fake_score 方向

```python
# 高 SNR = 真人 = 低分；低 SNR = 可疑 = 高分
def snr_to_fake_score(snr: float) -> float:
    return 1.0 - normalize(snr)   # normalize 到 [0,1]
```

### `__init__.py` 更新

`framework` branch merge 後會自動有，如果手動放檔案：

```python
# training/detectors/__init__.py 最底部加：
from .adb_rppg_detector import ADBRPPGDetector
```

### config 路徑

已有 `training/config/detector/adb_rppg.yaml`（`framework` branch），確認：

```yaml
model_name: 'adb_rppg'
rppg_pretrained: 'training/checkpoints/physnet_ubfc.pth'
frame_num: {train: 180, test: 180}   # rPPG 需要長時序
video_mode: true
clip_size: 180
```

---

## 五、音訊同步組要做什麼

### 要放的檔案

```
training/detectors/sync_detector.py           ← SyncDetector 核心實作
training/preprocessing/audio_extractor.py     ← AudioExtractor 工具
training/preprocessing/face_extractor.py      ← 跟子寰共用，放一份就好
training/checkpoints/latentsync_syncnet.pth   ← weights（gitignore）
evaluation/run_sync_inference.py               ← inference 腳本
evaluation/scores_avsync_sync_v0.csv           ← 跑出來的分數
```

### adapter 怎麼找到你的程式碼

`adb_sync_detector.py` 的第 39–47 行：

```python
_ADB_ROOT = Path(__file__).parent.parent   # = training/
sys.path.insert(0, str(_ADB_ROOT))
from detectors.sync_detector import SyncDetector as ADBSyncDetector_impl
from preprocessing.audio_extractor import AudioExtractor
from preprocessing.face_extractor import FaceTrack
from deepfakebench_adapters.adb_visual_detector import _dfb_batch_to_face_track
```

注意最後一行：還需要 visual adapter 的 `_dfb_batch_to_face_track`，這個在 `adb_visual_detector.py` 裡，merge `framework` 後就有了。

**你的 `sync_detector.py` 必須 export：**
```python
class SyncDetector:
    def predict(self, frames: torch.Tensor, audio: torch.Tensor) -> float:
        ...   # 回傳 sync_error 0~1（越高越不同步 = 越 fake）
```

### fake_score 方向

```python
# sync_error 越大 = 音視頻不同步 = 越可疑 = 高分
fake_score = sync_error   # 直接用，不用反轉
score_type = 'sync_error'
```

### config 路徑

已有 `training/config/detector/adb_sync.yaml`：

```yaml
model_name: 'adb_sync'
syncnet_path: 'training/checkpoints/latentsync_syncnet.pth'
with_audio: true
frame_num: {train: 25, test: 25}
video_mode: true
clip_size: 25
```

### `__init__.py` 更新

```python
from .adb_sync_detector import ADBSyncDetector
```

---

## 六、視覺組要做什麼

### 要放的檔案

```
training/detectors/visual_detector.py         ← VisualDetector 核心實作
training/preprocessing/face_extractor.py      ← 跟 rPPG / sync 共用
training/checkpoints/xception_ff_c23.pth      ← weights（gitignore）
evaluation/run_visual_inference.py             ← inference 腳本
evaluation/scores_visual_adb_v0.csv            ← 跑出來的分數
```

### adapter 怎麼找到你的程式碼

`adb_visual_detector.py` 的第 43–48 行：

```python
_ADB_ROOT = Path(__file__).parent.parent   # = training/
sys.path.insert(0, str(_ADB_ROOT))
from detectors.visual_detector import VisualDetector as ADBVisualDetector_impl
from preprocessing.face_extractor import UnifiedFaceExtractor, FaceTrack
```

**你的 `visual_detector.py` 必須 export：**
```python
class VisualDetector:
    def predict(self, image: torch.Tensor) -> float:
        ...   # 回傳 fake probability 0~1
```

### fake_score 方向

```python
# softmax 的 fake class 機率直接當 fake_score
fake_score = float(F.softmax(logits, dim=-1)[0, 1])
score_type = 'probability'
```

### config 路徑

已有 `training/config/detector/adb_visual.yaml`：

```yaml
model_name: 'adb_visual'
visual_pretrained: 'training/checkpoints/xception_ff_c23.pth'
resolution: 256
```

### `__init__.py` 更新

```python
from .adb_visual_detector import ADBVisualDetector
```

---

## 七、Inference 腳本格式（三個人都要寫）

### 模仿 `evaluation/run_syncnet_inference.py`

腳本必須：
1. 讀 `evaluation/mini_eval_metadata.csv` 或接受 `--videos` 參數
2. 跑 inference
3. 輸出 CSV 到 `evaluation/scores_模態_你的detector名_v0.csv`

### CSV 欄位（完全一樣，不能差一個字）

```
sample_id,dataset,label,detector_name,modality,fake_score,score_type,inference_time_ms,window_start_sec,window_end_sec,status,error_message
```

| 欄位 | rPPG 填什麼 | sync 填什麼 | visual 填什麼 |
|------|------------|------------|--------------|
| `modality` | `"rppg"` | `"av_sync"` | `"visual"` |
| `score_type` | `"snr"` | `"sync_error"` | `"probability"` |
| `fake_score` | 0~1（高=fake） | 0~1（高=不同步） | 0~1（高=fake） |
| `window_start_sec` | 分析起點（秒） | 分析起點（秒） | `"N/A"` |
| `status` | `"ok"` / `"failed"` / `"skipped"` | 同左 | 同左 |

### 確認 CSV 跟 Xception 的一模一樣

```bash
head -1 evaluation/scores_visual_xception_v0.csv
head -1 evaluation/scores_rppg_pos_v0.csv        # 子寰
head -1 evaluation/scores_avsync_sync_v0.csv     # 音訊
head -1 evaluation/scores_visual_adb_v0.csv      # 視覺
# 四行必須完全一樣
```

---

## 八、predict_with_timing() 介面（Inference 腳本呼叫用）

如果你的 detector 要被 inference 腳本直接呼叫，要加這個方法：

```python
def predict_with_timing(self, sample_id, ...) -> dict:
    return {
        'sample_id': sample_id,
        'detector_name': 'POS_rppg',     # 你的 detector 名字
        'modality': 'rppg',              # 'visual' | 'rppg' | 'av_sync'
        'fake_score': 0.3421,            # 0~1，越高越 fake
        'score_type': 'snr',             # 見上表
        'confidence': None,
        'inference_time_ms': 245.3,
        'window_start_sec': 0.0,
        'window_end_sec': 4.0,
        'status': 'ok',                  # 'ok' | 'failed' | 'skipped'
        'error_message': None,
    }
```

---

## 九、把 framework branch 的 adapter 合進來

adapter 檔案已在 `origin/framework`，先把它們抓進你的 branch：

```bash
git checkout main && git pull
git checkout -b feature/你的模態-detector

# 只把 framework 的 adapter 和 config 檔案 cherry-pick 進來
git checkout origin/framework -- training/detectors/adb_rppg_detector.py
git checkout origin/framework -- training/config/detector/adb_rppg.yaml
# （或者 sync / visual，視你的模態而定）

# 然後把你的核心實作複製進來（從實驗室機器 scp 過來）
cp /path/to/你的/rppg_detector.py training/detectors/rppg_detector.py
cp /path/to/你的/face_extractor.py training/preprocessing/face_extractor.py
```

---

## 十、從實驗室機器抓檔案

如果你的程式碼在實驗室機器（不在自己的筆電），用 scp：

```bash
# 在你的筆電上執行
scp 你的帳號@實驗室機器IP:/path/to/rppg_detector.py \
    /Users/你/DeepfakeBench/training/detectors/rppg_detector.py

scp 你的帳號@實驗室機器IP:/path/to/face_extractor.py \
    /Users/你/DeepfakeBench/training/preprocessing/face_extractor.py
```

或直接在實驗室機器上操作（git push from there）。

---

## 十一、Git 工作流程

```bash
git checkout main && git pull
git checkout -b feature/rppg-detector   # 改成你的模態名稱

# 只 add 你的檔案
git add training/detectors/rppg_detector.py
git add training/detectors/adb_rppg_detector.py   # 從 framework cherry-pick 來的
git add training/detectors/__init__.py
git add training/preprocessing/face_extractor.py
git add training/config/detector/adb_rppg.yaml
git add evaluation/run_rppg_inference.py
git add evaluation/scores_rppg_pos_v0.csv

git commit -m "feat(rPPG): add PhysNet rPPG detector and inference script"
git push -u origin feature/rppg-detector
```

PR 開好之後貼連結給 Nia review。

---

## 快速 checklist

開 PR 之前確認：

**程式碼：**
- [ ] `training/detectors/你的_detector.py` 存在，export 正確的 class 和 function
- [ ] `training/preprocessing/face_extractor.py` 存在（rPPG / sync / visual 三個都需要）
- [ ] `training/detectors/__init__.py` 最底部有 `from .adb_xxx_detector import ADBXxxDetector`
- [ ] `training/config/detector/adb_xxx.yaml` 存在，`model_name` 跟 `register_module` 一致

**Inference：**
- [ ] `evaluation/run_xxx_inference.py` 存在，可以跑
- [ ] `evaluation/scores_模態_你的detector_v0.csv` 存在
- [ ] CSV header 跟 `scores_visual_xception_v0.csv` 完全一致
- [ ] `fake_score` 方向正確（越高越 fake）

**資料集：**
- [ ] 用到的資料集放在 `datasets/rgb/資料集名稱/`，不在其他地方
- [ ] config 的 `rgb_dir` 指向 `datasets`（相對路徑）

**Smoke test：**
- [ ] `python training/detectors/adb_xxx_detector.py` 不爆炸
