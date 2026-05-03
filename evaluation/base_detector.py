"""
base_detector.py — Week 11 統一 Detector Interface v1
======================================================
作者: Nia（Framework & Integration）
日期: 2026-05-03
目的: 為三模態 detector（視覺紋理 / rPPG / 音視頻同步）定義統一輸出格式，
      使 fusion solver 可以直接讀取 score CSV 而不需要了解各 detector 內部實作。

[資料契約] 每個 detector 必須輸出以下欄位（見 §3.4 of Week 11 規格書）：
  sample_id       : str   — 對應 metadata.csv 的唯一識別碼
  detector_name   : str   — e.g. "Xception", "POS", "SyncNet"
  modality        : str   — "visual" | "rppg" | "av_sync"
  fake_score      : float — 0~1，越高越像 deepfake（統一方向）
  score_type      : str   — "probability" | "logit" | "distance" | "snr" | "sync_error"
  confidence      : float | None
  inference_time_ms: float
  window_start_sec: float | None — window-based detector 使用；單幀 detector 填 None
  window_end_sec  : float | None
  status          : str   — "ok" | "failed" | "skipped"
  error_message   : str | None

[設計原則]
- BaseDetector 是純 inference interface，不涉及 training pipeline
- 與 DeepfakeBench 的 AbstractDetector（training/detectors/base_detector.py）分開
- 可選擇在 predict() 內部呼叫 DeepfakeBench 的 forward()，但輸出必須轉換為本 schema
- 三個 Mock 實作示範如何讓三種模態輸出相同格式
"""

import time
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import csv
import json
from pathlib import Path


# ─────────────────────────────────────────────
# §1. 輸出 Schema（對應 PDF §3.4）
# ─────────────────────────────────────────────

@dataclass
class DetectorOutput:
    """
    單一樣本的 detector 輸出。所有欄位對應 score CSV schema。
    """
    sample_id: str
    detector_name: str
    modality: str                          # "visual" | "rppg" | "av_sync"
    fake_score: Optional[float]            # 0~1，越高越 fake；failed 時可為 None
    score_type: str                        # "probability" | "logit" | "distance" | "snr" | "sync_error"
    confidence: Optional[float] = None
    inference_time_ms: float = 0.0
    window_start_sec: Optional[float] = None
    window_end_sec: Optional[float] = None
    status: str = "ok"                     # "ok" | "failed" | "skipped"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """轉換為 dict，供 CSV exporter 使用。"""
        return {
            "sample_id": self.sample_id,
            "detector_name": self.detector_name,
            "modality": self.modality,
            "fake_score": self.fake_score if self.fake_score is not None else "N/A",
            "score_type": self.score_type,
            "confidence": self.confidence if self.confidence is not None else "N/A",
            "inference_time_ms": round(self.inference_time_ms, 3),
            "window_start_sec": self.window_start_sec if self.window_start_sec is not None else "N/A",
            "window_end_sec": self.window_end_sec if self.window_end_sec is not None else "N/A",
            "status": self.status,
            "error_message": self.error_message if self.error_message else "",
        }

    def to_json(self) -> str:
        """轉換為 JSON 字串，格式對齊 PDF §3.4。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────
# §2. 抽象 Base Class
# ─────────────────────────────────────────────

class BaseDetector(ABC):
    """
    所有 Week 11 deepfake detectors 的基礎抽象類別。

    子類別必須實作：
        predict(sample_id, input_data, **kwargs) -> DetectorOutput

    子類別可選擇實作：
        load_model()           — 載入模型權重
        preprocess(input_data) — 資料預處理
        postprocess(raw_output, score_direction) — 將原始輸出轉換為 fake_score
    """

    # 子類別須在 __init__ 中設定以下屬性
    detector_name: str = "UnnamedDetector"
    modality: str = "visual"          # "visual" | "rppg" | "av_sync"
    score_type: str = "probability"   # 原始輸出類型
    score_direction: str = "higher_is_fake"  # "higher_is_fake" | "higher_is_real" | "distance"

    @abstractmethod
    def predict(
        self,
        sample_id: str,
        input_data: Any,
        **kwargs,
    ) -> DetectorOutput:
        """
        對單一樣本執行推論，回傳標準化的 DetectorOutput。

        Args:
            sample_id  : metadata.csv 中的唯一識別碼
            input_data : 視模態而定
                         visual   → np.ndarray (H,W,3) 或 list of frames
                         rppg     → list of 30 frames (face crop sequences)
                         av_sync  → video clip path 或 (video_tensor, audio_tensor)
            **kwargs   : 額外參數，如 window_start_sec、window_end_sec

        Returns:
            DetectorOutput — 統一格式的輸出
        """
        raise NotImplementedError

    def normalize_score(self, raw_score: float) -> float:
        """
        將原始分數轉換為 fake_score（0~1，越高越 fake）。

        三種常見情況：
          - higher_is_fake : fake_score = sigmoid(raw_score) 或直接取 softmax[:,1]
          - higher_is_real : fake_score = 1 - normalized_score
          - distance       : 通常 distance 越大越 real，需個別處理
        """
        if self.score_direction == "higher_is_real":
            return 1.0 - float(raw_score)
        elif self.score_direction == "distance":
            # distance 類型需要 calibration；這裡留佔位符
            # 實作時請根據 detector 特性決定轉換方式，並記錄在 detector table
            return float(raw_score)
        else:  # higher_is_fake（預設）
            return float(raw_score)

    def _make_failed_output(
        self,
        sample_id: str,
        error_message: str,
        inference_time_ms: float = 0.0,
    ) -> DetectorOutput:
        """
        統一的 failed output 工廠方法。
        任何 Exception 都應呼叫此方法，確保 status 欄位正確填寫。
        """
        return DetectorOutput(
            sample_id=sample_id,
            detector_name=self.detector_name,
            modality=self.modality,
            fake_score=None,
            score_type=self.score_type,
            inference_time_ms=inference_time_ms,
            status="failed",
            error_message=error_message,
        )

    def _make_skipped_output(
        self,
        sample_id: str,
        reason: str,
    ) -> DetectorOutput:
        """
        統一的 skipped output 工廠方法。
        用於 clip_too_short_for_window 等預期性跳過。
        """
        return DetectorOutput(
            sample_id=sample_id,
            detector_name=self.detector_name,
            modality=self.modality,
            fake_score=None,
            score_type=self.score_type,
            status="skipped",
            error_message=reason,
        )


# ─────────────────────────────────────────────
# §3. Mock 實作 — 三模態各一個
#     用途：在真實模型整合前驗證 interface 輸出格式一致性
#           可以用 MockVisualDetector 跑 sanity check 確認 CSV exporter 正確
# ─────────────────────────────────────────────

class MockVisualDetector(BaseDetector):
    """
    視覺紋理模態 Mock（對應 Xception / EfficientNet-B4）。

    輸入: 單幀 224×224 RGB 圖像（np.ndarray）或 frame list
    時序: Sliding window，每幀獨立推論後 aggregate（window_start/end = None）
    分數類型: probability（softmax 輸出的 class 1 機率）
    """

    detector_name = "MockVisual"
    modality = "visual"
    score_type = "probability"
    score_direction = "higher_is_fake"

    def predict(
        self,
        sample_id: str,
        input_data: Any,
        **kwargs,
    ) -> DetectorOutput:
        t_start = time.perf_counter()

        try:
            # 模擬推論延遲 5~25ms
            time.sleep(random.uniform(0.005, 0.025))

            # 模擬臉部偵測失敗（5% 機率）
            if random.random() < 0.05:
                elapsed = (time.perf_counter() - t_start) * 1000
                return self._make_failed_output(
                    sample_id, "face_not_detected", elapsed
                )

            # 模擬 fake_score（真實模型替換此處）
            raw_score = random.uniform(0.0, 1.0)
            fake_score = self.normalize_score(raw_score)
            elapsed = (time.perf_counter() - t_start) * 1000

            return DetectorOutput(
                sample_id=sample_id,
                detector_name=self.detector_name,
                modality=self.modality,
                fake_score=round(fake_score, 4),
                score_type=self.score_type,
                confidence=round(abs(fake_score - 0.5) * 2, 4),  # 距離決策邊界的距離
                inference_time_ms=round(elapsed, 3),
                window_start_sec=None,
                window_end_sec=None,
                status="ok",
            )

        except Exception as e:
            elapsed = (time.perf_counter() - t_start) * 1000
            return self._make_failed_output(sample_id, str(e), elapsed)


class MockRPPGDetector(BaseDetector):
    """
    rPPG 生理訊號模態 Mock（對應 POS / TS-CAN / PhysMamba）。

    輸入: 連續 30 幀（約 1 秒）的臉部 crop 序列
    時序: 固定 window size=30，stride=15，需記錄 window_start/end_sec
    分數類型: snr（合成人臉的 rPPG SNR 通常顯著低於真實人臉）

    [注意] snr 方向是 higher_is_real（SNR 越高越像真人），
          因此 normalize_score 會做 1 - normalized_score 轉換。
          這必須記錄在 detector table 的 score_direction 欄位！
    """

    detector_name = "MockRPPG"
    modality = "rppg"
    score_type = "snr"
    score_direction = "higher_is_real"  # ← 重要！SNR 越高越 real

    # rPPG 需要的最低幀數（對應 Week 11 規格：minimum clip 長度 ≥ 2 秒 at 15fps = 30 幀）
    MIN_FRAMES = 30

    def predict(
        self,
        sample_id: str,
        input_data: Any,
        window_start_sec: Optional[float] = None,
        window_end_sec: Optional[float] = None,
        **kwargs,
    ) -> DetectorOutput:
        t_start = time.perf_counter()

        try:
            # 檢查幀數是否足夠
            if isinstance(input_data, list) and len(input_data) < self.MIN_FRAMES:
                return self._make_skipped_output(
                    sample_id,
                    f"clip_too_short_for_window: need {self.MIN_FRAMES} frames, got {len(input_data)}"
                )

            # 模擬 rPPG 推論延遲（較慢，50~150ms）
            time.sleep(random.uniform(0.05, 0.15))

            # 模擬 SNR 值（單位 dB，真實人臉通常 > 5dB，deepfake 通常 < 3dB）
            # 這裡用 0~1 normalize 後的值表示（真實模型輸出原始 SNR 後需做 calibration）
            raw_snr_normalized = random.uniform(0.0, 1.0)
            fake_score = self.normalize_score(raw_snr_normalized)  # 1 - snr_normalized
            elapsed = (time.perf_counter() - t_start) * 1000

            return DetectorOutput(
                sample_id=sample_id,
                detector_name=self.detector_name,
                modality=self.modality,
                fake_score=round(fake_score, 4),
                score_type=self.score_type,
                confidence=None,
                inference_time_ms=round(elapsed, 3),
                window_start_sec=window_start_sec,
                window_end_sec=window_end_sec,
                status="ok",
            )

        except Exception as e:
            elapsed = (time.perf_counter() - t_start) * 1000
            return self._make_failed_output(sample_id, str(e), elapsed)


class MockAVSyncDetector(BaseDetector):
    """
    音視頻同步模態 Mock（對應 SyncNet / LatentSync / AVAD）。

    輸入: 影片 clip（含音訊），至少 25 幀
    時序: 按音訊/視訊 frame rate 對齊，以 window 為單位
    分數類型: sync_error（唇音不同步誤差，越大越像 deepfake）

    [注意] sync_error 方向是 higher_is_fake（誤差越大越 fake），
           但需確認各 detector 的原始輸出定義後再決定 normalize 方式。
    """

    detector_name = "MockAVSync"
    modality = "av_sync"
    score_type = "sync_error"
    score_direction = "higher_is_fake"  # sync_error 越大越 fake

    # 音視頻同步需要的最低幀數
    MIN_FRAMES = 25

    def predict(
        self,
        sample_id: str,
        input_data: Any,
        window_start_sec: Optional[float] = None,
        window_end_sec: Optional[float] = None,
        **kwargs,
    ) -> DetectorOutput:
        t_start = time.perf_counter()

        try:
            # 模擬 audio decode 失敗（10% 機率，音訊模態風險較高）
            if random.random() < 0.10:
                elapsed = (time.perf_counter() - t_start) * 1000
                return self._make_failed_output(
                    sample_id, "audio_decode_failed", elapsed
                )

            # 模擬 AV sync 推論延遲（較慢，80~200ms，因需要處理音訊）
            time.sleep(random.uniform(0.08, 0.20))

            raw_sync_error = random.uniform(0.0, 1.0)
            fake_score = self.normalize_score(raw_sync_error)
            elapsed = (time.perf_counter() - t_start) * 1000

            return DetectorOutput(
                sample_id=sample_id,
                detector_name=self.detector_name,
                modality=self.modality,
                fake_score=round(fake_score, 4),
                score_type=self.score_type,
                confidence=None,
                inference_time_ms=round(elapsed, 3),
                window_start_sec=window_start_sec,
                window_end_sec=window_end_sec,
                status="ok",
            )

        except Exception as e:
            elapsed = (time.perf_counter() - t_start) * 1000
            return self._make_failed_output(sample_id, str(e), elapsed)


# ─────────────────────────────────────────────
# §4. Score CSV Exporter
# ─────────────────────────────────────────────

# CSV 欄位順序（嚴格遵守 PDF §3.3）
SCORE_CSV_FIELDS = [
    "sample_id", "dataset", "label",
    "detector_name", "modality",
    "fake_score", "score_type",
    "inference_time_ms",
    "window_start_sec", "window_end_sec",
    "status", "error_message",
]


def export_to_score_csv(
    outputs: List[DetectorOutput],
    output_path: str,
    dataset_name: str,
    labels: Dict[str, int],  # sample_id -> label (0=real, 1=fake)
    append: bool = False,
) -> None:
    """
    將 DetectorOutput list 匯出為標準 score CSV。

    Args:
        outputs     : predict() 的回傳值 list
        output_path : 輸出 CSV 路徑
        dataset_name: e.g. "FF++", "Celeb-DF"
        labels      : sample_id -> int(0 or 1)
        append      : 是否附加到既有 CSV（用於多個 detector 合併）
    """
    mode = "a" if append else "w"
    write_header = not (append and Path(output_path).exists())

    with open(output_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCORE_CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for output in outputs:
            row = output.to_dict()
            row["dataset"] = dataset_name
            row["label"] = labels.get(output.sample_id, -1)  # -1 表示 label 缺失
            # 僅保留 SCORE_CSV_FIELDS 中的欄位
            writer.writerow({k: row.get(k, "") for k in SCORE_CSV_FIELDS})


# ─────────────────────────────────────────────
# §5. 格式驗證（供交付物驗收使用）
# ─────────────────────────────────────────────

def validate_outputs_schema(outputs: List[DetectorOutput]) -> bool:
    """
    驗證三個模態的輸出 schema 是否一致。
    週末 Go/No-Go review 前可執行此函數確認 interface 正確。

    Returns:
        True 若所有 output 的欄位齊全且型態正確，否則 False（並 print 問題）
    """
    ok = True
    for i, o in enumerate(outputs):
        if o.modality not in ("visual", "rppg", "av_sync"):
            print(f"[ERROR] output[{i}] modality '{o.modality}' 非法")
            ok = False
        if o.status not in ("ok", "failed", "skipped"):
            print(f"[ERROR] output[{i}] status '{o.status}' 非法")
            ok = False
        if o.status == "ok" and o.fake_score is None:
            print(f"[ERROR] output[{i}] status=ok 但 fake_score 為 None")
            ok = False
        if o.status == "ok" and not (0.0 <= o.fake_score <= 1.0):
            print(f"[ERROR] output[{i}] fake_score={o.fake_score} 超出 [0,1] 範圍")
            ok = False
        if o.status in ("failed", "skipped") and not o.error_message:
            print(f"[WARNING] output[{i}] status={o.status} 但 error_message 為空")
    return ok


# ─────────────────────────────────────────────
# §6. Quick Self-Test（直接執行此檔案時驗證）
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import tempfile

    print("=" * 60)
    print("BaseDetector Interface v1 — Schema Validation Test")
    print("=" * 60)

    # 建立三個 mock detector
    detectors = [
        MockVisualDetector(),
        MockRPPGDetector(),
        MockAVSyncDetector(),
    ]

    # 模擬 10 個樣本
    sample_ids = [f"ffpp_{i:04d}" for i in range(10)]
    labels = {sid: (1 if i % 2 == 0 else 0) for i, sid in enumerate(sample_ids)}

    all_outputs: List[DetectorOutput] = []

    for detector in detectors:
        print(f"\n[{detector.detector_name}] modality={detector.modality}")
        outputs = []
        for sid in sample_ids:
            # rPPG mock 傳入 list of 30 frames（用 None 佔位）
            if detector.modality == "rppg":
                result = detector.predict(
                    sid, [None] * 30,
                    window_start_sec=0.0, window_end_sec=1.0
                )
            elif detector.modality == "av_sync":
                result = detector.predict(
                    sid, "mock_video.mp4",
                    window_start_sec=0.0, window_end_sec=1.0
                )
            else:
                result = detector.predict(sid, None)

            outputs.append(result)
            status_str = f"[{result.status.upper()}]"
            score_str = f"fake_score={result.fake_score}" if result.fake_score is not None else f"({result.error_message})"
            print(f"  {status_str} {result.sample_id}: {score_str} | {result.inference_time_ms:.1f}ms")

        all_outputs.extend(outputs)

    # 驗證 schema
    print("\n--- Schema Validation ---")
    passed = validate_outputs_schema(all_outputs)
    print(f"Schema check: {'✓ PASSED' if passed else '✗ FAILED'}")

    # 匯出 CSV
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, dir="/tmp"
    ) as tmp:
        csv_path = tmp.name

    export_to_score_csv(all_outputs, csv_path, dataset_name="FF++", labels=labels)

    # 讀回驗證 CSV 欄位
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        actual_fields = set(reader.fieldnames or [])

    expected_fields = set(SCORE_CSV_FIELDS)
    missing = expected_fields - actual_fields
    extra = actual_fields - expected_fields

    print(f"\n--- CSV Export ---")
    print(f"匯出路徑: {csv_path}")
    print(f"總行數: {len(rows)}")
    print(f"欄位完整性: {'✓ OK' if not missing else f'✗ 缺少 {missing}'}")
    if extra:
        print(f"額外欄位（無害）: {extra}")

    # 顯示前 3 行
    print("\n前 3 行預覽:")
    for row in rows[:3]:
        print(f"  {row}")

    print("\n[Test Complete] interface 可供 A 組與曉蓮確認欄位格式。")
    print("下一步: 將此檔案發給全組確認，然後進行 P1 dummy detector 整合。")
