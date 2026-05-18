"""
run_syncnet_inference.py — P5 SyncNet Audio-Visual Sync Evaluation
===================================================================
對給定的影片清單跑完整音訊 pipeline：
  1. ffmpeg 抽取 16kHz WAV（含 80ms leading-silence trim）
  2. ffmpeg 抽取影格（或讀取已有的 frames 目錄）
  3. SyncNet inference → sync_error (0~1)
  4. 輸出 scores_avsync_syncnet_v0.csv（與 Xception CSV schema 一致）

使用方式：
  python evaluation/run_syncnet_inference.py \
      --videos path/to/video1.mp4 path/to/video2.mp4 \
      --output evaluation/scores_avsync_syncnet_v0.csv

  # 或直接跑 LatentSync demo（不帶參數時的預設）：
  python evaluation/run_syncnet_inference.py
"""

import os
import sys
import csv
import time
import wave
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

# Import directly to avoid triggering dlib loading in detectors/__init__.py
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "syncnet_detector",
    Path(__file__).parent.parent / "training" / "detectors" / "syncnet_detector.py"
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
SyncNetDetector = _mod.SyncNetDetector

# ── 預設測試影片（LatentSync demo，AI lip-sync = fake）──────────────────
DEFAULT_VIDEOS = [
    ("/Users/nia/anti-deepfake-box/third_party/LatentSync/assets/demo1_video.mp4", 1, "LatentSync"),
    ("/Users/nia/anti-deepfake-box/third_party/LatentSync/assets/demo2_video.mp4", 1, "LatentSync"),
]

OUTPUT_CSV = Path(__file__).parent / "scores_avsync_syncnet_v0.csv"

SCORE_CSV_FIELDS = [
    "sample_id", "dataset", "label",
    "detector_name", "modality",
    "fake_score", "score_type",
    "inference_time_ms",
    "window_start_sec", "window_end_sec",
    "status", "error_message",
]

DEVICE = "cpu"
FPS = 25
CLIP_FRAMES = 5      # SyncNet standard window
AUDIO_WIN_SEC = 0.2  # 0.2s audio aligned to 5 frames @ 25fps
SR = 16000


# ── Step 1: 音訊抽取 ────────────────────────────────────────────────────

def extract_audio_to_tmpdir(video_path: str, tmp_dir: str, fps: float = 25.0) -> Optional[str]:
    """抽取 16kHz mono WAV，去除開頭 2 幀靜音（~80ms）。"""
    stem = Path(video_path).stem
    tmp_wav = os.path.join(tmp_dir, f"{stem}_raw.wav")
    final_wav = os.path.join(tmp_dir, f"{stem}.wav")

    # Step 1a: extract raw
    ret = subprocess.run(
        ["ffmpeg", "-i", video_path, "-vn",
         "-acodec", "pcm_s16le", "-ar", str(SR), "-ac", "1",
         "-y", tmp_wav],
        capture_output=True
    )
    if ret.returncode != 0 or not os.path.exists(tmp_wav):
        return None

    # Step 1b: trim leading 2 frames (~80ms @ 25fps)
    skip_samples = int((2 / fps) * SR)
    try:
        with wave.open(tmp_wav, "r") as wf:
            raw = wf.readframes(wf.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16)[skip_samples:]
        with wave.open(final_wav, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SR)
            wf.writeframes(audio.tobytes())
        os.remove(tmp_wav)
    except Exception:
        os.rename(tmp_wav, final_wav)

    return final_wav if os.path.exists(final_wav) else None


def load_wav_float32(wav_path: str) -> np.ndarray:
    """WAV → float32 ndarray [-1, 1]。"""
    with wave.open(wav_path, "r") as wf:
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


# ── Step 2: 影格抽取 ────────────────────────────────────────────────────

def extract_frames_to_tmpdir(video_path: str, tmp_dir: str,
                              fps: int = FPS, max_frames: int = 150) -> List[str]:
    """用 ffmpeg 抽取影格，最多 max_frames 幀，輸出到 tmp_dir。"""
    frames_dir = os.path.join(tmp_dir, Path(video_path).stem + "_frames")
    os.makedirs(frames_dir, exist_ok=True)
    ret = subprocess.run(
        ["ffmpeg", "-i", video_path, "-vf", f"fps={fps}",
         "-frames:v", str(max_frames), "-q:v", "2",
         os.path.join(frames_dir, "%04d.png"), "-y"],
        capture_output=True
    )
    if ret.returncode != 0:
        return []
    return sorted([
        os.path.join(frames_dir, f)
        for f in os.listdir(frames_dir)
        if f.endswith(".png")
    ])


# ── Step 3: 建立 clips（每 CLIP_FRAMES 幀 + 對齊的音訊片段）──────────────

def build_clips(frame_paths: List[str], audio: np.ndarray,
                clip_frames: int = CLIP_FRAMES, fps: int = FPS,
                audio_win_sec: float = AUDIO_WIN_SEC):
    """把影格序列切成 (5幀影格路徑, 對齊音訊ndarray) 的 clip 清單。"""
    clips = []
    step = clip_frames
    for start in range(0, len(frame_paths) - clip_frames + 1, step):
        frame_clip = frame_paths[start:start + clip_frames]
        # 取對齊的音訊：start_frame 對應的時間起點
        t_start = start / fps
        t_end = t_start + audio_win_sec
        s_start = int(t_start * SR)
        s_end = int(t_end * SR)
        win_samples = int(audio_win_sec * SR)
        if s_start >= len(audio):
            break
        audio_clip = audio[s_start:s_start + win_samples]
        if len(audio_clip) < win_samples:
            audio_clip = np.pad(audio_clip, (0, win_samples - len(audio_clip)))
        clips.append((frame_clip, audio_clip, t_start, t_end))
    return clips


# ── Step 4: 影格載入工具 ────────────────────────────────────────────────

def load_frames_tensor(frame_paths: List[str], size: int = 112) -> torch.Tensor:
    """載入 PNG 影格 → (T, C, H, W) float32 tensor。"""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required: pip install Pillow")
    frames = []
    for p in frame_paths:
        img = Image.open(p).convert("RGB").resize((size, size))
        arr = np.array(img, dtype=np.float32) / 255.0
        frames.append(torch.from_numpy(arr).permute(2, 0, 1))  # (C, H, W)
    return torch.stack(frames, dim=0)  # (T, C, H, W)


# ── Step 5: 單影片推論 ──────────────────────────────────────────────────

def run_inference_on_video(detector: SyncNetDetector,
                           video_path: str, tmp_dir: str,
                           sample_id: str) -> dict:
    """完整跑一支影片的音訊 pipeline，回傳 score row dict。"""
    t_start = time.perf_counter()

    # 1. 抽音訊
    wav_path = extract_audio_to_tmpdir(video_path, tmp_dir)
    if wav_path is None:
        return _error_row(sample_id, "audio extraction failed",
                          time.perf_counter() - t_start)

    # 2. 抽影格
    frame_paths = extract_frames_to_tmpdir(video_path, tmp_dir)
    if not frame_paths:
        return _error_row(sample_id, "frame extraction failed",
                          time.perf_counter() - t_start)

    # 3. 載入音訊
    audio_arr = load_wav_float32(wav_path)

    # 4. 切 clips 並推論
    clips = build_clips(frame_paths, audio_arr)
    if not clips:
        return _error_row(sample_id, "no valid clips built",
                          time.perf_counter() - t_start)

    sync_errors = []
    for frame_clip, audio_clip, t0, t1 in clips:
        try:
            image_t = load_frames_tensor(frame_clip).unsqueeze(0).to(DEVICE)  # (1, T, C, H, W)
            audio_t = torch.from_numpy(audio_clip).unsqueeze(0).to(DEVICE)   # (1, N)
            out = detector.predict_with_timing(
                sample_id=sample_id,
                image=image_t[0],
                audio=audio_t[0],
                device=DEVICE,
                window_start_sec=t0,
                window_end_sec=t1,
            )
            if out["status"] == "ok":
                sync_errors.append(out["fake_score"])
        except Exception as e:
            print(f"  clip error @ {t0:.2f}s: {e}")

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    if not sync_errors:
        return _error_row(sample_id, "all clips failed", elapsed_ms / 1000)

    fake_score = round(float(np.mean(sync_errors)), 4)
    return {
        "sample_id": sample_id,
        "detector_name": "SyncNet",
        "modality": "av_sync",
        "fake_score": fake_score,
        "score_type": "sync_error",
        "inference_time_ms": round(elapsed_ms, 2),
        "window_start_sec": "N/A",
        "window_end_sec": "N/A",
        "status": "ok",
        "error_message": "",
        "_n_clips": len(sync_errors),
        "_min": round(min(sync_errors), 4),
        "_max": round(max(sync_errors), 4),
    }


def _error_row(sample_id: str, msg: str, elapsed_sec: float) -> dict:
    return {
        "sample_id": sample_id,
        "detector_name": "SyncNet",
        "modality": "av_sync",
        "fake_score": "N/A",
        "score_type": "sync_error",
        "inference_time_ms": round(elapsed_sec * 1000, 2),
        "window_start_sec": "N/A",
        "window_end_sec": "N/A",
        "status": "failed",
        "error_message": msg,
    }


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SyncNet audio-visual inference")
    parser.add_argument("--videos", nargs="+", help="MP4 paths (label assumed 1=fake)")
    parser.add_argument("--output", default=str(OUTPUT_CSV))
    args = parser.parse_args()

    # 決定影片清單
    if args.videos:
        video_list = [(v, 1, "unknown") for v in args.videos]
    else:
        print("使用預設 LatentSync demo 影片（label=1, fake）")
        video_list = DEFAULT_VIDEOS

    # 初始化 detector（不載入 pretrained weights，random init 用於 pipeline 驗證）
    print("初始化 SyncNetDetector（random init，僅驗證 pipeline）...")
    config = {"pretrained": None, "with_audio": True}
    detector = SyncNetDetector(config)
    detector.eval()
    print("✓ Detector 初始化成功")

    rows = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for video_path, label, dataset in video_list:
            sample_id = Path(video_path).stem
            print(f"\n[{sample_id}] {video_path}")

            if not os.path.exists(video_path):
                print(f"  ✗ 找不到影片，跳過")
                row = _error_row(sample_id, "video not found", 0)
            else:
                row = run_inference_on_video(detector, video_path, tmp_dir, sample_id)
                if row["status"] == "ok":
                    print(f"  ✓ fake_score={row['fake_score']}  "
                          f"clips={row.get('_n_clips')}  "
                          f"range=[{row.get('_min')}, {row.get('_max')}]")
                else:
                    print(f"  ✗ {row['error_message']}")

            row["dataset"] = dataset
            row["label"] = label
            rows.append(row)

    # 寫 CSV（只保留 SCORE_CSV_FIELDS）
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCORE_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in SCORE_CSV_FIELDS})

    ok_rows = [r for r in rows if r["status"] == "ok"]
    print(f"\n完成：{len(ok_rows)}/{len(rows)} 成功 → {output_path}")
    if ok_rows:
        scores = [float(r["fake_score"]) for r in ok_rows]
        print(f"sync_error 統計：mean={np.mean(scores):.4f}  "
              f"min={min(scores):.4f}  max={max(scores):.4f}")
        print("（random init 模型，數值無意義；pipeline 正常跑通即為成功）")


if __name__ == "__main__":
    main()
