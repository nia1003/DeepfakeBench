"""
run_xception_inference.py — P2 Xception Mini Evaluation
用 timm Xception 直接跑 inference，輸出 scores_visual_xception_v0.csv
"""

import os
import csv
import time
import torch
import timm
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score

METADATA_CSV = "/Users/nia/DeepfakeBench/evaluation/mini_eval_metadata.csv"
OUTPUT_CSV   = "/Users/nia/DeepfakeBench/evaluation/scores_visual_xception_v0.csv"
FRAMES_PER_CLIP = 4
DEVICE = "cpu"

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

def load_model():
    # timm xception：pretrained=True 用 ImageNet weights（非 deepfake-finetuned）
    model = timm.create_model('xception', pretrained=True, num_classes=2)
    model.eval()
    return model

def predict_clip(model, frame_dir):
    frames_files = sorted(os.listdir(frame_dir))
    indices = [int(i * (len(frames_files) - 1) / (FRAMES_PER_CLIP - 1)) for i in range(FRAMES_PER_CLIP)]
    selected = [frames_files[i] for i in indices]

    scores = []
    t_start = time.perf_counter()
    for fname in selected:
        img = Image.open(os.path.join(frame_dir, fname)).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            logits = model(tensor)  # (1, 2)
            prob = torch.softmax(logits, dim=1)
            scores.append(float(prob[0][1]))

    elapsed = (time.perf_counter() - t_start) * 1000
    return round(sum(scores) / len(scores), 4), round(elapsed, 2)

def main():
    print("載入 Xception (timm, ImageNet pretrained)...")
    model = load_model()
    print("✓ 模型載入成功")

    with open(METADATA_CSV) as f:
        samples = list(csv.DictReader(f))
    print(f"共 {len(samples)} 個樣本，開始 inference...")

    rows = []
    for i, sample in enumerate(samples):
        try:
            fake_score, inference_time_ms = predict_clip(model, sample["path"])
            status = "ok"
            error = ""
        except Exception as e:
            fake_score, inference_time_ms, status, error = "N/A", 0, "failed", str(e)

        rows.append({
            "sample_id": sample["sample_id"],
            "dataset": sample["dataset"],
            "label": int(sample["label"]),
            "detector_name": "Xception_timm_imagenet",
            "modality": "visual",
            "fake_score": fake_score,
            "score_type": "probability",
            "inference_time_ms": inference_time_ms,
            "window_start_sec": "N/A",
            "window_end_sec": "N/A",
            "status": status,
            "error_message": error,
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(samples)}] {sample['sample_id']}: fake_score={fake_score}")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    y_true = [int(r["label"]) for r in ok_rows]
    y_score = [float(r["fake_score"]) for r in ok_rows]
    auc = roc_auc_score(y_true, y_score)

    print(f"\nAUC: {auc:.4f}  (ImageNet pretrained，非 deepfake-finetuned，AUC ~0.5 為正常)")
    print(f"完成：{len(rows)} 個樣本 → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
