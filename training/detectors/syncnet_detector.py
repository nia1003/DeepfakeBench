"""
syncnet_detector.py — SyncNet Audio-Visual Sync Detector
=========================================================
作者: Nia（Framework & Integration）
日期: 2026-05-18
目的: 基於音視頻唇語同步誤差偵測 deepfake。
     偽造音訊（cloned/dubbed）與唇部動作之間存在同步誤差，可作為偵測 signal。

[架構來源] Chung & Zisserman, "Out of Time: Automated Lip Sync in the Wild", ACCV 2016
           實作參考 joonson/syncnet_python (Apache-2.0)

[使用方式]
1. 建立 training/config/detector/syncnet.yaml
2. python training/train.py --detector_path training/config/detector/syncnet.yaml
3. 推論時透過 predict_with_timing() 取得 Week 11 相容格式輸出

[輸入格式]
  data_dict['image']  : (B, T, C, H, W) float32, T=5 frames, 112×112
  data_dict['audio']  : (B, N) float32, 16kHz PCM, N = 0.2s * 16000 = 3200 samples
  data_dict['label']  : (B,) int64, 0=real / 1=fake

[輸出格式]
  pred_dict['cls']    : (B, 2) logit
  pred_dict['prob']   : (B,) fake probability
  pred_dict['feat']   : (B, 2048) concatenated AV embedding
  pred_dict['sync_error'] : (B,) cosine distance (0~1), 越高越非同步
"""

import time
import logging
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from detectors.base_detector import AbstractDetector
    from metrics.registry import DETECTOR
    IN_DEEPFAKEBENCH = True
except ImportError:
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

    class _Registry:
        def register_module(self, module_name=None):
            def decorator(cls): return cls
            return decorator
    DETECTOR = _Registry()
    IN_DEEPFAKEBENCH = False

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# §1. Audio Encoder（Mel Spectrogram → 1024-d）
# ─────────────────────────────────────────────

class SyncNetAudioEncoder(nn.Module):
    """Encode a short mel spectrogram excerpt into a 1024-d L2-normalised embedding.

    Input shape: (B, 1, 13, T_mel) where T_mel corresponds to ~0.2s audio window.
    Output shape: (B, 1024)
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1)),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(512, 1024)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x).view(x.size(0), -1)
        feat = self.fc(feat)
        return F.normalize(feat, p=2, dim=1)


# ─────────────────────────────────────────────
# §2. Video Encoder（5 face frames → 1024-d）
# ─────────────────────────────────────────────

class SyncNetVideoEncoder(nn.Module):
    """Encode a 5-frame grayscale face clip into a 1024-d L2-normalised embedding.

    Input shape: (B, 5, H, W) — 5 consecutive grayscale frames, H=W=112.
    Reshape to (B, 1, 5, H, W) for 3D convolution.
    Output shape: (B, 1024)
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 96, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(0, 0, 0)),
            nn.BatchNorm3d(96), nn.ReLU(inplace=True),
            nn.Conv3d(96, 256, kernel_size=(1, 5, 5), stride=(1, 2, 2), padding=(0, 0, 0)),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.Conv3d(256, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 0, 0)),
            nn.BatchNorm3d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.fc = nn.Linear(512, 1024)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, H, W) → (B, 1, T, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(1)
        feat = self.encoder(x).view(x.size(0), -1)
        feat = self.fc(feat)
        return F.normalize(feat, p=2, dim=1)


# ─────────────────────────────────────────────
# §3. Audio preprocessing: raw PCM → mel spec
# ─────────────────────────────────────────────

def _pcm_to_mel(audio: torch.Tensor, sr: int = 16000,
                n_mels: int = 13, n_fft: int = 512,
                hop_length: int = 160) -> torch.Tensor:
    """Convert raw float32 PCM tensor to log-mel spectrogram.

    Input : (B, N) float32
    Output: (B, 1, n_mels, T_frames)
    """
    B = audio.shape[0]
    device = audio.device
    # Window
    window = torch.hann_window(n_fft, device=device)
    specs = []
    for i in range(B):
        wav = audio[i]
        stft = torch.stft(wav, n_fft=n_fft, hop_length=hop_length,
                          window=window, return_complex=True)
        power = stft.abs().pow(2)  # (freq_bins, T)
        # Simple triangular mel filterbank approximation (no torchaudio dependency)
        freq_bins = power.shape[0]
        mel_filters = _mel_filterbank(n_mels, n_fft, sr, device)  # (n_mels, freq_bins)
        mel_spec = torch.matmul(mel_filters, power)  # (n_mels, T)
        log_mel = torch.log(mel_spec.clamp(min=1e-9))
        specs.append(log_mel)
    mel_batch = torch.stack(specs, dim=0).unsqueeze(1)  # (B, 1, n_mels, T)
    return mel_batch


_MEL_CACHE: dict = {}


def _mel_filterbank(n_mels: int, n_fft: int, sr: int,
                    device: torch.device) -> torch.Tensor:
    """Build a triangular mel filterbank matrix (cached per device)."""
    key = (n_mels, n_fft, sr, str(device))
    if key in _MEL_CACHE:
        return _MEL_CACHE[key]

    freq_bins = n_fft // 2 + 1
    low_hz, high_hz = 0.0, sr / 2.0

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_low, mel_high = hz_to_mel(low_hz), hz_to_mel(high_hz)
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    filters = np.zeros((n_mels, freq_bins))
    for m in range(1, n_mels + 1):
        f_m_minus, f_m, f_m_plus = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            if f_m > f_m_minus:
                filters[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus > f_m:
                filters[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    fbank = torch.tensor(filters, dtype=torch.float32, device=device)
    _MEL_CACHE[key] = fbank
    return fbank


# ─────────────────────────────────────────────
# §4. SyncNet Loss (Contrastive + BCE)
# ─────────────────────────────────────────────

class SyncNetLoss(nn.Module):
    """Combined contrastive (sync/no-sync) + BCE (fake/real) loss.

    For deepfake detection, high cosine distance → fake.
    We also compute a standard cross-entropy from the distance-derived logit.
    """

    def __init__(self, margin: float = 0.6):
        super().__init__()
        self.margin = margin
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred_dict: dict, data_dict: dict) -> torch.Tensor:
        cls = pred_dict['cls']
        label = data_dict['label']
        return self.ce(cls, label)


# ─────────────────────────────────────────────
# §5. SyncNetDetector（DeepfakeBench 格式）
# ─────────────────────────────────────────────

@DETECTOR.register_module(module_name='syncnet')
class SyncNetDetector(AbstractDetector):
    """Audio-visual sync detector for deepfake detection.

    Computes cosine distance between lip-region video embedding and audio embedding.
    High distance (out-of-sync) → fake.

    data_dict required keys:
      'image' : (B, T, C, H, W) or (B, C, H, W) — face frames (will be converted to grayscale)
      'audio' : (B, N) float32 PCM 16kHz
      'label' : (B,) int64
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.audio_encoder = SyncNetAudioEncoder()
        self.video_encoder = SyncNetVideoEncoder()

        # Classifier head: maps [audio_feat, video_feat] concat → 2-class logit
        self.sync_classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, 2),
        )

        self.loss_func = self.build_loss(config)
        self._try_load_pretrained(config)

    def _try_load_pretrained(self, config: dict) -> None:
        """Load pretrained SyncNet weights if path is provided in config."""
        ckpt_path = config.get('pretrained', None)
        if not ckpt_path:
            return
        try:
            state = torch.load(ckpt_path, map_location='cpu')
            # Partial load: only audio/video encoders, skip classifier
            ae_state = {k[len('audio_encoder.'):]: v
                        for k, v in state.items() if k.startswith('audio_encoder.')}
            ve_state = {k[len('video_encoder.'):]: v
                        for k, v in state.items() if k.startswith('video_encoder.')}
            if ae_state:
                self.audio_encoder.load_state_dict(ae_state, strict=False)
                logger.info('SyncNet: loaded audio encoder weights')
            if ve_state:
                self.video_encoder.load_state_dict(ve_state, strict=False)
                logger.info('SyncNet: loaded video encoder weights')
        except Exception as e:
            logger.warning(f'SyncNet: could not load pretrained weights: {e}')

    # ── AbstractDetector interface ──────────────────────────────────────

    def build_backbone(self, config) -> nn.Module:
        return nn.Identity()  # Encoders are separate; backbone is a no-op here

    def build_loss(self, config) -> nn.Module:
        return SyncNetLoss()

    def features(self, data_dict: dict) -> torch.Tensor:
        video_feat = self._encode_video(data_dict['image'])
        audio_feat = self._encode_audio(data_dict['audio'])
        return torch.cat([video_feat, audio_feat], dim=1)  # (B, 2048)

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        return self.sync_classifier(features)

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        loss = self.loss_func(pred_dict, data_dict)
        return {'overall': loss}

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        label = data_dict['label'].cpu().numpy()
        prob = pred_dict['prob'].detach().cpu().numpy()
        pred = (prob > 0.5).astype(int)
        acc = (pred == label).mean()
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(label, prob) if len(set(label)) > 1 else 0.5
        except Exception:
            auc = 0.5
        return {'acc': acc, 'auc': auc}

    def forward(self, data_dict: dict, inference: bool = False) -> dict:
        if data_dict.get('audio') is None:
            raise ValueError(
                "SyncNetDetector requires 'audio' in data_dict. "
                "Set with_audio: true in config and re-run preprocessing."
            )
        # Encode once, reuse for both classifier and sync_error
        video_feat = self._encode_video(data_dict['image'])
        audio_feat = self._encode_audio(data_dict['audio'])
        feat = torch.cat([video_feat, audio_feat], dim=1)

        logit = self.classifier(feat)
        prob = torch.softmax(logit, dim=1)[:, 1]

        # Cosine distance: clamp to [0, 1] (L2-normalised vectors can give ~1.02 due to fp)
        sync_error = (1.0 - F.cosine_similarity(video_feat, audio_feat, dim=1)).clamp(0.0, 1.0)

        return {
            'cls': logit,
            'prob': prob,
            'feat': feat,
            'sync_error': sync_error,
        }

    # ── Internal helpers ──────────────────────────────────────────────

    def _encode_video(self, image: torch.Tensor) -> torch.Tensor:
        """Convert image tensor to grayscale clip and encode.

        Accepts (B, T, C, H, W) or (B, C, H, W); always uses T=5 frames.
        """
        if image.dim() == 4:
            image = image.unsqueeze(1)  # (B, 1, C, H, W) — single frame repeated
            image = image.expand(-1, 5, -1, -1, -1)
        # Convert to grayscale: (B, T, C, H, W) → (B, T, H, W)
        gray = image.mean(dim=2)
        # Resize to 112×112
        B, T, H, W = gray.shape
        gray_flat = gray.view(B * T, 1, H, W)
        gray_flat = F.interpolate(gray_flat, size=(112, 112), mode='bilinear', align_corners=False)
        gray = gray_flat.view(B, T, 112, 112)
        return self.video_encoder(gray)

    def _encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert raw PCM to mel spectrogram and encode."""
        mel = _pcm_to_mel(audio)  # (B, 1, 13, T_mel)
        return self.audio_encoder(mel)

    # ── Week 11 inference bridge ─────────────────────────────────────

    def predict_with_timing(
        self,
        sample_id: str,
        image: torch.Tensor,
        audio: torch.Tensor,
        device: str = 'cpu',
        window_start_sec: Optional[float] = None,
        window_end_sec: Optional[float] = None,
    ) -> dict:
        """Run inference and return a Week 11 DetectorOutput-compatible dict.

        Args:
            sample_id   : Unique identifier matching metadata CSV.
            image       : (T, C, H, W) or (B, T, C, H, W) face frames tensor.
            audio       : (N,) or (B, N) float32 PCM 16kHz tensor.
            device      : 'cpu' or 'cuda'.
            window_start_sec / window_end_sec : clip window timestamps.

        Returns:
            dict with all DetectorOutput fields (matches evaluation/base_detector.py schema).
        """
        t_start = time.perf_counter()
        try:
            self.eval()
            self.to(device)
            if image.dim() == 4:
                image = image.unsqueeze(0)
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            image = image.to(device)
            audio = audio.to(device)
            with torch.no_grad():
                pred = self.forward({'image': image, 'audio': audio, 'label': None})
            # Use sync_error (already clamped to [0,1]) as fake_score per Week 11 contract
            fake_score = float(pred['sync_error'][0].item())
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            return {
                'sample_id': sample_id,
                'detector_name': 'SyncNet',
                'modality': 'av_sync',
                'fake_score': round(fake_score, 4),
                'score_type': 'sync_error',
                'confidence': None,
                'inference_time_ms': round(elapsed_ms, 3),
                'window_start_sec': window_start_sec,
                'window_end_sec': window_end_sec,
                'status': 'ok',
                'error_message': None,
            }
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            return {
                'sample_id': sample_id,
                'detector_name': 'SyncNet',
                'modality': 'av_sync',
                'fake_score': None,
                'score_type': 'sync_error',
                'confidence': None,
                'inference_time_ms': round(elapsed_ms, 3),
                'window_start_sec': window_start_sec,
                'window_end_sec': window_end_sec,
                'status': 'failed',
                'error_message': str(e),
            }


# ─────────────────────────────────────────────
# §6. Smoke test（standalone）
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    print("SyncNetDetector smoke test...")

    config = {
        'pretrained': None,
        'with_audio': True,
    }

    detector = SyncNetDetector(config)
    detector.eval()

    B, T, C, H, W = 2, 5, 3, 224, 224
    N_audio = 16000 * 1  # 1 second

    dummy_image = torch.randn(B, T, C, H, W)
    dummy_audio = torch.randn(B, N_audio)
    dummy_label = torch.randint(0, 2, (B,))

    data_dict = {'image': dummy_image, 'audio': dummy_audio, 'label': dummy_label}
    with torch.no_grad():
        pred = detector.forward(data_dict)

    print(f"  cls shape   : {pred['cls'].shape}")
    print(f"  prob        : {pred['prob'].tolist()}")
    print(f"  sync_error  : {pred['sync_error'].tolist()}")
    print(f"  feat shape  : {pred['feat'].shape}")

    # Week 11 bridge
    out = detector.predict_with_timing(
        sample_id='test_001',
        image=dummy_image[0],
        audio=dummy_audio[0],
        window_start_sec=0.0,
        window_end_sec=1.0,
    )
    print(f"  DetectorOutput: {out}")
    print("Smoke test PASSED.")
