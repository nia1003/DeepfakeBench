import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

from metrics.registry import DETECTOR
from .utils import slowfast

from .facexray_detector import FaceXrayDetector
from .xception_detector import XceptionDetector
from .efficientnetb4_detector import EfficientDetector
from .resnet34_detector import ResnetDetector
from .f3net_detector import F3netDetector
from .meso4_detector import Meso4Detector
from .meso4Inception_detector import Meso4InceptionDetector
from .spsl_detector import SpslDetector
from .core_detector import CoreDetector
from .capsule_net_detector import CapsuleNetDetector
from .srm_detector import SRMDetector
from .ucf_detector import UCFDetector
from .recce_detector import RecceDetector
from .fwa_detector import FWADetector
from .ffd_detector import FFDDetector
from .videomae_detector import VideoMAEDetector
from .clip_detector import CLIPDetector
from .timesformer_detector import TimeSformerDetector
from .xclip_detector import XCLIPDetector
from .sbi_detector import SBIDetector
from .ftcn_detector import FTCNDetector
from .i3d_detector import I3DDetector
from .altfreezing_detector import AltFreezingDetector
from .stil_detector import STILDetector
from .lsda_detector import LSDADetector
from .sladd_detector import SLADDXceptionDetector
from .pcl_xception_detector import PCLXceptionDetector
from .iid_detector import IIDDetector
from .lrl_detector import LRLDetector
from .rfm_detector import RFMDetector
from .uia_vit_detector import UIAViTDetector
from .multi_attention_detector import MultiAttentionDetector
from .sia_detector import SIADetector
from .tall_detector import TALLDetector
from .effort_detector import EffortDetector

# ── Anti-Deepfake-Box 整合 (framework branch) ──────────────────────────────
# B-1: 最小可執行整合驗證 (dummy detector)
from .dummy_detector import DummyDetector

# B-2: 三模態 ADB Detector 適配器
# 需要將 anti-deepfake-box 根目錄加入 PYTHONPATH：
#   export PYTHONPATH=/path/to/anti-deepfake-box:$PYTHONPATH
try:
    from .adb_visual_detector import ADBVisualDetector
    from .adb_rppg_detector import ADBRPPGDetector
    from .adb_sync_detector import ADBSyncDetector
except ImportError as _e:
    import warnings
    warnings.warn(
        f"ADB detectors 未載入（請將 anti-deepfake-box 加入 PYTHONPATH）：{_e}",
        stacklevel=2,
    )
