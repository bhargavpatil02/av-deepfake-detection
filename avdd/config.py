"""
Central configuration for the Audio-Visual Deepfake Detection prototype.

Keeping every tunable number in one place makes the experiment reproducible:
the values used for feature extraction are saved inside the trained model
bundle, so a model can never be confused with features from other settings.
"""

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"            # data/real/*.mp4 and data/fake/*.mp4
OUTPUT_DIR = ROOT / "outputs"       # per-video plots + result.json
MODEL_DIR = ROOT / "models"         # trained model bundles
SAMPLES_DIR = ROOT / "samples"      # generated/collected test videos
REPORTS_DIR = ROOT / "reports"      # dataset CSV, metrics, diagrams

for _d in (DATA_DIR, OUTPUT_DIR, MODEL_DIR, SAMPLES_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_PATH = MODEL_DIR / "av_deepfake_rf.joblib"
# MediaPipe Tasks face-landmarker model (downloaded automatically on first use).
FACE_LANDMARKER_MODEL = MODEL_DIR / "face_landmarker.task"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# ---------------------------------------------------------------- input limits
SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}
MAX_FILE_SIZE_MB = 200
MAX_ANALYSIS_DURATION_S = 60.0    # only the first N seconds are analysed (speed)

# ---------------------------------------------------------------- audio
AUDIO_SR = 16_000                # resample everything to 16 kHz mono
AUDIO_N_FFT = 2048
AUDIO_HOP = 512                  # 32 ms hop at 16 kHz
AUDIO_N_MFCC = 20

# ---------------------------------------------------------------- visual
# Dense enough to see lip motion, sparse enough to stay fast on a CPU.
VISUAL_FPS_TARGET = 12.0
VISUAL_MAX_FRAMES = 240
VISUAL_WORK_WIDTH = 480          # frames are downscaled for detection speed
MIN_FACE_FRAMES = 5              # fewer detected faces than this -> visual unusable

# ---------------------------------------------------------------- sync
SYNC_FPS = 25.0                  # common resampling grid for AV correlation
SYNC_MAX_LAG_S = 0.40            # search +/- 400 ms
SYNC_MIN_OVERLAP_S = 1.0         # need at least this much overlap to report
SYNC_MIN_SAMPLES = 12

# ---------------------------------------------------------------- model
RANDOM_STATE = 42
TEST_SIZE = 0.25
CV_FOLDS = 5
