"""Shared configuration for the CCTV video enhancement pipeline."""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------
IS_COLAB = os.path.exists('/content/sample_data') or 'COLAB_GPU' in os.environ

if IS_COLAB:
    # On Colab: mirror the Jose/ structure under /content/jose/
    # Run this once in a Colab cell before any step:
    #   !mkdir -p /content/jose
    #   # Then upload your weights/ and dataset/ folders, or mount Drive.
    BASE_DIR = Path('/content/jose')
else:
    # Local: raiz del proyecto, tres niveles por encima de pipeline/
    # (pipeline -> reproducibility -> ColCom Paper -> raiz)
    BASE_DIR = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# Directory layout  (mirrors the Jose/ folder structure)
# ---------------------------------------------------------------------------
PIPELINE_DIR    = Path(__file__).resolve().parent
WEIGHTS_DIR     = BASE_DIR / 'weights'
DATASET_DIR     = BASE_DIR / 'dataset'
OUTPUT_DIR      = BASE_DIR / 'output'
PAPER_DIR       = BASE_DIR / 'ColCom Paper'
FIGURES_DIR     = PAPER_DIR / 'figures'

FRAMES_HR_DIR   = DATASET_DIR / 'frames_hr'
FRAMES_LR_DIR   = DATASET_DIR / 'frames_lr'
FRAMES_BIC_DIR  = OUTPUT_DIR / 'bicubic'
FRAMES_SR_DIR   = OUTPUT_DIR / 'sr'
FRAMES_RIFE_DIR = OUTPUT_DIR / 'rife'
RAW_VIDEO_DIR   = DATASET_DIR / 'raw_videos'
METRICS_DIR     = OUTPUT_DIR / 'metrics'
RIFE_DIR        = BASE_DIR / 'ECCV2022-RIFE'

for _d in [WEIGHTS_DIR, FRAMES_HR_DIR, FRAMES_LR_DIR, FRAMES_BIC_DIR,
           FRAMES_SR_DIR, FRAMES_RIFE_DIR, RAW_VIDEO_DIR, METRICS_DIR,
           FIGURES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)


def condition_of(frame_name: str) -> str:
    """Lighting/scene condition parsed from the frame filename prefix.

    Videos in dataset/raw_videos/ must be named <condition>_<name>.mp4
    (e.g. day_street.mp4, night_parking.mp4, indoor_lobby.mp4). Frames
    inherit the video stem, so the condition is the text before the
    first underscore. Unprefixed legacy names map to 'unspecified'.
    """
    prefix = frame_name.split('_', 1)[0].lower()
    return prefix if prefix in {'day', 'night', 'indoor'} else 'unspecified'

# ---------------------------------------------------------------------------
# Processing parameters  (fixed for reproducibility)
# ---------------------------------------------------------------------------
SCALE_FACTOR  = 4       # ESRGAN x4 upscaling
RIFE_EXP      = 1       # 2^1 = 2x frame-rate interpolation
NUM_FRAMES    = 60      # number of frames used in the experiment
FRAME_STRIDE  = 5       # sample every N-th frame from the source video

# Synthetic degradation (matches Real-ESRGAN training distribution)
BLUR_SIGMA    = 1.5
JPEG_QUALITY  = 50
NOISE_SIGMA   = 5.0

# ---------------------------------------------------------------------------
# Model weights
# ---------------------------------------------------------------------------
ESRGAN_WEIGHTS     = WEIGHTS_DIR / 'RealESRGAN_x4plus.pth'
ESRGAN_WEIGHTS_URL = (
    'https://github.com/xinntao/Real-ESRGAN'
    '/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
)

# RIFE v4.6 pretrained model (official megvii-research release)
RIFE_REPO_URL      = 'https://github.com/megvii-research/ECCV2022-RIFE.git'
RIFE_MODEL_GDRIVE  = '1APIzVeI-4ZZCEuIRE1m6WYfSCaOsi_7v'  # train_log.zip on GDrive

# SwinIR (Vision Transformer super-resolution, official release)
SWINIR_REPO_URL    = 'https://github.com/JingyunLiang/SwinIR.git'
SWINIR_DIR         = BASE_DIR / 'SwinIR'
SWINIR_WEIGHTS     = WEIGHTS_DIR / '003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth'
SWINIR_WEIGHTS_URL = (
    'https://github.com/JingyunLiang/SwinIR'
    '/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth'
)
FRAMES_SWINIR_DIR  = OUTPUT_DIR / 'swinir'

# Real-ESRGAN fine-tuning (thesis objective 3: model training)
REALESRGAN_REPO_URL   = 'https://github.com/xinntao/Real-ESRGAN.git'
REALESRGAN_DIR        = BASE_DIR / 'Real-ESRGAN'
ESRGAN_FT_WEIGHTS     = WEIGHTS_DIR / 'RealESRGAN_x4plus_finetuned.pth'
FRAMES_SR_FT_DIR      = OUTPUT_DIR / 'sr_finetuned'
FINETUNE_ITERS        = 5000
FINETUNE_LR           = 1e-4
FINETUNE_BATCH        = 4

# Thesis outputs
THESIS_FIGURES_DIR    = OUTPUT_DIR / 'thesis_figures'

# ---------------------------------------------------------------------------
# Figure settings  (IEEE IEEEtran two-column conference format)
# ---------------------------------------------------------------------------
FIG_SINGLE_W  = 3.5    # one-column width in inches
FIG_DOUBLE_W  = 7.16   # two-column width in inches
FIG_DPI       = 400   # con bbox tight quedan >300 DPI efectivos al ancho de dos columnas
