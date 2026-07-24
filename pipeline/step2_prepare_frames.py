"""
Step 2 — Frame extraction and synthetic degradation.

Reads all videos in dataset/raw_videos/, extracts NUM_FRAMES evenly-spaced
frames as the HR ground truth, applies the Real-ESRGAN synthetic degradation
pipeline to create matched LR frames, and produces bicubic-upscaled frames
as the quantitative baseline.

Outputs:
    dataset/frames_hr/   — HR ground truth  (1280x720 PNG)
    dataset/frames_lr/   — Degraded LR      (320x180  PNG)
    output/bicubic/      — Bicubic ×4 up    (1280x720 PNG)

All existing content in those directories is REPLACED.

Usage:
    python step2_prepare_frames.py
    python step2_prepare_frames.py --target-size 1920 1080   # keep 1080p
"""

import sys
import subprocess
import argparse
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    RAW_VIDEO_DIR, FRAMES_HR_DIR, FRAMES_LR_DIR, FRAMES_BIC_DIR,
    SCALE_FACTOR, NUM_FRAMES, FRAME_STRIDE,
    BLUR_SIGMA, JPEG_QUALITY, NOISE_SIGMA,
)


def install_deps():
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q', 'opencv-python', 'numpy', 'tqdm'],
        check=True
    )


def clear_dir(d: Path):
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)


def extract_frames(video_path: Path, out_dir: Path,
                   num_frames: int, target_wh) -> list:
    import cv2
    import numpy as np

    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)

    if total < num_frames:
        num_frames = total
    indices = sorted(set(
        int(i * (total - 1) / (num_frames - 1)) for i in range(num_frames)
    ))

    saved = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        if target_wh is not None:
            frame = cv2.resize(frame, target_wh, interpolation=cv2.INTER_LANCZOS4)
        else:
            # native resolution, cropped to a multiple of the scale factor
            h, w = frame.shape[:2]
            frame = frame[:h - h % SCALE_FACTOR, :w - w % SCALE_FACTOR]
        name  = f'{video_path.stem}_f{idx:06d}.png'
        dest  = out_dir / name
        cv2.imwrite(str(dest), frame)
        saved.append(dest)

    cap.release()
    print(f'  Extracted {len(saved)} frames  ({fps:.1f} fps source)')
    return saved


def degrade(hr_bgr):
    import cv2
    import numpy as np

    ksize = 2 * int(3 * BLUR_SIGMA + 0.5) + 1
    blurred = cv2.GaussianBlur(hr_bgr, (ksize, ksize), BLUR_SIGMA)

    h, w = blurred.shape[:2]
    lr_w, lr_h = w // SCALE_FACTOR, h // SCALE_FACTOR
    lr = cv2.resize(blurred, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    _, enc = cv2.imencode('.jpg', lr, encode_param)
    lr_jpg = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    noise = np.random.normal(0, NOISE_SIGMA, lr_jpg.shape).astype(np.float32)
    lr_noisy = np.clip(lr_jpg.astype(np.float32) + noise, 0, 255).astype(
        __import__('numpy').uint8
    )
    return lr_noisy


def bicubic_up(lr_bgr):
    import cv2
    h, w = lr_bgr.shape[:2]
    return cv2.resize(lr_bgr, (w * SCALE_FACTOR, h * SCALE_FACTOR),
                      interpolation=cv2.INTER_CUBIC)


def main():
    parser = argparse.ArgumentParser(description='Step 2: Frame extraction and degradation')
    parser.add_argument('--target-size', nargs=2, type=int, default=None,
                        metavar=('W', 'H'),
                        help='force HR frame size (default: native size per video)')
    args = parser.parse_args()

    install_deps()
    import cv2
    from tqdm import tqdm

    target_wh = tuple(args.target_size) if args.target_size else None

    print('=' * 60)
    print('STEP 2 — Frame extraction and synthetic degradation')
    print('=' * 60)
    if target_wh:
        print(f'  HR resolution : {target_wh[0]}x{target_wh[1]}')
        print(f'  LR resolution : {target_wh[0]//SCALE_FACTOR}x{target_wh[1]//SCALE_FACTOR}')
    else:
        print(f'  HR resolution : native per video (LR = HR / {SCALE_FACTOR})')
    print(f'  Degradation   : blur s={BLUR_SIGMA}, JPEG q={JPEG_QUALITY}, noise s={NOISE_SIGMA}')
    print()

    videos = sorted(RAW_VIDEO_DIR.glob('*.*'))
    videos = [v for v in videos if v.suffix.lower() in {'.mp4', '.avi', '.mkv', '.mov', '.webm'}]
    if not videos:
        print(f'ERROR: No video files found in {RAW_VIDEO_DIR}')
        print('       Run step1_get_dataset.py first.')
        sys.exit(1)

    # Clear previous runs
    for d in [FRAMES_HR_DIR, FRAMES_LR_DIR, FRAMES_BIC_DIR]:
        clear_dir(d)

    all_hr_paths = []
    for video in videos:
        print(f'Processing: {video.name}')
        paths = extract_frames(video, FRAMES_HR_DIR, NUM_FRAMES, target_wh)
        all_hr_paths.extend(paths)

    print(f'\nTotal HR frames: {len(all_hr_paths)}')

    print('\nCreating LR (degraded) and bicubic-upscaled frames...')
    for hr_path in tqdm(all_hr_paths, unit='frame'):
        hr  = cv2.imread(str(hr_path))
        if hr is None:
            continue

        lr  = degrade(hr)
        bic = bicubic_up(lr)

        lr_name  = hr_path.name
        bic_name = hr_path.name

        cv2.imwrite(str(FRAMES_LR_DIR  / lr_name),  lr)
        cv2.imwrite(str(FRAMES_BIC_DIR / bic_name), bic)

    print(f'\nDataset ready:')
    print(f'  HR  (ground truth) : {FRAMES_HR_DIR}  ({len(list(FRAMES_HR_DIR.glob("*.png")))} files)')
    print(f'  LR  (degraded)     : {FRAMES_LR_DIR}  ({len(list(FRAMES_LR_DIR.glob("*.png")))} files)')
    print(f'  BIC (bicubic ×{SCALE_FACTOR})   : {FRAMES_BIC_DIR} ({len(list(FRAMES_BIC_DIR.glob("*.png")))} files)')
    print('\nStep 2 complete. Run step3_esrgan_inference.py next.')


if __name__ == '__main__':
    main()
