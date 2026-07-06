"""
Step 3 — Real-ESRGAN super-resolution inference.

Runs the official Real-ESRGAN x4plus model on every LR frame produced by
Step 2 and saves the super-resolved frames to output/sr/.

The official realesrgan package is installed automatically.
Model weights are downloaded automatically if not found in weights/.

Usage:
    python step3_esrgan_inference.py
    python step3_esrgan_inference.py --tile 256   # reduce tile size on low-VRAM GPUs
"""

import sys
import subprocess
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    FRAMES_LR_DIR, FRAMES_SR_DIR,
    ESRGAN_WEIGHTS, ESRGAN_WEIGHTS_URL,
    SCALE_FACTOR,
)


def install_deps():
    pkgs = ['realesrgan', 'basicsr', 'facexlib', 'gfpgan', 'torch', 'torchvision', 'tqdm']
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + pkgs, check=True)


def download_weights():
    import urllib.request
    ESRGAN_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading Real-ESRGAN x4plus weights...')
    print(f'  Source : {ESRGAN_WEIGHTS_URL}')
    print(f'  Target : {ESRGAN_WEIGHTS}')
    urllib.request.urlretrieve(ESRGAN_WEIGHTS_URL, ESRGAN_WEIGHTS)
    print('  Download complete.')


def build_upsampler(tile: int, half_precision: bool):
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23,
        num_grow_ch=32, scale=SCALE_FACTOR,
    )
    upsampler = RealESRGANer(
        scale=SCALE_FACTOR,
        model_path=str(ESRGAN_WEIGHTS),
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=half_precision,
    )
    return upsampler


def main():
    parser = argparse.ArgumentParser(description='Step 3: ESRGAN super-resolution')
    parser.add_argument('--tile', type=int, default=0,
                        help='Tile size for inference (0=no tiling, 256 for low VRAM)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU inference (slow)')
    args = parser.parse_args()

    install_deps()
    import cv2
    import torch
    from tqdm import tqdm

    print('=' * 60)
    print('STEP 3 — Real-ESRGAN x4plus inference')
    print('=' * 60)

    if not ESRGAN_WEIGHTS.exists():
        download_weights()
    else:
        print(f'Weights found: {ESRGAN_WEIGHTS.name}')

    use_gpu  = torch.cuda.is_available() and not args.cpu
    half_prec = use_gpu
    device   = 'GPU' if use_gpu else 'CPU'
    print(f'Device        : {device}')
    if use_gpu:
        print(f'GPU           : {torch.cuda.get_device_name(0)}')
    print(f'Tile size     : {args.tile if args.tile > 0 else "none (full image)"}')
    print()

    lr_frames = sorted(FRAMES_LR_DIR.glob('*.png'))
    if not lr_frames:
        print(f'ERROR: No LR frames found in {FRAMES_LR_DIR}')
        print('       Run step2_prepare_frames.py first.')
        sys.exit(1)

    print(f'Loading model...')
    upsampler = build_upsampler(args.tile, half_prec)
    print(f'Model loaded. Processing {len(lr_frames)} frames...\n')

    errors = 0
    for lr_path in tqdm(lr_frames, unit='frame'):
        img = cv2.imread(str(lr_path), cv2.IMREAD_COLOR)
        if img is None:
            errors += 1
            continue

        try:
            output, _ = upsampler.enhance(img, outscale=SCALE_FACTOR)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f'\nGPU OOM on {lr_path.name}. Re-run with --tile 256')
                sys.exit(1)
            raise

        sr_path = FRAMES_SR_DIR / lr_path.name
        cv2.imwrite(str(sr_path), output)

    sr_count = len(list(FRAMES_SR_DIR.glob('*.png')))
    print(f'\nESRGAN inference complete.')
    print(f'  Processed  : {len(lr_frames)} frames')
    print(f'  Saved to   : {FRAMES_SR_DIR}  ({sr_count} files)')
    if errors:
        print(f'  Errors     : {errors} frames skipped (corrupt input)')
    print('\nStep 3 complete. Run step4_rife_inference.py next.')


if __name__ == '__main__':
    main()
