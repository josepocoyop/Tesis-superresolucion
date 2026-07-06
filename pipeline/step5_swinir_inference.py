"""
Step 5 — SwinIR (Vision Transformer) super-resolution inference.

Runs the official SwinIR-M x4 real-world model (GAN variant, trained with
the BSRGAN degradation model) on every LR frame produced by Step 2 and
saves the results to output/swinir/.

This provides the Vision Transformer branch of the thesis comparison:
    Bicubic (baseline)  vs  ESRGAN (GAN/CNN)  vs  SwinIR (Transformer)

The official SwinIR repository is cloned automatically and the pretrained
weights are downloaded if not found in weights/.

Usage:
    python step5_swinir_inference.py
    python step5_swinir_inference.py --tile 256   # reduce tile size on low-VRAM GPUs
"""

import sys
import subprocess
import argparse
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    FRAMES_LR_DIR, FRAMES_SWINIR_DIR,
    SWINIR_DIR, SWINIR_REPO_URL,
    SWINIR_WEIGHTS, SWINIR_WEIGHTS_URL,
    SCALE_FACTOR,
)


def install_deps():
    pkgs = ['torch', 'torchvision', 'opencv-python', 'numpy', 'timm', 'tqdm']
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + pkgs, check=True)


def clone_repo():
    if SWINIR_DIR.exists():
        print(f'SwinIR repo found: {SWINIR_DIR}')
        return
    print(f'Cloning SwinIR repository...')
    subprocess.run(['git', 'clone', '--depth', '1', SWINIR_REPO_URL, str(SWINIR_DIR)],
                   check=True)


def download_weights():
    import urllib.request
    SWINIR_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    print(f'Downloading SwinIR-M x4 real-world weights...')
    print(f'  Source : {SWINIR_WEIGHTS_URL}')
    print(f'  Target : {SWINIR_WEIGHTS}')
    urllib.request.urlretrieve(SWINIR_WEIGHTS_URL, SWINIR_WEIGHTS)
    print('  Download complete.')


def build_model():
    sys.path.insert(0, str(SWINIR_DIR))
    import torch
    from models.network_swinir import SwinIR

    # Parameters of the 003_realSR SwinIR-M x4 release
    model = SwinIR(
        upscale=SCALE_FACTOR, in_chans=3, img_size=64, window_size=8,
        img_range=1.0, depths=[6, 6, 6, 6, 6, 6], embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2,
        upsampler='nearest+conv', resi_connection='1conv',
    )
    state = torch.load(str(SWINIR_WEIGHTS), map_location='cpu')
    key = 'params_ema' if 'params_ema' in state else 'params'
    model.load_state_dict(state[key] if key in state else state, strict=True)
    model.eval()
    return model


def infer_frame(model, img_bgr, device, tile: int, window_size: int = 8):
    import torch
    import numpy as np
    import cv2

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    ten = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    # Pad input so height/width are multiples of the window size
    _, _, h, w = ten.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    ten = torch.nn.functional.pad(ten, (0, pad_w, 0, pad_h), mode='reflect')

    with torch.no_grad():
        if tile <= 0:
            out = model(ten)
        else:
            out = tiled_forward(model, ten, tile, window_size)

    out = out[:, :, :h * SCALE_FACTOR, :w * SCALE_FACTOR]
    out = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return cv2.cvtColor((out * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


def tiled_forward(model, ten, tile, window_size):
    import torch
    _, _, h, w = ten.shape
    tile = min(tile, h, w)
    tile = tile - tile % window_size
    stride  = tile - 32
    sf      = SCALE_FACTOR
    out     = torch.zeros(1, 3, h * sf, w * sf, device=ten.device)
    weight  = torch.zeros_like(out)
    ys = list(range(0, h - tile, stride)) + [h - tile]
    xs = list(range(0, w - tile, stride)) + [w - tile]
    for y in ys:
        for x in xs:
            patch = ten[:, :, y:y + tile, x:x + tile]
            o = model(patch)
            out[:, :, y * sf:(y + tile) * sf, x * sf:(x + tile) * sf] += o
            weight[:, :, y * sf:(y + tile) * sf, x * sf:(x + tile) * sf] += 1
    return out / weight


def main():
    parser = argparse.ArgumentParser(description='Step 5: SwinIR super-resolution')
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
    print('STEP 5 — SwinIR-M x4 (Vision Transformer) inference')
    print('=' * 60)

    clone_repo()
    if not SWINIR_WEIGHTS.exists():
        download_weights()
    else:
        print(f'Weights found: {SWINIR_WEIGHTS.name}')

    use_gpu = torch.cuda.is_available() and not args.cpu
    device  = torch.device('cuda' if use_gpu else 'cpu')
    print(f'Device        : {"GPU" if use_gpu else "CPU"}')
    if use_gpu:
        print(f'GPU           : {torch.cuda.get_device_name(0)}')
    print(f'Tile size     : {args.tile if args.tile > 0 else "none (full image)"}')
    print()

    lr_frames = sorted(FRAMES_LR_DIR.glob('*.png'))
    if not lr_frames:
        print(f'ERROR: No LR frames found in {FRAMES_LR_DIR}')
        print('       Run step2_prepare_frames.py first.')
        sys.exit(1)

    print('Loading model...')
    model = build_model().to(device)
    print(f'Model loaded. Processing {len(lr_frames)} frames...\n')

    FRAMES_SWINIR_DIR.mkdir(parents=True, exist_ok=True)
    errors = 0
    for lr_path in tqdm(lr_frames, unit='frame'):
        img = cv2.imread(str(lr_path), cv2.IMREAD_COLOR)
        if img is None:
            errors += 1
            continue
        try:
            output = infer_frame(model, img, device, args.tile)
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f'\nGPU OOM on {lr_path.name}. Re-run with --tile 256')
                sys.exit(1)
            raise
        cv2.imwrite(str(FRAMES_SWINIR_DIR / lr_path.name), output)

    count = len(list(FRAMES_SWINIR_DIR.glob('*.png')))
    print(f'\nSwinIR inference complete.')
    print(f'  Processed  : {len(lr_frames)} frames')
    print(f'  Saved to   : {FRAMES_SWINIR_DIR}  ({count} files)')
    if errors:
        print(f'  Errors     : {errors} frames skipped (corrupt input)')
    print('\nStep 5 complete. Run step6_finetune_esrgan.py next (or step7 for metrics).')


if __name__ == '__main__':
    main()
