"""
Measure per-frame inference time of ESRGAN (fine-tuned) and RIFE on the GPU.

Times only the model call (warm start, model already loaded). Results are
saved to output/metrics/inference_times.json and printed to console.

Usage:
    python measure_inference_time.py
    python measure_inference_time.py --frames 20
"""

import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'pipeline'))
from config import (
    FRAMES_LR_DIR, FRAMES_SR_DIR, METRICS_DIR,
    ESRGAN_FT_WEIGHTS, ESRGAN_WEIGHTS, RIFE_DIR, SCALE_FACTOR,
)


def time_esrgan(frames, n):
    import cv2
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    weights = ESRGAN_FT_WEIGHTS if ESRGAN_FT_WEIGHTS.exists() else ESRGAN_WEIGHTS
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                    num_grow_ch=32, scale=SCALE_FACTOR)
    upsampler = RealESRGANer(scale=SCALE_FACTOR, model_path=str(weights),
                             model=model, tile=0, tile_pad=10, pre_pad=0,
                             half=torch.cuda.is_available())

    imgs = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in frames[:n + 3]]

    # warmup
    for img in imgs[:3]:
        upsampler.enhance(img, outscale=SCALE_FACTOR)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for img in imgs[3:3 + n]:
        upsampler.enhance(img, outscale=SCALE_FACTOR)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n * 1000
    return ms, weights.name


def time_rife(frames, n):
    import cv2
    import torch
    from torch.nn import functional as F

    sys.path.insert(0, str(RIFE_DIR))
    from train_log.RIFE_HDv3 import Model

    cwd = Path.cwd()
    import os
    os.chdir(RIFE_DIR)
    model = Model()
    model.load_model('train_log', -1)
    model.eval()
    model.device()
    os.chdir(cwd)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def to_tensor(p):
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        t = (torch.tensor(img.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
        _, _, h, w = t.shape
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        return F.pad(t, (0, pw - w, 0, ph - h))

    pairs = [(to_tensor(a), to_tensor(b))
             for a, b in zip(frames[:n + 3], frames[1:n + 4])]

    with torch.no_grad():
        for a, b in pairs[:3]:
            model.inference(a, b)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for a, b in pairs[3:3 + n]:
            model.inference(a, b)
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n * 1000
    return ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=20,
                        help='number of timed frames per model')
    args = parser.parse_args()

    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f'Device: {gpu}')

    lr_frames = sorted(FRAMES_LR_DIR.glob('*.png'))
    sr_frames = sorted(FRAMES_SR_DIR.glob('*.png'))
    if len(lr_frames) < args.frames + 3 or len(sr_frames) < args.frames + 4:
        print('ERROR: not enough frames; run the pipeline (steps 2-3) first.')
        sys.exit(1)

    import cv2
    lr_shape = cv2.imread(str(lr_frames[0])).shape
    sr_shape = cv2.imread(str(sr_frames[0])).shape

    print(f'\nESRGAN x{SCALE_FACTOR}: input {lr_shape[1]}x{lr_shape[0]}, '
          f'{args.frames} frames...')
    esrgan_ms, esrgan_weights = time_esrgan(lr_frames, args.frames)
    print(f'  {esrgan_ms:.1f} ms/frame  ({1000 / esrgan_ms:.1f} fps)  [{esrgan_weights}]')

    print(f'\nRIFE: input {sr_shape[1]}x{sr_shape[0]}, {args.frames} pairs...')
    rife_ms = time_rife(sr_frames, args.frames)
    print(f'  {rife_ms:.1f} ms/interpolated frame  ({1000 / rife_ms:.1f} fps)')

    total_ms = esrgan_ms + rife_ms / 2  # RIFE adds one frame per input pair
    print(f'\nFull pipeline: ~{total_ms:.1f} ms per input frame '
          f'({1000 / total_ms:.1f} fps equivalent)')

    result = {
        'gpu': gpu,
        'timed_frames': args.frames,
        'lr_size': [lr_shape[1], lr_shape[0]],
        'sr_size': [sr_shape[1], sr_shape[0]],
        'esrgan_weights': esrgan_weights,
        'esrgan_ms_per_frame': round(esrgan_ms, 1),
        'rife_ms_per_frame': round(rife_ms, 1),
        'pipeline_ms_per_input_frame': round(total_ms, 1),
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / 'inference_times.json'
    out.write_text(json.dumps(result, indent=2))
    print(f'\nSaved -> {out}')


if __name__ == '__main__':
    main()
