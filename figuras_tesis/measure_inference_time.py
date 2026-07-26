"""
Measure per-frame inference time of every method compared in the thesis.

Covers bicubic upscaling, pretrained ESRGAN, fine-tuned ESRGAN, SwinIR and
RIFE. Times only the model call (warm start, model already loaded) so the
numbers are comparable between methods. Results go to
output/metrics/inference_times.json and are printed as a LaTeX-ready table.

Usage:
    python measure_inference_time.py
    python measure_inference_time.py --frames 20
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'pipeline'))
from config import (
    FRAMES_LR_DIR, FRAMES_SR_DIR, METRICS_DIR,
    ESRGAN_FT_WEIGHTS, ESRGAN_WEIGHTS, RIFE_DIR, SCALE_FACTOR,
    SWINIR_DIR, SWINIR_WEIGHTS,
)


def load_images(frames, n):
    import cv2
    return [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in frames[:n + 3]]


def time_bicubic(imgs, n):
    """Bicubic upscaling on CPU, the baseline every method is compared to."""
    import cv2

    def up(img):
        h, w = img.shape[:2]
        return cv2.resize(img, (w * SCALE_FACTOR, h * SCALE_FACTOR),
                          interpolation=cv2.INTER_CUBIC)

    for img in imgs[:3]:
        up(img)

    t0 = time.perf_counter()
    for img in imgs[3:3 + n]:
        up(img)
    return (time.perf_counter() - t0) / n * 1000


def time_esrgan(imgs, n, weights):
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                    num_grow_ch=32, scale=SCALE_FACTOR)
    upsampler = RealESRGANer(scale=SCALE_FACTOR, model_path=str(weights),
                             model=model, tile=0, tile_pad=10, pre_pad=0,
                             half=torch.cuda.is_available())

    for img in imgs[:3]:
        upsampler.enhance(img, outscale=SCALE_FACTOR)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for img in imgs[3:3 + n]:
        upsampler.enhance(img, outscale=SCALE_FACTOR)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n * 1000

    del upsampler, model
    torch.cuda.empty_cache()
    return ms


def time_swinir(imgs, n):
    """Uses the same builder and forward pass as step5, without tiling."""
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'pipeline'))
    from step5_swinir_inference import build_model, infer_frame

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model().to(device)

    for img in imgs[:3]:
        infer_frame(model, img, device, tile=0)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for img in imgs[3:3 + n]:
        infer_frame(model, img, device, tile=0)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n * 1000

    del model
    torch.cuda.empty_cache()
    return ms


def time_rife(frames, n):
    import cv2
    import torch
    from torch.nn import functional as F

    sys.path.insert(0, str(RIFE_DIR))
    from train_log.RIFE_HDv3 import Model

    cwd = Path.cwd()
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
    return (time.perf_counter() - t0) / n * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=20,
                        help='number of timed frames per model')
    args = parser.parse_args()

    import cv2
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f'Device: {gpu}')

    lr_frames = sorted(FRAMES_LR_DIR.glob('*.png'))
    sr_frames = sorted(FRAMES_SR_DIR.glob('*.png'))
    if len(lr_frames) < args.frames + 3 or len(sr_frames) < args.frames + 4:
        print('ERROR: not enough frames; run the pipeline (steps 2-3) first.')
        sys.exit(1)

    lr_shape = cv2.imread(str(lr_frames[0])).shape
    sr_shape = cv2.imread(str(sr_frames[0])).shape
    imgs = load_images(lr_frames, args.frames)

    n = args.frames
    print(f'\nInput {lr_shape[1]}x{lr_shape[0]} -> '
          f'{lr_shape[1] * SCALE_FACTOR}x{lr_shape[0] * SCALE_FACTOR}, '
          f'{n} frames per method\n')

    times = {}

    print('Bicubic (CPU)...')
    times['bicubic'] = time_bicubic(imgs, n)
    print(f'  {times["bicubic"]:.1f} ms/frame')

    print('ESRGAN pretrained...')
    times['esrgan_pretrained'] = time_esrgan(imgs, n, ESRGAN_WEIGHTS)
    print(f'  {times["esrgan_pretrained"]:.1f} ms/frame')

    if ESRGAN_FT_WEIGHTS.exists():
        print('ESRGAN fine-tuned...')
        times['esrgan_finetuned'] = time_esrgan(imgs, n, ESRGAN_FT_WEIGHTS)
        print(f'  {times["esrgan_finetuned"]:.1f} ms/frame')
    else:
        print('WARNING: fine-tuned weights not found, skipping.')

    if SWINIR_WEIGHTS.exists() and SWINIR_DIR.exists():
        print('SwinIR...')
        times['swinir'] = time_swinir(imgs, n)
        print(f'  {times["swinir"]:.1f} ms/frame')
    else:
        print('WARNING: SwinIR repo or weights not found, skipping.')

    print(f'RIFE ({sr_shape[1]}x{sr_shape[0]})...')
    times['rife'] = time_rife(sr_frames, n)
    print(f'  {times["rife"]:.1f} ms/interpolated frame')

    base = times.get('esrgan_finetuned', times['esrgan_pretrained'])
    total = base + times['rife'] / 2  # RIFE adds one frame per input pair
    print(f'\nFull pipeline: ~{total:.1f} ms per input frame '
          f'({1000 / total:.1f} fps equivalent)')

    print('\n' + '=' * 62)
    print('LaTeX table')
    print('=' * 62)
    labels = [
        ('bicubic', 'Bic\\\'ubico'),
        ('esrgan_pretrained', 'ESRGAN preentrenado'),
        ('swinir', 'SwinIR'),
        ('esrgan_finetuned', 'ESRGAN con ajuste fino'),
    ]
    for key, label in labels:
        if key in times:
            print(f'{label} & {times[key]:.1f} & {1000 / times[key]:.1f} \\\\')

    result = {
        'gpu': gpu,
        'timed_frames': n,
        'lr_size': [lr_shape[1], lr_shape[0]],
        'sr_size': [sr_shape[1], sr_shape[0]],
        'esrgan_weights': (ESRGAN_FT_WEIGHTS.name if ESRGAN_FT_WEIGHTS.exists()
                           else ESRGAN_WEIGHTS.name),
        'ms_per_frame': {k: round(v, 1) for k, v in times.items()},
        'pipeline_ms_per_input_frame': round(total, 1),
        # kept for backwards compatibility with the first measurement
        'esrgan_ms_per_frame': round(base, 1),
        'rife_ms_per_frame': round(times['rife'], 1),
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / 'inference_times.json'
    out.write_text(json.dumps(result, indent=2))
    print(f'\nSaved -> {out}')


if __name__ == '__main__':
    main()
