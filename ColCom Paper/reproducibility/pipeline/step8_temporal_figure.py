"""
Step 8 - Temporal interpolation figure from consecutive frames.

Steps 2 to 4 sample 60 frames spread over the whole recording, which is what
the spatial metrics need but is not a valid input for temporal interpolation:
two of those frames are seconds apart. This script builds the temporal figure
from frames that really are consecutive, so the interpolated frame corresponds
to the instant t+0.5 of the original sequence.

It reads a short run of consecutive frames from the source video, applies the
same degradation as Step 2, enhances frames t and t+1 with ESRGAN, and
interpolates the middle one with RIFE.

Inputs  : dataset/raw_videos/
Outputs : ColCom Paper/figures/fig5_rife.png

Usage:
    python step8_temporal_figure.py
    python step8_temporal_figure.py --video day_street.mp4 --frame 1269
"""

import os
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    RAW_VIDEO_DIR, FIGURES_DIR, ESRGAN_WEIGHTS, RIFE_DIR, SCALE_FACTOR,
    BLUR_SIGMA, JPEG_QUALITY, NOISE_SIGMA,
)

FIG_DOUBLE_W = 7.16  # ancho de dos columnas en IEEEtran, en pulgadas


def degrade(hr_bgr):
    """Same synthetic degradation as Step 2."""
    import cv2
    import numpy as np

    ksize = 2 * int(3 * BLUR_SIGMA + 0.5) + 1
    blurred = cv2.GaussianBlur(hr_bgr, (ksize, ksize), BLUR_SIGMA)

    h, w = blurred.shape[:2]
    lr = cv2.resize(blurred, (w // SCALE_FACTOR, h // SCALE_FACTOR),
                    interpolation=cv2.INTER_CUBIC)

    _, enc = cv2.imencode('.jpg', lr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    lr = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    noise = np.random.normal(0, NOISE_SIGMA, lr.shape).astype(np.float32)
    return np.clip(lr.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def build_upsampler():
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                    num_grow_ch=32, scale=SCALE_FACTOR)
    return RealESRGANer(scale=SCALE_FACTOR, model_path=str(ESRGAN_WEIGHTS),
                        model=model, tile=0, tile_pad=10, pre_pad=0,
                        half=torch.cuda.is_available())


def interpolate(frame_a, frame_b):
    """Middle frame between two BGR images, at the original size."""
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
    h, w = frame_a.shape[:2]
    ph = ((h - 1) // 32 + 1) * 32
    pw = ((w - 1) // 32 + 1) * 32

    def to_tensor(img):
        t = torch.tensor(img.transpose(2, 0, 1)).to(device).float() / 255.
        return F.pad(t.unsqueeze(0), (0, pw - w, 0, ph - h))

    with torch.no_grad():
        mid = model.inference(to_tensor(frame_a), to_tensor(frame_b))

    out = mid[0, :, :h, :w].clamp(0, 1).cpu().numpy().transpose(1, 2, 0)
    return (out * 255).round().astype('uint8')


def build_figure(frame_t, frame_interp, frame_t1):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    panels = [
        (frame_t,      'Fotograma $t$\n(ESRGAN)',                  'black',   False),
        (frame_interp, 'Fotograma $t\\!+\\!0.5$\n(RIFE, sintetizado)', '#b22222', True),
        (frame_t1,     'Fotograma $t\\!+\\!1$\n(ESRGAN)',           'black',   False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(FIG_DOUBLE_W, 2.3))
    for ax, (img, title, color, bold) in zip(axes, panels):
        ax.imshow(img[:, :, ::-1])
        ax.set_title(title, fontsize=7, color=color, pad=3,
                     fontweight='bold' if bold else 'normal', linespacing=1.4)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('red' if bold else '#aaaaaa')
            spine.set_linewidth(1.0 if bold else 0.5)
        ax.set_frame_on(True)

    fig.tight_layout(pad=0.25, w_pad=0.4)
    return fig


def main():
    parser = argparse.ArgumentParser(description='Rebuild the temporal figure from consecutive frames')
    parser.add_argument('--video', default='day_street.mp4')
    parser.add_argument('--frame', type=int, default=1269,
                        help='index of frame t; t+1 is the next one in the video')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    import cv2
    import numpy as np
    import torch

    video_path = RAW_VIDEO_DIR / args.video
    if not video_path.exists():
        print(f'ERROR: {video_path} not found')
        sys.exit(1)

    np.random.seed(args.seed)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok_a, hr_a = cap.read()
    ok_b, hr_b = cap.read()
    cap.release()
    if not (ok_a and ok_b):
        print(f'ERROR: could not read frames {args.frame} and {args.frame + 1}')
        sys.exit(1)

    h, w = hr_a.shape[:2]
    hr_a = hr_a[:h - h % SCALE_FACTOR, :w - w % SCALE_FACTOR]
    hr_b = hr_b[:h - h % SCALE_FACTOR, :w - w % SCALE_FACTOR]

    print('=' * 60)
    print('Temporal figure from consecutive frames')
    print('=' * 60)
    print(f'  Source : {video_path.name} ({fps:.1f} fps)')
    print(f'  Frames : {args.frame} and {args.frame + 1} (consecutive, {1000 / fps:.1f} ms apart)')
    print(f'  Device : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    print()

    upsampler = build_upsampler()
    sr_a, _ = upsampler.enhance(degrade(hr_a), outscale=SCALE_FACTOR)
    sr_b, _ = upsampler.enhance(degrade(hr_b), outscale=SCALE_FACTOR)
    del upsampler
    torch.cuda.empty_cache()

    mid = interpolate(sr_a, sr_b)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / 'fig5_rife.png'
    fig = build_figure(sr_a, mid, sr_b)
    fig.savefig(str(out), dpi=300, bbox_inches='tight')
    print(f'Saved -> {out}')


if __name__ == '__main__':
    main()
