"""
Quality evaluation of the frames synthesized by RIFE.

The interpolated frames have no ground truth in the pipeline: RIFE inserts them
between two real frames, so there is nothing to compare them against. The usual
way around this is a hold-out test. Consecutive triplets (t-1, t, t+1) are taken
from the source video, RIFE interpolates the middle one from the two outer
frames, and the result is compared with the real frame t that was hidden from
the model.

The test is slightly harder than deployment: here the two inputs are 2 frame
intervals apart, while in the pipeline they are 1 interval apart, so the motion
RIFE has to resolve is twice as large.

Only triplets that actually contain motion are evaluated. On a fixed camera
most of the footage is static, and on a static triplet averaging the two
neighbours already reproduces the hidden frame, so those segments say nothing
about the interpolation.

Usage:
    python step9_evaluate_rife.py
    python step9_evaluate_rife.py --video day_street.mp4 --triplets 60
"""

import os
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_VIDEO_DIR, METRICS_DIR, RIFE_DIR
from step7_compute_metrics import compute_psnr, compute_ssim, compute_lpips


def load_rife():
    import torch

    sys.path.insert(0, str(RIFE_DIR))
    from train_log.RIFE_HDv3 import Model

    cwd = Path.cwd()
    os.chdir(RIFE_DIR)
    model = Model()
    model.load_model('train_log', -1)
    model.eval()
    model.device()
    os.chdir(cwd)
    return model


def interpolate(model, frame_a, frame_b):
    """Middle frame between two BGR images, returned at the original size."""
    import torch
    from torch.nn import functional as F

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


def main():
    parser = argparse.ArgumentParser(description='Hold-out evaluation of RIFE interpolation')
    parser.add_argument('--video', default='day_street.mp4')
    parser.add_argument('--start', type=int, default=0, help='first frame index')
    parser.add_argument('--triplets', type=int, default=60,
                        help='number of (t-1, t, t+1) triplets to evaluate')
    parser.add_argument('--stride', type=int, default=15,
                        help='frames between one candidate triplet and the next')
    parser.add_argument('--min-motion', type=float, default=3.0,
                        help='mean absolute difference (DN) required between t-1 and t+1')
    parser.add_argument('--no-lpips', action='store_true')
    args = parser.parse_args()

    import cv2
    import numpy as np
    import torch

    video_path = RAW_VIDEO_DIR / args.video
    if not video_path.exists():
        print(f'ERROR: {video_path} not found')
        sys.exit(1)

    print('=' * 60)
    print('RIFE hold-out evaluation')
    print('=' * 60)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'  Source  : {video_path.name}  ({total} frames, {fps:.1f} fps)')
    print(f'  Triplets: {args.triplets}, candidates every {args.stride} frames,')
    print(f'            kept only if motion >= {args.min_motion} DN')
    print(f'  Device  : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    print()

    lpips_fn = None
    if not args.no_lpips:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex')

    model = load_rife()

    scores = {'rife': [], 'blend': []}
    motions = []
    used = 0
    idx = args.start
    while used < args.triplets and idx + 2 < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        triplet = []
        for _ in range(3):
            ok, frame = cap.read()
            if not ok:
                break
            triplet.append(frame)
        if len(triplet) < 3:
            break
        idx += args.stride

        before, truth, after = triplet
        motion = float(np.mean(cv2.absdiff(before, after)))
        if motion < args.min_motion:
            continue
        motions.append(motion)

        rife = interpolate(model, before, after)
        # averaging the two neighbours is the naive alternative to interpolation
        blend = cv2.addWeighted(before, 0.5, after, 0.5, 0)

        for name, pred in (('rife', rife), ('blend', blend)):
            row = [compute_psnr(truth, pred), compute_ssim(truth, pred)]
            row.append(compute_lpips(lpips_fn, truth, pred) if lpips_fn else float('nan'))
            scores[name].append(row)
        used += 1

    cap.release()

    if not used:
        print('ERROR: no triplets evaluated')
        sys.exit(1)

    print(f'{used} triplets evaluated, mean motion {np.mean(motions):.1f} DN\n')
    print(f'{"Method":<28}{"PSNR":>10}{"SSIM":>10}{"LPIPS":>10}')
    print('-' * 58)
    summary = {'video': video_path.name, 'triplets': used, 'stride': args.stride,
               'min_motion': args.min_motion,
               'mean_motion': round(float(np.mean(motions)), 2)}
    for name, label in (('blend', 'Promedio de vecinos'), ('rife', 'RIFE (interpolado)')):
        m = np.nanmean(np.array(scores[name]), axis=0)
        print(f'{label:<28}{m[0]:>10.2f}{m[1]:>10.4f}{m[2]:>10.4f}')
        summary[name] = {'psnr': round(float(m[0]), 2),
                         'ssim': round(float(m[1]), 4),
                         'lpips': round(float(m[2]), 4)}

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = METRICS_DIR / 'rife_holdout.json'
    out.write_text(json.dumps(summary, indent=2))
    print(f'\nSaved -> {out}')


if __name__ == '__main__':
    main()
