"""
Degradation and restoration test on a clip that was never used for training.

Takes the original recording as ground truth, degrades it by a chosen factor
(same protocol as step2: blur, downscaling, JPEG, noise), restores it with the
fine-tuned ESRGAN and measures how close the result is to the original. Because
the ground truth exists here, this test does produce PSNR, SSIM and LPIPS,
which the plain enhancement in enhance_video.py cannot.

The network is x4. For a factor of 2 the output is resampled to the requested
size, which is the standard Real-ESRGAN behaviour for a non-native scale.

Usage:
    python degrade_and_restore.py --factor 2
    python degrade_and_restore.py --factor 4 --frames 60 --no-lpips
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'pipeline'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OUTPUT_DIR, BLUR_SIGMA, JPEG_QUALITY, NOISE_SIGMA
from step7_compute_metrics import compute_psnr, compute_ssim, compute_lpips
from make_demo_video import build_upsampler, FINETUNED_WEIGHTS
from enhance_video import resolve_video, DEMO_INPUT_DIR

DEMO_DIR = OUTPUT_DIR / 'demo_videos'


def degrade_by(hr_bgr, factor):
    """step2 degradation with the downscaling factor as a parameter."""
    import cv2
    import numpy as np

    ksize = 2 * int(3 * BLUR_SIGMA + 0.5) + 1
    blurred = cv2.GaussianBlur(hr_bgr, (ksize, ksize), BLUR_SIGMA)

    h, w = blurred.shape[:2]
    lr = cv2.resize(blurred, (w // factor, h // factor), interpolation=cv2.INTER_CUBIC)

    _, enc = cv2.imencode('.jpg', lr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    lr_jpg = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    noise = np.random.normal(0, NOISE_SIGMA, lr_jpg.shape).astype(np.float32)
    return np.clip(lr_jpg.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description='Degrade a real clip and restore it')
    parser.add_argument('--video', default='test_video.mp4')
    parser.add_argument('--start', type=int, default=120)
    parser.add_argument('--frames', type=int, default=150)
    parser.add_argument('--factor', type=int, default=2, help='degradation factor')
    parser.add_argument('--roi', nargs=4, type=int, default=None, metavar=('X', 'Y', 'W', 'H'))
    parser.add_argument('--tile', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no-lpips', action='store_true')
    parser.add_argument('--no-video', action='store_true', help='metrics only')
    args = parser.parse_args()

    import cv2
    import numpy as np
    import torch
    from tqdm import tqdm

    video_path = resolve_video(args.video)
    if video_path is None:
        print(f'ERROR: {args.video} not found in {DEMO_INPUT_DIR}')
        sys.exit(1)
    if not FINETUNED_WEIGHTS.exists():
        print(f'ERROR: {FINETUNED_WEIGHTS} not found. Run step6_finetune_esrgan.py first.')
        sys.exit(1)

    f = args.factor
    np.random.seed(args.seed)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    if args.roi:
        x, y, w, h = args.roi
    else:
        x = y = 0
        w, h = src_w, src_h
    w -= w % f
    h -= h % f

    print('=' * 60)
    print(f'Degradation x{f} and restoration')
    print('=' * 60)
    print(f'  Source     : {video_path.name}  ({src_w}x{src_h}, {fps:.1f} fps)')
    print(f'  Clip       : frames {args.start}-{args.start + args.frames - 1}')
    print(f'  Reference  : {w}x{h}   ->   degraded {w // f}x{h // f}')
    print(f'  Degradation: blur s={BLUR_SIGMA}, bicubic /{f}, JPEG q={JPEG_QUALITY}, noise s={NOISE_SIGMA}')
    print(f'  Device     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    print()

    lpips_fn = None
    if not args.no_lpips:
        import lpips
        lpips_fn = lpips.LPIPS(net='alex')

    upsampler = build_upsampler(FINETUNED_WEIGHTS, args.tile)

    writers = {}
    if not args.no_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        for tag in ('reference', 'degraded', 'restored'):
            path = DEMO_DIR / f'{video_path.stem}_x{f}_{tag}.mp4'
            writers[tag] = (path, cv2.VideoWriter(str(path), fourcc, fps, (w, h)))

    scores = {'bicubic': [], 'esrgan_ft': []}
    for _ in tqdm(range(args.frames), unit='frame'):
        ok, frame = cap.read()
        if not ok:
            break
        ref = frame[y:y + h, x:x + w]

        lr = degrade_by(ref, f)
        bicubic = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
        restored, _ = upsampler.enhance(lr, outscale=f)

        for name, pred in (('bicubic', bicubic), ('esrgan_ft', restored)):
            row = [compute_psnr(ref, pred), compute_ssim(ref, pred)]
            row.append(compute_lpips(lpips_fn, ref, pred) if lpips_fn else float('nan'))
            scores[name].append(row)

        if writers:
            # nearest-neighbour keeps the degraded input honest on screen
            writers['reference'][1].write(ref)
            writers['degraded'][1].write(
                cv2.resize(lr, (w, h), interpolation=cv2.INTER_NEAREST))
            writers['restored'][1].write(restored)

    cap.release()
    for path, wr in writers.values():
        wr.release()

    print(f'\n{"Method":<24}{"PSNR":>10}{"SSIM":>10}{"LPIPS":>10}')
    print('-' * 54)
    for name, label in (('bicubic', f'Bicubic x{f}'), ('esrgan_ft', f'ESRGAN fine-tuned x{f}')):
        m = np.nanmean(np.array(scores[name]), axis=0)
        print(f'{label:<24}{m[0]:>10.2f}{m[1]:>10.4f}{m[2]:>10.4f}')

    if writers:
        print(f'\nWritten to {DEMO_DIR}:')
        for path, _ in writers.values():
            print(f'  {path.name}  ({path.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
