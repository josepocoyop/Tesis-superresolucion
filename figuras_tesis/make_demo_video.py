"""
Demo clips for the thesis presentation.

Takes a stretch of consecutive frames from one of the source videos, applies
the same synthetic degradation used in step2, runs the fine-tuned ESRGAN on
every frame and writes two mp4 files: the degraded input and the restored
output. Unlike the 60 frames sampled by step2 (which are spread over the whole
video and would play as a timelapse), these frames are consecutive, so the
clips play at the original speed.

The LR clip is written at the same pixel size as the output using
nearest-neighbour upscaling, so both videos play side by side at the same size
and the input keeps its blocky look instead of being smoothed by the player.

Usage:
    python make_demo_video.py                       # indoor_tienda, 180 frames
    python make_demo_video.py --video day_street.mp4 --frames 120 --start 300
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'pipeline'))
from config import RAW_VIDEO_DIR, OUTPUT_DIR, SCALE_FACTOR, ESRGAN_WEIGHTS
from step2_prepare_frames import degrade

DEMO_DIR = OUTPUT_DIR / 'demo_videos'
FINETUNED_WEIGHTS = ESRGAN_WEIGHTS.parent / 'RealESRGAN_x4plus_finetuned.pth'


def build_upsampler(weights, tile):
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=SCALE_FACTOR)
    return RealESRGANer(
        scale=SCALE_FACTOR,
        model_path=str(weights),
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available(),
    )


def read_consecutive(video_path, start, count):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    frames = []
    for _ in range(count):
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        frames.append(frame[:h - h % SCALE_FACTOR, :w - w % SCALE_FACTOR])
    cap.release()
    return frames, fps, total


def main():
    parser = argparse.ArgumentParser(description='Demo clips: degraded input vs fine-tuned ESRGAN')
    parser.add_argument('--video', default='indoor_tienda.mp4')
    parser.add_argument('--start', type=int, default=0, help='first frame index')
    parser.add_argument('--frames', type=int, default=180, help='number of consecutive frames')
    parser.add_argument('--tile', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0, help='seed for the degradation noise')
    args = parser.parse_args()

    import cv2
    import numpy as np
    import torch
    from tqdm import tqdm

    video_path = RAW_VIDEO_DIR / args.video
    if not video_path.exists():
        print(f'ERROR: {video_path} not found')
        sys.exit(1)
    if not FINETUNED_WEIGHTS.exists():
        print(f'ERROR: {FINETUNED_WEIGHTS} not found. Run step6_finetune_esrgan.py first.')
        sys.exit(1)

    np.random.seed(args.seed)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('Demo clips')
    print('=' * 60)

    frames, fps, total = read_consecutive(video_path, args.start, args.frames)
    if not frames:
        print('ERROR: no frames read')
        sys.exit(1)

    h, w = frames[0].shape[:2]
    print(f'  Source     : {video_path.name}  ({total} frames, {fps:.1f} fps)')
    print(f'  Clip       : frames {args.start}-{args.start + len(frames) - 1}  ({len(frames) / fps:.1f} s)')
    print(f'  HR size    : {w}x{h}')
    print(f'  LR size    : {w // SCALE_FACTOR}x{h // SCALE_FACTOR}')
    print(f'  Device     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    print()

    upsampler = build_upsampler(FINETUNED_WEIGHTS, args.tile)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    stem = video_path.stem
    lr_path = DEMO_DIR / f'{stem}_lr_input.mp4'
    sr_path = DEMO_DIR / f'{stem}_esrgan_finetuned.mp4'
    lr_writer = cv2.VideoWriter(str(lr_path), fourcc, fps, (w, h))
    sr_writer = cv2.VideoWriter(str(sr_path), fourcc, fps, (w, h))

    for frame in tqdm(frames, unit='frame'):
        lr = degrade(frame)
        sr, _ = upsampler.enhance(lr, outscale=SCALE_FACTOR)
        # nearest-neighbour so the player does not smooth the degraded input
        lr_big = cv2.resize(lr, (w, h), interpolation=cv2.INTER_NEAREST)
        lr_writer.write(lr_big)
        sr_writer.write(sr)

    lr_writer.release()
    sr_writer.release()

    print(f'\nWritten to {DEMO_DIR}:')
    for p in (lr_path, sr_path):
        print(f'  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
