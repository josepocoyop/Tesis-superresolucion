"""
Enhancement of a real recording with the fine-tuned ESRGAN.

Unlike make_demo_video.py, this script does NOT degrade the input: it feeds the
video exactly as it comes out of the camera and writes the enhanced result.
Use it on footage that was never part of the training set.

With --roi it crops a region of interest before the network, which is the
forensic use case: a small area of the frame (a face, a plate, a doorway) blown
up to a size where it can actually be read.

Both clips are written at the same pixel size so they can be compared frame by
frame. The input is enlarged with nearest-neighbour, which does not invent any
detail, so nothing in the comparison favours the network artificially.

Usage:
    python enhance_video.py --video test_video.mp4 --start 120 --frames 150
    python enhance_video.py --video test_video.mp4 --roi 280 150 480 270 --outscale 4
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'pipeline'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RAW_VIDEO_DIR, DATASET_DIR, OUTPUT_DIR
from make_demo_video import build_upsampler, FINETUNED_WEIGHTS

DEMO_DIR = OUTPUT_DIR / 'demo_videos'
# clips used only for demonstrations, kept out of raw_videos/ so step2 does not
# pick them up as part of the dataset
DEMO_INPUT_DIR = DATASET_DIR / 'demo_input'


def resolve_video(name):
    """Accepts a full path, or a file name in demo_input/ or raw_videos/."""
    for candidate in (Path(name), DEMO_INPUT_DIR / name, RAW_VIDEO_DIR / name):
        if candidate.is_file():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(description='Enhance a real clip with the fine-tuned ESRGAN')
    parser.add_argument('--video', default='test_video.mp4')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--frames', type=int, default=150)
    parser.add_argument('--outscale', type=float, default=2,
                        help='output size relative to the input (network is x4)')
    parser.add_argument('--roi', nargs=4, type=int, default=None, metavar=('X', 'Y', 'W', 'H'),
                        help='crop this region before enhancing')
    parser.add_argument('--tile', type=int, default=512,
                        help='tile size, keeps VRAM bounded on large frames')
    parser.add_argument('--suffix', default='', help='added to the output file names')
    args = parser.parse_args()

    import cv2
    import torch
    from tqdm import tqdm

    video_path = resolve_video(args.video)
    if video_path is None:
        print(f'ERROR: {args.video} not found in {DEMO_INPUT_DIR} or {RAW_VIDEO_DIR}')
        sys.exit(1)
    if not FINETUNED_WEIGHTS.exists():
        print(f'ERROR: {FINETUNED_WEIGHTS} not found. Run step6_finetune_esrgan.py first.')
        sys.exit(1)

    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    if args.roi:
        x, y, w, h = args.roi
        in_w, in_h = w, h
    else:
        x = y = 0
        in_w, in_h = src_w, src_h
    out_w = int(in_w * args.outscale)
    out_h = int(in_h * args.outscale)

    print('=' * 60)
    print('Enhancement of a real clip')
    print('=' * 60)
    print(f'  Source   : {video_path.name}  ({src_w}x{src_h}, {total} frames, {fps:.1f} fps)')
    print(f'  Clip     : frames {args.start}-{args.start + args.frames - 1}  ({args.frames / fps:.1f} s)')
    if args.roi:
        print(f'  ROI      : {in_w}x{in_h} at ({x}, {y})')
    print(f'  Input    : {in_w}x{in_h}   ->   output {out_w}x{out_h}  (x{args.outscale:g})')
    print(f'  Device   : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
    print()

    upsampler = build_upsampler(FINETUNED_WEIGHTS, args.tile)

    tag = args.suffix or ('roi' if args.roi else 'full')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    in_path = DEMO_DIR / f'{video_path.stem}_{tag}_input.mp4'
    out_path = DEMO_DIR / f'{video_path.stem}_{tag}_enhanced.mp4'
    in_writer = cv2.VideoWriter(str(in_path), fourcc, fps, (out_w, out_h))
    out_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))

    written = 0
    for _ in tqdm(range(args.frames), unit='frame'):
        ok, frame = cap.read()
        if not ok:
            break
        if args.roi:
            frame = frame[y:y + h, x:x + w]

        enhanced, _ = upsampler.enhance(frame, outscale=args.outscale)
        enlarged = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        in_writer.write(enlarged)
        out_writer.write(enhanced)
        written += 1

    cap.release()
    in_writer.release()
    out_writer.release()

    print(f'\n{written} frames written to {DEMO_DIR}:')
    for p in (in_path, out_path):
        print(f'  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
