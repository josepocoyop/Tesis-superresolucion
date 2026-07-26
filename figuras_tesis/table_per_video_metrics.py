"""
Per-video metric tables for the thesis annex.

step7 reports global and per-condition averages. This script breaks the same
per-frame CSV down by source video and adds the standard deviation, which is
what an annex with the complete results needs.

Reads  : output/metrics/thesis_metrics.csv
Writes : output/metrics/per_video_metrics.csv
Prints : one LaTeX table per metric, ready to paste

Usage:
    python table_per_video_metrics.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'pipeline'))
from config import METRICS_DIR

METHODS = [
    ('bic', 'Bicubico'),
    ('sr', 'ESRGAN preentrenado'),
    ('swinir', 'SwinIR'),
    ('sr_ft', 'ESRGAN con ajuste fino'),
]

METRICS = [
    ('psnr', 'PSNR (dB)', 3),
    ('ssim', 'SSIM', 4),
    ('lpips', 'LPIPS', 4),
]


def video_of(frame_name):
    """day_street_f000066.png -> day_street"""
    return frame_name.rsplit('_f', 1)[0]


def main():
    src = METRICS_DIR / 'thesis_metrics.csv'
    if not src.exists():
        print(f'ERROR: {src} not found. Run step7_compute_metrics.py first.')
        sys.exit(1)

    df = pd.read_csv(src)
    df['video'] = df['frame'].map(video_of)

    cols = [f'{m}_{k}' for m, _, _ in METRICS for k, _ in METHODS]
    grouped = df.groupby('video')[cols]
    mean, std = grouped.mean(), grouped.std()
    counts = df.groupby('video').size()

    out = mean.copy()
    for c in cols:
        out[f'{c}_std'] = std[c]
    out.insert(0, 'n_frames', counts)
    dst = METRICS_DIR / 'per_video_metrics.csv'
    out.to_csv(dst)

    videos = list(mean.index)
    print(f'Videos: {", ".join(videos)}')
    print(f'Frames: {", ".join(f"{v}={counts[v]}" for v in videos)}\n')

    for key, title, dec in METRICS:
        print('=' * 70)
        print(f'{title}  (media +/- desviacion estandar)')
        print('=' * 70)
        header = ' & '.join(['Metodo'] + videos)
        print(f'{header} \\\\')
        print('\\hline')
        for m, label in METHODS:
            cells = []
            for v in videos:
                mu = mean.loc[v, f'{key}_{m}']
                sd = std.loc[v, f'{key}_{m}']
                cells.append(f'{mu:.{dec}f} $\\pm$ {sd:.{dec}f}')
            print(f'{label} & ' + ' & '.join(cells) + ' \\\\')
        print()

    print(f'Saved -> {dst}')


if __name__ == '__main__':
    main()
