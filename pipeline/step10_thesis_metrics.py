"""
Step 10 — Thesis metrics: all methods, overall and per lighting condition.

Computes PSNR, SSIM and LPIPS against the HR ground truth for every method
whose output directory exists:
    output/bicubic/        Bicubic x4          (baseline)
    output/sr/             ESRGAN x4           (pretrained, GAN)
    output/swinir/         SwinIR x4           (Transformer)     [step8]
    output/sr_finetuned/   ESRGAN x4 fine-tuned                  [step9]

Results are grouped by lighting condition parsed from the frame filename
prefix (day_, night_, indoor_; see config.condition_of). To satisfy the
thesis objective of a multi-condition dataset, name the source videos
accordingly, e.g. day_street.mp4, night_parking.mp4, indoor_lobby.mp4.

Outputs:
    output/metrics/thesis_metrics.csv     per-frame, per-method
    output/metrics/thesis_summary.json    aggregate overall + per condition
    (prints a LaTeX-ready table for the thesis document)

Usage:
    python step10_thesis_metrics.py
    python step10_thesis_metrics.py --no-lpips
"""

import sys
import subprocess
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    FRAMES_HR_DIR, FRAMES_BIC_DIR, FRAMES_SR_DIR,
    FRAMES_SWINIR_DIR, FRAMES_SR_FT_DIR, METRICS_DIR,
    condition_of,
)

METHODS = [
    # (column key, display label, output directory)
    ('bic',    'Bicúbica ×4 (referencia)',   FRAMES_BIC_DIR),
    ('sr',     'ESRGAN ×4 (preentrenado)',   FRAMES_SR_DIR),
    ('swinir', 'SwinIR ×4 (Transformer)',    FRAMES_SWINIR_DIR),
    ('sr_ft',  'ESRGAN ×4 (fine-tuned)',     FRAMES_SR_FT_DIR),
]


def install_deps():
    pkgs = ['scikit-image', 'lpips', 'torch', 'torchvision', 'numpy', 'pandas', 'tqdm']
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + pkgs, check=True)


def compute_psnr(ref, pred):
    from skimage.metrics import peak_signal_noise_ratio
    return float(peak_signal_noise_ratio(ref, pred, data_range=255))


def compute_ssim(ref, pred):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(ref, pred, data_range=255, channel_axis=2))


def compute_lpips(fn, ref_bgr, pred_bgr):
    import torch
    import numpy as np
    import cv2

    def to_tensor(img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        d = fn(to_tensor(ref_bgr), to_tensor(pred_bgr))
    return float(d)


def main():
    parser = argparse.ArgumentParser(description='Step 10: Thesis metrics (all methods)')
    parser.add_argument('--no-lpips', action='store_true',
                        help='Skip LPIPS computation (runs on CPU, can be slow)')
    args = parser.parse_args()

    install_deps()
    import cv2
    import pandas as pd
    from tqdm import tqdm

    print('=' * 60)
    print('STEP 10 — Thesis metrics (all methods, per condition)')
    print('=' * 60)

    active = [(k, lbl, d) for k, lbl, d in METHODS
              if d.exists() and any(d.glob('*.png'))]
    if len(active) < 2:
        print('ERROR: Need at least the bicubic baseline plus one SR method.')
        print('       Run steps 2, 3 (and optionally 8, 9) first.')
        sys.exit(1)
    print('Methods found:')
    for k, lbl, d in active:
        print(f'  {lbl:<32} {d}')
    missing = [lbl for k, lbl, d in METHODS if (k, lbl, d) not in active]
    for lbl in missing:
        print(f'  (missing, skipped)               {lbl}')
    print()

    lpips_fn = None
    if not args.no_lpips:
        print('Loading LPIPS model (AlexNet)...')
        import lpips
        lpips_fn = lpips.LPIPS(net='alex')
        print('LPIPS ready.\n')

    hr_frames = {p.name: p for p in FRAMES_HR_DIR.glob('*.png')}
    # Only frames present in every active method, for a fair comparison
    common = set(hr_frames)
    for _, _, d in active:
        common &= {p.name for p in d.glob('*.png')}
    common = sorted(common)
    if not common:
        print('ERROR: No frames common to all methods.')
        sys.exit(1)
    print(f'Evaluating {len(common)} frames x {len(active)} methods...\n')

    records = []
    for name in tqdm(common, unit='frame'):
        hr = cv2.imread(str(hr_frames[name]))
        if hr is None:
            continue
        row = {'frame': name, 'condition': condition_of(name)}
        for key, _, d in active:
            pred = cv2.imread(str(d / name))
            if pred is None:
                continue
            h = min(hr.shape[0], pred.shape[0])
            w = min(hr.shape[1], pred.shape[1])
            ref, p = hr[:h, :w], pred[:h, :w]
            row[f'psnr_{key}'] = compute_psnr(ref, p)
            row[f'ssim_{key}'] = compute_ssim(ref, p)
            if lpips_fn is not None:
                row[f'lpips_{key}'] = compute_lpips(lpips_fn, ref, p)
        records.append(row)

    df = pd.DataFrame(records)
    csv_path = METRICS_DIR / 'thesis_metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nPer-frame metrics saved → {csv_path}')

    def aggregate(sub):
        out = {'n_frames': len(sub)}
        for key, _, _ in active:
            for m in ['psnr', 'ssim', 'lpips']:
                col = f'{m}_{key}'
                if col in sub.columns:
                    out[col] = float(sub[col].mean())
        return out

    summary = {'overall': aggregate(df), 'by_condition': {}}
    for cond, sub in df.groupby('condition'):
        summary['by_condition'][cond] = aggregate(sub)

    summary_path = METRICS_DIR / 'thesis_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Summary JSON → {summary_path}\n')

    # Console table
    def print_block(title, agg):
        print(f'--- {title} (n={agg["n_frames"]}) ---')
        hdr = f'{"Método":<32} {"PSNR (dB)":>10} {"SSIM":>8}'
        if lpips_fn is not None:
            hdr += f' {"LPIPS":>8}'
        print(hdr)
        for key, lbl, _ in active:
            line = f'{lbl:<32} {agg.get(f"psnr_{key}", float("nan")):>10.2f} ' \
                   f'{agg.get(f"ssim_{key}", float("nan")):>8.4f}'
            if lpips_fn is not None:
                line += f' {agg.get(f"lpips_{key}", float("nan")):>8.4f}'
            print(line)
        print()

    print('=' * 60)
    print('AGGREGATE RESULTS')
    print('=' * 60)
    print_block('General', summary['overall'])
    for cond, agg in summary['by_condition'].items():
        print_block(f'Condición: {cond}', agg)

    # LaTeX table for the thesis document
    print('=' * 60)
    print('LATEX TABLE (paste into the thesis)')
    print('=' * 60)
    cols = 'lccc' if lpips_fn is not None else 'lcc'
    print(f'\\begin{{tabular}}{{{cols}}}')
    print('\\hline')
    hdr = 'Método & PSNR (dB) $\\uparrow$ & SSIM $\\uparrow$'
    if lpips_fn is not None:
        hdr += ' & LPIPS $\\downarrow$'
    print(hdr + ' \\\\')
    print('\\hline')
    agg = summary['overall']
    for key, lbl, _ in active:
        line = f'{lbl} & {agg.get(f"psnr_{key}", 0):.2f} & {agg.get(f"ssim_{key}", 0):.4f}'
        if lpips_fn is not None:
            line += f' & {agg.get(f"lpips_{key}", 0):.4f}'
        print(line + ' \\\\')
    print('\\hline')
    print('\\end{tabular}')

    print('\nStep 10 complete. Run step11_thesis_figures.py next.')


if __name__ == '__main__':
    main()
