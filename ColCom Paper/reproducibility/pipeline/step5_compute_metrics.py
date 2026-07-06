"""
Step 5 — Image quality metrics: PSNR, SSIM, LPIPS.

Computes per-frame and aggregate metrics for:
  - Bicubic ×4 upscaling  (baseline)
  - ESRGAN ×4             (proposed, spatial only)

Writes results to output/metrics/metrics.csv and prints a formatted summary.

Note on PSNR/SSIM vs LPIPS:
  GAN-based methods like ESRGAN optimise perceptual quality at the cost of
  pixel-level fidelity (Blau & Michaeli, CVPR 2018). ESRGAN may therefore
  yield lower PSNR/SSIM than bicubic while LPIPS (a perceptual metric) shows
  clear improvement — this is the expected and scientifically valid result.

Usage:
    python step5_compute_metrics.py
    python step5_compute_metrics.py --no-lpips   # skip LPIPS (faster)
"""

import sys
import subprocess
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import FRAMES_HR_DIR, FRAMES_BIC_DIR, FRAMES_SR_DIR, METRICS_DIR


def install_deps():
    pkgs = ['scikit-image', 'lpips', 'torch', 'torchvision', 'numpy', 'pandas', 'tqdm']
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + pkgs, check=True)


def compute_psnr(ref, pred):
    from skimage.metrics import peak_signal_noise_ratio
    return float(peak_signal_noise_ratio(ref, pred, data_range=255))


def compute_ssim(ref, pred):
    from skimage.metrics import structural_similarity
    return float(structural_similarity(ref, pred, data_range=255, channel_axis=2))


def build_lpips_fn():
    import lpips
    return lpips.LPIPS(net='alex')


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


def match_frames(hr_dir: Path, other_dir: Path) -> list:
    hr_names    = {p.name: p for p in hr_dir.glob('*.png')}
    other_names = {p.name: p for p in other_dir.glob('*.png')}
    common      = sorted(hr_names.keys() & other_names.keys())
    return [(hr_names[n], other_names[n]) for n in common]


def main():
    parser = argparse.ArgumentParser(description='Step 5: Compute quality metrics')
    parser.add_argument('--no-lpips', action='store_true',
                        help='Skip LPIPS computation (runs on CPU, can be slow)')
    args = parser.parse_args()

    install_deps()
    import cv2
    import numpy as np
    import pandas as pd
    from tqdm import tqdm

    print('=' * 60)
    print('STEP 5 — Image quality metrics (PSNR / SSIM / LPIPS)')
    print('=' * 60)

    lpips_fn = None
    if not args.no_lpips:
        print('Loading LPIPS model (AlexNet)...')
        lpips_fn = build_lpips_fn()
        print('LPIPS ready.')
    else:
        print('LPIPS disabled (--no-lpips).')
    print()

    bic_pairs = match_frames(FRAMES_HR_DIR, FRAMES_BIC_DIR)
    sr_pairs  = match_frames(FRAMES_HR_DIR, FRAMES_SR_DIR)

    if not bic_pairs:
        print(f'ERROR: No matching frames in {FRAMES_BIC_DIR}. Run step2 first.')
        sys.exit(1)
    if not sr_pairs:
        print(f'ERROR: No matching frames in {FRAMES_SR_DIR}. Run step3 first.')
        sys.exit(1)

    # Use only frames present in both methods for fair comparison
    common_names = {p[1].name for p in bic_pairs} & {p[1].name for p in sr_pairs}
    bic_pairs = [(h, p) for h, p in bic_pairs if p.name in common_names]
    sr_pairs  = [(h, p) for h, p in sr_pairs  if p.name in common_names]

    print(f'Evaluating {len(bic_pairs)} frames...\n')

    records = []
    for (hr_path, bic_path), (_, sr_path) in tqdm(zip(bic_pairs, sr_pairs), total=len(bic_pairs), unit='frame'):
        hr  = cv2.imread(str(hr_path))
        bic = cv2.imread(str(bic_path))
        sr  = cv2.imread(str(sr_path))

        if hr is None or bic is None or sr is None:
            continue

        # Ensure same spatial size (crop to minimum)
        h  = min(hr.shape[0], bic.shape[0], sr.shape[0])
        w  = min(hr.shape[1], bic.shape[1], sr.shape[1])
        hr  = hr[:h, :w]
        bic = bic[:h, :w]
        sr  = sr[:h, :w]

        row = {
            'frame':      hr_path.name,
            'psnr_bic':   compute_psnr(hr, bic),
            'ssim_bic':   compute_ssim(hr, bic),
            'psnr_sr':    compute_psnr(hr, sr),
            'ssim_sr':    compute_ssim(hr, sr),
        }
        if lpips_fn is not None:
            row['lpips_bic'] = compute_lpips(lpips_fn, hr, bic)
            row['lpips_sr']  = compute_lpips(lpips_fn, hr, sr)
        records.append(row)

    df = pd.DataFrame(records)
    csv_path = METRICS_DIR / 'metrics.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nPer-frame metrics saved → {csv_path}')

    # Summary table
    print()
    print('=' * 60)
    print('AGGREGATE RESULTS')
    print('=' * 60)
    print(f'{"Method":<30} {"PSNR (dB)":>12} {"SSIM":>10}', end='')
    if lpips_fn is not None:
        print(f' {"LPIPS":>10}', end='')
    print()
    print('-' * (52 + (12 if lpips_fn else 0)))

    def fmt(col):
        m = df[col].mean()
        s = df[col].std()
        return f'{m:.4f} ± {s:.4f}'

    rows = [
        ('Bicúbica ×4 (referencia)',   'psnr_bic', 'ssim_bic', 'lpips_bic'),
        ('ESRGAN ×4 (propuesto)',      'psnr_sr',  'ssim_sr',  'lpips_sr'),
    ]
    for label, pk, sk, lk in rows:
        psnr_s = fmt(pk)
        ssim_s = fmt(sk)
        print(f'{label:<30} {psnr_s:>22} {ssim_s:>15}', end='')
        if lpips_fn is not None and lk in df.columns:
            print(f' {fmt(lk):>18}', end='')
        print()

    print()
    print('NOTE: ESRGAN is a GAN-based perceptual model. If PSNR(ESRGAN) <')
    print('      PSNR(bicubic), this is the known perception-distortion tradeoff')
    print('      (Blau & Michaeli, CVPR 2018). Check LPIPS for perceptual quality.')
    print()

    # Save a clean summary JSON for step7
    summary = {
        'n_frames':     len(df),
        'psnr_bic':     float(df['psnr_bic'].mean()),
        'ssim_bic':     float(df['ssim_bic'].mean()),
        'psnr_sr':      float(df['psnr_sr'].mean()),
        'ssim_sr':      float(df['ssim_sr'].mean()),
    }
    if lpips_fn is not None:
        summary['lpips_bic'] = float(df['lpips_bic'].mean())
        summary['lpips_sr']  = float(df['lpips_sr'].mean())

    import json
    summary_path = METRICS_DIR / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Summary JSON   → {summary_path}')
    print('\nStep 5 complete. Run step6_generate_figures.py next.')


if __name__ == '__main__':
    main()
