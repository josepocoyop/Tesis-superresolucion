"""
Step 6 — Publication-quality figure generation.

Generates all paper figures at 300 DPI, IEEE IEEEtran two-column width (7.16 in).
Saves everything directly into ColCom Paper/figures/ so the LaTeX source can
include them without any manual copying.

Figures produced:
  fig4_comparison.png  — 4-panel: LR input | Bicubic | ESRGAN | HR reference
  fig5_zoom.png        — ROI zoom: full frames + patch detail (2×3 grid)
  fig6_rife.png        — Temporal interpolation: frame t | RIFE t+0.5 | frame t+1
  fig7_metrics.png     — Bar chart: PSNR and SSIM for both methods (single column)

Usage:
    python step6_generate_figures.py
    python step6_generate_figures.py --frame-idx 12   # use a specific frame
    python step6_generate_figures.py --no-rife         # skip Figure 6 if Step 4 was skipped
"""

import sys
import subprocess
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    FRAMES_HR_DIR, FRAMES_LR_DIR, FRAMES_BIC_DIR, FRAMES_SR_DIR,
    FRAMES_RIFE_DIR, METRICS_DIR, FIGURES_DIR,
    FIG_SINGLE_W, FIG_DOUBLE_W, FIG_DPI,
    SCALE_FACTOR,
)


def install_deps():
    pkgs = ['matplotlib', 'opencv-python', 'numpy', 'pandas']
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + pkgs, check=True)


def setup_mpl():
    import matplotlib
    matplotlib.rcParams.update({
        'font.family':       'serif',
        'font.serif':        ['Times New Roman', 'DejaVu Serif'],
        'font.size':         8,
        'axes.titlesize':    8,
        'axes.labelsize':    8,
        'xtick.labelsize':   7,
        'ytick.labelsize':   7,
        'figure.dpi':        FIG_DPI,
        'savefig.dpi':       FIG_DPI,
        'savefig.bbox':      'tight',
        'savefig.pad_inches': 0.03,
        'axes.linewidth':    0.5,
    })


def load_image(path):
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Cannot read: {path}')
    return img


def bgr_to_rgb(img):
    import cv2
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def nn_upsample(img, scale):
    import cv2
    h, w = img.shape[:2]
    return cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)


def select_best_frame(hr_dir: Path, sr_dir: Path) -> str:
    """Return the frame name with highest edge density in the HR image (most interesting scene)."""
    import cv2
    import numpy as np

    candidates = sorted(hr_dir.glob('*.png'))
    if not candidates:
        raise FileNotFoundError(f'No PNG files in {hr_dir}')

    best_name  = candidates[0].name
    best_score = -1.0

    for hr_path in candidates[:40]:   # check first 40 frames to pick the richest
        img  = cv2.imread(str(hr_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        gx   = cv2.Sobel(img, cv2.CV_64F, 1, 0)
        gy   = cv2.Sobel(img, cv2.CV_64F, 0, 1)
        score = float(np.sqrt(gx**2 + gy**2).mean())
        if score > best_score:
            best_score = score
            best_name  = hr_path.name

    return best_name


def find_best_roi(img, patch_w=240, patch_h=135):
    """Slide a window over the image; return (x, y, w, h) with highest edge energy."""
    import cv2
    import numpy as np

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx    = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy    = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    mag   = np.sqrt(gx**2 + gy**2)

    h, w  = mag.shape
    step  = 32
    best  = 0.0
    bx, by = 0, 0

    for y in range(0, h - patch_h, step):
        for x in range(0, w - patch_w, step):
            s = float(mag[y:y + patch_h, x:x + patch_w].mean())
            if s > best:
                best = s
                bx, by = x, y

    return bx, by, patch_w, patch_h


def add_border(ax, color='#aaaaaa', lw=0.5):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(lw)
        spine.set_edgecolor(color)


def figure4_comparison(hr, lr, bic, sr,
                        psnr_bic, ssim_bic, psnr_sr, ssim_sr,
                        lpips_bic=None, lpips_sr=None):
    import matplotlib.pyplot as plt
    import cv2

    lr_display = nn_upsample(lr, SCALE_FACTOR)
    h = min(hr.shape[0], bic.shape[0], sr.shape[0], lr_display.shape[0])
    w = min(hr.shape[1], bic.shape[1], sr.shape[1], lr_display.shape[1])
    lr_display = lr_display[:h, :w]
    bic = bic[:h, :w]
    sr  = sr[:h, :w]
    hr  = hr[:h, :w]

    panels = [
        (lr_display, 'Entrada LR\n(320×180, NN escalada)', 'black', False),
        (bic,        f'Bicúbico ×4\nPSNR: {psnr_bic:.2f} dB  SSIM: {ssim_bic:.3f}', '#444444', False),
        (sr,         f'ESRGAN ×4 (propuesto)\nPSNR: {psnr_sr:.2f} dB  SSIM: {ssim_sr:.3f}', '#b22222', True),
        (hr,         'Referencia HR\n(1280×720)', 'black', False),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(FIG_DOUBLE_W, 2.3))
    for ax, (img, title, color, bold) in zip(axes, panels):
        ax.imshow(bgr_to_rgb(img))
        ax.set_title(title, fontsize=6.5, color=color, pad=3,
                     fontweight='bold' if bold else 'normal', linespacing=1.4)
        ax.axis('off')
        add_border(ax)

    fig.tight_layout(pad=0.2, w_pad=0.4)
    return fig


def figure5_zoom(hr, bic, sr, roi):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import cv2

    h = min(hr.shape[0], bic.shape[0], sr.shape[0])
    w = min(hr.shape[1], bic.shape[1], sr.shape[1])
    hr  = hr[:h, :w]
    bic = bic[:h, :w]
    sr  = sr[:h, :w]

    rx, ry, rw, rh = roi

    fig, axes = plt.subplots(2, 3, figsize=(FIG_DOUBLE_W, 3.8))

    # Top row: full frames with ROI rectangle
    for ax, img, title, color in zip(
        axes[0],
        [bic, sr, hr],
        ['Bicúbico ×4', 'ESRGAN ×4 (propuesto)', 'Referencia HR'],
        ['#444444', '#b22222', 'black'],
    ):
        ax.imshow(bgr_to_rgb(img))
        rect = mpatches.FancyBboxPatch(
            (rx, ry), rw, rh,
            linewidth=1.5, edgecolor='red', facecolor='none',
            boxstyle='square,pad=0',
        )
        ax.add_patch(rect)
        ax.set_title(title, fontsize=7, color=color,
                     fontweight='bold' if color == '#b22222' else 'normal')
        ax.axis('off')
        add_border(ax)

    # Bottom row: zoomed patches
    for ax, img, title, color in zip(
        axes[1],
        [bic, sr, hr],
        ['Detalle — Bicúbico', 'Detalle — ESRGAN (propuesto)', 'Detalle — HR'],
        ['#444444', '#b22222', 'black'],
    ):
        patch = img[ry:ry + rh, rx:rx + rw]
        ax.imshow(bgr_to_rgb(patch))
        ax.set_title(title, fontsize=7, color=color,
                     fontweight='bold' if color == '#b22222' else 'normal')
        ax.axis('off')
        add_border(ax, color='red' if color == '#b22222' else '#aaaaaa', lw=1.0)

    fig.tight_layout(pad=0.25, h_pad=0.5, w_pad=0.4)
    return fig


def figure6_rife(frame_t, frame_interp, frame_t1):
    import matplotlib.pyplot as plt

    h = min(frame_t.shape[0], frame_interp.shape[0], frame_t1.shape[0])
    w = min(frame_t.shape[1], frame_interp.shape[1], frame_t1.shape[1])

    panels = [
        (frame_t[:h, :w],      'Fotograma $t$\n(original ESRGAN)',       'black',   False),
        (frame_interp[:h, :w], 'Fotograma $t\!+\!0.5$\n(RIFE interpolado)', '#b22222', True),
        (frame_t1[:h, :w],     'Fotograma $t\!+\!1$\n(original ESRGAN)', 'black',   False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(FIG_DOUBLE_W, 2.3))
    for ax, (img, title, color, bold) in zip(axes, panels):
        ax.imshow(bgr_to_rgb(img))
        ax.set_title(title, fontsize=7, color=color, pad=3,
                     fontweight='bold' if bold else 'normal', linespacing=1.4)
        ax.axis('off')
        add_border(ax, color='red' if bold else '#aaaaaa', lw=1.0 if bold else 0.5)

    fig.suptitle('Interpolación temporal RIFE ($\\times$2 fps)',
                 fontsize=8, y=1.01, fontstyle='italic')
    fig.tight_layout(pad=0.25, w_pad=0.4)
    return fig


def figure7_metrics(summary: dict):
    import matplotlib.pyplot as plt
    import numpy as np

    has_lpips = 'lpips_bic' in summary

    metrics   = ['PSNR (dB)', 'SSIM']
    bic_vals  = [summary['psnr_bic'], summary['ssim_bic']]
    sr_vals   = [summary['psnr_sr'],  summary['ssim_sr']]

    if has_lpips:
        metrics.append('LPIPS')
        bic_vals.append(summary['lpips_bic'])
        sr_vals.append(summary['lpips_sr'])

    x     = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(FIG_SINGLE_W, 2.4))
    bars_bic = ax.bar(x - width / 2, bic_vals, width, label='Bicúbico ×4',
                      color='#5b9bd5', edgecolor='white', linewidth=0.5)
    bars_sr  = ax.bar(x + width / 2, sr_vals,  width, label='ESRGAN ×4',
                      color='#b22222', edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=7)
    ax.set_ylabel('Valor de métrica', fontsize=7)
    ax.legend(fontsize=6.5, framealpha=0.9, edgecolor='#cccccc')
    ax.yaxis.grid(True, color='#eeeeee', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(length=2)

    # Value labels on bars
    for bar in list(bars_bic) + list(bars_sr):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01 * max(bic_vals + sr_vals),
                f'{h:.3f}', ha='center', va='bottom', fontsize=5.5)

    if has_lpips:
        ax.annotate('↓ mejor', xy=(x[-1], min(bic_vals[-1], sr_vals[-1]) * 0.7),
                    fontsize=5.5, color='#555555', ha='center')

    fig.tight_layout(pad=0.3)
    return fig


def main():
    parser = argparse.ArgumentParser(description='Step 6: Generate paper figures')
    parser.add_argument('--frame-idx', type=int, default=-1,
                        help='Use a specific frame index (0-based). -1 = auto-select.')
    parser.add_argument('--no-rife', action='store_true',
                        help='Skip Figure 6 (if Step 4 was not run)')
    args = parser.parse_args()

    install_deps()
    setup_mpl()
    import json
    import matplotlib.pyplot as plt
    import cv2

    print('=' * 60)
    print('STEP 6 — Publication-quality figure generation')
    print('=' * 60)

    # Load metrics summary
    summary_path = METRICS_DIR / 'summary.json'
    summary = {}
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        print(f'Metrics loaded from {summary_path.name}')
    else:
        print('WARNING: metrics/summary.json not found. Run step5 first for annotated metrics.')
        print('         Figures will be generated without PSNR/SSIM annotations.\n')

    # Select the representative frame
    hr_frames = sorted(FRAMES_HR_DIR.glob('*.png'))
    if not hr_frames:
        print(f'ERROR: No HR frames in {FRAMES_HR_DIR}. Run step2 first.')
        sys.exit(1)

    if args.frame_idx >= 0:
        frame_name = hr_frames[min(args.frame_idx, len(hr_frames) - 1)].name
        print(f'Using specified frame: {frame_name}')
    else:
        print('Auto-selecting richest frame (highest edge density)...')
        frame_name = select_best_frame(FRAMES_HR_DIR, FRAMES_SR_DIR)
        print(f'Selected: {frame_name}')

    # Load the four versions of the selected frame
    hr_path  = FRAMES_HR_DIR  / frame_name
    lr_path  = FRAMES_LR_DIR  / frame_name
    bic_path = FRAMES_BIC_DIR / frame_name
    sr_path  = FRAMES_SR_DIR  / frame_name

    for p, name in [(hr_path, 'HR'), (lr_path, 'LR'), (bic_path, 'Bicubic'), (sr_path, 'ESRGAN')]:
        if not p.exists():
            print(f'ERROR: {name} frame not found: {p}')
            sys.exit(1)

    hr  = load_image(hr_path)
    lr  = load_image(lr_path)
    bic = load_image(bic_path)
    sr  = load_image(sr_path)

    psnr_bic = summary.get('psnr_bic', float('nan'))
    ssim_bic = summary.get('ssim_bic', float('nan'))
    psnr_sr  = summary.get('psnr_sr',  float('nan'))
    ssim_sr  = summary.get('ssim_sr',  float('nan'))
    lpips_bic = summary.get('lpips_bic')
    lpips_sr  = summary.get('lpips_sr')

    roi = find_best_roi(hr)
    print(f'ROI for zoom: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}')
    print()

    # --- Figure 4 ---
    print('Generating Figure 4 (4-panel comparison)...')
    fig4 = figure4_comparison(hr, lr, bic, sr, psnr_bic, ssim_bic, psnr_sr, ssim_sr,
                               lpips_bic, lpips_sr)
    p4 = FIGURES_DIR / 'fig4_comparison.png'
    fig4.savefig(str(p4))
    plt.close(fig4)
    print(f'  Saved → {p4}')

    # --- Figure 5 ---
    print('Generating Figure 5 (ROI zoom detail)...')
    fig5 = figure5_zoom(hr, bic, sr, roi)
    p5 = FIGURES_DIR / 'fig5_zoom.png'
    fig5.savefig(str(p5))
    plt.close(fig5)
    print(f'  Saved → {p5}')

    # --- Figure 6 (RIFE) ---
    if not args.no_rife:
        rife_frames = sorted(FRAMES_RIFE_DIR.glob('*.png'))
        orig_frames  = [f for f in rife_frames if '_orig' in f.name]
        interp_frames = [f for f in rife_frames if '_rife' in f.name]

        if len(orig_frames) >= 2 and len(interp_frames) >= 1:
            print('Generating Figure 6 (RIFE temporal interpolation)...')
            ft  = load_image(orig_frames[0])
            fi  = load_image(interp_frames[0])
            ft1 = load_image(orig_frames[1])
            fig6 = figure6_rife(ft, fi, ft1)
            p6 = FIGURES_DIR / 'fig6_rife.png'
            fig6.savefig(str(p6))
            plt.close(fig6)
            print(f'  Saved → {p6}')
        else:
            print('Figure 6 skipped: no RIFE output frames found. Run step4 first,')
            print('or pass --no-rife to suppress this message.')
    else:
        print('Figure 6 skipped (--no-rife).')

    # --- Figure 7 (metrics bar chart) ---
    if summary:
        print('Generating Figure 7 (metrics bar chart)...')
        fig7 = figure7_metrics(summary)
        p7 = FIGURES_DIR / 'fig7_metrics.png'
        fig7.savefig(str(p7))
        plt.close(fig7)
        print(f'  Saved → {p7}')
    else:
        print('Figure 7 skipped (no metrics summary — run step5 first).')

    print()
    print('All figures saved to:')
    print(f'  {FIGURES_DIR}')
    print()
    print('Next: update main.tex image paths and run step7_compile_results.py')
    print('      to get the ready-to-paste LaTeX table.')


if __name__ == '__main__':
    main()
